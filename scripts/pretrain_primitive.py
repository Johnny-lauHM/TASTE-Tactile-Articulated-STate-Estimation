"""Supervised warm start from complete successful diagnostic trajectories.

Uses privileged actor inputs and scripted teacher actions. Held-out action error
is only a fitting diagnostic; autonomous simulation evaluation is still required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn

p = argparse.ArgumentParser()
p.add_argument('data')
p.add_argument('--checkpoint', required=True)
p.add_argument('--out', required=True)
p.add_argument('--epochs', type=int, default=100)
p.add_argument('--seed', type=int, default=123)
p.add_argument('--device', default='cpu')
p.add_argument('--batch', type=int, default=256)
a = p.parse_args()
torch.manual_seed(a.seed)
np.random.seed(a.seed)
torch.set_num_threads(4)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
with h5py.File(a.data, 'r') as f:
    uid, gid = f['episode_uid'][:], f['grasp_idx'][:]
    success_ids = np.unique(uid[f['primitive_done'][:].astype(bool)])
    for key in ('dropped', 'broken', 'cycle_failed'):
        if key in f:
            success_ids = np.setdiff1d(success_ids, uid[f[key][:].astype(bool)])
    good = np.isin(uid, success_ids)
    groups = np.unique(gid[good])
    if len(groups) < 2:
        raise ValueError('Need complete successes from at least two grasps')
    rng = np.random.default_rng(a.seed)
    rng.shuffle(groups)
    val_groups = groups[:max(1, len(groups)//5)]
    train_mask = good & ~np.isin(gid, val_groups)
    val_mask = good & np.isin(gid, val_groups)
    x = torch.tensor(f['policy_obs'][:], device=a.device)
    y = torch.tensor(f['policy_action'][:], device=a.device)
    cfg = json.loads(f.attrs['config_json'])
    # A latch activates the teacher only on the FOLLOWING action. Use pre-action
    # policy observations (latch column -8 in this version), never post-step flags.
    returning = x[:, -8] > 0.5
ckpt = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
state = ckpt['model_state_dict']
if any('normalizer' in key for key in state):
    raise ValueError('This pretrainer requires an unnormalized actor checkpoint')
weights = sorted((int(k.split('.')[1]), v) for k,v in state.items() if k.startswith('actor.') and k.endswith('.weight'))
layers = []
for i, (_, weight) in enumerate(weights):
    layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
    if i < len(weights)-1:
        layers.append(nn.ELU())
actor = nn.Sequential(*layers).to(a.device)
actor.load_state_dict({k[len('actor.'):]: v for k,v in state.items() if k.startswith('actor.')})
train_ids = torch.tensor(np.flatnonzero(train_mask), device=a.device)
val_ids = torch.tensor(np.flatnonzero(val_mask), device=a.device)
opt = torch.optim.AdamW(actor.parameters(), lr=1e-4, weight_decay=1e-5)
# Equalize turn and return influence; otherwise many nearly identical hold frames
# can dominate the smaller turn phase.
w = torch.ones(len(x), device=a.device)
for phase in (False, True):
    count = (returning[train_ids] == phase).sum().clamp(min=1)
    w[returning == phase] = len(train_ids) / (2*count)
report = {'dataset': a.data, 'dataset_sha256': hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
          'warm_start': a.checkpoint, 'seed': a.seed, 'success_episodes': len(success_ids),
          'train_rows': len(train_ids), 'val_rows': len(val_ids), 'val_grasps': val_groups.tolist(),
          'train_grasps': groups[~np.isin(groups, val_groups)].tolist(),
          'scope': 'privileged actor imitation of scripted teacher, not autonomous success', 'epochs': []}
best = float('inf')
for epoch in range(a.epochs):
    actor.train()
    permutation = train_ids[torch.randperm(len(train_ids), device=a.device)]
    loss_sum = 0.0
    for ids in permutation.split(a.batch):
        err = (actor(x[ids]) - y[ids]).square().mean(-1)
        loss = (err*w[ids]).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
        loss_sum += float(loss.detach()) * len(ids)
    actor.eval()
    with torch.no_grad():
        val = float((actor(x[val_ids]).clamp(-1,1)-y[val_ids]).square().mean())
    report['epochs'].append({'epoch': epoch+1, 'train_mse': loss_sum/len(train_ids), 'val_mse': val})
    if val < best:
        best = val
        for key, value in actor.state_dict().items():
            state['actor.'+key] = value.detach().clone()
        state['log_std'] = torch.full_like(state['log_std'], np.log(0.15))
        ckpt['iter'] = 0
        ckpt['infos'] = {'pretraining': 'scripted_teacher_successes', 'best_epoch': epoch+1}
        torch.save(ckpt, out/'model.pt')
    if epoch % 10 == 0 or epoch == a.epochs-1:
        print(report['epochs'][-1], flush=True)
(out/'env.json').write_text(json.dumps(cfg, indent=2)+'\n')
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print('[pretrain] finished', out, 'best held-out action MSE', best, flush=True)
