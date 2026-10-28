"""Distill full successful teacher episodes into a tactile/proprioceptive GRU.
Offline action error is not an autonomous-success metric.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source/rubik_lab'))
from rubik_lab.blind_actor import BlindPrimitiveActor, student_features
p = argparse.ArgumentParser()
p.add_argument('data')
p.add_argument('--out', required=True)
p.add_argument('--epochs', type=int, default=200)
p.add_argument('--seed', type=int, default=123)
p.add_argument('--hidden', type=int, default=128)
p.add_argument('--batch', type=int, default=16)
p.add_argument('--split_by', choices=['grasp_idx','episode_uid'], default='grasp_idx')
p.add_argument('--device', default='cpu')
p.add_argument('--tactile_delay_augmentation', type=int, default=0)
a = p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=False)
with h5py.File(a.data,'r') as f:
    uid, gid, step = f['episode_uid'][:], f['grasp_idx'][:], f['step'][:]
    good = np.unique(uid[f['primitive_done'][:].astype(bool)])
    for key in ('dropped', 'broken', 'cycle_failed'):
        if key in f:
            good = np.setdiff1d(good, uid[f[key][:].astype(bool)])
    groups = np.unique(gid[np.isin(uid, good)]) if a.split_by=='grasp_idx' else good.copy()
    if len(groups) < 2:
        raise ValueError('Need successful complete episodes from multiple grasps')
    rng.shuffle(groups)
    val_groups = groups[:max(1,len(groups)//5)]
    obs = torch.from_numpy(f['policy_obs'][:])
    features = student_features(obs)
    actions = torch.from_numpy(f['policy_action'][:])
    cfg = json.loads(f.attrs['config_json'])
    execution_protocol=json.loads(f.attrs.get('execution_protocol_json','{}'))
episodes = []
for episode in good:
    rows = np.flatnonzero(uid == episode)
    rows = rows[np.argsort(step[rows])]
    if not np.array_equal(step[rows], np.arange(len(rows))):
        raise ValueError('Student training requires complete contiguous histories')
    episodes.append((features[rows], actions[rows], bool((gid[rows[0]] if a.split_by=='grasp_idx' else episode) in val_groups)))
train = [e for e in episodes if not e[2]]
val = [e for e in episodes if e[2]]
model = BlindPrimitiveActor(a.hidden).to(a.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
def batch_loss(batch, augment=False):
    lengths = torch.tensor([len(e[0]) for e in batch], device=a.device)
    inputs = [e[0].clone() for e in batch]
    if augment and a.tactile_delay_augmentation:
        for sequence in inputs:
            delay = int(rng.integers(a.tactile_delay_augmentation+1))
            if delay:
                for start,end in ((44,74),(118,148)):
                    source = sequence[:,start:end].clone()
                    sequence[:delay,start:end] = 0
                    sequence[delay:,start:end] = source[:-delay]
    x = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True).to(a.device)
    y = torch.nn.utils.rnn.pad_sequence([e[1] for e in batch], batch_first=True).to(a.device)
    pred, _ = model(x)
    mask = torch.arange(x.shape[1],device=a.device)[None] < lengths[:,None]
    error = (pred-y).square().mean(-1)
    # Equal episode influence, regardless of how many settle frames were observed.
    return ((error*mask).sum(-1)/lengths).mean()
report = {'dataset':a.data, 'dataset_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
          'train_episodes':len(train), 'val_episodes':len(val), 'split_by':a.split_by,'val_groups':val_groups.tolist(),
          'val_grasps':val_groups.tolist() if a.split_by=='grasp_idx' else [],
          'seed':a.seed, 'training_args':vars(a),'execution_protocol':execution_protocol,
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'actor_source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'source/rubik_lab/rubik_lab/blind_actor.py').read_bytes()).hexdigest(), 'inputs':'finger q/qd, delivered F/T, move ID, previous action',
          'scope':'offline successful-teacher imitation, must evaluate autonomous rollouts', 'epochs':[]}
best = float('inf')
for epoch in range(a.epochs):
    model.train()
    order = rng.permutation(len(train))
    total = 0.
    for begin in range(0,len(order),a.batch):
        batch = [train[i] for i in order[begin:begin+a.batch]]
        loss = batch_loss(batch, augment=True)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        total += float(loss.detach())*len(batch)
    model.eval()
    with torch.no_grad():
        val_loss = sum(float(batch_loss(val[i:i+a.batch]))*len(val[i:i+a.batch]) for i in range(0,len(val),a.batch))/len(val)
    report['epochs'].append({'epoch':epoch+1,'train_mse':total/len(train),'val_mse':val_loss})
    if val_loss < best:
        best = val_loss
        torch.save({'format':'blind_primitive_gru_v1','hidden':a.hidden,'model':model.state_dict(),
                    'feature_layout':'cycle_250_to_sensor_173_v1','epoch':epoch+1,'execution_protocol':execution_protocol},out/'model.pt')
    if epoch%20 == 0 or epoch==a.epochs-1:
        print(report['epochs'][-1],flush=True)
(out/'env.json').write_text(json.dumps(cfg,indent=2)+'\n')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('[blind_train] finished',out,flush=True)
