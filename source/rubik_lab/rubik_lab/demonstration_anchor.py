"""Alternating PPO and supervised rehearsal to retain a warm-started primitive.

This is an explicit training-only addition; runtime actor architecture is unchanged.
Only complete successful episodes enter rehearsal. Held-out grasps stay excluded.
"""
import h5py
import numpy as np
import torch


def attach_anchor(algorithm, path, excluded_grasps=(), steps=10, batch=256, lr=5e-5):
    device = next(algorithm.policy.parameters()).device
    with h5py.File(path, 'r') as f:
        uid = f['episode_uid'][:]
        success = np.unique(uid[f['primitive_done'][:].astype(bool)])
        for key in ('dropped','broken','cycle_failed'):
            if key in f:
                success = np.setdiff1d(success,uid[f[key][:].astype(bool)])
        keep = np.isin(uid,success) & ~np.isin(f['grasp_idx'][:],excluded_grasps)
        if not keep.any():
            raise ValueError('No successful training demonstration rows for rehearsal')
        observations = torch.tensor(f['policy_obs'][:][keep],device=device)
        targets = torch.tensor(f['policy_action'][:][keep],device=device)
    optimizer = torch.optim.Adam(algorithm.policy.actor.parameters(),lr=lr)
    ppo_update = algorithm.update
    def update_with_rehearsal():
        losses = ppo_update()
        total = 0.
        for _ in range(steps):
            ids = torch.randint(len(targets),(min(batch,len(targets)),),device=device)
            prediction = algorithm.policy.act_inference({'policy':observations[ids]})
            loss = (prediction-targets[ids]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(algorithm.policy.actor.parameters(),1.)
            optimizer.step()
            total += float(loss.detach())
        losses['demo_rehearsal'] = total/max(steps,1)
        return losses
    algorithm.update = update_with_rehearsal
    return {'rows':len(targets),'steps_per_ppo_update':steps,'batch':batch,'learning_rate':lr,
            'excluded_grasps':list(excluded_grasps), 'data':path}
