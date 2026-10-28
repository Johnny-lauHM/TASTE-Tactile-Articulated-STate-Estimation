"""Learn one representative successful action sequence per initial finger posture.
This baseline uses initial proprioception and a clock, without online cube state.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
p=argparse.ArgumentParser()
p.add_argument('data')
p.add_argument('--out',required=True)
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
with h5py.File(a.data,'r') as f:
    uid,gid,step=f['episode_uid'][:],f['grasp_idx'][:],f['step'][:]
    good=np.unique(uid[f['primitive_done'][:].astype(bool)])
    for key in ('cycle_failed','broken','dropped'):
        if key in f:good=np.setdiff1d(good,uid[f[key][:].astype(bool)])
    actions=f['policy_action'][:]
    observations=f['policy_obs'][:]
    config=json.loads(f.attrs['config_json'])
prototypes,anchors,metadata=[],[],[]
for group in np.unique(gid[np.isin(uid,good)]):
    sequences,firsts,episode_ids=[],[],[]
    for episode in good:
        rows=np.flatnonzero(uid==episode)
        if gid[rows[0]]!=group:continue
        rows=rows[np.argsort(step[rows])]
        sequences.append(actions[rows]);firsts.append(observations[rows[0]]);episode_ids.append(int(episode))
    length=max(map(len,sequences))
    padded=np.stack([np.concatenate([s,np.repeat(s[-1:],length-len(s),axis=0)]) for s in sequences])
    flat=padded.reshape(len(padded),-1)
    distance=((flat[:,None]-flat[None])**2).mean(-1).mean(-1)
    best=int(distance.argmin())
    prototypes.append(torch.tensor(sequences[best]))
    anchors.append(np.r_[firsts[best][:22],firsts[best][102:124]])
    metadata.append({'grasp_idx':int(group),'episodes':len(sequences),'selected_medoid_episode':episode_ids[best],'steps':len(sequences[best])})
if not prototypes:raise ValueError('No complete successful trajectories')
lengths=torch.tensor([len(p) for p in prototypes])
max_length=int(lengths.max())
trajectories=torch.stack([torch.cat([p,p[-1:].repeat(max_length-len(p),1)]) for p in prototypes])
torch.save({'format':'sensor_prototype_trajectory_v1','anchors':torch.tensor(np.stack(anchors)),
            'trajectories':trajectories,'lengths':lengths},out/'model.pt')
(out/'env.json').write_text(json.dumps(config,indent=2)+'\n')
(out/'report.json').write_text(json.dumps({'data':a.data,'selection':'action-sequence medoid among complete successful training episodes',
    'scope':'initial finger posture and clock; requires fresh autonomous evaluation','prototypes':metadata},indent=2)+'\n')
print('[trajectory_train] finished',out,'prototypes',len(prototypes))
