"""Compare paired environment traces across fixed-quota evaluation trials.
This diagnoses divergence, without attributing it to a specific physics mechanism.
"""
import argparse,hashlib,json
from pathlib import Path
import h5py
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('dataset')
p.add_argument('--num_envs',type=int,required=True)
p.add_argument('--steps',type=int,default=20)
p.add_argument('--out',required=True)
a=p.parse_args()
with h5py.File(a.dataset,'r') as f:
    data={key:f[key][:] for key in ('episode_uid','env_id','step','policy_obs','policy_action','twist_error','tactile_ft')}
rows=[]
trial=data['episode_uid']//a.num_envs
for step in range(a.steps):
    indices=[np.flatnonzero((trial==t)&(data['step']==step)) for t in (0,1)]
    common=np.intersect1d(*[data['env_id'][idx] for idx in indices])
    if not len(common):continue
    paired=[]
    for idx in indices:
        lookup={int(data['env_id'][i]):int(i) for i in idx}
        paired.append(np.array([lookup[int(env)] for env in common]))
    left,right=paired
    obs=data['policy_obs']
    rows.append({'step':step,'paired_envs':len(common),
        'pre_obs_max_abs_difference':float(np.max(abs(obs[left]-obs[right]))),
        'action_mean_abs_difference':float(np.mean(abs(data['policy_action'][left]-data['policy_action'][right]))),
        'post_step_mean_angle_error_deg':[float(np.rad2deg(data['twist_error'][r]).mean()) for r in paired],
        'post_step_tactile_mean_abs_difference':float(np.mean(abs(data['tactile_ft'][left]-data['tactile_ft'][right])))})
report={'dataset':a.dataset,'sha256':hashlib.sha256(Path(a.dataset).read_bytes()).hexdigest(),
        'num_envs':a.num_envs,'comparison':'trial 0 versus trial 1, matched environment IDs',
        'limitation':'Divergence timing is evidence; it does not identify the physics or sensor mechanism.',
        'steps':rows}
Path(a.out).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
