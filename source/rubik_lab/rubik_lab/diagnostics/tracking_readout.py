"""Exact-input IK convergence audit for continuation tracking captures."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from rubik_lab.flick import FlickPlanner
from rubik_lab.constants import FINGER_JOINTS
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
z=np.load(a.input.with_suffix('.npz'));ref=np.load(a.reference.with_suffix('.npz'))
equality={k:bool(np.array_equal(z[k],ref[k])) for k in ('trace','sensory','contact','poses','commands')}
planner=FlickPlanner('cuda:0');cache=np.load('cache/g70s_cycle_demo1_v3.npz');scale=json.load(open('logs/rsl_rl/rubik_primitive_cycle/U_blind_full/env.json'))['residual_finger_range']
cols=[i for i,n in enumerate(FINGER_JOINTS) if n.startswith('index_')]
summary={}
def stats(x):
 x=np.asarray(x);return {'median':float(np.median(x)),'p95':float(np.percentile(x,95)),'max':float(x.max())}
for h,(side,move) in enumerate((('right','U'),('left',"U'"))):
 m=z['tracking_index'][:,3]==h;r=torch.tensor(z['tracking'][m],device='cuda:0');idx=z['tracking_index'][m]
 if not len(r):continue
 wp,wr,q,seed,target,recorded=r[:,:3],r[:,3:12].reshape(-1,3,3),r[:,12:34],r[:,34:56],r[:,56:59],r[:,59:81]
 pad=lambda joints:planner.fk[side].pads(planner.fk[side].forward(wp,wr,joints))['index'][0]
 solved={it:planner.track(move,wp,wr,q,seed.clone(),target,iters=it) for it in (12,60)}
 distances={'actual_to_requested_mm':(pad(q)-target).norm(dim=-1)*1000,'recorded_ik_error_mm':(pad(recorded)-target).norm(dim=-1)*1000}
 for it,v in solved.items():distances['ik_'+str(it)+'_error_mm']=(pad(v)-target).norm(dim=-1)*1000
 grasp=torch.tensor(cache[side+'_fingers'][0],device='cuda:0');clipped=recorded.clone();clipped[:,cols]=grasp[cols]+(recorded[:,cols]-grasp[cols]).clamp(-scale,scale)
 distances['action_clipping_pad_shift_mm']=(pad(clipped)-pad(recorded)).norm(dim=-1)*1000
 distances={k:v.detach().cpu().numpy() for k,v in distances.items()}
 regions={}
 for name,mask in [('inside_cube_bounding_sphere',(target.norm(dim=-1)<=np.sqrt(3)*.035).cpu().numpy()),('outside_cube_bounding_sphere',(target.norm(dim=-1)>np.sqrt(3)*.035).cpu().numpy())]:
  if mask.any():regions[name]={'frames':int(mask.sum()),'metrics':{k:stats(v[mask]) for k,v in distances.items()}}
 summary[move]={'target_regions':regions,'frames':len(r),'recomputed12_q_max_difference_rad':float((solved[12]-recorded).abs().max()),'all_frames':{k:stats(v) for k,v in distances.items()},'rounds':{str(t):{k:stats(v[idx[:,0]==t]) for k,v in distances.items()} for t in np.unique(idx[:,0])}}
result={'scope':'Offline exact pre-step inputs. Actual-to-requested gap includes intended upcoming movement and loading; it is not alone actuator tracking failure. No physical acceptance claim.','measurement_arrays_equal_reference':equality,'recorded_solver_iterations':json.loads(a.input.with_suffix('.json').read_text()).get('tracking_iters',12),'target_region_note':'Geometric target inside/outside70mm cube bounding sphere, not proof of contact. Separates near-cube motion from distant withdrawal targets.','inputs':str(a.input),'reference':str(a.reference),'summary':summary}
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
