import argparse,h5py,numpy as np,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--reports',nargs='+',default=['U_tactile_event_v4','sequence_tactile_fullready_v5b','sequence_tactile_fullready_v7']);parser.add_argument('--out',default='outputs/primitive_cycle/premature_trigger_diagnosis.json');args=parser.parse_args()
out=Path(args.out)
if out.exists():raise FileExistsError(out)
sources=args.reports
reports={}
for name in sources:
 p=Path('outputs/primitive_cycle')/(name+'.json')
 if not p.exists():continue
 r=json.loads(p.read_text());bad=[e for e in r['episodes'] if e['reason']=='premature_trigger'];path=p.with_suffix('.h5')
 if not path.exists():continue
 with h5py.File(path,'r') as f:
  uid=f['episode_uid'][:];done=f['detector_done'][:].astype(bool);truth=f['primitive_done'][:].astype(bool);stage=f['sequence_index'][:] if 'sequence_index' in f else np.zeros(len(uid))
  idx=np.flatnonzero(done&~truth);count={}
  checks={'finger_position':f['finger_ready_error'][:]>.2,'finger_speed':f['finger_speed'][:]>1,'finger_not_released':f['release_force'][:]>.5,'layer_angle':abs(f['twist_error'][:])>np.deg2rad(5),'layer_speed':abs(f['twist_velocity'][:])>.25,'motion_limit':f['cycle_failed'][:].astype(bool),'instant_ready_but_dwell_incomplete':f['cycle_ready'][:].astype(bool)&~truth}
  for k,v in checks.items():count[k]=int(v[idx].sum())
  reports[name]={'recorded_premature_trigger_frames':len(idx),'report_premature_episodes':len(bad),'overlapping_causes':count,'events':[{'uid':int(uid[i]),'stage':int(stage[i]),'angle_error_deg':float(np.rad2deg(f['twist_error'][i])),'stable_steps':int(f['cycle_stable_steps'][i]),'ready_probability':float(f['detector_ready_probability'][i])} for i in idx]}
out.write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports))
