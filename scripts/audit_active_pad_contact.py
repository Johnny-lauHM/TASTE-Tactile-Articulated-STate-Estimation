import argparse,hashlib,h5py,numpy as np,json,sys
from pathlib import Path
sys.path.insert(0,'source/rubik_lab')
from rubik_lab.moves import MOVE_INDEX
parser=argparse.ArgumentParser();parser.add_argument('--reports',nargs='+',default=['sequence_four_Up50','sequence_four_Up200']);parser.add_argument('--stage',type=int,default=None);parser.add_argument('--out',default='outputs/primitive_cycle/active_pad_contact_audit.json');args=parser.parse_args()
destination=Path(args.out)
if destination.exists():raise FileExistsError(destination)
sources=args.reports
report={'scope':'Privileged same-step contact audit. Active whole-finger force and pad-layer force come from pre-reset snapshots; no delivered-sensor latency comparison. Threshold0.5N; not a policy input.','sources':{}}
for name in sources:
 path=Path('outputs/primitive_cycle')/(name+'.h5')
 if not path.exists() or not path.with_suffix('.json').exists():continue
 with h5py.File(path,'r') as f:
  moves=f['move_id'][:];release=f['release_force'][:];pads=f['tactile_layer_force'][:];err=f['twist_error'][:]
  out={}
  for move,hand in [('U',1),("U'",0)]:
   sel=moves==MOVE_INDEX[move]
   if args.stage is not None:sel &= f['sequence_index'][:]==args.stage
   contact=sel&(release>.5);pad=pads[:,hand,1,1]
   out[move]={'whole_finger_contact_frames':int(contact.sum()),'pad_contact_during_whole_finger_fraction':float((pad[contact]>.5).mean()) if contact.any() else None,'whole_force_N_p50_p90':np.percentile(release[contact],[50,90]).tolist() if contact.any() else None,'pad_force_N_p50_p90':np.percentile(pad[contact],[50,90]).tolist() if contact.any() else None,'sum_pad_force_over_sum_whole_force':float(pad[contact].sum()/release[contact].sum()) if contact.any() else None}
  report['sources'][name]={'stage_filter':args.stage,'report_sha256':hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest(),'moves':out}
destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
