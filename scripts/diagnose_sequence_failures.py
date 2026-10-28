"""Summarize observed stage failure mechanisms; no causal attribution or relabeling."""
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np
p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out)
if out.exists():raise FileExistsError(out)
report=Path(a.report);raw=report.read_bytes();r=json.loads(raw)
with h5py.File(report.with_suffix('.h5')) as f:
 d={k:f[k][:] for k in ('episode_uid','sequence_index','twist_error','move_done','layer_done_latch','primitive_done','finger_ready_error','finger_speed','release_force','cycle_ready','cycle_failed')}
rows=[]
for uid in np.unique(d['episode_uid']):
 for stage in np.unique(d['sequence_index'][d['episode_uid']==uid]):
  mask=(d['episode_uid']==uid)&(d['sequence_index']==stage)
  ids=np.flatnonzero(mask);best=ids[np.argmin(abs(d['twist_error'][ids]))];last=ids[-1]
  rows.append({'uid':int(uid),'stage':int(stage),'full_ready_observed':bool(d['primitive_done'][ids].any()),
   'layer_ready_observed':bool(d['move_done'][ids].any()),'angle_within5deg_observed':bool((abs(d['twist_error'][ids])<np.deg2rad(5)).any()),
   'minimum_angle_error_deg':float(np.rad2deg(abs(d['twist_error'][best]))),
   'final_angle_error_deg':float(np.rad2deg(abs(d['twist_error'][last]))),
   'final_finger_ready_error_rad':float(d['finger_ready_error'][last]),'final_finger_speed_rad_s':float(d['finger_speed'][last]),
   'final_release_normal_force_N':float(d['release_force'][last]),'motion_failure_observed':bool(d['cycle_failed'][ids].any())})
summary={}
for stage in sorted(set(x['stage'] for x in rows)):
 rr=[x for x in rows if x['stage']==stage];failed=[x for x in rr if not x['full_ready_observed']]
 s={'observed':len(rr),'full_ready_observed':sum(x['full_ready_observed'] for x in rr),'not_full_ready':len(failed)}
 for key in ('layer_ready_observed','angle_within5deg_observed','motion_failure_observed'):
  s['not_full_ready_'+key]=sum(x[key] for x in failed)
 for key in ('minimum_angle_error_deg','final_angle_error_deg','final_finger_ready_error_rad','final_finger_speed_rad_s','final_release_normal_force_N'):
  s['not_full_ready_'+key+'_p10_p50_p90']=np.percentile([x[key] for x in failed],[10,50,90]).tolist() if failed else None
 summary[str(stage)]=s
result={'scope':'Observed trajectory diagnostics. Early aborts are censored; readiness can precede a later failure. Counts are not replacement acceptance outcomes.',
 'report':str(report),'report_sha256':hashlib.sha256(raw).hexdigest(),'summary':summary,'fragments':rows}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))
