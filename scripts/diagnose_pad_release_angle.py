"""Compare pad and whole-finger last contact in successfully completed stages."""
import argparse,hashlib,json
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
if out.exists():raise FileExistsError(out)
path=Path(a.report);raw=path.read_bytes();report=json.loads(raw);dt=report['config']['sim']['dt']*report['config']['decimation']
with h5py.File(path.with_suffix('.h5')) as f:
 uid=f['episode_uid'][:];stage=f['sequence_index'][:];steps=f['step'][:];whole=f['release_force'][:];err=np.rad2deg(f['twist_error'][:]);done=f['primitive_done'][:].astype(bool);move=f['move_id'][:]
 if not np.isin(move,[6,7]).all():raise ValueError('This diagnostic supports U/Up only')
 pads=f['tactile_layer_force'][:];pad=pads[np.arange(len(move)),np.where(move==6,1,0),1,1]
 order=np.lexsort((steps,stage,uid));keys=np.stack([uid[order],stage[order]],1);starts=np.r_[0,np.flatnonzero(np.any(keys[1:]!=keys[:-1],axis=1))+1];ends=np.r_[starts[1:],len(order)]
 rows=[]
 for start,end in zip(starts,ends):
  ids=order[start:end];completed=np.flatnonzero(done[ids])
  if not len(completed):continue
  ids=ids[:completed[0]+1];row={'uid':int(uid[ids[0]]),'stage':int(stage[ids[0]])}
  for name,force in [('pad',pad),('whole_finger',whole)]:
   contact=np.flatnonzero(force[ids]>.5)
   row[name]=None if not len(contact) else {'last_error_deg':float(err[ids[contact[-1]]]),'seconds_to_completion':float((steps[ids[-1]]-steps[ids[contact[-1]]])*dt)}
  rows.append(row)
summary={}
for stage in sorted(set(x['stage'] for x in rows)):
 part=[x for x in rows if x['stage']==stage];s={'completed_stages':len(part)}
 for name in ('pad','whole_finger'):
  values=[x[name] for x in part if x[name] is not None]
  s[name]={'no_observed_contact':len(part)-len(values),'last_abs_error_deg_p10_p50_p90':np.percentile([abs(x['last_error_deg']) for x in values],[10,50,90]).tolist() if values else None,'seconds_to_completion_p10_p50_p90':np.percentile([x['seconds_to_completion'] for x in values],[10,50,90]).tolist() if values else None}
 summary[str(stage)]=s
result={'scope':'Retrospective successful-stage diagnostic. Normal-force threshold0.5N at control boundaries; brief contacts between samples may be missed. Not delivered sensory signals, not causal attribution.', 'source':str(path),'report_sha256':hashlib.sha256(raw).hexdigest(),'summary':summary,'rows':rows}
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))
