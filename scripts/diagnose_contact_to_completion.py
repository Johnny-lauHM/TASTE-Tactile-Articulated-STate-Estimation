import h5py,numpy as np,json
from pathlib import Path
path=Path('outputs/primitive_cycle/sequence_four_Up50.h5')
with h5py.File(path,'r') as f:
 uid=f['episode_uid'][:];stage=f['sequence_index'][:];steps=f['step'][:];force=f['release_force'][:];err=np.rad2deg(f['twist_error'][:]);done=f['primitive_done'][:].astype(bool);move=f['move_id'][:]
 order=np.lexsort((steps,stage,uid));keys=np.stack([uid[order],stage[order]],1);starts=np.r_[0,np.flatnonzero(np.any(keys[1:]!=keys[:-1],axis=1))+1];ends=np.r_[starts[1:],len(order)]
 rows=[]
 for start,end in zip(starts,ends):
  r=order[start:end];completed=np.flatnonzero(done[r])
  if not len(completed):continue
  r=r[:completed[0]+1];contacts=np.flatnonzero(force[r]>.5)
  if not len(contacts):continue
  last=contacts[-1];firstnear=np.flatnonzero(abs(err[r])<5)
  rows.append({'uid':int(uid[r[0]]),'stage':int(stage[r[0]]),'move_id':int(move[r[0]]),'last_contact_error_deg':float(err[r[last]]),'last_contact_to_complete_s':float((steps[r[-1]]-steps[r[last]])*.025),'last_contact_to_first_arrival_s':float((steps[r[firstnear[0]]]-steps[r[last]])*.025) if len(firstnear) else None})
 summary={}
 for s in sorted(set(x['stage'] for x in rows)):
  part=[x for x in rows if x['stage']==s]
  summary[str(s)]={'episodes':len(part),'abs_error_at_last_contact_deg_p10_p50_p90':np.percentile([abs(x['last_contact_error_deg']) for x in part],[10,50,90]).tolist(),'last_contact_to_complete_s_p10_p50_p90':np.percentile([x['last_contact_to_complete_s'] for x in part],[10,50,90]).tolist(),'last_contact_before_first_arrival_fraction':float(np.mean([x['last_contact_to_first_arrival_s']>0 for x in part]))}
report={'source':str(path),'scope':'Privileged retrospective contact/angle diagnosis of successful stages; not a deployed completion signal. Active whole-finger layer-force threshold0.5N, sampled40Hz.','summary':summary,'rows':rows}
Path('outputs/primitive_cycle/contact_to_completion_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary))
