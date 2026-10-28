"""Retrospective sampled motion audit; preserves the established acceptance scores."""
import argparse,hashlib,json
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('reports',nargs='+');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
if out.exists():raise FileExistsError(out)
results=[]
for filename in a.reports:
 path=Path(filename);raw=path.read_bytes()
 with h5py.File(path.with_suffix('.h5')) as f:
  uid=f['episode_uid'][:];stage=f['sequence_index'][:];st=f['step'][:];q=np.rad2deg(f['twist'][:]);done=f['primitive_done'][:].astype(bool);move=f['move_id'][:];rows=[]
  if not np.isin(move,[6,7]).all():raise ValueError('U/Up only')
  order=np.lexsort((st,stage,uid));key=np.stack([uid[order],stage[order]],1);starts=np.r_[0,np.flatnonzero(np.any(key[1:]!=key[:-1],axis=1))+1];ends=np.r_[starts[1:],len(order)]
  for begin,end in zip(starts,ends):
   ids=order[begin:end];completed=np.flatnonzero(done[ids])
   if not len(completed):continue
   ids=ids[:completed[0]+1];direction=-1 if move[ids[0]]==6 else 1;progress=direction*(q[ids]-q[ids[0]])
   rows.append([float(progress[-1]),float(np.abs(np.diff(q[ids])).sum()),float(progress.min()),float(progress.max())])
 x=np.array(rows)
 if not len(x):results.append({'source':str(path),'completed_fragments':0});continue
 results.append({'source':str(path),'report_sha256':hashlib.sha256(raw).hexdigest(),'completed_fragments':len(x),'net_progress_deg_p0_p50_p100':np.percentile(x[:,0],[0,50,100]).tolist(),'sampled_path_length_deg_p50_p90_p100':np.percentile(x[:,1],[50,90,100]).tolist(),'reverse_excursion_over10deg':int((x[:,2]<-10).sum()),'sampled_path_over180deg':int((x[:,1]>180).sum())})
result={'scope':'Observed fragments through first full ground-truth completion, including eventual tactile misses. Post-step40Hz samples omit the initial interval and between-sample motion; path length is a sampled lower bound. Diagnostic thresholds10/180deg do not change version3 acceptance.', 'results':results}
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(results))
