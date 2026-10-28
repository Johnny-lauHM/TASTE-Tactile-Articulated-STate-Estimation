"""Separate observed physical readiness from observer acknowledgement per stage."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('--out',required=True);a=p.parse_args()
r=json.loads(Path(a.report).read_text());n=r['config']['scene']['num_envs'];records={e['trial']*n+e['env']:e for e in r['episodes']}
with h5py.File(Path(a.report).with_suffix('.h5'),'r') as f:
 uid=f['episode_uid'][:];st=f['sequence_index'][:];truth=f['primitive_done'][:].astype(bool);det=f['detector_done'][:].astype(bool);prob=f['detector_ready_probability'][:]
 result={}
 for stage,move in enumerate(r['sequence']):
  ids=np.unique(uid[st==stage]);stats={'move':move,'attempted':len(ids),'observed_true_readiness':0,'acknowledged':0,'premature':0,'timeout_after_true_readiness':0,'timeout_without_observed_readiness':0,'other_endings':0}
  for e in ids:
   mask=(uid==e)&(st==stage);ready=bool(truth[mask].any());ack=bool((truth[mask]&det[mask]).any());early=bool((~truth[mask]&det[mask]).any());reason=records[int(e)]['reason']
   stats['observed_true_readiness']+=ready
   if ack:stats['acknowledged']+=1
   elif early:stats['premature']+=1
   elif reason=='timeout':stats['timeout_after_true_readiness' if ready else 'timeout_without_observed_readiness']+=1
   else:stats['other_endings']+=1
  vals=prob[(st==stage)&truth];stats['ready_score_on_true_ready_p10_p50_p90']=np.percentile(vals,[10,50,90]).tolist() if len(vals) else None
  result[str(stage)]=stats
out={'source':a.report,'scope':'Retrospective truth diagnostic. Premature-trigger runs are censored; absence of later readiness cannot be interpreted as inherent physical failure.','stages':result}
Path(a.out).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(result))
