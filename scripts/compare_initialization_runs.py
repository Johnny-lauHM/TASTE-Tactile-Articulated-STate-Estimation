"""Compare predeclared same-seed startup-hold controls without selection by outcome."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('control');p.add_argument('--rounds',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
if out.exists():raise FileExistsError(out)
raw=[Path(x).read_bytes() for x in (a.baseline,a.control)];b,c=map(json.loads,raw)
fields=('seed','checkpoint_sha256','checkpoint_up_sha256','trigger','detector_sha256','detector_up_sha256','detector_threshold','detector_up_threshold','detector_hold_steps','sequence','synchronize_each_primitive','synchronize_tactile_clock','per_primitive_timeout_seconds','numerics')
for key in fields:
 if b.get(key)!=c.get(key):raise ValueError('Confounded comparison: '+key)
canonical_cfg=lambda value:re.sub(r'0x[0-9a-fA-F]+','0xADDRESS',json.dumps(value,sort_keys=True))
if canonical_cfg(b['config'])!=canonical_cfg(c['config']):raise ValueError('Different environment configs beyond process-local callable addresses')
if a.rounds<1:raise ValueError('rounds must be positive')
rounds=list(range(a.rounds));bs={(x['trial'],x['env']):x for x in b['episodes'] if x['trial'] in rounds};cs={(x['trial'],x['env']):x for x in c['episodes'] if x['trial'] in rounds}
if set(bs)!=set(cs) or set(t for t,e in bs)!=set(rounds):raise ValueError('Different or missing trial identities')
def summarize(rows):
 return {'n':len(rows),'successes':sum(x['success'] for x in rows),'premature':sum(x['reason']=='premature_trigger' for x in rows),'timeouts':sum(x['reason']=='timeout' for x in rows)}
result={'scope':'Predeclared matched seeds and env identities; startup minimum-hold setting differs. Stochastic simulator evolution and shared environments preclude treating this as an IID significance test. Only requested rounds are compared, chosen before outcomes.',
 'sources':[{'path':path,'sha256':hashlib.sha256(content).hexdigest()} for path,content in zip((a.baseline,a.control),raw)],'matched_fields':list(fields)+['config, ignoring process-local callable addresses'],'rounds':rounds,
 'minimum_hold_steps':[b['initial_settle_steps'],c['initial_settle_steps']],
 'actual_startup_hold_steps':[b['actual_startup_hold_steps'][:a.rounds],c['actual_startup_hold_steps'][:a.rounds]],
 'baseline':summarize(list(bs.values())),'control':summarize(list(cs.values())),
 'control_only_success':sum(cs[k]['success'] and not bs[k]['success'] for k in bs),
 'baseline_only_success':sum(bs[k]['success'] and not cs[k]['success'] for k in bs),
 'per_round':{str(t):{'baseline':summarize([v for (r,e),v in bs.items() if r==t]),'control':summarize([v for (r,e),v in cs.items() if r==t])} for t in rounds}}
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
