"""Assess a fixed-quota same-face tactile demo, never a full-cube solve."""
import argparse,json,hashlib,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('report');p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out)
if out.exists():raise FileExistsError(out)
raw=Path(a.report).read_bytes();r=json.loads(raw);rows=r['episodes'];cfg=r['config'];n=len(rows);success=[x for x in rows if x['success']];premature=sum(x['reason']=='premature_trigger' for x in rows)
checks=[]
def check(name,passed,actual):checks.append({'criterion':name,'passed':bool(passed),'actual':actual})
check('Fixed-quota evaluation',r.get('mode')=='fixed_quota_evaluation',r.get('mode'))
check('U then U-prime, without sequence reset',r.get('sequence')==['U',"U'"],r.get('sequence'))
check('Tactile-triggered transitions',r.get('trigger')=='tactile_detector',r.get('trigger'))
scopes=[r.get('actor_observation_scope'),r.get('actor_observation_scope_up')]
check('Both actors restricted to tactile/proprioceptive inputs',all(s=='tactile_proprioceptive' for s in scopes),scopes)
identities=[(x['trial'],x['env']) for x in rows];trials=sorted(set(t for t,e in identities));env_count=cfg['scene']['num_envs']
quota=len(set(identities))==n and all({x['env'] for x in rows if x['trial']==t}==set(range(env_count)) for t in trials)
check('At least512 complete trials across at least4 full rounds',n>=512 and len(trials)>=4 and quota,{'trials':n,'rounds':len(trials),'quota_consistent':quota})
check('Reported counts agree with raw episode records',r['summary']['n']==n and abs(r['summary']['success_rate']-len(success)/max(n,1))<1e-12,{'successes':len(success),'trials':n})
check('At least95% complete tactile sequences',len(success)/max(n,1)>=.95,len(success)/max(n,1))
check('At most1% premature requests per trial',premature/max(n,1)<=.01,{'premature':premature,'rate':premature/max(n,1)})
check('Six-second maximum per primitive',0<r.get('per_primitive_timeout_seconds',float('inf'))<=6,r.get('per_primitive_timeout_seconds'))
check('No artificial notch or detent torque',cfg.get('notch_torque')==0 and cfg.get('detent_torque')==0,{'notch':cfg.get('notch_torque'),'detent':cfg.get('detent_torque')})
check('Version3 physical dwell contract',cfg.get('cycle_contract_version')==3 and cfg.get('cycle_stability_mode')=='pose_window' and cfg.get('cycle_stable_s',0)>=.25,{'version':cfg.get('cycle_contract_version'),'stable_seconds':cfg.get('cycle_stable_s')})
bounds={'move_success_tol':math.radians(5),'move_success_vel':.25,'move_core_ang_speed':1,'cycle_finger_tol':.2,'cycle_finger_speed':1,'cycle_release_force':.5,'cycle_wrist_pos':.01,'cycle_wrist_rot':math.radians(10),'cycle_cube_pos':.015,'cycle_cube_rot':math.radians(15),'cycle_stable_pos':.002,'cycle_stable_rot':math.radians(2)}
check('Physical thresholds have not been relaxed',all(k in cfg and 0<=cfg[k]<=v+1e-12 for k,v in bounds.items()),{k:cfg.get(k) for k in bounds})
consistent=all(len(x.get('primitives',[]))==2 and all(abs(p['error_deg'])<=5+1e-5 and p['seconds']<=6+1e-8 for p in x['primitives']) and x['peak_wrist_mm']<=10+1e-5 and x['peak_wrist_deg']<=10+1e-5 and x['peak_cube_mm']<=15+1e-5 and x['peak_cube_deg']<=15+1e-5 for x in success)
check('Successful records satisfy both-stage angle/time and whole-chain motion limits',consistent,len(success))
report={'source':a.report,'source_sha256':hashlib.sha256(raw).hexdigest(),'accepted_for_canonical_two_step_simulation_demo':all(x['passed'] for x in checks),'checks':checks,'scope':'Selected canonical grasp, simulation-only U/U-prime chain. Point-estimate engineering targets; passing does not establish real-world reliability, six-face solving, calibrated tactile sensing, or recovery.','scoring':'Ground truth scores and aborts premature acknowledgements; this is not a deployed safety supervisor.','strict_calm_success_rate_separate':r['summary'].get('strict_calm_success_rate')}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'accepted':report['accepted_for_canonical_two_step_simulation_demo'],'failed':[x['criterion'] for x in checks if not x['passed']]}))
