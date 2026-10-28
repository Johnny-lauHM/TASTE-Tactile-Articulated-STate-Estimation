"""Render explicit GT-trigger vs tactile-trigger acceptance comparisons."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
if out.with_suffix('.png').exists() or out.with_suffix('.json').exists():raise FileExistsError(out)
base=Path('outputs/primitive_cycle')
groups=[('Single U: ground-truth completion',[
 ('Original U','U_clock_sync_v1'),('U transition50','U_transition50_initial'),('U transition100','U_transition100_initial')]),
 ("U then U': ground-truth switching",[
 ('Original Up','sequence_clock_sync_v1'),('Up PPO50','sequence_Up_ppo50_clock'),('Up PPO100','sequence_Up100_clock'),('Up PPO200','sequence_Up200_clock'),('Up200 repeat','sequence_Up200_verify'),('Up final repeat','sequence_Upfinal_verify'),('Up pad50','sequence_UpPad50'),('Up pad final','sequence_UpPad_final')]),
 ("Four moves: ground-truth switching",[
 ('Original U + Up50','sequence_four_Up50'),('U transition50 + Up50','sequence_four_Utransition50'),('Original U + Up200','sequence_four_Up200'),('U self25 + Up200','sequence_four_Uself25'),('U self75 + Up200','sequence_four_Uself75'),('U valid25 + Up200','sequence_four_Uvalid25'),('Original U + Up final','sequence_four_Upfinal'),('U/Up200, hold4','sequence_four_hold4_control'),('U/Up200, matched hold0','sequence_four_hold0_matched'),('U self final + Up200','sequence_four_Uself_final'),('U valid final + Up200','sequence_four_Uvalid_final')]),
 ("U then U': tactile switching",[
 ('v5b / Up50','sequence_tactile_fullready_v5b'),('v7 / Up50','sequence_tactile_fullready_v7'),('v7 hold3 / Up50','sequence_tactile_v7_debounce3'),('v8 / Up200','sequence_tactile_v8_Up200'),('v8 hold3 / Up200','sequence_tactile_v8_hold3'),('v9 Up hold3 / Up200','sequence_tactile_v9_hold3'),('U7/Up9 hold3 / Up final','sequence_tactile_U7Up9_Upfinal'),('v9 larger repeat','sequence_tactile_v9_verify'),('v9, startup hold4','sequence_tactile_v9_initialhold4'),('hold4 larger repeat','sequence_tactile_hold4_verify_canonical')])]
fig,axes=plt.subplots(2,2,figsize=(17,15));artifacts=[]
for ax,(title,items) in zip(axes.flat,groups):
 rows=[]
 for label,name in items:
  path=base/(name+'.json')
  if not path.exists():continue
  raw=path.read_bytes();r=json.loads(raw);s=r['summary'];rows.append((label,s))
  artifacts.append({'label':label,'panel':title,'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'trigger':r['trigger'],'summary':s,'per_stage':r.get('per_primitive_stage'),'cache':r['config']['grasp_cache_path'],'seed':r['seed'],'initial_settle_steps':r.get('initial_settle_steps'),'checkpoint':r['checkpoint'],'checkpoint_up':r.get('checkpoint_up')})
 y=np.arange(len(rows));rates=np.array([s['success_rate']*100 for _,s in rows]);ci=np.array([s['wilson95'] for _,s in rows])*100
 if len(rows):
  ax.barh(y,rates,color='#286c9b' if 'ground-truth' in title else '#25805c',alpha=.8)
  ax.errorbar(rates,y,xerr=np.stack([rates-ci[:,0],ci[:,1]-rates]),fmt='none',color='black',capsize=3)
  for i,(label,s) in enumerate(rows):
   count=s.get('reasons',{}).get('success',round(s['success_rate']*s['n']))
   ax.text(min(ci[i,1]+2,101),i,f"{count}/{s['n']}" + (f"; {s.get('reasons',{}).get('premature_trigger',0)} early" if 'tactile' in title else ''),va='center',fontsize=9)
 ax.set_yticks(y,[label for label,s in rows]);ax.invert_yaxis();ax.set_xlim(0,125);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Complete trials (%)');ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Fixed-wrist primitive acceptance: simulation only',fontsize=16)
fig.text(.5,.015,'Selected canonical grasp; 6 s per move; no physical reset inside a sequence.\nGT switching is not tactile autonomy. Intervals are descriptive Wilson intervals; trials share environments and grasp.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.065,1,.95]);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out.with_suffix('.png'),dpi=160)
out.with_suffix('.json').write_text(json.dumps({'scope':'Fixed predeclared comparison list; missing unfinished runs omitted. All source artifacts preserved.','results':artifacts},indent=2)+'\n');print(out.with_suffix('.png'))
