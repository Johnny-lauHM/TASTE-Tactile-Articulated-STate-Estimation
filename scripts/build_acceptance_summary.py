"""Build a fixed-list handoff inventory without altering source evaluations."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
if out.exists():raise FileExistsError(out)
base=Path('outputs/primitive_cycle')
groups={
 'ground_truth_two_step':['sequence_Up200_verify','sequence_Upfinal_verify','sequence_UpPad50','sequence_UpPad_final'],
 'ground_truth_four_step':['sequence_four_Up50','sequence_four_Up200','sequence_four_Uself25','sequence_four_Uself75','sequence_four_Uself25_Up50','sequence_four_hold4_control','sequence_four_hold0_matched','sequence_four_Uvalid25','sequence_four_Upfinal','sequence_four_Uself_final','sequence_four_Uvalid_final'],
 'exploratory_broad_grasp':['sequence_tactile_hold4_verify'],
 'pair_coverage':['pair_U_U_v1','pair_Up_U_v1','pair_Up_Up_v1'],
 'tactile_two_step':['sequence_tactile_fullready_v5b','sequence_tactile_fullready_v7','sequence_tactile_v7_debounce3','sequence_tactile_v8_Up200','sequence_tactile_v8_hold3','sequence_tactile_v9_hold3','sequence_tactile_U7Up9_Upfinal','sequence_tactile_v9_verify','sequence_tactile_v9_initialhold4','sequence_tactile_hold4_verify_canonical']}
results={}
for group,names in groups.items():
 results[group]=[]
 for name in names:
  path=base/(name+'.json')
  if not path.exists():
   results[group].append({'name':name,'status':'no_completed_report'});continue
  raw=path.read_bytes();r=json.loads(raw)
  results[group].append({'name':name,'status':'reported','path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
   'trigger':r['trigger'],'checkpoint':r['checkpoint'],'checkpoint_up':r.get('checkpoint_up'),'detector':r.get('detector'),'detector_up':r.get('detector_up'),
   'sequence':r.get('sequence'),'cache':r['config']['grasp_cache_path'],'seed':r['seed'],'summary':r['summary'],'per_stage':r.get('per_primitive_stage'),
   'detector_hold_steps':r.get('detector_hold_steps'),'initial_settle_steps':r.get('initial_settle_steps')})
result={'updated_utc':datetime.now(timezone.utc).isoformat(),'scope':'Simulation, same-face U/Up. Main comparisons use selected canonical grasp; exploratory_broad_grasp uses18-grasp checkpoint default after an omitted cache override, disclosed separately. No full-cube solve, live initial vision, calibrated real tactile sensing or autonomous recovery claimed.',
 'interpretation':'Ground-truth switching tests motor execution; tactile switching tests completion-trigger autonomy. Same-face pair coverage does not imply multi-face coverage. Larger repeats take precedence over a favorable small batch. Candidate scores do not automatically select or deploy a checkpoint.',
 'engineering_target':{'tactile_two_step_success':.95,'premature_trial_rate_max':.01,'minimum_trials':512,'minimum_rounds':4},'groups':results,
 'audit':str(base/'audit.json'),'memory':'MEMORY.md','pair_coverage_figure':str(base/'pair_coverage_v2.png')}
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:[{'name':x['name'],'success':x.get('summary',{}).get('success_rate'),'early':x.get('summary',{}).get('reasons',{}).get('premature_trigger',0)} for x in v] for k,v in results.items()}))
