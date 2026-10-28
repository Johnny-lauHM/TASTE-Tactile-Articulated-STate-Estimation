import torch,h5py,numpy as np,json,sys
from pathlib import Path
sys.path.insert(0,'source/rubik_lab')
from rubik_lab.blind_actor import BlindPrimitiveActor,student_features
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
ck=torch.load('logs/rsl_rl/rubik_primitive_cycle/U_blind_full/model.pt',map_location='cpu',weights_only=False);model=BlindPrimitiveActor(ck['hidden']);model.load_state_dict(ck['model']);model.eval();model=model.cuda()
rows=[]
with h5py.File('outputs/primitive_cycle/U_first_and_third_self_demos_v1.h5','r') as f:
 uid=f['episode_uid'][:];good=np.unique(uid[f['primitive_done'][:].astype(bool)]);stage=f['sequence_index'][:]
 for s in (0,2):
  chosen=good[np.isin(good,uid[stage==s])][:16]
  for e in chosen:
   idx=np.flatnonzero(uid==e);obs=torch.from_numpy(f['policy_obs'][idx]);actions=torch.from_numpy(f['policy_action'][idx]).cuda();obs=obs.cuda()
   with torch.inference_mode():pred=model(student_features(obs)[None])[0][0]
   rows.append({'uid':int(e),'stage':s,'max_abs_error':float((pred-actions).abs().max()),'mse':float((pred-actions).square().mean())})
report={'scope':'Offline original actor replay of32 extracted full histories, including third-stage boundaries','max_abs_error':max(r['max_abs_error'] for r in rows),'rows':rows}
Path('outputs/primitive_cycle/U_self_demo_replay_gpu_tf32.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
if report['max_abs_error']>1e-3:raise ValueError('Extracted histories do not reproduce original actor actions')
