import torch,h5py,numpy as np,json,sys
from pathlib import Path
sys.path.insert(0,'source/rubik_lab')
from rubik_lab.blind_actor import BlindPrimitiveActor,student_features
torch.set_num_threads(4);ck=torch.load('logs/rsl_rl/rubik_primitive_cycle/U_blind_full/model.pt',map_location='cpu',weights_only=False)
model=BlindPrimitiveActor(ck['hidden']).cuda();model.load_state_dict(ck['model']);model.eval();inputs=[];targets=[];stages=[]
with h5py.File('outputs/primitive_cycle/U_first_and_third_self_demos_v1.h5','r') as f:
 uid=f['episode_uid'][:];good=np.unique(uid[f['primitive_done'][:].astype(bool)]);stage=f['sequence_index'][:]
 for s in (0,2):
  for e in good[np.isin(good,uid[stage==s])][:16]:
   idx=np.flatnonzero(uid==e);inputs.append(student_features(torch.from_numpy(f['policy_obs'][idx])));targets.append(torch.from_numpy(f['policy_action'][idx]).cuda());stages.append(s)
x=torch.nn.utils.rnn.pad_sequence(inputs,batch_first=True).cuda();x=torch.cat([x,x.new_zeros(128-len(x),x.shape[1],x.shape[2])],0)
results={}
for cudnn_tf32 in (False,True):
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=cudnn_tf32
 hidden=None;prediction=[]
 with torch.inference_mode():
  for t in range(x.shape[1]):
   value,hidden=model(x[:,t:t+1],hidden);prediction.append(value[:,0])
 prediction=torch.stack(prediction,1)
 errors=[float((prediction[i,:len(y)]-y).abs().max()) for i,y in enumerate(targets)]
 results[str(cudnn_tf32)]={'max_abs_error':max(errors),'per_stage_max':{str(s):max(e for e,ss in zip(errors,stages) if ss==s) for s in (0,2)}}
report={'scope':'Stepwise GPU replay with original batch size128. Matmul TF32 disabled; compare cudnn precision modes. 32 first/third-stage histories.','results':results}
Path('outputs/primitive_cycle/U_self_demo_replay_stepwise.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if min(v['max_abs_error'] for v in results.values())>1e-4:raise ValueError('Replay discrepancy needs investigation')
