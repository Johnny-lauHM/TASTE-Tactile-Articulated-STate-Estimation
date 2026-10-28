"""Validation-only modality ablation: does readiness prediction use tactile history?"""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rubik_lab'))
from rubik_lab.detector.data import TactileWindows,N_MOVES
from rubik_lab.detector.streaming import StreamingReadiness
from rubik_lab.detector.metrics import trigger_metrics
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('data');p.add_argument('--out',required=True);p.add_argument('--threshold',type=float,required=True);p.add_argument('--device',default='cuda');p.add_argument('--hold_steps',type=int,default=None)
a=p.parse_args();torch.set_num_threads(4);out=Path(a.out)
if out.exists():raise FileExistsError(out)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
if ck.get('kinematic_ft') or ck.get('preserve_history_between_moves') or ck.get('angle_gate_deg') is not None:raise ValueError('Plain per-move observers only')
gate=StreamingReadiness(ck,16,a.device,a.threshold)
if a.hold_steps is not None:
 if a.hold_steps<1:raise ValueError('hold_steps must be positive')
 gate.hold_steps=a.hold_steps
data=TactileWindows(a.data,window=1,label_key=ck['label_key'])
ids=np.asarray(ck['val_episodes']);rows=[np.flatnonzero(data.uid_sorted==uid) for uid in ids]
results={}
for mode in ('normal','ft_mean','prop_mean','ft_reversed'):
 truth=[];uids=[];probs=[]
 with torch.inference_mode():
  for begin in range(0,len(ids),16):
   part=rows[begin:begin+16]
   pad=lambda values:torch.nn.utils.rnn.pad_sequence(values,batch_first=True).to(a.device)
   ft=[data.ft[r]*.2 for r in part]
   if mode=='ft_reversed':ft=[v.flip(0) for v in ft]
   x={'ft':pad(ft),'prop':pad([data.prop[r] for r in part]),'move':torch.nn.functional.one_hot(torch.stack([data.move[r[0]] for r in part]),N_MOVES).float().to(a.device)}
   if mode=='ft_mean':x['ft'][:]=gate.model.ft_mean
   if mode=='prop_mean':x['prop'][:]=gate.model.prop_mean
   pred=gate.model.forward_sequence(x)[0].softmax(-1)[:,:,1].cpu().numpy()
   for i,r in enumerate(part):
    uids.extend([int(ids[begin+i])]*len(r));truth.extend(data.label[r].tolist());probs.extend(pred[i,:len(r)])
 results[mode]=trigger_metrics(uids,truth,probs,a.threshold,gate.hold_steps)
 print('[ablation]',mode,json.dumps(results[mode]),flush=True)
report={'scope':'Ablation on reused validation data, not independent physical acceptance. Mean/reversal are out-of-distribution diagnostic interventions.','args':vars(a),'results':results,'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'data_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest()}
out.write_text(json.dumps(report,indent=2)+'\n')
