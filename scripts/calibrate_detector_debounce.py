"""Calibrate a fixed full-window observer with a short extra prediction debounce.
This reuses validation data; only a fresh physical rollout can establish performance.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rubik_lab'))
from rubik_lab.detector.data import TactileWindows,N_MOVES
from rubik_lab.detector.streaming import StreamingReadiness
from rubik_lab.detector.metrics import trigger_metrics
p=argparse.ArgumentParser()
p.add_argument('checkpoint');p.add_argument('data');p.add_argument('--out',required=True)
p.add_argument('--hold_steps',type=int,default=3);p.add_argument('--max_premature_rate',type=float,default=.01)
p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=16)
a=p.parse_args();torch.set_num_threads(4)
out=Path(a.out)
if out.exists() or Path(str(out)+'.json').exists():raise FileExistsError(out)
checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
if checkpoint.get('label_key')!='primitive_done' or checkpoint.get('preserve_history_between_moves') or checkpoint.get('kinematic_ft') or checkpoint.get('angle_gate_deg') is not None:
    raise ValueError('This calibration expects plain per-primitive completed-window observers')
if a.hold_steps<1:raise ValueError('Positive debounce required')
gate=StreamingReadiness(checkpoint,a.batch,a.device)
data=TactileWindows(a.data,window=1,label_key='primitive_done')
ids=np.asarray(checkpoint['val_episodes'])
rows=[np.flatnonzero(data.uid_sorted==uid) for uid in ids]
if any(not len(r) for r in rows):raise ValueError('Validation IDs do not match dataset')
uid_output=[];truth=[];probability=[]
with torch.inference_mode():
    for start in range(0,len(ids),a.batch):
        part=rows[start:start+a.batch]
        pad=lambda values:torch.nn.utils.rnn.pad_sequence(values,batch_first=True).to(a.device)
        x={'ft':pad([data.ft[r]*.2 for r in part]),'prop':pad([data.prop[r] for r in part]),
           'move':torch.nn.functional.one_hot(torch.stack([data.move[r[0]] for r in part]),N_MOVES).float().to(a.device)}
        prediction=gate.model.forward_sequence(x)[0].softmax(-1)[:,:,1].cpu().numpy()
        for i,r in enumerate(part):
            uid_output.extend([int(ids[start+i])]*len(r));truth.extend(data.label[r].tolist());probability.extend(prediction[i,:len(r)])
metrics=[trigger_metrics(uid_output,truth,probability,threshold,a.hold_steps) for threshold in (.5,.8,.9,.95,.97,.99,.995,.999,.9995,.9999)]
feasible=[m for m in metrics if m['premature_episode_rate']<=a.max_premature_rate and m['correct_first_triggers']>0]
selected=max(feasible,key=lambda m:(m['correct_first_triggers'],-m['premature_triggers'],m['threshold'])) if feasible else None
report={'checkpoint':a.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        'data':a.data,'data_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'metrics':metrics,'selected':selected,
        'scope':'Calibration on already-used validation groups; not independent acceptance','args':vars(a)}
out.parent.mkdir(parents=True,exist_ok=True)
if selected:
    checkpoint['hold_steps']=a.hold_steps;checkpoint['calibration_threshold']=selected['threshold']
    checkpoint['validation_event_metrics']=selected;checkpoint['calibration_source']=report['checkpoint_sha256']
    torch.save(checkpoint,out)
Path(str(out)+'.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(str(out)+'.validation.npz',episode_uid=np.asarray(uid_output),label=np.asarray(truth),probability=np.asarray(probability))
print('[calibration]',json.dumps(selected),flush=True)
