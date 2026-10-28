"""Train tactile instantaneous-readiness recognition over complete episode histories.
The executor requires consecutive ready predictions before a completion trigger.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rubik_lab'))
from rubik_lab.detector import CompletionDetector, TactileWindows
from rubik_lab.detector.data import split_episodes,N_MOVES
from rubik_lab.detector.metrics import trigger_metrics
from rubik_lab.detector.model import AngleCompletionDetector
p=argparse.ArgumentParser()
p.add_argument('data')
p.add_argument('--out',required=True)
p.add_argument('--epochs',type=int,default=50)
p.add_argument('--batch',type=int,default=32)
p.add_argument('--seed',type=int,default=123)
p.add_argument('--device',default='cuda')
p.add_argument('--hold_steps',type=int,default=10)
p.add_argument('--label_key',choices=['cycle_ready','primitive_done'],default='cycle_ready')
p.add_argument('--split_by',choices=['grasp_idx','episode_uid','source_file'],default='grasp_idx')
p.add_argument('--angle_head',action='store_true')
p.add_argument('--angle_weight',type=float,default=5.)
p.add_argument('--angle_gate_deg',type=float,default=3.)
p.add_argument('--angle_near_weight',type=float,default=1.)
p.add_argument('--checkpoint',default=None)
p.add_argument('--normalize_inputs',action='store_true')
p.add_argument('--kinematic_ft',action='store_true',help='append encoder-derived wrist-aligned pad force/moment vectors')
a=p.parse_args()
if a.hold_steps<1:p.error("hold_steps must be positive")
torch.set_num_threads(4)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
path=Path(a.out)
path.parent.mkdir(parents=True,exist_ok=True)
source_dir=Path(str(path)+'.sources')
source_dir.mkdir(exist_ok=False)
root=Path(__file__).resolve().parents[1]
for source_path in [Path(__file__),root/'source/rubik_lab/rubik_lab/detector/model.py',root/'source/rubik_lab/rubik_lab/detector/data.py',root/'source/rubik_lab/rubik_lab/detector/metrics.py',root/'source/rubik_lab/rubik_lab/detector/kinematic_features.py']:
    destination=source_dir/source_path.name
    destination.write_bytes(source_path.read_bytes())
if path.exists():
    raise FileExistsError(path)
train_ep,val_ep=split_episodes(a.data,group_key=a.split_by,seed=a.seed)
data=TactileWindows(a.data,window=1,label_key=a.label_key,timeout_is_failure=False)
kinematics_hashes=None
if a.kinematic_ft:
    from rubik_lab.detector.kinematic_features import PadWrenchRotation
    rotation=PadWrenchRotation().to(a.device)
    transformed=[]
    with torch.inference_mode():
        for begin in range(0,len(data.ft),8192):
            ft=data.ft[begin:begin+8192].reshape(-1,2,5,6).to(a.device)
            q=data.prop[begin:begin+8192,:44].reshape(-1,2,22).to(a.device)
            transformed.append(rotation(ft,q).flatten(1).cpu())
    data.ft=torch.cat([data.ft,torch.cat(transformed)],-1)
    kinematics_hashes=rotation.urdf_sha256
    del rotation,transformed
continuous_history=bool(data.f.attrs.get('continuous_history',False))
stage_truth=data.f['sequence_index'][:][data.order] if 'sequence_index' in data.f else np.zeros(len(data.order),dtype=np.int64)
stage_count=int(stage_truth.max())+1
angle_truth=data.f['twist_error'][:][data.order]
full=data.f['primitive_done'][:][data.order].astype(np.int64)
full[data.label.numpy()==2]=2
episodes={}
for uid in data.uids:
    rows=np.flatnonzero(data.uid_sorted==uid)
    if not continuous_history and len(torch.unique(data.move[rows]))!=1:
        raise ValueError('Expected one commanded primitive per episode')
    episodes[int(uid)]=rows

def batches(ids,shuffle=False):
    ids=np.array(ids).copy()
    if shuffle:
        np.random.shuffle(ids)
    for begin in range(0,len(ids),a.batch):
        selected=ids[begin:begin+a.batch]
        rows=[episodes[int(uid)] for uid in selected]
        pad=lambda values,padding=0:torch.nn.utils.rnn.pad_sequence(values,batch_first=True,padding_value=padding).to(a.device)
        x={'ft':pad([data.ft[r]*0.2 for r in rows]),'prop':pad([data.prop[r] for r in rows]),
           'move':pad([torch.nn.functional.one_hot(data.move[r],N_MOVES).float() for r in rows]) if continuous_history else torch.nn.functional.one_hot(torch.stack([data.move[r[0]] for r in rows]),N_MOVES).float().to(a.device)}
        labels=pad([data.label[r] for r in rows],-100)
        yield x,labels,rows,selected
train_rows=np.isin(data.uid_sorted,train_ep)
counts=torch.bincount(data.label[train_rows],minlength=3)
if (counts==0).any():
    raise ValueError('Need all readiness/physical-failure classes in training split')
weights=counts.sum()/counts.float()
weights=(weights/weights.sum()*3).to(a.device)
model=(AngleCompletionDetector(ft_dim=data.ft.shape[-1]) if a.angle_head else CompletionDetector(ft_dim=data.ft.shape[-1])).to(a.device)
if a.checkpoint:
    model.load_state_dict(torch.load(a.checkpoint,map_location=a.device,weights_only=False)["model"])
normalization=None
if a.normalize_inputs:
    normalization={}
    for name,values,floor in [('ft',data.ft[train_rows]*0.2,1e-4),('prop',data.prop[train_rows],1e-3)]:
        normalization[name+'_mean']=values.mean(0).tolist()
        normalization[name+'_scale']=values.std(0).clamp(min=floor).tolist()
    model.set_normalization(normalization)
elif a.checkpoint:
    normalization=torch.load(a.checkpoint,map_location='cpu',weights_only=False).get('feature_normalization')
    model.set_normalization(normalization)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
loss_fn=torch.nn.CrossEntropyLoss(weight=weights,ignore_index=-100)
report={'data':a.data,'data_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
        'training_args':vars(a),'kinematic_ft':a.kinematic_ft,'kinematics_urdf_sha256':kinematics_hashes,'trainer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':a.seed,'label_key':a.label_key,'timeout_is_failure':False,
        'split_by':a.split_by,'split_grouping':'raw source trial (all primitive fragments together)' if a.split_by=='episode_uid' and 'source_episode_uid' in data.f else a.split_by,'history':'whole sequence, stable counter resets at primitive boundaries' if continuous_history else 'whole primitive, reset at primitive boundaries','hold_steps':a.hold_steps,
        'train_episodes':train_ep.tolist(),'val_episodes':val_ep.tolist(),
        'class_counts':counts.tolist(),'feature_normalization':normalization,'checkpoint':a.checkpoint,'angle_near_weight':a.angle_near_weight,'angle_head':a.angle_head,'angle_weight':a.angle_weight,'angle_gate_deg':a.angle_gate_deg if a.angle_head else None,'scope':a.split_by+' held-out validation, not closed-loop acceptance','epochs':[]}

def predict_loss(x,y,rows):
    if a.angle_head:
        logits,angle,_=model.predict_sequence(x)
        targets=torch.nn.utils.rnn.pad_sequence([torch.tensor(angle_truth[r]) for r in rows],batch_first=True).to(a.device)
        mask=(y!=-100)&torch.isfinite(targets)
        error=torch.atan2((angle-targets).sin(),(angle-targets).cos())
        per_frame=torch.nn.functional.smooth_l1_loss(error[mask],torch.zeros_like(error[mask]),beta=0.1,reduction="none")
        emphasis=torch.where(targets[mask].abs()<torch.deg2rad(torch.tensor(20.,device=a.device)),a.angle_near_weight,1.)
        regression=(per_frame*emphasis).sum()/emphasis.sum()
    else:
        logits,_=model.forward_sequence(x)
        angle=None
        regression=0.
    return logits,angle,loss_fn(logits.transpose(1,2),y)+a.angle_weight*regression

def validate():
    model.eval()
    probabilities,truth,uids=[],[],[]
    angle_errors,ready_angle_errors=[],[]
    total=frames=0
    with torch.no_grad():
        for x,y,rows,selected in batches(val_ep):
            logits,angle,validation_loss=predict_loss(x,y,rows)
            n=int((y!=-100).sum())
            total+=float(validation_loss)*n
            frames+=n
            probs=logits.softmax(-1)[:,:,1].cpu().numpy()
            angle_np=angle.cpu().numpy() if angle is not None else None
            for i,r in enumerate(rows):
                if angle_np is not None:
                    delta=angle_np[i,:len(r)]-angle_truth[r]
                    degrees=np.abs(np.rad2deg(np.arctan2(np.sin(delta),np.cos(delta))))
                    angle_errors.extend(degrees[np.isfinite(degrees)])
                    ready_angle_errors.extend(degrees[data.label[r].numpy()==1])
                    probs[i,:len(r)]*=np.abs(np.rad2deg(angle_np[i,:len(r)]))<=a.angle_gate_deg
                probabilities.extend(probs[i,:len(r)])
                truth.extend(full[r])
                uids.extend((int(selected[i])*stage_count+stage_truth[r]).tolist() if continuous_history else [int(selected[i])]*len(r))
    metrics=[trigger_metrics(uids,truth,probabilities,t,a.hold_steps) for t in (0.5,0.8,0.9,0.95,0.99)]
    angle_metrics={"all_error_deg_p50_p90_p95":np.percentile(angle_errors,[50,90,95]).tolist() if angle_errors else None,
                   "ready_error_deg_p50_p90_p95":np.percentile(ready_angle_errors,[50,90,95]).tolist() if ready_angle_errors else None}
    return total/max(frames,1),metrics,angle_metrics
def checkpoint_payload(epoch):
    return {'format':('streaming_tactile_ready_angle' if a.angle_head else 'streaming_tactile_ready')+('_normalized_v1' if normalization is not None else '_v1'),
            'feature_normalization':normalization,'preserve_history_between_moves':continuous_history,
            'kinematic_ft':a.kinematic_ft,'ft_dim':int(data.ft.shape[-1]),'kinematics_urdf_sha256':kinematics_hashes,
            'angle_gate_deg':a.angle_gate_deg if a.angle_head else None,'model':model.state_dict(),'images':False,
            'label_key':a.label_key,'hold_steps':a.hold_steps,'epoch':epoch+1,
            'train_episodes':train_ep,'val_episodes':val_ep,'schema_version':3}

best=float('inf')
event_best=None
for epoch in range(a.epochs):
    model.train()
    total=frames=0
    for x,y,rows,selected in batches(train_ep,True):
        logits,angle,loss=predict_loss(x,y,rows)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        n=int((y!=-100).sum())
        total+=float(loss.detach())*n
        frames+=n
    val_loss,metrics,angle_metrics=validate()
    row={'epoch':epoch+1,'train_loss':total/max(frames,1),'val_loss':val_loss,'triggers':metrics,'angle_metrics':angle_metrics}
    report['epochs'].append(row)
    if val_loss<best:
        best=val_loss
        torch.save(checkpoint_payload(epoch),path)
    for metric in metrics:
        if metric['premature_episode_rate']<=.01 and metric['correct_first_triggers']>0:
            score=(metric['correct_first_triggers'],-metric['premature_triggers'],metric['threshold'])
            if event_best is None or score>event_best:
                event_best=score
                event_path=path.with_name(path.stem+'.event'+path.suffix)
                payload=checkpoint_payload(epoch)
                payload['calibration_threshold']=metric['threshold']
                payload['validation_event_metrics']=metric
                torch.save(payload,event_path)
                report['selected_event']={'path':str(event_path),'epoch':epoch+1,'metrics':metric,
                    'selection':'maximize held-out correct first triggers subject to observed premature episode rate <= 1%; fresh rollout required'}

    if epoch%5==0 or epoch==a.epochs-1:
        print(json.dumps(row),flush=True)
best_state=torch.load(path,map_location=a.device,weights_only=False)
model.load_state_dict(best_state['model'])
report['selected_epoch']=best_state['epoch']
report['selected_val_loss'],report['selected_triggers'],report['selected_angle_metrics']=validate()
Path(a.out+'.json').write_text(json.dumps(report,indent=2)+'\n')
print('[sequence_detector] finished',a.out,json.dumps(report['selected_triggers']),flush=True)
