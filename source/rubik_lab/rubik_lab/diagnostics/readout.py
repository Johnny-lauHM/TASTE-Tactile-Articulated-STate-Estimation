"""Nested, round-separated linear readout audit; no simulator or policy changes."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--independent');p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out)
if out.exists():raise FileExistsError(out)
z=np.load(Path(a.input).with_suffix('.npz'));h=z['sensory'];y=z['truth'][:,:,-1,0];good=z['valid'][:,:,-8:].all(-1)&np.isfinite(h).all(axis=(-1,-2))
other=np.load(Path(a.independent).with_suffix('.npz')) if a.independent else None
alphas=[0.1,1.,10.,100.,1000.,10000.]
def design(h):return np.concatenate([h[:,:,-8:].mean(2),h[:,:,-8:].std(2),h[:,:,-1]-h[:,:,0]],-1)
def fit(x,y,alpha):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-5);xx=(x-mean)/scale;bias=y.mean();yy=y-bias
    # Dual ridge avoids underdetermined inverse when features exceed episodes.
    coef=xx.T@np.linalg.solve(xx@xx.T+alpha*np.eye(len(xx)),yy)
    return mean,scale,bias,coef
def predict(model,x):
    mean,scale,bias,coef=model;return (x-mean)/scale@coef+bias
def metrics(pred,true):
    near=np.abs(true)<5;guess=np.abs(pred)<5
    tp=int((near&guess).sum());fp=int((~near&guess).sum());fn=int((near&~guess).sum());tn=int((~near&~guess).sum())
    return {'n':len(true),'angle_MAE_deg':float(abs(pred-true).mean()),'angle_p95_abs_error_deg':float(np.percentile(abs(pred-true),95)),'balanced_accuracy':0.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1)),'angle_confusion':{'TP':tp,'FP':fp,'FN':fn,'TN':tn}}

def round_interval(rows):
    rng=np.random.default_rng(1235)
    draws=rng.integers(0,len(rows),size=(10000,len(rows)))
    c=np.array([[r['angle_confusion'][k] for k in ('TP','FP','FN','TN')] for r in rows])
    sums=c[draws].sum(1);tp,fp,fn,tn=sums.T
    balanced=0.5*(tp/np.maximum(tp+fn,1)+tn/np.maximum(tn+fp,1))
    return {'unit':'whole test rounds, percentile bootstrap; descriptive within this single-grasp simulation distribution','balanced_accuracy_95':np.percentile(balanced,[2.5,97.5]).tolist()}

results={}
for name,cols in [('proprio',slice(0,88)),('ft',slice(88,148)),('combined',slice(0,148))]:
    x=design(h[:,:,:,cols]);cv=[]
    for alpha in alphas:
        errors=[]
        for held in range(4):
            rows=[j for j in range(4) if j!=held]
            model=fit(x[rows][good[rows]],y[rows][good[rows]],alpha)
            errors.extend(abs(predict(model,x[held][good[held]])-y[held][good[held]]).tolist())
        cv.append(float(np.mean(errors)))
    alpha=alphas[int(np.argmin(cv))];xx=x[:4][good[:4]];yy=y[:4][good[:4]];model=fit(xx,yy,alpha)
    if other is None:
        xt=x[4:];yt=y[4:];valid=good[4:]
    else:
        xt=design(other['sensory'][:,:,:,cols]);yt=other['truth'][:,:,-1,0];valid=other['valid'][:,:,-8:].all(-1)&np.isfinite(other['sensory']).all(axis=(-1,-2))
    pred=predict(model,xt[valid]);true=yt[valid]
    shuffled=fit(xx,np.random.default_rng(1234).permutation(yy),alpha)
    results[name]={'selected_alpha':alpha,'train_round_cv_MAE':dict(zip(map(str,alphas),cv)),'test':metrics(pred,true),'constant_mean_baseline':metrics(np.full(len(true),yy.mean()),true),'shuffled_training_labels_control':metrics(predict(shuffled,xt[valid]),true),'per_test_round':[metrics(predict(model,xt[j][valid[j]]),yt[j][valid[j]]) for j in range(len(xt))]}
for value in results.values():value['round_bootstrap']=round_interval(value['per_test_round'])
r={'scope':'Angle-information readout audit only; never full completion acceptance. All alpha selection uses first4trainingrounds only. Existing heldout streams are development data after earlier inspection; independent input is required for new confirmation.','training_input':a.input,'test_input':a.independent or a.input,'test_scope':'new independent stream' if a.independent else 'last4developmentrounds','feature_summary':'last8 mean/std plus entire-history endpointdelta','selection':'leaveone-traininground-out mean absolute angle error over fixedridgepenalties','results':results}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:{'alpha':v['selected_alpha'],'test':v['test'],'shuffle':v['shuffled_training_labels_control']} for k,v in results.items()}))
