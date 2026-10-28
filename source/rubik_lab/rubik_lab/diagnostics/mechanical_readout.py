"""Summarize full pre-reset mechanical traces without modifying them."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
from rubik_lab.evaluation_metrics import summarize_mechanical
p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--summary-only',action='store_true')
p.add_argument('--momentum-only',action='store_true',help='Rigid body momentum balance with runtime mass properties')
p.add_argument('--substeps-only',action='store_true',help='Check physics-rate orientation/velocity consistency and measured contact impulse')
p.add_argument('--terminal-only',action='store_true',help='Compact terminal and retry-state evidence from every active trial')
a=p.parse_args()
if a.out.exists(): raise FileExistsError(a.out)
r=json.loads(a.input.with_suffix('.json').read_text())
summary=None if r.get('physics_fixture') else summarize_mechanical(r)
if a.summary_only:
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(source=str(a.input),summary=summary),indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    raise SystemExit(0)
z=np.load(a.input.with_suffix('.npz'))
t=z['trace']; states=z['states']; plan=z['planner_state']; contacts=z['contacts']
if a.momentum_only:
    meta=r.get('cube_runtime_mass_properties')
    if meta is None:raise ValueError('Runtime mass/inertia/COM required; no silent analytic substitution')
    b=z['substep_body_state'];idx=z['substep_index'];active=z['substep_active'];ww=z['substep_wrenches'];dt=r['configuration_sources']['physics_dt']
    mass=np.asarray(meta['mass_kg']);inertia=np.asarray(meta['inertia']).reshape(*mass.shape,3,3);com=np.asarray(meta['com_local_pose'])
    if not np.allclose(com[...,3:],[0,0,0,1]):raise ValueError('Nonidentity principal COM orientation needs explicit handling')
    def rotation(q):
        q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,v=np.moveaxis(q,-1,0)
        return np.stack([1-2*(y*y+v*v),2*(x*y-w*v),2*(x*v+w*y),2*(x*y+w*v),1-2*(x*x+v*v),2*(y*v-w*x),2*(x*v-w*y),2*(y*v+w*x),1-2*(x*x+y*y)],-1).reshape(*q.shape[:-1],3,3)
    results=[];core=r['substep_schema']['body_names'].index(r['substep_schema'].get('reference_body','core'))
    for trial in np.unique(idx[:,0]):
        for env in range(b.shape[1]):
            rows=np.flatnonzero((idx[:,0]==trial)&active[:,env])
            if len(rows)<2:continue
            if not np.all(np.diff(idx[rows,1])==1):raise ValueError('Nonconsecutive active substeps')
            state=b[rows,env].astype(np.float64);R=rotation(state[:,:,3:7]);origin=state[0,core,:3].copy()
            pos=state[:,:,:3]+np.einsum('tbij,bj->tbi',R,com[env,:,:3]);vel=state[:,:,7:10];omega=state[:,:,10:13]
            iw=R@inertia[env]@np.swapaxes(R,-1,-2)
            linear=(mass[env,:,None]*vel).sum(1)
            angular=(np.einsum('tbij,tbj->tbi',iw,omega)+np.cross(pos-origin,mass[env,:,None]*vel)).sum(1)
            contact=ww[rows,env].astype(np.float64).sum((1,2));force=contact[:,:3]
            torque=contact[:,3:]+np.cross(state[:,core,:3]-origin,force)
            gravity=mass[env,:,None]*np.array([0.,0.,-9.81])
            gt=np.cross(pos-origin,gravity).sum(1)
            J=np.cumsum((force[1:]+gravity.sum(0))*dt,axis=0)
            K=np.cumsum((torque[1:]+(gt[:-1]+gt[1:])/2)*dt,axis=0)
            dp=linear[1:]-linear[0];dl=angular[1:]-angular[0]
            results.append(dict(trial=int(trial),env=env,seconds=float((idx[rows[-1],1]-idx[rows[0],1])*dt),linear_delta_kg_m_s=dp[-1].tolist(),external_linear_impulse_Ns=J[-1].tolist(),linear_residual_norm_Ns=float(np.linalg.norm(dp[-1]-J[-1])),angular_delta_kg_m2_s=dl[-1].tolist(),external_angular_impulse_Nms=K[-1].tolist(),angular_residual_norm_Nms=float(np.linalg.norm(dl[-1]-K[-1])),angular_peak_cumulative_residual_Nms=float(np.linalg.norm(dl-K,axis=1).max())))
    result=dict(source=str(a.input),results=results,scope='Two rigid-body momentum about fixed initial core origin; includes COM offsets, spin/orbital momentum, all measured hand contacts and9.81gravity. Omits unrecorded damping/stabilization/other contacts; not an exact solver conservation proof. Joint armature is not modeled as a separate physical rotor.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(0)
if a.substeps_only:
    index=z['substep_index'];body=z['substep_body_state'];active=z['substep_active'];wrench=z['substep_wrenches']
    if not len(index):raise ValueError('No physics-rate records')
    dt=r['configuration_sources']['physics_dt'];results=[]
    for trial in np.unique(index[:,0]):
        for env in range(body.shape[1]):
            rows=np.flatnonzero((index[:,0]==trial)&active[:,env])
            if len(rows)<2:continue
            end=rows[1:][np.diff(index[rows,1])==1];start=end-1
            for bi,name in enumerate(r['substep_schema']['body_names']):
                q0=body[start,env,bi,3:7];q1=body[end,env,bi,3:7]
                q0=q0/np.linalg.norm(q0,axis=-1,keepdims=True);q1=q1/np.linalg.norm(q1,axis=-1,keepdims=True)
                scalar=(q0*q1).sum(-1)
                vector=-q1[:,:1]*q0[:,1:]+q0[:,:1]*q1[:,1:]-np.cross(q1[:,1:],q0[:,1:])
                vector*=np.where(scalar<0,-1,1)[:,None];size=np.linalg.norm(vector,axis=-1)
                omega=vector/np.maximum(size[:,None],1e-12)*(2*np.arctan2(size,np.abs(scalar))/dt)[:,None]
                reported=(body[start,env,bi,10:13]+body[end,env,bi,10:13])/2
                mismatch=omega-reported
                results.append(dict(trial=int(trial),env=env,body=name,pairs=len(end),orientation_rate_rms_rad_s=float(np.sqrt(np.mean(omega**2))),reported_rate_rms_rad_s=float(np.sqrt(np.mean(reported**2))),mismatch_rms_rad_s=float(np.sqrt(np.mean(mismatch**2))),mismatch_peak_norm_rad_s=float(np.linalg.norm(mismatch,axis=-1).max()),contact_angular_impulse_about_moving_core_Nms=(wrench[end,env,:,:,3:].sum((1,2))*dt).sum(0).tolist()))
    result=dict(source=str(a.input),results=results,scope='Finite-difference world orientation versus trapezoidal COM angular velocity. Contact impulse about moving core is descriptive, not a complete angular-momentum balance; gravity and moving-origin terms excluded. body_state_w position is link-origin, velocities are COM.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));raise SystemExit(0)
if a.terminal_only:
    terminal=[]; retry_events=[]
    for ep in r['episodes']:
        env=ep['env']; rows=np.flatnonzero((t[:,env,0]==ep['trial'])&(t[:,env,2]>0))
        if not len(rows):raise ValueError('Episode has no active trace')
        last=rows[-1]
        terminal.append(dict(trial=ep['trial'],env=env,move=ep['move'],reason=ep['reason'],phase=int(t[last,env,3]),retry_count=int(plan[last,env,5]),withdrawal_stage=int(z['withdrawal_stages'][last,env]) if z['withdrawal_stages'].size else None))
        events=rows[1:][np.diff(plan[rows,env,5])>0]
        for i in events:
            retry_events.append(dict(trial=ep['trial'],env=env,move=ep['move'],step=int(t[i,env,1]),previous_finger_ready_error_rad=float(t[i-1,env,6]),previous_release_force_N=float(t[i-1,env,7]),previous_angle_error_deg=float(t[i-1,env,4]),previous_withdrawal_stage=int(z['withdrawal_stages'][i-1,env]) if z['withdrawal_stages'].size else None))
    result=dict(source=str(a.input),summary=summary,terminal=terminal,retry_events=retry_events,scope='All active trials; previous tick is before retry transition. Descriptive evidence, not proof of causality.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(episodes=len(terminal),retry_events=len(retry_events))))
    raise SystemExit(0)
module=ast.parse(Path('source/rubik_lab/rubik_lab/cube_contacts.py').read_text())
links=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='LINKS' for x in n.targets))
blocks=[]
for env in range(t.shape[1]):
    active=np.flatnonzero(t[:,env,2]>0)
    if not len(active): continue
    cuts=np.flatnonzero((np.diff(active)!=1)|(np.diff(t[active,env,0])!=0)|(np.diff(t[active,env,3])!=0))+1
    for rows in np.split(active,cuts):
        trial=int(t[rows[0],env,0]); ep=next(e for e in r['episodes'] if e['trial']==trial and e['env']==env)
        hand=1 if ep['move']=='U' else 0
        q=states[rows,env,21:65].reshape(-1,2,22)
        target=states[rows,env,109:153].reshape(-1,2,22)
        err=np.abs(q[:,hand,5:9]-target[:,hand,5:9]).max(1)
        top=contacts[rows,env,1].mean(0)
        ranked=sorted([(float(top[h,j]),('left','right')[h]+'_'+links[j]) for h in range(2) for j in range(len(links))],reverse=True)[:6]
        ps=plan[rows,env]
        blocks.append(dict(trial=trial,env=env,move=ep['move'],phase=int(t[rows[0],env,3]),step_start=int(t[rows[0],env,1]),step_end=int(t[rows[-1],env,1]),angle_error_start_end_deg=t[[rows[0],rows[-1]],env,4].tolist(),active_joint_tracking_error_median_max_rad=[float(np.median(err)),float(err.max())],mean_top_normal_force_by_link_N=ranked,active_index_top_force_median_N=float(np.median(contacts[rows,env,1,hand,6:11].sum(1))),planner_first_last=ps[[0,-1]].tolist(),fraction_at_or_past_path_end=float(np.mean(ps[:,0]>=ps[:,1]))))
if 'wrenches' in z and z['wrenches'].size:
    for block in blocks:
        env=block['env']
        rows=np.flatnonzero((t[:,env,0]==block['trial'])&(t[:,env,1]>=block['step_start'])&(t[:,env,1]<=block['step_end'])&(plan[:,env,0]>=12))
        if not len(rows) or block['phase']!=11: continue
        quat=states[rows,env,3:7]; w,x,y,zz=quat.T
        axis=np.stack([2*(x*zz+w*y),2*(y*zz-w*x),1-2*(x*x+y*y)],-1)
        torque=(z['wrenches'][rows,env,1,:,:,3:]*axis[:,None,None,:]).sum(-1)*(-1 if block['move']=='U' else 1)
        mean=torque.mean(0)
        block['directed_top_torque_mean_Nm']={'normal':float(mean[0].sum()),'friction':float(mean[1].sum()),'sum':float(mean.sum())}
        order=np.argsort(-np.abs(mean.sum(0)))[:6]
        block['top_pair_torques_Nm']=[dict(link=('left','right')[int(j)//27]+'_'+links[int(j)%27],normal=float(mean[0,j]),friction=float(mean[1,j])) for j in order]
release_events=[]
for env in range(t.shape[1]):
    changes=np.flatnonzero((t[1:,env,3]==12)&(t[:-1,env,3]==11)&(t[1:,env,2]>0)&(t[:-1,env,2]>0)&(t[1:,env,0]==t[:-1,env,0]))+1
    for i in changes:
        trial=int(t[i,env,0]); ep=next(e for e in r['episodes'] if e['trial']==trial and e['env']==env)
        quats=states[i-1:i+1,env,3:7];dot=abs(float(np.dot(quats[0],quats[1])))/(np.linalg.norm(quats[0])*np.linalg.norm(quats[1]))
        release_events.append(dict(trial=trial,env=env,move=ep['move'],step=int(t[i,env,1]),error_before_after_deg=t[i-1:i+1,env,4].tolist(),layer_speed_before_after_rad_s=t[i-1:i+1,env,5].tolist(),core_orientation_jump_deg=float(np.rad2deg(2*np.arccos(np.clip(dot,0,1)))),failed_at_release=bool(t[i,env,9])))
result=dict(summary=summary,release_events=release_events,scope='Descriptive phase blocks from all active trials; force norms exclude friction and cannot alone establish resisting torque.',source=str(a.input),planner_columns=r['planner_state_columns'],blocks=blocks)
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
