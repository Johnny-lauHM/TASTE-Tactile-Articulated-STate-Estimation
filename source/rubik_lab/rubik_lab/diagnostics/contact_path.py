"""Diagnostic side-contact candidates for staged U/Up; kinematics only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rubik_lab.flick import FlickPlanner, FLICKS, NORMALS
from rubik_lab.hand_kinematics import quat_to_mat
p=argparse.ArgumentParser()
p.add_argument('--check-paths',type=Path,help='Read saved residual paths and report sampled active-index box penetration')
p.add_argument('--continuation',type=Path,help='Actual stopped UU failure states: inspect signed residual correction paths')
p.add_argument('--rigid-clearance',action='store_true')
p.add_argument('--surface-contact',action='store_true',help='Constrain pad to a side-face patch instead of a fixed material point')
p.add_argument('--wrist-assist',action='store_true',help='Coordinate wrist and holding fingers within remaining original motion budget')
p.add_argument('--recontact-plans',action='store_true',help='Use every actual pre-plan state recorded by the live recontact runner')
p.add_argument('--dense-geometry',action='store_true')
p.add_argument('--feasible-prefix-only',action='store_true',help='Audit the stored feasible partial stroke, not the rejected remainder')
p.add_argument('--ik-iters',type=int,default=60)
p.add_argument('--contact-press',type=float,default=.003)
p.add_argument('--contact-face',choices=['B','F','L','R'],help='Override contact surface only for actual-state continuation IK')
p.add_argument('--underturns',action='store_true',help='All failed UU negative residuals after peak speed and actual right-index readiness')
p.add_argument('--max-correction-deg',type=int,default=15)
p.add_argument('--faces',nargs='+',choices=['B','F','L','R'],default=['B','F','L','R'])
p.add_argument('--hand-selection',choices=['default','opposite'],default='default')
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.feasible_prefix_only and not a.check_paths:raise ValueError("Prefix audit requires --check-paths")
if a.out.with_suffix('.npz').exists() or a.out.with_suffix('.json').exists(): raise FileExistsError(a.out)
if a.check_paths:
    from rubik_lab.hand_kinematics import HandFK,axis_angle_to_mat
    import re
    asset=Path('assets/rubik_cube/rubik_cube_2body.usda').read_text()
    boxes={}
    for name in ('core','top'):
        section=asset.split('def Xform "'+name+'"',1)[1].split('def Xform "visuals"',1)[0]
        xyz=lambda label:torch.tensor([float(x) for x in re.search(label+r' = \(([^)]+)\)',section).group(1).split(',')])
        boxes[name]=(xyz('xformOp:translate'),xyz('xformOp:scale')*.5)
    z=np.load(a.check_paths.with_suffix('.npz'));report=json.loads(a.check_paths.with_suffix('.json').read_text())
    moving_whole_hand=bool(report.get('wrist_assist',False))
    dense_fingers=('thumb','index','middle','ring','pinky','hand') if moving_whole_hand else ('index',)
    fk={side:HandFK(side,'cpu',pts_per_link=256,seed=0,dense_fingers=dense_fingers if a.dense_geometry else ()) for side in ('left','right')}
    rows=[]
    for r in report['results']:
        key='U' if r['move']=='U' else 'Up';index=z[key+'_source_index'];match=np.flatnonzero((index[:,0]==r['trial'])&(index[:,1]==r['env']))
        if len(match)!=1:raise ValueError('Ambiguous source path')
        i=int(match[0]);end=12+r['required_prefix_degrees']
        if a.feasible_prefix_only:
            if r['first_failed_waypoint'] is not None:end=min(end,r['first_failed_waypoint']-1)
            if end<=12:
                rows.append(dict(trial=r['trial'],env=r['env'],move=r['move'],positive_turn_prefix=False,last_feasible_waypoint=end,turn_degrees=max(0,end-12)))
                continue
        q=torch.tensor(z[key+'_q'][i,:end+1])
        dp=torch.tensor(z[key+'_wrist_delta_pos'][i,:end+1]) if key+'_wrist_delta_pos' in z else torch.zeros(len(q),3)
        dr=torch.tensor(z[key+'_wrist_delta_rot'][i,:end+1]) if key+'_wrist_delta_rot' in z else torch.zeros(len(q),3)
        turn=torch.cat([torch.zeros(12),torch.arange(end-12+1)])
        if a.dense_geometry:
            weights=torch.arange(5,dtype=q.dtype)/5
            dp=torch.cat([(dp[:-1,None,:]+weights[None,:,None]*(dp[1:]-dp[:-1])[:,None,:]).flatten(0,1),dp[-1:]])
            dr=torch.cat([(dr[:-1,None,:]+weights[None,:,None]*(dr[1:]-dr[:-1])[:,None,:]).flatten(0,1),dr[-1:]])
            q=torch.cat([(q[:-1,None,:]+weights[None,:,None]*(q[1:]-q[:-1])[:,None,:]).flatten(0,1),q[-1:]])
            turn=torch.cat([(turn[:-1,None]+weights[None,:]*(turn[1:]-turn[:-1])[:,None]).flatten(),turn[-1:]])
        wp=torch.tensor(z[key+'_wrist_pos'][i]).expand(len(q),3);wr=torch.tensor(z[key+'_wrist_rot'][i]).expand(len(q),3,3)
        selected_links=[n for n in fk[r['side']].link_pts if moving_whole_hand or '_index_' in n]
        points=fk[r['side']].points(fk[r['side']].forward(wp+dp,axis_angle_to_mat(dr)@wr,q),links=selected_links)
        initial=fk[r['side']].points(fk[r['side']].forward(wp[:1],wr[:1],torch.tensor(z[key+'_initial_q'][i:i+1])),links=selected_links)
        angle=torch.deg2rad(-r['error_deg']+(1 if r['error_deg']>0 else -1)*turn)
        R=axis_angle_to_mat(torch.stack([torch.zeros_like(angle),torch.zeros_like(angle),angle],-1))
        worst={}
        for body,(center,half) in boxes.items():
            for link,pts in points.items():
                local=torch.einsum('bji,bpj->bpi',R,pts) if body=='top' else pts
                depth=(half-(local-center).abs()).amin(-1).clamp(min=0)
                value,flat=depth.flatten().max(0);step=int(flat)//depth.shape[1]
                initial_local=torch.einsum('bji,bpj->bpi',R[:1],initial[link]) if body=='top' else initial[link]
                initial_depth=(half-(initial_local-center).abs()).amin(-1).clamp(min=0).max()
                worst[body+'/'+link]={'depth_mm':float(value*1000),'waypoint':step,'initial_depth_mm':float(initial_depth*1000)}
        rigid_depths=[v['depth_mm'] for name,v in worst.items() if not name.endswith('_elastomer') or (name.startswith('core/') and '_index_' in name)]
        rows.append({'trial':r['trial'],'env':r['env'],'move':r['move'],'prefix_feasible':r['correction_prefix_feasible'],'positive_turn_prefix':end>12,'last_checked_waypoint':end,'turn_degrees':max(0,end-12),'rigid_and_active_pad_core_max_mm':max(rigid_depths),'sampled_clearance_pass':max(rigid_depths)<=.1,'penetration':worst})
    payload={'feasible_prefix_only':a.feasible_prefix_only,'moving_whole_hand':moving_whole_hand,'wrist_deltas_applied':True,'dense_geometry':a.dense_geometry,'dense_definition':'All collision-mesh vertices and triangle centroids, 5 joint/wrist-position/axis-angle-vector linear subdivisions per waypoint edge; sampled intersection evidence only, not exact continuous collision detection.','scope':'Deterministic collision-surface points for activeindex (wholehand when wristmoves) against actualUSD body boxes (256 samples normally; dense_definition when dense_geometry is true); positive penetration is evidence of intersection, zero is NOT proof of collision-free motion. Top angle follows hypothetical intended correction; no dynamics/self-collision check.', 'source':str(a.check_paths),'rows':rows}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.with_suffix('.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload));raise SystemExit(0)

if a.wrist_assist and not a.continuation:raise ValueError('Wrist assistance requires actual-state continuation input')
if a.underturns and (not a.continuation or a.recontact_plans):raise ValueError('Underturn selection requires continuation and excludes event replay')
if a.recontact_plans and not a.continuation:raise ValueError('Recontact plan extraction needs continuation input')

if a.continuation:
    z=np.load(a.continuation.with_suffix('.npz'))
    report=json.loads(a.continuation.with_suffix('.json').read_text())
    samples=[];selection=[]
    if a.recontact_plans:
        sources=[]
        for event in report['recontact_events']:
            if event['event']!='plan':continue
            env=event['env']
            match=np.flatnonzero((z['trace'][:,env,0]==event['trial'])&(z['trace'][:,env,1]==event['step']-1))
            if len(match)!=1:raise ValueError('Missing unique actual pre-plan state')
            frame=int(match[0]);sources.append((event['trial'],env,frame))
        for trial,env,frame in sources:
            t=z['trace'][frame,env];error=float(t[5]);side='left' if error>0 else 'right'
            if a.hand_selection=='opposite':side='right' if side=='left' else 'left'
            hand=0 if side=='left' else 1
            pose=z['poses'][frame,env];rot=quat_to_mat(torch.tensor(pose[3:7])).numpy()
            wp=rot.T@(pose[7+7*hand:10+7*hand]-z['env_origins'][env]-pose[:3])
            wr=rot.T@quat_to_mat(torch.tensor(pose[10+7*hand:14+7*hand])).numpy()
            fingers=z['sensory'][frame,env,22*hand:22*(hand+1)]
            samples.append(dict(trial=trial,env=env,source_frame=frame,source_step=int(t[1]),error_deg=error,side=side,move="U'" if error>0 else 'U',wp=wp,wr=wr,q=fingers))
    else:
        for episode in report['episodes']:
            if episode['pair']!=['U','U'] or episode['completed_stages']<1 or episode['reason']=='success':continue
            env=episode['env']
            ix=np.flatnonzero((z['trace'][:,env,0]==episode['trial']) & (z['trace'][:,env,3]==1) & (z['trace'][:,env,4]>0))
            t=z['trace'][ix,env]
            good=(np.abs(t[:,6])<.25)&(np.abs(t[:,5])>3)&(np.abs(t[:,5])<a.max_correction_deg)&(t[:,11]==0)
            if a.underturns:
                if episode['final_error_deg']>=0:continue
                ready=z['ready_targets'][episode['trial'],env,1,5:9]
                q=z['sensory'][ix,env,27:31];qd=z['sensory'][ix,env,71:75]
                force=z['contact'][ix,env,1,1,6:11].sum(-1)
                finger_ready=(np.abs(q-ready).max(-1)<=.2)&(np.abs(qd).max(-1)<=1)&(force<=.5)
                good&=(t[:,5]<-5)&finger_ready&(np.arange(len(t))>int(np.argmax(np.abs(t[:,6]))))
            windows=[j for j in range(3,len(t)) if good[j-3:j+1].all()]
            selection.append(dict(trial=episode['trial'],env=env,selected=bool(windows),final_error_deg=episode['final_error_deg']))
            if not windows:continue
            j=windows[0];frame=int(ix[j]);error=float(t[j,5])
            side='left' if error>0 else 'right'
            if a.hand_selection=='opposite':side='right' if side=='left' else 'left'
            hand=0 if side=='left' else 1
            pose=z['poses'][frame,env];rot=quat_to_mat(torch.tensor(pose[3:7])).numpy()
            wp=rot.T@(pose[7+7*hand:10+7*hand]-z['env_origins'][env]-pose[:3])
            wr=rot.T@quat_to_mat(torch.tensor(pose[10+7*hand:14+7*hand])).numpy()
            fingers=z['sensory'][frame,env,22*hand:22*(hand+1)]
            samples.append(dict(trial=episode['trial'],env=env,source_frame=frame,source_step=int(t[j,1]),error_deg=error,side=side,move="U'" if error>0 else 'U',wp=wp,wr=wr,q=fingers))
    for sample in samples:
        expected_side='right' if sample['move']=='U' else 'left'
        if a.hand_selection=='opposite':expected_side='left' if expected_side=='right' else 'right'
        if sample['side']!=expected_side:raise RuntimeError('Actual-state hand routing disagrees with requested selection')
    planner=FlickPlanner('cuda:0',steps_deg=1,max_deg=a.max_correction_deg,contact_press=a.contact_press)
    if a.rigid_clearance:
        from rubik_lab.hand_kinematics import HandFK
        planner.fk={side:HandFK(side,'cuda:0',pts_per_link=256,seed=0) for side in ('left','right')}
    results=[];raw={}
    for move in ('U',"U'"):
        selected=[x for x in samples if x['move']==move]
        if not selected:continue
        tensor=lambda key:torch.tensor(np.stack([x[key] for x in selected]),device='cuda:0')
        error=np.array([x['error_deg'] for x in selected])
        original_flick=FLICKS[move]
        FLICKS[move]=(selected[0]["side"],*original_flick[1:])
        if a.contact_face:
            cn=np.array(NORMALS[a.contact_face]);ax=np.array(NORMALS['U']);t=np.cross(cn,ax)
            t=t*(1 if t@np.array(NORMALS['L'])>0 else -1)
            sign=1 if move.endswith("'") else -1
            lat=float(sign*np.sign(np.cross(t,-cn)@ax))
            FLICKS[move]=(selected[0]["side"],original_flick[1],a.contact_face,False,lat)
        if a.wrist_assist:
            spec=FLICKS[move];FLICKS[move]=(*spec[:3],True,spec[4])
        peak=torch.tensor([[z['trace'][x['source_frame'],x['env'],14],z['trace'][x['source_frame'],x['env'],15]] for x in selected],device='cuda:0')
        budget=(.010-peak[:,0],np.deg2rad(10)-peak[:,1]) if a.wrist_assist else None
        # The target is an ideal quarterturn multiple. The physical square is
        # invariant under90deg, so current residual layer phase is -error.
        phi=torch.tensor(-np.abs(np.deg2rad(error)),device='cuda:0',dtype=torch.float32)
        q,dp,dr,ok=planner.solve(move,tensor('wp'),tensor('wr'),tensor('q'),phi=phi,iters=a.ik_iters,rigid_clearance=a.rigid_clearance,wrist_budget=budget,surface_contact=a.surface_contact)
        kinematic_ok=ok.clone()
        motion_ok=(peak[:,0,None]+dp.norm(dim=-1)<=.010)&(peak[:,1,None]+dr.norm(dim=-1)<=np.deg2rad(10))
        ok=ok&motion_ok.long().cumprod(1).bool()
        key='U' if move=='U' else 'Up'
        for field,value in dict(q=q,wrist_delta_pos=dp,wrist_delta_rot=dr,motion_ok=motion_ok,prefix_ok=ok,raw_ok=planner.last_ok_raw,pad=planner.last_pads,target=planner.last_targets,normal_alignment=planner.last_norm).items():raw[key+'_'+field]=value.cpu().numpy()
        raw[key+'_source_index']=np.array([[x['trial'],x['env'],x['source_frame']] for x in selected])
        raw[key+'_wrist_pos']=tensor('wp').cpu().numpy();raw[key+'_wrist_rot']=tensor('wr').cpu().numpy();raw[key+'_initial_q']=tensor('q').cpu().numpy()
        for i,x in enumerate(selected):
            end=planner.n_app+int(np.ceil(abs(x['error_deg'])))
            results.append({k:v for k,v in x.items() if k not in ('wp','wr','q')} | dict(approach_feasible=bool(ok[i,planner.n_app-1]),correction_prefix_feasible=bool(ok[i,end]),required_prefix_degrees=int(np.ceil(abs(x['error_deg']))),first_failed_waypoint=int((~ok[i]).nonzero()[0]) if (~ok[i]).any() else None,max_required_pad_error_m=float((planner.last_pads[i,:end+1]-planner.last_targets[i,:end+1]).norm(dim=-1).max())))
            results[-1]['kinematic_prefix_feasible']=bool(kinematic_ok[i,end])
            results[-1]['wrist_motion']=dict(additional_position_max_m=float(dp[i,:end+1].norm(dim=-1).max()),additional_rotation_max_deg=float(torch.rad2deg(dr[i,:end+1].norm(dim=-1)).max()),original_reference_position_bound_m=float(peak[i,0]+dp[i,:end+1].norm(dim=-1).max()),original_reference_rotation_bound_deg=float(torch.rad2deg(peak[i,1]+dr[i,:end+1].norm(dim=-1).max())),all_within_budget=bool(motion_ok[i,:end+1].all()))
            bad=results[-1]['first_failed_waypoint']
            if bad is not None:
                from rubik_lab.flick import rigid_finger_box_penetration
                from rubik_lab.hand_kinematics import axis_angle_to_mat
                with torch.no_grad():
                    frames=planner.fk[x['side']].forward(tensor('wp')[i:i+1]+dp[i:i+1,bad],axis_angle_to_mat(dr[i:i+1,bad])@tensor('wr')[i:i+1],q[i:i+1,bad])
                    angle=-x['error_deg']+(1 if x['error_deg']>0 else -1)*max(0,bad-planner.n_app)
                    rotation=axis_angle_to_mat(q.new_tensor([[0.,0.,np.deg2rad(angle)]]))
                    depth=float(rigid_finger_box_penetration(planner.fk[x['side']],frames,'index',rotation)[0])
                results[-1]['first_failure_components']=dict(pad_error_m=float((planner.last_pads[i,bad]-planner.last_targets[i,bad]).norm()),pad_tolerance_m=.006 if bad<planner.n_app else .004,normal_dot_face=float(planner.last_norm[i,bad,0]),normal_dot_axis=float(planner.last_norm[i,bad,1]),normal_gate_applies=bad>=planner.n_app-1,rigid_core_penetration_m=depth,clearance_tolerance_m=.0001)
        FLICKS[move]=original_flick
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out.with_suffix('.npz'),**raw)
    payload=dict(surface_contact=a.surface_contact,wrist_assist=a.wrist_assist,wrist_budget_definition="Conservative triangle bound: recorded original-reference whole-sequence peak plus proposed displacement/rotation; optimizer projects into remaining10mm/10deg budget with1e-6margin. No state/reference writes.",hand_selection=a.hand_selection,contact_face=a.contact_face,underturns=a.underturns,max_correction_deg=a.max_correction_deg,selection=selection,contact_press_m=a.contact_press,source=str(a.continuation),recontact_plans=a.recontact_plans,rigid_clearance=a.rigid_clearance,ik_iterations=a.ik_iters,scope='Actual stopped failure states, fixed measured wrists, existing single-index planner; no dynamics, full-link collision or global infeasibility proof. All qualifying failures retained; no state reset executed.',results=results)
    a.out.with_suffix('.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload))
    raise SystemExit(0)

c=np.load('cache/g70s_cycle_demo1_v3.npz')
phases=np.array([0,30,45,60,75],dtype=np.float32)
results=[]; raw={}
planner=FlickPlanner('cuda:0',max_deg=45,contact_press=a.contact_press)
R=quat_to_mat(torch.tensor(c['cube_quat'][0])).numpy()
for move,side,sgn in [('U','right',-1),("U'",'left',1)]:
    if a.hand_selection=='opposite': side='left' if side=='right' else 'right'
    position=R.T@(c[side+'_ee_pos'][0]-c['cube_pos'][0])
    rotation=R.T@quat_to_mat(torch.tensor(c[side+'_ee_quat'][0])).numpy()
    inputs=[torch.tensor(np.repeat(x[None],len(phases),axis=0),device='cuda:0',dtype=torch.float32) for x in (position,rotation,c[side+'_fingers'][0])]
    phi=torch.tensor(np.deg2rad(phases),device='cuda:0')
    original=FLICKS[move]
    for face in a.faces:
        cn=np.array(NORMALS[face]); ax=np.array(NORMALS['U'])
        tang=np.cross(cn,ax); t=tang*(1 if tang@np.array(NORMALS['L'])>0 else -1)
        # Positive compression acts opposite the outer surface normal.
        lat=float(sgn*np.sign(np.cross(t,-cn)@ax))
        FLICKS[move]=(side,'index',face,False,lat)
        q,dp,dr,ok=planner.solve(move,*inputs,phi=phi,iters=a.ik_iters)
        key=('U' if move=='U' else 'Up')+'_'+face
        reach=planner.reach_deg(ok).cpu().numpy()
        raw[key+'_raw_ok']=planner.last_ok_raw.cpu().numpy();raw[key+'_normal_alignment']=planner.last_norm.cpu().numpy()
        raw[key+'_q']=q.cpu().numpy();raw[key+'_prefix_ok']=ok.cpu().numpy()
        raw[key+'_pad']=planner.last_pads.cpu().numpy();raw[key+'_target']=planner.last_targets.cpu().numpy()
        raw[key+'_normal']=planner.last_tn.cpu().numpy()
        torque=np.cross(raw[key+'_target'][:,planner.n_app:],-raw[key+'_normal'][:,planner.n_app:])[:,:,2]*sgn
        results.append(dict(move=move,hand=side,face=face,lateral_sign=lat,initial_progress_degrees=phases.tolist(),additional_prefix_degrees=reach.tolist(),positive_compression_torque_arm_min_m=float(torque.min())))
        print(json.dumps(results[-1]),flush=True)
    FLICKS[move]=original
a.out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.out.with_suffix('.npz'),**raw)
a.out.with_suffix('.json').write_text(json.dumps(dict(hand_selection=a.hand_selection,ik_iterations=a.ik_iters,scope='Canonical fixed wrists/index and hypothetical rotated layer. No dynamics, contact collision or recontact feasibility proof.',results=results),indent=2)+'\n')
