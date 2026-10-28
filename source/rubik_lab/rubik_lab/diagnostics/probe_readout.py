"""Read-only measured-motion audit of whole-hand diagnostic probes.

Requested targets are not motion evidence. Match each request to its recorded
control frame and actuator target before analyzing actual angle and contacts.
"""
import argparse,json
from pathlib import Path
import numpy as np


def summarize_probe(report,raw,phase_boundaries=None):
    probe=report.get('whole_hand_probe') or {}
    accepted=[e for e in probe.get('events',[]) if e['event']=='start_accepted']
    targets=np.asarray(raw.get('whole_hand_targets',np.empty((0,33))))
    result=dict(scope=('Development privileged correction andreturn; originalphysicalmonitor evidence only, no blind-policy/real-time/acceptance claim' if probe.get('ready_return') else 'Development privileged probe only; no full-action, blind-policy, real-time or acceptance claim'),requested_target_frames=len(targets),start_accepted=bool(accepted),confirmed_target_delivery=False,target_application_status='unconfirmed' if accepted and len(targets) else 'not_requested')
    if not accepted or not len(targets):return result
    if len(accepted)!=1:raise ValueError('Expected one preregistered probe start')
    event=accepted[0];trial,env,start=int(event['trial']),int(event['env']),int(event['step'])
    trace=raw['trace'][:,env];history=np.flatnonzero((trace[:,0]==trial)&(trace[:,1]==start-1))
    if len(history)!=1:raise ValueError('Missing unique pre-intervention frame')
    frames=[];elapsed=[];errors=[];missing=[]
    for target in targets:
        if tuple(target[:3].astype(int))!=(trial,int(target[1]),env):raise ValueError('Unexpected requested target cohort')
        found=np.flatnonzero((trace[:,0]==trial)&(trace[:,1]==target[1]))
        if len(found)!=1:missing.append(int(target[1]));continue
        i=int(found[0]);q_error=float(np.max(np.abs(raw['motor_targets'][i,env,1]-target[4:26])))
        wrist_error=float(np.max(np.abs(raw['wrist_targets'][i,env,1]-target[26:33])))
        errors.append(max(q_error,wrist_error))
        if not np.isfinite(errors[-1]) or errors[-1]>1e-6:continue
        frames.append(i);elapsed.append(float(target[3]))
    result.update(missing_requested_steps=missing,matched_actuator_frames=len(frames),max_logged_target_error=max(errors,default=None))
    if not frames:return result
    result['confirmed_target_delivery']=True
    result['target_application_status']='confirmed' if len(frames)==len(targets) else 'partially_confirmed'
    if len(set(frames))!=len(frames):raise ValueError('Duplicate requested frame')
    frames=np.array(frames);t=trace[frames];initial_error=float(trace[history[0],5]);dt=1/float(report['control_hz'])
    contacts=raw['contact'][frames,env]
    if not np.isfinite(t).all() or not np.isfinite(contacts).all() or not np.isfinite(initial_error):raise ValueError('Nonfinite measured motion/contact')
    pad_links=[5,10,15,20,26]
    per_hand=np.take(contacts,pad_links,axis=-1).sum(axis=1).max(axis=-1)
    top_pad=contacts[:,1,1,10];top_rigid=contacts[:,1,1,6:10].sum(-1)
    invalid=(t[:,11]>0)|(t[:,12]>.015)|(t[:,13]>np.deg2rad(15))|(t[:,14]>.010)|(t[:,15]>np.deg2rad(10))
    desired=np.sign(initial_error)*(initial_error-t[:,5])
    result.update(trial=trial,env=env,first_step=int(t[0,1]),last_step=int(t[-1,1]),initial_error_deg=initial_error,final_error_deg=float(t[-1,5]),minimum_abs_error_deg=float(np.abs(t[:,5]).min()),max_desired_progress_deg=float(desired.max()),final_desired_progress_deg=float(desired[-1]),entered_angle_tolerance=bool((np.abs(t[:,5])<5).any()),motion_or_drop_invalid_frames=int(invalid.sum()),per_hand_contact_missing_frames=(per_hand<=.05).sum(axis=0).tolist(),active_pad_top_contact_frames=int((top_pad>.05).sum()),active_pad_top_peak_N=float(top_pad.max()),active_pad_top_impulse_Ns=float(top_pad.sum()*dt),active_rigid_top_impulse_Ns=float(top_rigid.sum()*dt),peak_cube_position_mm=float(t[:,12].max()*1000),peak_cube_rotation_deg=float(np.rad2deg(t[:,13].max())),peak_wrist_position_mm=float(t[:,14].max()*1000),peak_wrist_rotation_deg=float(np.rad2deg(t[:,15].max())),contact_measurement_scope='Control-frame normal forces; rectangle-rule impulse estimate, not all physics-substep forces',actual_primitive_done_frames=int((t[:,10]>0).sum()))
    if phase_boundaries is not None:
        elapsed=np.array(elapsed);approach_end,turn_end=phase_boundaries
        returning=[e for e in probe.get('events',[]) if e['event']=='return_start_accepted']
        returned=[e for e in probe.get('events',[]) if e['event']=='return_targets_finished_holding_ready']
        prefix=np.ones(len(t),dtype=bool) if not returning else t[:,1]<returning[0]['step']
        phase_masks={'approach':prefix&(elapsed<=approach_end),'nominal_turn':prefix&(elapsed>approach_end)&(elapsed<=turn_end),'endpoint_hold':prefix&(elapsed>turn_end)}
        if returning:
            phase_masks['withdrawal']=(~prefix)&(t[:,1]<=returned[0]['step'] if returned else True)
            phase_masks['ready_hold']=(t[:,1]>returned[0]['step']) if returned else np.zeros(len(t),dtype=bool)
        changes=-np.sign(initial_error)*np.diff(np.r_[initial_error,t[:,5]])
        result['planned_phase_boundaries_seconds']=[float(approach_end),float(turn_end)]
        result['per_planned_phase']={name:dict(frames=int(mask.sum()),measured_progress_deg=float(changes[mask].sum()),pad_top_impulse_Ns=float(top_pad[mask].sum()*dt),rigid_top_impulse_Ns=float(top_rigid[mask].sum()*dt)) for name,mask in phase_masks.items()}
    finished=[e for e in probe.get('events',[]) if e['event']=='prefix_targets_finished_holding_endpoint']
    if finished:
        step=int(finished[0]['step']);at=t[t[:,1]==step]
        result['prefix_requested_finished_step']=step
        result['measured_error_at_prefix_finish_deg']=float(at[0,5]) if len(at)==1 else None
    return result


def summarize_wrenches(report,raw):
    """Sum moments at individual normal/friction points about measured axis."""
    frames=np.asarray(raw['probe_wrench_frames']);normal=np.asarray(raw['probe_normal_contacts']);friction=np.asarray(raw['probe_friction_contacts'])
    finished=[e for e in (report.get('whole_hand_probe') or {}).get('events',[]) if e['event']=='prefix_targets_finished_holding_endpoint']
    finish_step=int(finished[0]['step']) if finished else None
    rows=[]
    for frame in frames:
        key=frame[:3];n=normal[(normal[:,:3]==key).all(1)];f=friction[(friction[:,:3]==key).all(1)]
        if len(n)!=int(frame[14]) or len(f)!=int(frame[15]):raise ValueError('Manifold frame count mismatch')
        origin=frame[3:6];axis=frame[6:9];axis=axis/np.linalg.norm(axis)
        nf=n[:,4,None]*n[:,8:11];ff=f[:,4:7]
        nt=float(np.cross(n[:,5:8]-origin,nf).sum(0)@axis)
        ft=float(np.cross(f[:,7:10]-origin,ff).sum(0)@axis)
        # This is an estimate from recorded damping and speed, not a direct joint wrench sensor.
        damping=-float(frame[11]*frame[10]);explicit=float(frame[12])
        row=[*key,nt,ft,nt+ft,damping,explicit,float(frame[10]),float(frame[13])]
        if not np.isfinite(row).all():raise ValueError('Nonfinite measured wrench')
        rows.append(row)
    rows=np.asarray(rows).reshape(-1,10)
    def stats(mask):
        values=rows[mask]
        if not len(values):return dict(frames=0)
        return dict(frames=len(values),normal_torque_mean_Nm=float(values[:,3].mean()),friction_torque_mean_Nm=float(values[:,4].mean()),contact_torque_mean_Nm=float(values[:,5].mean()),contact_torque_range_Nm=[float(values[:,5].min()),float(values[:,5].max())],damping_torque_estimate_mean_Nm=float(values[:,6].mean()),explicit_effort_mean_Nm=float(values[:,7].mean()),mean_abs_layer_speed_rad_s=float(np.abs(values[:,8]).mean()),max_normal_sum_error_N=float(values[:,9].max()))
    targets=np.asarray(raw.get('whole_hand_targets',np.empty((0,33))))
    requested_keys={tuple(t[:3]) for t in targets}
    requested=np.array([tuple(row[:3]) in requested_keys for row in rows],dtype=bool)
    masks={'all_recorded':np.ones(len(rows),dtype=bool),'requested_target_frames':requested}
    returning=[e for e in (report.get('whole_hand_probe') or {}).get('events',[]) if e['event']=='return_start_accepted']
    returned=[e for e in (report.get('whole_hand_probe') or {}).get('events',[]) if e['event']=='return_targets_finished_holding_ready']
    prefix=np.ones(len(rows),dtype=bool) if not returning else rows[:,1]<returning[0]['step']
    if finish_step is not None:masks['endpoint_hold_after_0p25s']=requested & prefix & (rows[:,1]>=finish_step+int(round(.25*report['control_hz'])))
    if returning:masks['withdrawal']=requested & ~prefix & ((rows[:,1]<=returned[0]['step']) if returned else True)
    if returned:masks['ready_hold_after_0p25s']=requested & (rows[:,1]>=returned[0]['step']+int(round(.25*report['control_hz'])))
    return dict(scope='Per-contact moment sums about measured axis; normal andfriction separately. Control-frame samples, not fullsubstepimpulses. Hold subset starts fixed .25s after prefix target finish and is restricted to requested-target frames, not selected by motion outcome.',groups={name:stats(mask) for name,mask in masks.items()},per_frame_columns='trial,step,env,normal_torque_Nm,friction_torque_Nm,contact_torque_Nm,damping_torque_estimate_Nm,explicit_effort_Nm,layer_speed_rad_s,normal_sum_error_N',per_frame=rows.tolist())


def summarize_feedback(report,raw,path):
    """Check feedback against prior measured state and actual actuator records."""
    feedback=np.asarray(raw.get('whole_hand_feedback',np.empty((0,23))))
    if not len(feedback):return dict(frames=0)
    if feedback.ndim!=2 or feedback.shape[1]!=23 or not np.isfinite(feedback).all():raise ValueError('Invalid feedback records')
    if len(np.unique(feedback[:,:3],axis=0))!=len(feedback):raise ValueError('Duplicate feedback records')
    if path is None:raise ValueError('Feedback requires independently reconstructed physical reference')
    events=[e for e in report['whole_hand_probe']['events'] if e['event']=='start_accepted']
    if len(events)!=1:raise ValueError('Feedback requires one accepted start')
    bias=np.asarray(events[0]['finger_bias_rad'])[5:9];rows=[]
    targets=np.asarray(raw['whole_hand_targets'])
    for row in feedback:
        trial,step,env=map(int,row[:3]);trace=raw['trace'][:,env]
        if (trial,env)!=(events[0]['trial'],events[0]['env']):raise ValueError('Unexpected feedback cohort')
        limited=(np.abs(row[15:19]-row[11:15])>1e-12).astype(float)
        if not np.array_equal(limited,row[19:23]):raise ValueError('Feedback limiting flags mismatch')
        before=np.flatnonzero((trace[:,0]==trial)&(trace[:,1]==step-1))
        after=np.flatnonzero((trace[:,0]==trial)&(trace[:,1]==step))
        requested=targets[(targets[:,:3]==row[:3]).all(1)]
        if len(before)!=1 or len(after)!=1 or len(requested)!=1:raise ValueError('Missing feedback frame alignment')
        i,j=int(before[0]),int(after[0]);elapsed=float(requested[0,3])
        if elapsed<path.duration+.25:raise ValueError('Feedback before fixed hold window')
        reference=path.sample(elapsed)['fingers'][5:9]
        measured=raw['sensory'][i,env,27:31]
        command=raw['motor_targets'][j,env,1,5:9]
        previous=raw['motor_targets'][i,env,1,5:9]
        lower=raw['actuator_lower'][env,1,5:9];upper=raw['actuator_upper'][env,1,5:9]
        differences=[np.max(np.abs(reference-row[3:7])),np.max(np.abs(measured-row[7:11])),np.max(np.abs(reference-measured-row[11:15])),np.max(np.abs(reference+bias+row[15:19]-command))]
        if not np.isfinite(differences).all() or max(differences)>1e-6:raise ValueError('Feedback state/reference/command mismatch')
        # Rebuild command independently, including cap, actuator limits and rate.
        dt=1/report['control_hz'];desired=reference+bias+np.clip(reference-measured,-.1,.1)
        expected=np.clip(desired,np.maximum(lower,previous-dt),np.minimum(upper,previous+dt))
        if np.max(np.abs(expected-command))>1e-6:raise ValueError('Feedback rule mismatch')
        if (command<lower).any() or (command>upper).any():raise ValueError('Feedback actuator limit exceeded')
        rate=float(np.max(np.abs(command-previous))/dt)
        if rate>1+1e-5:raise ValueError('Feedback command rate exceeded')
        rows.append([step,float(np.max(np.abs(reference-measured))),float(np.max(np.abs(row[15:19]))),rate,int(np.count_nonzero(row[19:23]))])
    values=np.asarray(rows)
    return dict(frames=len(values),first_step=int(values[0,0]),last_step=int(values[-1,0]),prior_state_reference_and_actual_command_verified=True,max_index_error_rad=float(values[:,1].max()),mean_index_error_rad=float(values[:,1].mean()),max_applied_correction_rad=float(values[:,2].max()),max_command_rate_rad_s=float(values[:,3].max()),limited_joint_frames=int(values[:,4].sum()),scope='Proprioception from previous completed frame; fixed path reference, original actuator limits and gain1/rate/cap independently rebuilt. Not action acceptance.')


def summarize_tracking(report,raw,path):
    """Batch FK of measured joints/wrists versus the fixed physical path.

    This is not a direct physical pad-body pose calibration. Counterfactual
    q-only/wrist-only displacements are not additive causal contributions.
    """
    import torch
    from scipy.spatial.transform import Rotation
    from rubik_lab.hand_kinematics import HandFK
    targets=np.asarray(raw.get('whole_hand_targets',np.empty((0,33))))
    returning=[e for e in report['whole_hand_probe']['events'] if e['event']=='return_start_accepted']
    if returning:targets=targets[targets[:,1]<returning[0]['step']]
    if not len(targets):return dict(frames=0)
    if path is None:raise ValueError('Tracking requires reconstructed path')
    trial,env=int(targets[0,0]),int(targets[0,2]);trace=raw['trace'][:,env]
    frames=[]
    for target in targets:
        if (int(target[0]),int(target[2]))!=(trial,env):raise ValueError('Mixed tracking cohort')
        found=np.flatnonzero((trace[:,0]==trial)&(trace[:,1]==target[1]))
        if len(found)!=1:raise ValueError('Missing tracking frame')
        frames.append(int(found[0]))
    poses=raw['poses'][frames,env];q=raw['sensory'][frames,env,22:44]
    wp=poses[:,14:17]-raw['env_origins'][env]
    wr=Rotation.from_quat(poses[:,17:21][:,[1,2,3,0]]).as_matrix()
    planned=[path.sample(t) for t in targets[:,3]]
    rq=np.stack([v['fingers'] for v in planned]);rp=np.stack([v['wrist_position'] for v in planned])
    rr=Rotation.from_quat(np.stack([v['wrist_quaternion'] for v in planned])[:,[1,2,3,0]]).as_matrix()
    fk=HandFK('right','cpu',pts_per_link=8,seed=0)
    for table in (fk.T0,fk.axis,fk.link_pts):
        for key in table:table[key]=table[key].double()
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    def pad(q,p,r):return fk.pads(fk.forward(t(p),t(r),t(q)))['index'][0].numpy()
    actual=pad(q,wp,wr);nominal=pad(rq,rp,rr)
    q_only=pad(q,rp,rr);wrist_only=pad(rq,wp,wr)
    cube=Rotation.from_quat(path.cube_quaternion[[1,2,3,0]])
    angle=np.deg2rad(-path.source_error_deg-max(0,len(path.variables)-14))
    delta=Rotation.from_rotvec([0,0,angle]).inv().apply(cube.inv().apply(actual-nominal))*1000
    face=report['whole_hand_probe']['contact_face']
    if face not in ('R','B'):raise ValueError('Unknown tracking contact face')
    normal=np.array([1,0,0]) if face=='B' else np.array([0,-1,0]);tangent=np.cross([0,0,1],normal)
    directional=np.column_stack([delta@normal,delta@tangent,delta[:,2]])
    error=np.linalg.norm(actual-nominal,axis=1)*1000
    finished=[e for e in report['whole_hand_probe']['events'] if e['event']=='prefix_targets_finished_holding_endpoint']
    if len(finished)>1:raise ValueError('Ambiguous prefix finish')
    finish=int(finished[0]['step']) if finished else None
    hold=(targets[:,1]>=finish+int(round(.25*report['control_hz']))) if finish is not None else np.zeros(len(targets),dtype=bool)
    at=np.flatnonzero(targets[:,1]==finish) if finish is not None else []
    values=np.column_stack([error,np.linalg.norm(q_only-nominal,axis=1)*1000,np.linalg.norm(wrist_only-nominal,axis=1)*1000,np.max(np.abs(q[:,5:9]-rq[:,5:9]),axis=1),directional])
    if not np.isfinite(values).all():raise ValueError('Nonfinite FK tracking readout')
    result=dict(frames=len(values),hold_frames=int(hold.sum()),scope='Pre-withdrawal measured q/wrist FK versus fixed physical path; no direct pad-body pose calibration. q-only/wrist-only geometry is not additive causal attribution. Hold starts fixed .25s after this run own prefix finish; no outcome selection.')
    if len(at)==1:result['pad_error_at_prefix_finish_mm']=float(error[at[0]])
    if hold.any():
        mean=values[hold].mean(0)
        result.update(hold_mean_pad_error_mm=float(mean[0]),hold_mean_q_only_error_mm=float(mean[1]),hold_mean_wrist_only_error_mm=float(mean[2]),hold_mean_index_error_rad=float(mean[3]),hold_mean_error_normal_tangent_height_mm=mean[4:].tolist(),hold_max_pad_error_mm=float(error[hold].max()))
    return result


def summarize_return(report,raw,path):
    targets=np.asarray(raw.get('whole_hand_return_targets',np.empty((0,33))))
    if not len(targets):return dict(requested_frames=0,original_full_physical_monitor_passed=False)
    if path is None:raise ValueError('Missing audited return reference')
    events=report['whole_hand_probe']['events'];starts=[e for e in events if e['event']=='return_start_accepted'];ends=[e for e in events if e['event']=='return_targets_finished_holding_ready']
    if len(starts)!=1 or len(ends)>1:raise ValueError('Ambiguous return events')
    event=starts[0];trial,env=int(event['trial']),int(event['env']);trace=raw['trace'][:,env]
    bias=np.asarray(event['finger_bias_rad']);indices=[];max_error=0.;max_rate=0.
    source=np.flatnonzero((trace[:,0]==trial)&(trace[:,1]==event['step']-1))
    if len(source)!=1:raise ValueError('Missing return source frame')
    source=int(source[0]);support=[i for i in range(22) if i not in range(5,9)]
    for target in targets:
        if (int(target[0]),int(target[2]))!=(trial,env):raise ValueError('Mixed return cohort')
        current=np.flatnonzero((trace[:,0]==trial)&(trace[:,1]==target[1]));before=np.flatnonzero((trace[:,0]==trial)&(trace[:,1]==target[1]-1))
        if len(current)!=1 or len(before)!=1:raise ValueError('Missing actual return command frame')
        i,j=int(current[0]),int(before[0]);indices.append(i)
        motor=raw['motor_targets'][i,env,1];wrist=raw['wrist_targets'][i,env,1]
        reference=path.sample(float(target[3]))['fingers']+bias
        error=max(float(np.max(np.abs(motor-target[4:26]))),float(np.max(np.abs(wrist-target[26:33]))),float(np.max(np.abs(motor-reference))),float(np.max(np.abs(motor[support]-raw['motor_targets'][source,env,1,support]))),float(np.max(np.abs(wrist-raw['wrist_targets'][source,env,1]))))
        rate=float(np.max(np.abs(motor[5:9]-raw['motor_targets'][j,env,1,5:9]))*report['control_hz'])
        if not np.isfinite([error,rate]).all() or error>1e-6 or rate>1+1e-5:raise ValueError('Return actuator/reference/support/rate mismatch')
        max_error=max(max_error,error);max_rate=max(max_rate,rate)
    feedback=np.asarray(raw.get('whole_hand_feedback',np.empty((0,23))))
    if len(feedback) and ((feedback[:,0]==trial)&(feedback[:,2]==env)&(feedback[:,1]>=event['step'])).any():raise ValueError('Pushing feedback continued during withdrawal')
    t=trace[indices];last=indices[-1];ready=raw['ready_targets'][trial,env,1,5:9]
    completed=bool(ends and ((t[:,1]>=ends[0]['step'])&(t[:,10]>0)&(t[:,11]==0)&(t[:,3]==1)).any())
    episodes=[e for e in report.get('episodes',[]) if e['trial']==trial and e['env']==env]
    return dict(requested_frames=len(targets),actual_commands_verified=True,max_command_error=max_error,max_index_command_rate_rad_s=max_rate,support_and_wrist_commands_preserved=True,pushing_feedback_after_return=False,target_sequence_finished=bool(ends),original_full_physical_monitor_passed=completed,final_ready_error_rad=float(np.max(np.abs(raw['sensory'][last,env,27:31]-ready))),final_index_speed_rad_s=float(np.max(np.abs(raw['sensory'][last,env,71:75]))),final_top_index_release_force_N=float(raw['contact'][last,env,1,1,6:11].sum()),final_whole_index_release_force_N=float(raw['contact'][last,env,:,1,6:11].sum()),release_monitor_scope='Original runtime monitor is top-only; whole-index value independently sums core and top',episode=episodes[0] if len(episodes)==1 else None,scope='Audited privileged withdrawal; originalphysicalmonitor, not blinddetector, generalization, repeatability oracceptance evidence.')


def summarize_post_return(report,raw,stage=2,event_name='post_return_handoff',completed_stages=3):
    """Read only the appended action, using its actual participating hand."""
    events=[e for e in report.get('post_return_events',[]) if e['event']==event_name]
    holds=[e for e in report.get('post_return_events',[]) if e['event']=='post_return_motor_hold' and stage==2]
    if not events:return dict(attempted=False)
    if len(events)!=1 or len(events[0]['envs'])!=1:raise ValueError('Expected one predeclared post-return action')
    event=events[0];trial=event['trial'];env=event['envs'][0];step=event['step'];move=event['move']
    if move not in ('U',"U'"):raise ValueError('Unsupported post-return move')
    hand=1 if move=='U' else 0
    trace=np.asarray(raw['trace'])[:,env]
    selected=(trace[:,0]==trial)&(trace[:,3]==stage)&(trace[:,4]>0)
    idx=np.flatnonzero(selected)
    if not len(idx):raise ValueError('Handoff event without third-action frames')
    t=trace[idx]
    if not np.array_equal(t[:,1],np.arange(step+1,step+1+len(t))):raise ValueError('Missing post-return control frames')
    for key in ('whole_hand_targets','whole_hand_return_targets'):
        targets=np.asarray(raw.get(key,np.empty((0,33))))
        if len(targets) and ((targets[:,0]==trial)&(targets[:,2]==env)&(targets[:,1]>step)).any():raise ValueError('Old probe targets continued after handoff')
    c=np.asarray(raw['contact'])[idx,env];q=np.asarray(raw['sensory'])[idx,env]
    commands=np.asarray(raw['commands'])[idx,env];motors=np.asarray(raw['motor_targets'])[idx,env,hand,5:9]
    speed=np.abs(q[-1,44+22*hand+5:44+22*hand+9]).max()
    done=(t[:,10]>0)&(t[:,11]==0);elapsed=(t[-1,1]-step)/report['control_hz']
    episode=[e for e in report['episodes'] if e['trial']==trial and e['env']==env]
    if len(episode)!=1:raise ValueError('Missing unique sequence result')
    passed=bool(not holds and done.any() and elapsed<=report['timeout_per_primitive_s']+1e-6 and episode[0]['reason']=='success' and episode[0]['completed_stages']==completed_stages)
    return dict(attempted=True,motor_hold_interventions=holds,trial=trial,env=env,move=move,hand='right' if hand else 'left',frames=len(idx),original_elapsed_seconds=float(elapsed),original_full_monitor_passed=passed,final_error_deg=float(t[-1,5]),minimum_abs_error_deg=float(np.abs(t[:,5]).min()),motion_failure_frames=int((t[:,11]>0).sum()),pad_contact_frames=int((c[:,1,hand,10]>.05).sum()),peak_pad_force_N=float(c[:,1,hand,10].max()),final_top_index_release_force_N=float(c[-1,1,hand,6:11].sum()),final_whole_index_release_force_N=float(c[-1,:,hand,6:11].sum()),release_monitor_scope='Original runtime monitor is top-only; whole-index value independently sums core and top',final_index_speed_rad_s=float(speed),final_ready_error_rad=float(t[-1,7]),command_range=np.ptp(commands,axis=0).tolist(),motor_range=np.ptp(motors,axis=0).tolist(),last10_ready=t[-10:,9].tolist(),missing_support_frames=[int((c[:,:,h,:][:,:,[5,10,15,20,26]].max(axis=(1,2))<=.05).sum()) for h in range(2)],old_probe_targets_after_handoff=False,episode=episode[0],scope='One appended-action diagnostic; truth-gated handoff and original physical monitor, not blind switching or acceptance.')


def summarize_escape(report,raw,control):
    """Verify a single-target causal intervention against its unchanged-history control."""
    events=[e for e in report.get('post_return_events',[]) if e['event'] in ('post_return_escape_start','post_return_relax_start')]
    if len(events)!=1:raise ValueError('Expected one escape intervention')
    e=events[0];trial,env,step=int(e['trial']),int(e['env']),int(e['step'])
    t=np.asarray(raw['trace'])[:,env];b=np.asarray(control['trace'])[:,env]
    before=np.flatnonzero((t[:,0]==trial)&(t[:,1]<step))
    other=np.flatnonzero((b[:,0]==trial)&(b[:,1]<step))
    if not len(before) or t[before[-1],1]!=step-1 or not np.array_equal(t[before,1],b[other,1]):raise ValueError('Missing matched intervention prefix')
    comparisons={}
    for key in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed'):
        x=np.asarray(raw[key])[before,env];y=np.asarray(control[key])[other,env]
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.array_equal(x,y):raise ValueError('Changed preintervention history: '+key)
        comparisons[key]=0.
    idx=np.flatnonzero((t[:,0]==trial)&(t[:,1]>=step)&(t[:,3]==2)&(t[:,4]>0))
    if not len(idx) or not np.array_equal(t[idx,1],np.arange(step,step+len(idx))):raise ValueError('Missing escape frames')
    source=before[-1];motor=np.asarray(raw['motor_targets']);initial=np.asarray(e['source_motor_target'])
    if not np.allclose(initial,motor[source,env,0,5:9],atol=1e-7,rtol=0):raise ValueError('Escape target did not start from previous motor target')
    expected=np.repeat(initial[None],len(idx),axis=0);elapsed=(t[idx,1]-step+1)/report['control_hz']
    relaxation=e['event']=='post_return_relax_start'
    if relaxation:
        reference=np.asarray(e['physical_reference'])
        if not np.array_equal(reference,np.asarray(raw['sensory'])[source,env,5:9]):raise ValueError('Relaxation reference was not previous physical q')
        expected+=np.clip(reference-initial,-elapsed[:,None],elapsed[:,None])
    else:expected[:,1]-=np.minimum(elapsed,.4)
    actual=motor[idx,env,0,5:9];error=float(np.max(np.abs(actual-expected)))
    if not np.isfinite(actual).all() or error>1e-6:raise ValueError('Escape target not delivered')
    rate=float(np.max(np.abs(np.diff(np.vstack([initial,actual]),axis=0)))*report['control_hz'])
    if rate>1+1e-5:raise ValueError('Escape target rate exceeded')
    lower=np.asarray(raw['actuator_lower'])[env,0,5:9];upper=np.asarray(raw['actuator_upper'])[env,0,5:9]
    if np.any(actual<lower) or np.any(actual>upper):raise ValueError('Escape target outside actuator limits')
    q=np.asarray(raw['sensory'])[idx,env,5:9];contacts=np.asarray(raw['contact'])[idx,env]
    if not np.isfinite(q).all() or not np.isfinite(contacts).all() or not np.isfinite(t[idx]).all():raise ValueError('Nonfinite escape measurements')
    support=[j for j in range(22) if j not in range(5,9)]
    support_error=float(np.max(np.abs(motor[idx,env,0][:,support]-motor[source,env,0,support])))
    other_error=float(np.max(np.abs(motor[idx,env,1]-motor[source,env,1])))
    wrist_error=float(np.max(np.abs(np.asarray(raw['wrist_targets'])[idx,env]-np.asarray(raw['wrist_targets'])[source,env])))
    force=t[idx,8];core_force=contacts[:,0,0,6:11].sum(axis=1);top_force=contacts[:,1,0,6:11].sum(axis=1);total_force=core_force+top_force
    good=(force<=.5)&(total_force<=.5)&(np.abs(t[idx,5])<5)&(np.abs(t[idx,6])<.25)&(np.asarray(raw['core_angular_speed'])[idx,env]<1)&(t[idx,11]==0)
    missing=[int((contacts[:,:,h,:][:,:,[5,10,15,20,26]].max(axis=(1,2))<=.05).sum()) for h in range(2)]
    return dict(intervention_kind=e['event'],prefix_exact_match=comparisons,source_step=step-1,intervention_step=step,frames=len(idx),motor_max_error_rad=error,motor_max_rate_rad_s=rate,support_motor_change_rad=support_error,opposite_hand_motor_change_rad=other_error,wrist_target_change=wrist_error,source_angle_error_deg=float(t[source,5]),final_angle_error_deg=float(t[idx[-1],5]),max_abs_angle_error_deg=float(np.abs(t[idx,5]).max()),final_release_force_N=float(force[-1]),release_monitor_scope='Original trace release metric; report both bodies independently',final_index_core_force_N=float(core_force[-1]),peak_index_core_force_N=float(core_force.max()),final_index_top_force_N=float(top_force[-1]),final_index_total_force_N=float(total_force[-1]),index_total_force_over_limit_frames=int((total_force>.5).sum()),peak_release_force_N=float(force.max()),release_force_over_limit_frames=int((force>.5).sum()),final_index_q=q[-1].tolist(),index_q_change=(q[-1]-np.asarray(raw['sensory'])[source,env,5:9]).tolist(),last10_unloaded_and_quiet=good[-10:].tolist(),missing_support_frames=missing,motion_failure_frames=int((t[idx,11]>0).sum()),first_segment_supported=bool(len(idx)>=16 and np.all(good[-10:]) and np.all(np.abs(t[idx,5])<5) and not any(missing) and np.all(t[idx,11]==0) and support_error<=1e-6 and other_error<=1e-6 and wrist_error<=1e-6),scope='First-segment causal diagnosis only; not full withdrawal, blind completion, repeatability or acceptance.')


def summarize_left_return(report,raw,control,path):
    """Independent measured delivery and full-monitor readout for left withdrawal."""
    starts=[e for e in report.get('post_return_events',[]) if e['event']=='left_return_start']
    ends=[e for e in report.get('post_return_events',[]) if e['event']=='left_return_targets_finished']
    if len(starts)!=1 or len(ends)>1:raise ValueError('Ambiguous left return events')
    e=starts[0];trial,env,step=int(e['trial']),int(e['env']),int(e['step'])
    t=np.asarray(raw['trace'])[:,env];bt=np.asarray(control['trace'])[:,env]
    before=np.flatnonzero((t[:,0]==trial)&(t[:,1]<step));old=np.flatnonzero((bt[:,0]==trial)&(bt[:,1]<step))
    if not len(before) or t[before[-1],1]!=step-1 or not np.array_equal(t[before,1],bt[old,1]):raise ValueError('Missing left return prefix')
    for key in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed'):
        x=np.asarray(raw[key])[before,env];y=np.asarray(control[key])[old,env]
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.array_equal(x,y):raise ValueError('Left return prefix mismatch: '+key)
    idx=np.flatnonzero((t[:,0]==trial)&(t[:,1]>=step)&(t[:,3]==2)&(t[:,4]>0))
    if not len(idx) or not np.array_equal(t[idx,1],np.arange(step,step+len(idx))):raise ValueError('Missing left return frames')
    elapsed=(t[idx,1]-step+1)/report['control_hz'];source=before[-1]
    expected=np.stack([np.interp(elapsed,path.times,path.motor[:,j]) for j in range(5,9)],axis=1)
    motor=np.asarray(raw['motor_targets']);actual=motor[idx,env,0,5:9]
    error=float(np.max(np.abs(actual-expected)))
    rate=float(np.max(np.abs(np.diff(np.vstack([motor[source,env,0,5:9],actual]),axis=0)))*report['control_hz'])
    if not np.isfinite(actual).all() or error>1e-6 or rate>1+1e-5:raise ValueError('Left return actual target or rate mismatch')
    support=[j for j in range(22) if j not in range(5,9)]
    if not np.array_equal(motor[idx,env,0][:,support],np.repeat(motor[source,env,0,support][None],len(idx),axis=0)) or not np.array_equal(motor[idx,env,1],np.repeat(motor[source,env,1][None],len(idx),axis=0)):raise ValueError('Other finger targets changed')
    wrist=np.asarray(raw['wrist_targets'])
    if not np.array_equal(wrist[idx,env],np.repeat(wrist[source,env][None],len(idx),axis=0)):raise ValueError('Wrist targets changed')
    lo=np.asarray(raw['actuator_lower'])[env,0,5:9];hi=np.asarray(raw['actuator_upper'])[env,0,5:9]
    if np.any(actual<lo-1e-7) or np.any(actual>hi+1e-7):raise ValueError('Motor bound violation')
    contacts=np.asarray(raw['contact'])[idx,env];core=contacts[:,0,0,6:11].sum(axis=1);top=contacts[:,1,0,6:11].sum(axis=1);total=core+top
    q=np.asarray(raw['sensory'])[idx,env];ready=np.asarray(raw['ready_targets'])[trial,env,0,5:9]
    qerror=np.max(np.abs(q[:,5:9]-ready),axis=1);qspeed=np.max(np.abs(q[:,49:53]),axis=1)
    missing=[contacts[:,:,h,:][:,:,[5,10,15,20,26]].max(axis=(1,2))<=.05 for h in range(2)]
    finished=elapsed>=path.duration-1e-9
    if ends and (not finished.any() or ends[0]['step']!=t[idx[np.flatnonzero(finished)[0]],1]):raise ValueError('Incorrect finish event timing')
    good=finished&(t[idx,9]>0)&(t[idx,11]==0)&(np.abs(t[idx,5])<5)&(np.abs(t[idx,6])<.25)&(np.asarray(raw['core_angular_speed'])[idx,env]<1)&(qerror<=.2)&(qspeed<=1)&(total<=.5)&~missing[0]&~missing[1]
    if not np.isfinite(np.r_[total,qerror,qspeed,t[idx,5]]).all():raise ValueError('Nonfinite completion measurements')
    episodes=[v for v in report['episodes'] if v['trial']==trial and v['env']==env]
    if len(episodes)!=1:raise ValueError('Missing sequence result')
    handoffs=[v for v in report['post_return_events'] if v['event']=='post_return_handoff']
    if len(handoffs)!=1:raise ValueError('Missing original third clock')
    seconds=(t[idx[-1],1]-handoffs[0]['step'])/report['control_hz']
    fourth=[v for v in report['post_return_events'] if v['event']=='after_left_return_handoff']
    advanced=bool(len(fourth)==1 and fourth[0]['step']==t[idx[-1],1] and episodes[0]['completed_stages']>=3)
    terminal=episodes[0]['reason']=='success' and episodes[0]['completed_stages']==3
    passed=bool(ends and len(good)>=10 and good[-10:].all() and t[idx[-1],10]>0 and seconds<=report['timeout_per_primitive_s']+1e-6 and (advanced or terminal))
    return dict(prefix_exact_match=True,frames=len(idx),max_motor_error_rad=error,max_motor_rate_rad_s=rate,other_targets_preserved=True,original_elapsed_seconds=float(seconds),final_angle_error_deg=float(t[idx[-1],5]),final_ready_error_rad=float(qerror[-1]),final_index_speed_rad_s=float(qspeed[-1]),final_core_force_N=float(core[-1]),final_top_force_N=float(top[-1]),peak_total_force_N=float(total.max()),last10_full_conditions=good[-10:].tolist(),motion_failure_frames=int((t[idx,11]>0).sum()),missing_support_frames=[int(x.sum()) for x in missing],target_path_finished=bool(ends),original_full_physical_monitor_passed=passed,episode=episodes[0],scope='One privileged full withdrawal diagnostic, not blind policy or acceptance.')


def summarize_fourth(report,raw,control=None):
    events=[e for e in report.get('post_return_events',[]) if e['event']=='after_left_return_handoff']
    if len(events)!=1 or len(events[0]['envs'])!=1:raise ValueError('Missing unique fourth handoff')
    e=events[0];env=e['envs'][0];trial=e['trial'];step=e['step']
    if not e.get('diagnostic_targets_cleared') or e.get('preserved_fields',0)<42:raise ValueError('Incomplete fourth handoff audit')
    t=np.asarray(raw['trace'])[:,env]
    if control is not None:
        bt=np.asarray(control['trace'])[:,env]
        before=np.flatnonzero((t[:,0]==trial)&(t[:,1]<=step));old=np.flatnonzero((bt[:,0]==trial)&(bt[:,1]<=step))
        if not len(before) or t[before[-1],1]!=step or not np.array_equal(t[before,1],bt[old,1]):raise ValueError('Missing fourth prefix')
        for key in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed'):
            x=np.asarray(raw[key])[before,env];y=np.asarray(control[key])[old,env]
            if x.shape!=y.shape or not np.isfinite(x).all() or not np.array_equal(x,y):raise ValueError('Fourth prefix mismatch: '+key)
    result=summarize_post_return(report,raw,stage=3,event_name='after_left_return_handoff',completed_stages=4)
    idx=np.flatnonzero((t[:,0]==trial)&(t[:,3]==3)&(t[:,4]>0));hand=1 if e['move']=='U' else 0
    contacts=np.asarray(raw['contact'])[idx,env,:,hand,6:11];force=contacts.sum(axis=(1,2))
    release=bool(len(force)>=10 and np.isfinite(force).all() and (force[-10:]<=.5).all())
    result.update(prefix_exact_match=control is not None,prefix_comparison_scope='Independent control compared' if control is not None else 'No independent matched-control prefix; actual-state handoff audit only',handoff_preserved_fields=e['preserved_fields'],diagnostic_targets_cleared=True,final_all_body_index_force_N=float(force[-1]),last10_all_body_released=release)
    result['original_full_monitor_passed']=bool(result['original_full_monitor_passed'] and release)
    return result


def summarize_preload_brake(report,raw):
    """Check causal trigger and actual preload-removal commands without a matched-control claim."""
    events=[e for e in report.get('post_return_events',[]) if e['event']=='post_return_relax_start']
    if len(events)!=1:raise ValueError('Missing unique preload trigger')
    e=events[0];env=e['env'];trial=e['trial'];step=e['step'];threshold=report['diagnostic_preload_at_error_deg']
    t=np.asarray(raw['trace'])[:,env];eligible=(t[:,0]==trial)&(t[:,3]==2)&(t[:,4]>0)&(t[:,5]>0)&(t[:,5]<=threshold)&(t[:,6]>0)
    triggers=np.flatnonzero(eligible)
    if not len(triggers) or t[triggers[0],1]!=step-1:raise ValueError('Trigger was not first completed eligible state')
    source=int(triggers[0]);idx=np.flatnonzero((t[:,0]==trial)&(t[:,3]==2)&(t[:,4]>0)&(t[:,1]>=step))
    if not len(idx) or not np.array_equal(t[idx,1],np.arange(step,step+len(idx))):raise ValueError('Missing preload command frames')
    initial=np.asarray(e['source_motor_target']);reference=np.asarray(e['physical_reference']);motor=np.asarray(raw['motor_targets']);sensory=np.asarray(raw['sensory'])
    if not np.array_equal(initial,motor[source,env,0,5:9]) or not np.array_equal(reference,sensory[source,env,5:9]):raise ValueError('Reference not taken from preceding state')
    elapsed=(t[idx,1]-step+1)/report['control_hz'];expected=initial+np.clip(reference-initial,-elapsed[:,None],elapsed[:,None]);actual=motor[idx,env,0,5:9]
    error=float(np.max(np.abs(expected-actual)));rate=float(np.max(np.abs(np.diff(np.vstack([initial,actual]),axis=0)))*report['control_hz'])
    if not np.isfinite(actual).all() or error>1e-6 or rate>1+1e-5:raise ValueError('Preload motor delivery/rate failed')
    lo=np.asarray(raw['actuator_lower'])[env,0,5:9];hi=np.asarray(raw['actuator_upper'])[env,0,5:9]
    if np.any(actual<lo-1e-7) or np.any(actual>hi+1e-7):raise ValueError('Preload motor bounds failed')
    support=[j for j in range(22) if j not in range(5,9)]
    for current,prior in [(motor[idx,env,0][:,support],motor[source,env,0,support]),(motor[idx,env,1],motor[source,env,1]),(np.asarray(raw['wrist_targets'])[idx,env],np.asarray(raw['wrist_targets'])[source,env])]:
        if not np.array_equal(current,np.repeat(prior[None],len(idx),axis=0)):raise ValueError('Other actuator targets changed')
    c=np.asarray(raw['contact'])[idx,env];core=c[:,0,0,6:11].sum(axis=1);top=c[:,1,0,6:11].sum(axis=1);total=core+top
    missing=[c[:,:,h,:][:,:,[5,10,15,20,26]].max(axis=(1,2))<=.05 for h in range(2)]
    omega=np.asarray(raw['core_angular_speed'])[idx,env];duration=float(np.max(np.abs(reference-initial)))
    finite=np.isfinite(c).all() and np.isfinite(t[idx]).all() and np.isfinite(omega).all()
    if not finite:raise ValueError('Nonfinite preload measurement')
    good=(elapsed>=duration)&(np.abs(t[idx,5])<5)&(np.abs(t[idx,6])<.25)&(omega<1)&(total<=.5)&(t[idx,11]==0)&~missing[0]&~missing[1]
    stable=bool(len(good)>=10 and good[-10:].all() and not (t[idx,11]>0).any() and not any(x.any() for x in missing))
    return dict(trigger_step=step,source_step=step-1,source_error_deg=float(t[source,5]),source_layer_speed_rad_s=float(t[source,6]),first_eligible_trigger_verified=True,max_motor_error_rad=error,max_motor_rate_rad_s=rate,other_targets_preserved=True,ramp_duration_s=duration,final_error_deg=float(t[idx[-1],5]),max_overshoot_deg=float(max(0,-t[idx,5].min())),final_core_force_N=float(core[-1]),final_top_force_N=float(top[-1]),peak_total_force_N=float(total.max()),last10_quiet_released=good[-10:].tolist(),braking_hypothesis_supported=stable,matched_control_verified=False,scope='Preload braking component only; no fullready/fullaction/acceptance claim. Matched-control comparison must be reported separately.')



def summarize_residual_escape(report,raw,control,physical_delta):
    """Independent command and paired-trajectory audit of a small residual escape."""
    events=[e for e in report.get('post_return_events',[]) if e['event']=='preload_residual_escape_start']
    if len(events)!=1:raise ValueError('Missing unique residual escape event')
    e=events[0];env=e['env'];trial=e['trial'];step=e['step']
    t=np.asarray(raw['trace'])[:,env];bt=np.asarray(control['trace'])[:,env]
    before=np.flatnonzero((t[:,0]==trial)&(t[:,1]<step));old=np.flatnonzero((bt[:,0]==trial)&(bt[:,1]<step))
    if not len(before) or t[before[-1],1]!=step-1 or not np.array_equal(t[before,1],bt[old,1]):raise ValueError('Missing residual prefix')
    for key in ('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed'):
        x=np.asarray(raw[key])[before,env];y=np.asarray(control[key])[old,env]
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.array_equal(x,y):raise ValueError('Residual prefix mismatch: '+key)
    idx=np.flatnonzero((t[:,0]==trial)&(t[:,1]>=step)&(t[:,3]==2)&(t[:,4]>0))
    old=np.flatnonzero((bt[:,0]==trial)&(bt[:,1]>=step)&(bt[:,3]==2)&(bt[:,4]>0))
    if not len(idx) or not np.array_equal(t[idx,1],bt[old,1]) or not np.array_equal(t[idx,1],np.arange(step,step+len(idx))):raise ValueError('Missing paired residual frames')
    delta=np.asarray(physical_delta)
    if delta.shape!=(4,) or not np.isfinite(delta).all() or not 0<np.max(np.abs(delta))<=.020000001:raise ValueError('Invalid residual delta')
    source=before[-1];motor=np.asarray(raw['motor_targets']);initial=motor[source,env,0,5:9];elapsed=(t[idx,1]-step+1)/report['control_hz']
    expected=initial+np.clip(delta,-elapsed[:,None],elapsed[:,None]);actual=motor[idx,env,0,5:9]
    error=float(np.max(np.abs(expected-actual)));rate=float(np.max(np.abs(np.diff(np.vstack([initial,actual]),axis=0)))*report['control_hz'])
    if not np.isfinite(actual).all() or error>1e-6 or rate>1+1e-5:raise ValueError('Residual motor delivery/rate failed')
    if np.any(actual<np.asarray(raw['actuator_lower'])[env,0,5:9]-1e-7) or np.any(actual>np.asarray(raw['actuator_upper'])[env,0,5:9]+1e-7):raise ValueError('Residual motor bounds failed')
    support=[j for j in range(22) if j not in range(5,9)]
    bm=np.asarray(control['motor_targets'])
    if not np.array_equal(motor[idx,env,0][:,support],bm[old,env,0][:,support]) or not np.array_equal(motor[idx,env,1],bm[old,env,1]) or not np.array_equal(np.asarray(raw['wrist_targets'])[idx,env],np.asarray(control['wrist_targets'])[old,env]):raise ValueError('Unrelated target trajectory differs from control')
    c=np.asarray(raw['contact'])[idx,env];force=c[:,:,0,6:11].sum(axis=(1,2));omega=np.asarray(raw['core_angular_speed'])[idx,env];q=np.asarray(raw['sensory'])[idx,env]
    if not np.isfinite(c).all() or not np.isfinite(t[idx]).all() or not np.isfinite(omega).all() or not np.isfinite(q).all():raise ValueError('Nonfinite residual data')
    missing=[c[:,:,h,:][:,:,[5,10,15,20,26]].max(axis=(1,2))<=.05 for h in range(2)]
    good=(np.abs(t[idx,5])<5)&(np.abs(t[idx,6])<.25)&(omega<1)&(force<=.5)&(np.abs(q[:,49:53]).max(axis=1)<=1)&(t[idx,11]==0)&~missing[0]&~missing[1]
    supported=bool(len(good)>=10 and good[-10:].all() and not (t[idx,11]>0).any() and not any(x.any() for x in missing))
    return dict(prefix_exact_match=True,other_target_trajectories_match=True,frames=len(idx),max_motor_error_rad=error,max_motor_rate_rad_s=rate,final_error_deg=float(t[idx[-1],5]),control_final_error_deg=float(bt[old[-1],5]),final_whole_index_force_N=float(force[-1]),peak_whole_index_force_N=float(force.max()),last10_quiet_released=good[-10:].tolist(),hypothesis_supported=supported,motion_failure_frames=int((t[idx,11]>0).sum()),missing_support_frames=[int(v.sum()) for v in missing],scope='Matched residual-contact intervention only; no complete withdrawal, blind control, or acceptance credit.')


def audit_live_left_timing(report,raw):
    """Independently reconstruct first eligible completed-state decisions."""
    events=report.get('post_return_events',[])
    relax=[e for e in events if e['event']=='post_return_relax_start']
    if len(relax)!=1:raise ValueError('Missing live preload event')
    e=relax[0];env=e['env'];trial=e['trial'];hz=report['control_hz']
    t=np.asarray(raw['trace'])[:,env];q=np.asarray(raw['sensory'])[:,env];c=np.asarray(raw['contact'])[:,env];omega=np.asarray(raw['core_angular_speed'])[:,env]
    trigger=np.flatnonzero((t[:,0]==trial)&(t[:,3]==2)&(t[:,4]>0)&(t[:,5]>0)&(t[:,5]<=report['diagnostic_preload_at_error_deg'])&(t[:,6]>0))
    if not len(trigger) or t[trigger[0],1]!=e['step']-1:raise ValueError('Live preload trigger was not first eligible')
    i=trigger[0];reference=np.asarray(e['physical_reference']);initial=np.asarray(e['source_motor_target'])
    if not np.array_equal(reference,q[i,5:9]) or not np.array_equal(initial,np.asarray(raw['motor_targets'])[i,env,0,5:9]):raise ValueError('Noncausal live preload reference')
    minimum=e['step']+int(np.ceil(np.max(np.abs(reference-initial))*hz-1e-9))-1
    total=c[:,:,0,6:11].sum(axis=(1,2))
    finite=np.isfinite(t).all(axis=1)&np.isfinite(q).all(axis=1)&np.isfinite(c).all(axis=(1,2,3))&np.isfinite(omega)
    quiet=finite&(t[:,0]==trial)&(t[:,3]==2)&(t[:,4]>0)&(np.abs(t[:,5])<5)&(np.abs(t[:,6])<.25)&(omega<1)&(np.abs(q[:,49:53]).max(axis=1)<=1)&(t[:,11]==0)
    for h in range(2):quiet &= c[:,:,h,:][:,:,[5,10,15,20,26]].max(axis=(1,2))>.05
    def first_end(minimum,release):
        valid=quiet&(t[:,1]>=minimum)
        if release:valid &= total<=.5
        for j in range(3,len(t)):
            if valid[j-3:j+1].all() and np.array_equal(t[j-3:j+1,1],np.arange(t[j,1]-3,t[j,1]+1)):return j
        return None
    plans=[v for v in events if v['event'] in ('preload_residual_escape_start','left_return_start','left_live_plan_rejected')]
    first=first_end(minimum,False)
    if first is None:
        if plans:raise ValueError('Live plan without quiet window')
        return dict(first_eligible_decisions_verified=True,attempts=0)
    kind='full' if (total[first-3:first+1]<=.5).all() else 'residual'
    def check(plan,kind,source):
        actual=plan.get('kind') if plan['event']=='left_live_plan_rejected' else ('full' if plan['event']=='left_return_start' else 'residual')
        if actual!=kind or plan['step']!=t[source,1]+1:raise ValueError('Live decision was not first eligible actual state')
    if not plans:raise ValueError('Missing eligible live plan')
    check(plans[0],kind,first)
    expected=1
    if kind=='residual' and plans[0]['event']!='left_live_plan_rejected':
        minimum=int(t[first,1])+int(np.ceil(plans[0]['duration_seconds']*hz-1e-9))
        second=first_end(minimum,True)
        if second is not None:
            if len(plans)<2:raise ValueError('Missing eligible full return')
            check(plans[1],'full',second);expected=2
    if len(plans)!=expected:raise ValueError('Extra live path attempt')
    return dict(first_eligible_decisions_verified=True,attempts=expected,initial_kind=kind,initial_source_step=int(t[first,1]),scope='Truth-based staging audit only; no blind detector claim')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control-input',type=Path);parser.add_argument('--kinematic-tracking',action='store_true');parser.add_argument('--input',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    report=json.loads(a.input.with_suffix('.json').read_text())
    boundaries=None;path=None;return_path=None
    audited=[e for e in (report.get('whole_hand_probe') or {}).get('events',[]) if e['event']=='live_plan_audited']
    if audited:
        from rubik_lab.diagnostics.whole_hand_path import WholeHandPath
        event=audited[0];path=WholeHandPath.from_result(event['source'],event['trial'],event['env'],stop_waypoint=event['requested_end'])
        boundaries=(path.times[13],path.duration)
    returned=[e for e in (report.get('whole_hand_probe') or {}).get('events',[]) if e['event']=='return_plan_audited']
    if returned:
        from rubik_lab.diagnostics.whole_hand_path import WholeHandPath
        event=returned[0];return_path=WholeHandPath.from_result(event['source'],event['trial'],event['env'],stop_waypoint=0)
    with np.load(a.input.with_suffix('.npz')) as raw:
        result=summarize_probe(report,raw,boundaries)
        if 'probe_wrench_frames' in raw:result['wrenches']=summarize_wrenches(report,raw)
        if 'whole_hand_feedback' in raw:result['feedback']=summarize_feedback(report,raw,path)
        if a.kinematic_tracking:result['tracking']=summarize_tracking(report,raw,path)
        if 'whole_hand_return_targets' in raw:result['return']=summarize_return(report,raw,return_path)
        if report.get('post_return_events'):result['post_return']=summarize_post_return(report,raw)
        if report.get('diagnostic_preload_at_error_deg') is not None and not report.get('diagnostic_preload_escape_path') and not report.get('live_left_withdrawal'):result['preload_brake']=summarize_preload_brake(report,raw)
        if report.get('live_left_withdrawal'):
            result['live_left_timing']=audit_live_left_timing(report,raw)
            starts=[e for e in report.get('post_return_events',[]) if e['event']=='left_return_start']
            if starts:
                from rubik_lab.diagnostics.left_return import LeftReturnPath
                if len(starts)!=1:raise ValueError('Ambiguous live left return')
                live_path=LeftReturnPath(starts[0]['source'])
                with np.load(live_path.raw_prefix.with_suffix('.npz')) as actual_source:
                    result['left_return']=summarize_left_return(report,raw,actual_source,live_path)
                result['left_return']['prefix_scope']='Compared to actual paused planning source, not an independent matched-control run'
                result['post_return']['original_full_monitor_passed']=result['left_return']['original_full_physical_monitor_passed']
            else:result['left_return']=dict(original_full_physical_monitor_passed=False,reason='No audited full withdrawal started',rejections=[e for e in report.get('post_return_events',[]) if e['event']=='left_live_plan_rejected'])
            if any(e['event']=='after_left_return_handoff' for e in report.get('post_return_events',[])):result['fourth']=summarize_fourth(report,raw)
        if a.control_input:
            with np.load(a.control_input.with_suffix('.npz')) as control:
                if report.get('live_left_withdrawal'):
                    result['fourth']=summarize_fourth(report,raw,control)
                elif report.get('diagnostic_preload_escape_path') and not report.get('post_return_left_path'):
                    with np.load(str(report['diagnostic_preload_escape_path'])+'.npz') as path_raw:
                        delta=(path_raw['solution'][0]-path_raw['initial'][0])[5:9]
                    result['residual_escape']=summarize_residual_escape(report,raw,control,delta)
                elif report.get('post_return_left_path'):
                    from rubik_lab.diagnostics.left_return import LeftReturnPath
                    result['left_return']=summarize_left_return(report,raw,control,LeftReturnPath(report['post_return_left_path']))
                    result['post_return']['original_full_monitor_passed']=result['left_return']['original_full_physical_monitor_passed']
                    if report.get('after_left_return_move'):result['fourth']=summarize_fourth(report,raw,control)
                else:result['escape']=summarize_escape(report,raw,control)
        if report.get('fourth_recontact'):
            from rubik_lab.diagnostics.left_recontact import summarize_recontact
            if report.get('recycle_completed_stroke'):
                from rubik_lab.diagnostics.stroke_cycle import _summarize_internal
                result['completed_cycle_recontact']=_summarize_internal(report,raw,recontact=True,event_name='cycle_recontact_start')
            elif report.get('second_stroke'):
                from rubik_lab.diagnostics.left_recontact import summarize_contact_turn
                result['second_contact_turn']=summarize_contact_turn(report,raw,event_name='second_contact_turn_start')
            elif report.get('stroke_recontact_path') or report.get('live_stroke_cycle'):
                from rubik_lab.diagnostics.stroke_cycle import summarize_stroke_recontact
                result['stroke_recontact']=summarize_stroke_recontact(report,raw)
            elif report.get('stroke_lift_path'):
                from rubik_lab.diagnostics.stroke_cycle import summarize_stroke_lift
                result['stroke_lift']=summarize_stroke_lift(report,raw)
            elif report.get('contact_turn'):
                from rubik_lab.diagnostics.left_recontact import summarize_contact_turn
                if a.control_input:
                    with np.load(a.control_input.with_suffix('.npz')) as control:result['contact_turn']=summarize_contact_turn(report,raw,control)
                else:result['contact_turn']=summarize_contact_turn(report,raw)
            else:result['fourth_recontact']=summarize_recontact(report,raw)
            if 'fourth' in result:
                result['fourth']['scope']='Privileged contact component; not frozen actor execution'
                result['fourth']['original_full_monitor_passed']=False
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
