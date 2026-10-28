"""Explicit-constraint IK audit of recorded failed poses; never writes simulator state."""
import argparse,json
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize
import torch
from rubik_lab.hand_kinematics import HandFK,axis_angle_to_mat,quat_to_mat
from rubik_lab.flick import surface_patch_target,rigid_finger_box_penetration,LAT_MIN,LAT_MAX
from rubik_lab.constants import FINGER_JOINTS
from rubik_lab.diagnostics.whole_hand_path import tangential_transport

p=argparse.ArgumentParser();p.add_argument('--paths',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--verify-solutions',type=Path);p.add_argument('--whole-hand',action='store_true',help='All22finger joints and dense moved-rigid-link constraints')
p.add_argument('--sequential',action='store_true',help='Solve every waypoint in order from the actual initial hand pose')
p.add_argument('--segment-constraints',action='store_true',help='Constrain four segment interior samples during sequential optimization')
p.add_argument('--cooked-hulls',type=Path,help='Actual PhysX convex hulls with whole-hull SAT cube clearance')
p.add_argument('--approach-orientation',action='store_true',help='Ramp pad-facing constraint during approach rather than introducing it only at contact')
p.add_argument('--batched-jacobian',action='store_true',help='Compute constraint Jacobian in one batched reverse-autograd call')
p.add_argument('--adaptive-segment',action='store_true',help='Add violated interior samples with a shared 200-iteration waypoint budget')
p.add_argument('--optimizer-clearance-margin-m',type=float,default=0.,help='Tighten optimization clearance only; independent .1mm scoring remains unchanged')
p.add_argument('--prefix-degrees',type=int,help='Explicit diagnostic partial-stroke request; never full-correction evidence')
p.add_argument('--support-contact',action='store_true',help='Replace support position proxy with initial-contact overlap/normal/footprint constraints')
p.add_argument('--support-optimizer-reserve-m',type=float,default=0.,help='Tighten only the optimized support overlap band; independent50um gate unchanged')
p.add_argument('--contact-lateral-max-m',type=float,default=LAT_MAX,help='Narrow side-contact corridor within original19..28mm range')
p.add_argument('--contact-face',choices=['R','B'],help='Explicit alternative U contact family; rebuild approach from same actual source')
p.add_argument('--hand-pairs-from',type=Path,help='Add independently discovered rigid hand pairs to optimization; full independentaudit still required')
p.add_argument('--contact-transport',action='store_true',help='Require .5mm tangential transport after contact; no virtualobjectrotation with stationaryfinger')
p.add_argument('--transport-optimizer-reserve-m',type=float,default=0.,help='Tighten optimization only inside unchanged.5mm transport scoring band')
a=p.parse_args()
if not LAT_MIN-1e-12<=a.contact_lateral_max_m<=LAT_MAX+1e-12:raise ValueError('Contact corridor must narrow originalrange')
if not np.isclose(a.contact_lateral_max_m,LAT_MAX,atol=1e-12,rtol=0) and not a.contact_face:raise ValueError('Narrow corridor requires rebuilt explicitcontactfamily')
if not 0<=a.transport_optimizer_reserve_m<.0005 or (a.transport_optimizer_reserve_m and not a.contact_transport):raise ValueError('Invalid transport reserve')
if a.contact_transport and not a.sequential:raise ValueError('Contact transport requires sequential solve')
if a.hand_pairs_from and not (a.whole_hand and a.sequential and a.cooked_hulls):raise ValueError('Handpairs require cooked wholehand sequential solve')
if a.support_optimizer_reserve_m and not a.support_contact:raise ValueError('Support reserve requires support contact')
if a.support_contact and (not a.cooked_hulls or not a.whole_hand or not a.sequential):raise ValueError('Support contact requires cooked whole-hand sequential solve')
if a.prefix_degrees is not None and (not a.sequential or not 1<=a.prefix_degrees<=90):raise ValueError('Prefix degrees require sequential mode and1..90')
if not 0<=a.optimizer_clearance_margin_m<=.0001:raise ValueError('Optimizer margin must be within0..0.1mm')
if a.adaptive_segment and not a.segment_constraints:raise ValueError('Adaptive segment requires segment constraints')
if a.approach_orientation and not a.sequential:raise ValueError('Approach orientation requires sequential mode')
if a.cooked_hulls and (not a.whole_hand or a.verify_solutions):raise ValueError('Cooked SAT requires whole-hand solve, not dense URDF verification')
if a.segment_constraints and not a.sequential:raise ValueError('Segment constraints require sequential mode')
if a.sequential and (not a.whole_hand or a.verify_solutions):raise ValueError('Sequential mode requires whole-hand solve, not single-pose verification')
if a.out.with_suffix('.json').exists() or a.out.with_suffix('.npz').exists():raise FileExistsError(a.out)
r=json.loads(a.paths.with_suffix('.json').read_text());z=np.load(a.paths.with_suffix('.npz'))
if not (r['surface_contact'] and r['wrist_assist'] and r['contact_face'] in ('R','B')):raise ValueError('Audit expects a declared coordinated side surface-patch cohort')
contact_face=a.contact_face or r['contact_face']
if a.contact_face and (not a.sequential or a.verify_solutions):raise ValueError('Contact-family override requires sequential solve')
source=np.load(Path(r['source']).with_suffix('.npz'))
hand_pairs=[]
if a.hand_pairs_from:
    pair_audit=json.loads(a.hand_pairs_from.read_text())
    prior_report=json.loads(Path(pair_audit['source']).with_suffix('.json').read_text())
    if pair_audit.get('cooked_hulls')!=str(a.cooked_hulls):raise ValueError('Handpair audit uses different cooked geometry')
    # Transfer only pair identities; reconstruct both poses from this actual source. No prior solution is reused.
    found={tuple(pair) for pair in prior_report.get('constrained_hand_pairs',[])}
    for audited in pair_audit['results']:
        for sample in [dict(collisions=audited['candidate_collisions']),*audited.get('segment_samples',[])]:
            for collision in sample['collisions']:
                if not collision['rigid_pair']:continue
                overlap=collision['sat_signed_overlap_m']
                if overlap is None or not np.isfinite(overlap):raise ValueError('Unknown auditpair depth')
                if overlap>.0001:found.add(tuple(collision['pair']))
    hand_pairs=sorted(found)
    if not hand_pairs:raise ValueError('No rejected rigid pair in supplied audit')
geometry_indices=[1,2,3,6,7]+list(range(8,8+len(hand_pairs)+int(a.contact_transport)))
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
fk=HandFK('right','cpu',pts_per_link=256,seed=0,dense_fingers=('thumb','index','middle','ring','pinky','hand') if a.whole_hand else ())
for table in (fk.T0,fk.axis,fk.link_pts):
    for name in table:table[name]=table[name].double()
fk.lower=fk.lower.double();fk.upper=fk.upper.double()
verify=None
if a.verify_solutions:
    from types import SimpleNamespace
    verify=np.load(a.verify_solutions.with_suffix('.npz'))
    verification_report=json.loads(a.verify_solutions.with_suffix('.json').read_text())
    dense=HandFK('right','cpu',pts_per_link=256,seed=0,dense_fingers=('thumb','index','middle','ring','pinky','hand'))
    for table in (dense.T0,dense.axis,dense.link_pts):
        for name in table:table[name]=table[name].double()
    import re
    asset=Path('assets/rubik_cube/rubik_cube_2body.usda').read_text();boxes={}
    for body in ('core','top'):
        section=asset.split('def Xform "'+body+'"',1)[1].split('def Xform "visuals"',1)[0]
        xyz=lambda label:torch.tensor([float(v) for v in re.search(label+r' = \(([^)]+)\)',section).group(1).split(',')])
        boxes[body]=(xyz('xformOp:translate'),xyz('xformOp:scale')*.5)

left_fk=None
if hand_pairs:
    left_fk=HandFK('left','cpu',pts_per_link=8,seed=0)
    for table in (left_fk.T0,left_fk.axis,left_fk.link_pts):
        for name in table:table[name]=table[name].double()
cooked_features=None
if a.cooked_hulls:
    import trimesh
    from rubik_lab.diagnostics.convex_geometry import convex_features,convex_box_overlap_torch,convex_overlap_torch
    cooked_report=json.loads(a.cooked_hulls.with_suffix('.json').read_text())
    if not cooked_report['probe_mesh_local_coordinate_check'] or len(cooked_report['colliders'])!=54:raise ValueError('Incomplete cooked geometry')
    cooked_raw=np.load(a.cooked_hulls.with_suffix('.npz'));cooked_features={}
    for link in sorted(set(fk.link_pts)|{link for pair in hand_pairs for link in pair}):
        key=link+'_vertices'
        if key not in cooked_raw:raise ValueError('Missing cooked hull: '+link)
        mesh=trimesh.Trimesh(vertices=cooked_raw[key],process=False).convex_hull
        cooked_features[link]=tuple(torch.tensor(v,dtype=torch.float64) for v in convex_features(mesh.vertices,mesh.faces))

cols=list(range(22)) if a.whole_hand else [i for i,n in enumerate(FINGER_JOINTS) if n.split('_')[0] in ('index','thumb','middle')]
normal=torch.tensor([0.,-1.,0.] if contact_face=='R' else [1.,0.,0.]);axis=torch.tensor([0.,0.,1.]);tangent=torch.linalg.cross(axis,normal)
rows=[];solutions=[];initials=[];residuals=[];sequences=[]
for i,row in enumerate(r['results']):
    if row['side']!='right' or row['move']!='U':raise ValueError('Unexpected hand or move')
    k=row.get('first_failed_waypoint')
    if k is None and not a.sequential:raise ValueError('Single-pose audit requires a known failed waypoint')
    if k is None:k=0
    q0=torch.tensor(z['U_initial_q'][i:i+1],dtype=torch.float64)
    wp=torch.tensor(z['U_wrist_pos'][i:i+1],dtype=torch.float64);wr=torch.tensor(z['U_wrist_rot'][i:i+1],dtype=torch.float64)
    initial_frames=fk.forward(wp,wr,q0)
    initial_pads=fk.pads(initial_frames)
    support_contact=None
    if a.support_contact:
        from rubik_lab.diagnostics.support_contact import SupportContact
        support_contact=SupportContact(fk,cooked_features,initial_frames,initial_pads,a.support_optimizer_reserve_m)
    targets=z['U_target'][i]
    if a.contact_face:
        from rubik_lab.diagnostics.whole_hand_path import contact_approach_targets
        targets=contact_approach_targets(initial_pads['index'][0][0].numpy(),row['error_deg'],contact_face,r['contact_press_m'],a.contact_lateral_max_m)
    initial_layer_rotation=axis_angle_to_mat(torch.tensor([[0.,0.,np.deg2rad(-row['error_deg'])]]))
    initial_normal_dot=float((initial_pads['index'][1]*(initial_layer_rotation@normal).reshape(3)).sum())
    peak=source['trace'][row['source_frame'],row['env'],14:16]
    bp=.010-float(peak[0])-1e-6;br=np.deg2rad(10)-float(peak[1])-1e-6
    if min(bp,br)<=0:raise ValueError('No remaining original-reference wrist budget')
    joint_lower=fk.lower.numpy().copy();joint_upper=fk.upper.numpy().copy()
    actuator_keys=('U_actuator_lower','U_actuator_upper','U_target_bias')
    present=[key in z for key in actuator_keys]
    if any(present) and not all(present):raise ValueError('Incomplete actuator-state planning input')
    if all(present):
        if not a.whole_hand:raise ValueError('Actuator-aware planning requires whole-hand scope')
        from rubik_lab.diagnostics.whole_hand_path import actuator_compatible_bounds
        joint_lower,joint_upper=actuator_compatible_bounds(joint_lower,joint_upper,*(z[key][i] for key in actuator_keys))
    left_frames={}
    if hand_pairs:
        pose=source['poses'][row['source_frame'],row['env']];cube_rotation=quat_to_mat(torch.tensor(pose[3:7],dtype=torch.float64))
        left_position=cube_rotation.T@torch.tensor(pose[7:10]-source['env_origins'][row['env']]-pose[:3],dtype=torch.float64)
        left_rotation=cube_rotation.T@quat_to_mat(torch.tensor(pose[10:14],dtype=torch.float64))
        left_q=torch.tensor(source['sensory'][row['source_frame'],row['env'],:22],dtype=torch.float64)
        left_frames=left_fk.forward(left_position[None],left_rotation[None],left_q[None])
    contact_anchor=None
    first_row=len(rows)
    previous_x=np.concatenate([q0[0,cols].numpy(),np.zeros(6)])
    full_required_end=12+int(np.ceil(abs(row['error_deg'])))
    required_end=full_required_end if a.prefix_degrees is None else min(full_required_end,12+a.prefix_degrees)
    waypoints=range(required_end+1) if a.sequential else [k]
    for k in waypoints:
        angle=np.deg2rad(-row['error_deg']-max(0,k-12));rotation=axis_angle_to_mat(torch.tensor([[0.,0.,angle]]))
        tn=(rotation@normal).reshape(3)
        tol=.006 if k<12 else .004
        face_bound=-.1 if k>=11 else None
        if a.approach_orientation and 1<=k<=10:
            start_bound=max(-.1,initial_normal_dot)
            face_bound=start_bound+min(k/10.,1.)*(-.1-start_bound)
        fixed_target=torch.tensor(targets[k:k+1],dtype=torch.float64)
        def measurements(x, local_rotation=None, optimization=False):
            rotation_used=rotation if local_rotation is None else local_rotation
            normal_used=(rotation_used@normal).reshape(3)
            q=q0.clone();q[:,cols]=x[:len(cols)]
            dp=x[len(cols):len(cols)+3]*bp;dr=x[len(cols)+3:]*br
            frames=fk.forward(wp+dp,axis_angle_to_mat(dr[None])@wr,q);pads=fk.pads(frames);c,m=pads['index']
            target=surface_patch_target(c,rotation_used,normal,axis,tangent,-1.,(LAT_MIN,a.contact_lateral_max_m),r['contact_press_m']) if k>=11 else fixed_target
            error=(c-target).square().sum()
            hold=torch.stack([(pads[f][0]-initial_pads[f][0]).square().sum() for f in ('thumb','middle')])
            if cooked_features is not None:
                depths=[]
                for link,features in cooked_features.items():
                    if not link.startswith('right_'):continue
                    pad=link.endswith('_elastomer')
                    if pad and '_index_' not in link:continue
                    position,orientation=frames[link]
                    for body,center,half in (('core',[0.,0.,-.035/3],[.035,.035,2*.035/3]),('top',[0.,0.,2*.035/3],[.035,.035,.035/3])):
                        if pad and body=='top':continue
                        box_rotation=rotation_used[0] if body=='top' else torch.eye(3,dtype=x.dtype)
                        box_center=box_rotation@x.new_tensor(center)
                        depths.append(convex_box_overlap_torch(features,orientation[0],position[0],box_rotation,box_center,x.new_tensor(half)))
                depth=torch.stack(depths).amax().clamp(min=0)
            else:
                depth=rigid_finger_box_penetration(fk,frames,'index',rotation_used)[0]
                if a.whole_hand:
                    depths=[depth]
                    for link,pts in fk.points(frames).items():
                        if link.endswith('_elastomer'):continue
                        for center,half,R in (([0.,0.,-.035/3],[.035,.035,2*.035/3],None),([0.,0.,2*.035/3],[.035,.035,.035/3],rotation_used)):
                            local=torch.einsum('bji,bpj->bpi',R,pts) if R is not None else pts
                            depths.append((pts.new_tensor(half)-(local-pts.new_tensor(center)).abs()).amin(-1).amax())
                    depth=torch.stack(depths).max().clamp(min=0)
            dot=(m*normal_used).sum();ax=(m*axis).sum()
            support_margins=support_contact.measure(frames,pads,optimization=optimization)[0] if support_contact else 1-hold/.004**2
            cons=torch.cat([torch.stack([1-error/tol**2,(.0001-depth)/.001]),support_margins,
                torch.stack([face_bound-dot if face_bound is not None else x.new_tensor(1.),.8**2-ax**2 if k>=11 else x.new_tensor(1.),1-(dp/bp).square().sum(),1-(dr/br).square().sum()])])
            if hand_pairs:
                hand_depths=[]
                for first,second in hand_pairs:
                    first_p,first_r=(frames if first.startswith('right_') else left_frames)[first]
                    second_p,second_r=(frames if second.startswith('right_') else left_frames)[second]
                    hand_depths.append(convex_overlap_torch(cooked_features[first],cooked_features[second],first_r[0],first_p[0],second_r[0],second_p[0]))
                cons=torch.cat([cons,(.0001-torch.stack(hand_depths))/.001])
            if a.contact_transport:
                transport=tangential_transport(c[0],rotation_used[0],initial_layer_rotation[0],contact_anchor,tangent,.0005-a.transport_optimizer_reserve_m if optimization else .0005)[0] if contact_anchor is not None else x.new_tensor(1.)
                cons=torch.cat([cons,transport.reshape(1)])
            return cons,(error,hold,depth,dot,ax,dp,dr,q)
        x0=previous_x.copy() if a.sequential else np.concatenate([z['U_q'][i,k,cols],z['U_wrist_delta_pos'][i,k]/bp,z['U_wrist_delta_rot'][i,k]/br]).astype(np.float64)
        bounds=list(zip(joint_lower[cols],joint_upper[cols]))+[(-1.,1.)]*6
        cache={}
        constrained_fractions=[.2,.4,.6,.8]
        refinement_attempts=[]
        total_iterations=0
        def constraint_eval(x, need_jac=False):
            if 'x' not in cache or not np.array_equal(x,cache['x']):
                tx=torch.tensor(x,requires_grad=True);value=measurements(tx,optimization=True)[0]
                value=value-value.new_tensor([0.,a.optimizer_clearance_margin_m/.001,0.,0.,0.,0.,0.,0.]+[a.optimizer_clearance_margin_m/.001]*len(hand_pairs)+([0.] if a.contact_transport else []))
                if a.segment_constraints:
                    interior=[]
                    prior_angle=np.deg2rad(-row['error_deg']-max(0,k-1-12))
                    for fraction in constrained_fractions:
                        intermediate=torch.tensor(previous_x)+fraction*(tx-torch.tensor(previous_x))
                        intermediate_rotation=axis_angle_to_mat(torch.tensor([[0.,0.,prior_angle+fraction*(angle-prior_angle)]]))
                        interior_value=measurements(intermediate,intermediate_rotation,optimization=True)[0][geometry_indices]
                        interior.append(interior_value-interior_value.new_tensor([a.optimizer_clearance_margin_m/.001,0.,0.,0.,0.]+[a.optimizer_clearance_margin_m/.001]*len(hand_pairs)+([0.] if a.contact_transport else [])))
                    value=torch.cat([value,torch.stack(interior).amin(0)])
                cache.clear()
                cache.update(x=x.copy(),value=value.detach().numpy(),tensor=value,input=tx)
            if need_jac and 'jac' not in cache:
                value,tx=cache['tensor'],cache['input']
                jac=(torch.autograd.grad(value,tx,grad_outputs=torch.eye(len(value),dtype=tx.dtype),is_grads_batched=True)[0] if a.batched_jacobian else torch.stack([torch.autograd.grad(v,tx,retain_graph=True)[0] for v in value]))
                cache['jac']=jac.detach().numpy()
            return cache
        def objective(x):
            delta=x-x0;weights=np.ones_like(x);weights[:len(cols)]=.01
            return .5*np.dot(weights*delta,delta),weights*delta
        initial=measurements(torch.tensor(x0))[0].detach().numpy()
        if verify is None:
            start=x0.copy()
            for refinement in range(4 if a.adaptive_segment else 1):
                cache.clear()
                answer=minimize(objective,start,jac=True,method='SLSQP',bounds=bounds,
                    constraints=[dict(type='ineq',fun=lambda x:constraint_eval(x)['value']-1e-6,jac=lambda x:constraint_eval(x,need_jac=True)['jac'])],
                    options=dict(maxiter=200-total_iterations,ftol=1e-10,disp=False))
                total_iterations+=int(answer.nit)
                if not a.adaptive_segment:break
                with torch.no_grad():
                    endpoint=measurements(torch.tensor(answer.x))[0].numpy()
                    endpoint_ok=bool(np.isfinite(endpoint).all() and (endpoint>=0).all() and all(lo<=v<=hi for v,(lo,hi) in zip(answer.x,bounds)))
                    violated=[]
                    if endpoint_ok:
                        prior_angle=np.deg2rad(-row['error_deg']-max(0,k-1-12))
                        for fraction in np.arange(1,10)/10:
                            intermediate=torch.tensor(previous_x+fraction*(answer.x-previous_x))
                            intermediate_rotation=axis_angle_to_mat(torch.tensor([[0.,0.,prior_angle+fraction*(angle-prior_angle)]]))
                            margins=measurements(intermediate,intermediate_rotation)[0][geometry_indices].numpy()
                            if not (np.isfinite(margins).all() and (margins>=0).all()):violated.append(float(fraction))
                added=[f for f in violated if all(abs(f-g)>1e-9 for g in constrained_fractions)]
                refinement_attempts.append(dict(fractions=list(constrained_fractions),iterations=int(answer.nit),solver_status=int(answer.status),endpoint_feasible=endpoint_ok,violated_fractions=violated,solution=answer.x.tolist()))
                if not endpoint_ok or not added or total_iterations>=200 or refinement==3:break
                constrained_fractions=sorted(constrained_fractions+added)
                start=answer.x.copy()
        else:
            previous=verification_report['results'][i]
            if verification_report.get('whole_hand',False)!=a.whole_hand:raise ValueError('Verification variable/constraint scope mismatch')
            if (previous['trial'],previous['env'],previous['waypoint'])!=(row['trial'],row['env'],k):raise ValueError('Verification cohort mismatch')
            answer=SimpleNamespace(x=verify['solution'][i],success=previous['solver_success'],status=previous['solver_status'],message='Read-only stored-solution verification',nit=previous['iterations'])
        optimized_margins=constraint_eval(answer.x)['value'].copy()
        with torch.no_grad():
            values,detail=measurements(torch.tensor(answer.x));error,hold,depth,dot,ax,dp,dr,q=detail
            command_domain_ok=bool((q[:,cols]>=torch.tensor(joint_lower[cols])).all() and (q[:,cols]<=torch.tensor(joint_upper[cols])).all())
            valid=bool(torch.isfinite(values).all() and (values>=0).all() and command_domain_ok)
            record=dict(actuator_bounds_used=all(present),joint_domain_feasible=command_domain_ok,optimizer_margin_satisfied=bool(np.isfinite(optimized_margins).all() and (optimized_margins>=0).all()),approach_face_bound=face_bound,initial_normal_dot_face=initial_normal_dot,trial=row['trial'],env=row['env'],waypoint=k,source_frame=row['source_frame'],solver_success=bool(answer.success),solver_status=int(answer.status),message=str(answer.message),iterations=int(total_iterations if verify is None else answer.nit),measured_feasible=valid,
                initial_normalized_constraints=initial.tolist(),final_normalized_constraints=values.tolist(),pad_error_m=float(error.sqrt()),hold_pad_error_m=hold.sqrt().tolist(),rigid_core_penetration_m=float(depth),normal_dot_face=float(dot),normal_dot_axis=float(ax),additional_wrist_position_m=float(dp.norm()),additional_wrist_rotation_deg=float(torch.rad2deg(dr.norm())),original_reference_position_bound_m=float(peak[0]+dp.norm()),original_reference_rotation_bound_deg=float(np.rad2deg(peak[1])+torch.rad2deg(dr.norm())))
            if a.contact_transport:
                record['transport_anchor_cube_m']=contact_anchor.tolist() if contact_anchor is not None else None
                if contact_anchor is not None:
                    transport_frames=fk.forward(wp+dp,axis_angle_to_mat(dr[None])@wr,q)
                    transport_pad=fk.pads(transport_frames)['index'][0][0]
                    record['tangential_transport_error_m']=float(tangential_transport(transport_pad,rotation[0],initial_layer_rotation[0],contact_anchor,tangent)[1])
            if hand_pairs:record['constrained_hand_pair_overlap_m']={first+'/'+second:float(.0001-values[8+j]*.001) for j,(first,second) in enumerate(hand_pairs)}
            if support_contact:
                final_frames=fk.forward(wp+dp,axis_angle_to_mat(dr[None])@wr,q)
                record['support_contact']=support_contact.serialize(support_contact.measure(final_frames,fk.pads(final_frames))[1])
                record['original_position_proxy_pass']=bool((hold<=.004**2).all())
            if a.adaptive_segment:record['refinement_attempts']=refinement_attempts
            if verify is not None:
                points=dense.points(dense.forward(wp+dp,axis_angle_to_mat(dr[None])@wr,q))
                initial_points=dense.points(dense.forward(wp,wr,q0))
                initial_rotation=axis_angle_to_mat(torch.tensor([[0.,0.,np.deg2rad(-row['error_deg'])]]))
                penetrations={}
                for body,(center,half) in boxes.items():
                    for link,pts in points.items():
                        local=torch.einsum('bji,bpj->bpi',rotation,pts) if body=='top' else pts
                        initial_local=torch.einsum('bji,bpj->bpi',initial_rotation,initial_points[link]) if body=='top' else initial_points[link]
                        depth=(half-(local-center).abs()).amin(-1).clamp(min=0).max()
                        initial_depth=(half-(initial_local-center).abs()).amin(-1).clamp(min=0).max()
                        penetrations[body+'/'+link]=dict(depth_mm=float(depth*1000),initial_depth_mm=float(initial_depth*1000))
                record['dense_whole_hand_penetration']=penetrations
                active=[v['depth_mm'] for name,v in penetrations.items() if '_index_' in name and (name.startswith('core/') or not name.endswith('_elastomer'))]
                record['dense_active_max_mm']=max(active)
                record['dense_active_pass']=max(active)<=.1
                rigid=[v['depth_mm'] for name,v in penetrations.items() if not name.endswith('_elastomer')]
                record['dense_whole_rigid_max_mm']=max(rigid)
                record['dense_whole_hand_pass']=max(active)<=.1 and max(rigid)<=.1
            if a.sequential and valid:
                endpoint_rotation=rotation;endpoint_normal=tn
                segment_min=np.full(len(geometry_indices),np.inf)
                for fraction in (np.arange(1,20)/20 if a.adaptive_segment else ((.1,.2,.3,.4,.5,.6,.7,.8,.9) if a.segment_constraints else (.2,.4,.6,.8))):
                    prior_angle=np.deg2rad(-row['error_deg']-max(0,k-1-12))
                    intermediate_angle=prior_angle+fraction*(angle-prior_angle)
                    rotation=axis_angle_to_mat(torch.tensor([[0.,0.,intermediate_angle]]));tn=(rotation@normal).reshape(3)
                    intermediate=torch.tensor(previous_x+fraction*(answer.x-previous_x))
                    margins=measurements(intermediate)[0][geometry_indices].numpy()
                    segment_min=np.minimum(segment_min,margins)
                rotation=endpoint_rotation;tn=endpoint_normal
                record['segment_min_geometry_hold_wrist_constraints']=segment_min.tolist()
                record['segment_pass']=bool(np.isfinite(segment_min).all() and (segment_min>=0).all())
                record['max_finger_increment_rad']=float(np.max(np.abs(answer.x[:len(cols)]-previous_x[:len(cols)])))
            rows.append(record);solutions.append(answer.x);initials.append(x0);residuals.append(values.numpy())
        print(json.dumps(record),flush=True)
        if a.sequential:
            if not valid or not record.get("segment_pass",False):break
            previous_x=answer.x.copy()
            if a.contact_transport and k==12:
                contact_anchor=fk.pads(fk.forward(wp+dp,axis_angle_to_mat(dr[None])@wr,q))['index'][0][0].detach().clone()
    if a.sequential:
        attempted=rows[first_row:]
        sequences.append(dict(contact_transport_anchor_cube_m=contact_anchor.tolist() if contact_anchor is not None else None,full_correction_requested=required_end==full_required_end,full_correction_required_waypoints=full_required_end+1,trial=row['trial'],env=row['env'],required_waypoints=required_end+1,attempted_waypoints=len(attempted),
            complete_sampled_geometry_path=len(attempted)==required_end+1 and all(x['measured_feasible'] and x.get('segment_pass',False) for x in attempted),
            final_waypoint=attempted[-1]['waypoint'],endpoint_feasible=attempted[-1]['measured_feasible'],segment_pass=attempted[-1].get('segment_pass',False),
            finger_travel_time_lower_bound_s_at_1rad_s=sum(x.get('max_finger_increment_rad',0.) for x in attempted)))
a.out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.out.with_suffix('.npz'),initial=np.array(initials),solution=np.array(solutions),normalized_constraints=np.array(residuals))
a.out.with_suffix('.json').write_text(json.dumps(dict(contact_lateral_max_m=a.contact_lateral_max_m,transport_optimizer_reserve_m=a.transport_optimizer_reserve_m,contact_transport=a.contact_transport,transport_tolerance_m=.0005,hand_pairs_from=str(a.hand_pairs_from) if a.hand_pairs_from else None,constrained_hand_pairs=hand_pairs,contact_face=contact_face,contact_face_override=a.contact_face,support_optimizer_reserve_m=a.support_optimizer_reserve_m,support_contact=a.support_contact,support_contact_rule=('initial signed cooked/core overlap +/-50um, initial normal dot>=.95, exact core side-face padcenter footprint with2mm inset; geometry only, not force guarantee' if a.support_contact else None),requested_prefix_degrees=a.prefix_degrees,optimizer_clearance_margin_m=a.optimizer_clearance_margin_m,independent_clearance_limit_m=.0001,lazy_constraint_jacobian=True,adaptive_segment=a.adaptive_segment,adaptive_refinement_limit=3,iteration_budget_per_waypoint=200,batched_jacobian=a.batched_jacobian,approach_orientation=a.approach_orientation,approach_orientation_rule='k0unchanged; k1..10linear bound from max(initial_face_dot,-.1) to -.1; contact bound unchanged',cooked_hulls=str(a.cooked_hulls) if a.cooked_hulls else None,clearance_model='PhysX cooked convex SAT vs cube boxes' if a.cooked_hulls else 'dense URDF point penetration',segment_constraints=a.segment_constraints,optimization_segment_fractions=[.2,.4,.6,.8] if a.segment_constraints else [],verification_segment_fractions=(np.arange(1,20)/20).tolist() if a.adaptive_segment else ([.1,.2,.3,.4,.5,.6,.7,.8,.9] if a.segment_constraints else [.2,.4,.6,.8]),sequential=a.sequential,sequences=sequences,whole_hand=a.whole_hand,variable_joint_indices=cols,verify_solutions=str(a.verify_solutions) if a.verify_solutions else None,source=str(a.paths),scipy_version=scipy.__version__,method='SLSQP',max_iterations=200,constraint_names=['pad_position','rigid_clearance','thumb_hold','middle_hold','normal_face','normal_axis','wrist_translation','wrist_rotation']+['hand_pair:'+'/'.join(pair) for pair in hand_pairs]+(['tangential_transport'] if a.contact_transport else []),scope='Single-pose audit by default; sequential mode starts at actual source hand pose, retains original references and checks segment-interior geometry/holding/wrist samples (four normally, nine with segment-constraints), stopping at first failure. No continuous-collision proof. Local solver failure is not an infeasibility certificate. No continuous-path, dynamics, self/interhand collision or hardware claim; independent dense cube-intersection readout only when verify-solutions is supplied.',results=rows),indent=2)+'\n')
