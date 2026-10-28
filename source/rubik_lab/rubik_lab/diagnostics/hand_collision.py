"""Independent mesh checks of the active hand against the measured opposite hand."""
import argparse,json,itertools,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch,trimesh,hppfcl as fcl
from rubik_lab.constants import ASSET_DIR
from rubik_lab.diagnostics.convex_geometry import convex_features,convex_overlap
from rubik_lab.hand_kinematics import HandFK,_origin,PROJECT_DIR,axis_angle_to_mat,quat_to_mat
def validate_active_geometry_scope(side,report):
    if side=='left':
        phase=report.get('phase','')
        if not (phase.startswith('withdrawal') or phase in ('recontact_geometry','turn_geometry')) or report.get('contact_transport') or 'contact_lateral_max_m' in report:
            raise ValueError('Left audit supports withdrawal/recontact/turn geometry only, not contact transport or patch validation')


def audited_layer_angle(source,report,waypoint=None):
    """Intended geometric layer angle; never a physical state update."""
    base=-float(source['error_deg'])
    if report.get('phase')=='turn_geometry':
        if source['side']!='left' or source['move']!="U'":raise ValueError('Explicit turn geometry currently requires left Uprime')
        values=np.asarray(report.get('layer_angle_deltas_deg',[]),dtype=float)
        if values.ndim!=1 or len(values)!=len(report['results']) or not len(values) or not np.isfinite(values).all() or np.any(values<=0) or np.any(values>90) or np.any(np.diff(values)<=0):raise ValueError('Invalid positive layer-angle deltas')
        if waypoint is not None and (waypoint<0 or waypoint>=len(values)):raise ValueError('Missing layer-angle waypoint')
        return np.deg2rad(base+(0 if waypoint is None else values[waypoint]))
    return np.deg2rad(base-(0 if waypoint is None else max(0,waypoint-12)))

p=argparse.ArgumentParser();p.add_argument('--solutions',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--usd-convex',action='store_true',help='Read USD points and filters; query solid convex hulls');p.add_argument('--cooked-hulls',type=Path,help='Exported PhysX hull vertices in hand link frames');p.add_argument('--cube-clearance',action='store_true',help='Independent SAT against exact USD cube boxes');p.add_argument('--segment-samples',type=int,default=0,help='Independent uniformly spaced interior checks of sequential solutions');a=p.parse_args()
if a.segment_samples<0:raise ValueError('Segment sample count must be nonnegative')
if a.cube_clearance and not a.cooked_hulls:raise ValueError('Cube clearance audit requires cooked hulls')
if a.cooked_hulls and not a.usd_convex:raise ValueError('Cooked hulls require USD filters via --usd-convex')
cooked=None
if a.cooked_hulls:
    cooked_report=json.loads(a.cooked_hulls.with_suffix('.json').read_text())
    if not cooked_report['probe_mesh_local_coordinate_check'] or len(cooked_report['colliders'])!=54:raise ValueError('Incomplete cooked geometry export')
    cooked=np.load(a.cooked_hulls.with_suffix('.npz'))
if a.out.with_suffix('.json').exists():raise FileExistsError(a.out)
report=json.loads(a.solutions.with_suffix('.json').read_text());solutions=np.load(a.solutions.with_suffix('.npz'))
if a.segment_samples and not report.get('sequential'):raise ValueError('Segment audit requires sequential solutions')
paths=Path(report['source']);pr=json.loads(paths.with_suffix('.json').read_text());z=np.load(paths.with_suffix('.npz'));raw=np.load(Path(pr['source']).with_suffix('.npz'))
sides={row['side'] for row in pr['results']}
if len(sides)!=1 or not sides<={'left','right'}:raise ValueError('Expected one active hand side')
active_side=next(iter(sides));other_side='left' if active_side=='right' else 'right'
validate_active_geometry_scope(active_side,report)
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
tensor=lambda value:torch.as_tensor(value,dtype=torch.float64)
fks={};meshes={};clusters={};adjacent={};usd_filters=set();collider_metadata=[];convex_data={}
for side in ('left','right'):
    fk=HandFK(side,'cpu',pts_per_link=8)
    for table in (fk.T0,fk.axis,fk.link_pts):
        for name in table:table[name]=table[name].double()
    fks[side]=fk
    urdf=ET.parse(Path(ASSET_DIR)/f'sharpa_wave_float_base/{side}/{side}_sharpa_wave_with_float_base_with_wrist.urdf').getroot()
    parent={el.get('name'):el.get('name') for el in urdf.findall('link')}
    def root(name):
        while parent[name]!=name:
            parent[name]=parent[parent[name]];name=parent[name]
        return name
    joints=urdf.findall('joint')
    for joint in joints:
        if joint.get('type')=='fixed':parent[root(joint.find('child').get('link'))]=root(joint.find('parent').get('link'))
    clusters[side]={name:root(name) for name in parent}
    adjacent[side]={frozenset((root(j.find('parent').get('link')),root(j.find('child').get('link')))) for j in joints if j.get('type')!='fixed'}
    meshes[side]={};convex_data[side]={}
    usd_points={}
    if a.usd_convex:
        from pxr import Usd,UsdGeom
        stage=Usd.Stage.Open(str(Path(ASSET_DIR)/f'sharpa_wave_float_base/{side}/{side}_sharpa_wave_with_float_base_with_wrist.usda'))
        xforms=UsdGeom.XformCache()
        for prim in stage.TraverseAll():
            relation=prim.GetRelationship('physics:filteredPairs')
            if relation and relation.HasAuthoredTargets():
                for target in relation.GetTargets():usd_filters.add(frozenset((prim.GetName(),str(target).split('/')[-1])))
            if prim.GetTypeName()!='Mesh' or not prim.GetAttribute('physics:collisionEnabled').Get():continue
            approximation=prim.GetAttribute('physics:approximation').Get()
            if approximation!='convexHull':raise ValueError('Unsupported collider approximation: '+str(prim.GetPath()))
            link_name=str(prim.GetPath()).split('/')[2]
            link_prim=stage.GetPrimAtPath('/'+str(prim.GetPath()).split('/')[1]+'/'+link_name)
            transform=np.asarray(xforms.ComputeRelativeTransform(prim,link_prim)[0])
            points=np.asarray(prim.GetAttribute('points').Get(),dtype=np.float64)
            points=points@transform[:3,:3]+transform[3,:3]
            if link_name in usd_points:raise ValueError('Multiple colliders require separate hulls: '+link_name)
            usd_points[link_name]=points
            collider_metadata.append(dict(path=str(prim.GetPath()),approximation=approximation,points=len(points)))
    for link in urdf.findall('link'):
        name=link.get('name');col=link.find('collision')
        if col is None or name not in fk.order:continue
        path=Path(PROJECT_DIR)/f'sharpa_wave/{side}_sharpa_wave/meshes'/Path(col.find('geometry/mesh').get('filename')).name
        mesh=trimesh.load(path,force='mesh');mesh.apply_transform(_origin(col.find('origin')))
        if a.usd_convex:
            if name not in usd_points:raise ValueError('Missing USD collider: '+name)
            points=cooked[name+'_vertices'] if cooked is not None else usd_points[name]
            mesh=trimesh.Trimesh(vertices=points,process=False).convex_hull
        if a.usd_convex:convex_data[side][name]=convex_features(mesh.vertices,mesh.faces)
        vertices=fcl.StdVec_Vec3f();triangles=fcl.StdVec_Triangle()
        for v in mesh.vertices:vertices.append(np.asarray(v,dtype=np.float64))
        for t in mesh.faces:triangles.append(fcl.Triangle(*map(int,t)))
        model=fcl.BVHModelOBBRSS();model.beginModel(len(mesh.faces),len(mesh.vertices));model.addSubModel(vertices,triangles);model.endModel()
        meshes[side][name]=fcl.Convex(vertices,triangles) if a.usd_convex else model
pairs=[]
for first,second in itertools.combinations(meshes[active_side],2):
    c1,c2=clusters[active_side][first],clusters[active_side][second]
    if c1==c2 or frozenset((c1,c2)) in adjacent[active_side]:continue
    if a.usd_convex and frozenset((first,second)) in usd_filters:continue
    pairs.append((active_side,first,active_side,second))
pairs.extend((active_side,r,other_side,l) for r in meshes[active_side] for l in meshes[other_side])
request=fcl.CollisionRequest();request.enable_contact=True;request.num_max_contacts=8

def query(frames):
    transforms={side:{name:fcl.Transform3f(R[0].numpy(),p[0].numpy()) for name,(p,R) in frames[side].items() if name in meshes[side]} for side in frames}
    collisions=[]
    for sa,la,sb,lb in pairs:
        result=fcl.CollisionResult()
        if fcl.collide(meshes[sa][la],transforms[sa][la],meshes[sb][lb],transforms[sb][lb],request,result):
            contacts=list(result.getContacts())
            depths=[float(c.penetration_depth) for c in contacts]
            depth_valid=bool(depths) and all(np.isfinite(d) and abs(d)<1. for d in depths)
            sat_overlap=None
            if a.usd_convex:
                pa,ra=frames[sa][la];pb,rb=frames[sb][lb]
                sat_overlap=convex_overlap(convex_data[sa][la],convex_data[sb][lb],ra[0].numpy(),pa[0].numpy(),rb[0].numpy(),pb[0].numpy())
            collisions.append(dict(sat_signed_overlap_m=sat_overlap,pair=[la,lb],kind='self' if sa==sb else 'interhand',rigid_pair=not la.endswith('_elastomer') and not lb.endswith('_elastomer'),reported_contact_depth_max_m=max(depths) if depth_valid else None,reported_depth_valid=depth_valid))
    return collisions
cube_boxes={};cube_halves={}
if a.cube_clearance:
    import re
    asset=(Path(ASSET_DIR)/'rubik_cube/rubik_cube_2body.usda').read_text()
    for body in ('core','top'):
        section=asset.split('def Xform "'+body+'"',1)[1].split('def Xform "visuals"',1)[0]
        xyz=lambda label:np.array([float(v) for v in re.search(label+r' = \(([^)]+)\)',section).group(1).split(',')])
        center=xyz('xformOp:translate');scale=xyz('xformOp:scale')
        box=trimesh.creation.box(extents=scale)
        cube_boxes[body]=(center,convex_features(box.vertices,box.faces))
        cube_halves[body]=scale/2

def cube_query(frames,angle):
    turn=axis_angle_to_mat(tensor([[0.,0.,angle]]))[0].numpy()
    margins=[];violations=[];support_pad_overlaps={};active_pad_top_overlap=None
    for link,features in convex_data[active_side].items():
        pad=link.endswith('_elastomer')
        position,rotation=frames[link]
        if pad and '_index_' not in link:
            support_pad_overlaps[link]={}
            for body,(center,box) in cube_boxes.items():
                body_rotation=turn if body=='top' else np.eye(3)
                support_pad_overlaps[link][body]=convex_overlap(features,box,rotation[0].numpy(),position[0].numpy(),body_rotation,body_rotation@center)*1000
            continue
        for body,(center,box) in cube_boxes.items():
            if pad and body=='top':
                active_pad_top_overlap=convex_overlap(features,box,rotation[0].numpy(),position[0].numpy(),turn,turn@center)*1000
                continue
            body_rotation=turn if body=='top' else np.eye(3)
            overlap=convex_overlap(features,box,rotation[0].numpy(),position[0].numpy(),body_rotation,body_rotation@center)
            margins.append(overlap)
            if overlap>.0001:violations.append(dict(link=link,body=body,overlap_mm=overlap*1000))
    return dict(active_pad_top_signed_overlap_mm=active_pad_top_overlap,max_overlap_mm=max(margins)*1000,pass_at_0p1mm=not violations,violations=violations,support_pad_signed_overlap_mm=support_pad_overlaps)

def support_query(frames,initial_frames,cube_result,initial_cube_result):
    """Independent NumPy readout of the optional contact candidate, no optimization."""
    pads=fks[active_side].pads(frames);original=fks[active_side].pads(initial_frames)
    center=cube_boxes['core'][0];half=cube_halves['core'];details={}
    for finger in ('thumb','middle'):
        link=fks[active_side].pad_links[finger]
        overlap=cube_result['support_pad_signed_overlap_mm'][link]['core']/1000
        initial_overlap=initial_cube_result['support_pad_signed_overlap_mm'][link]['core']/1000
        position=pads[finger][0][0].numpy();normal=pads[finger][1][0].numpy()
        initial_position=original[finger][0][0].numpy();initial_normal=original[finger][1][0].numpy()
        drift=overlap-initial_overlap;alignment=float(normal@initial_normal)
        footprint=half[1:]-.002-np.abs(position[1:]-center[1:])
        face_margin=float(position[0]*np.sign(initial_position[0])-.025)
        measurements=[drift,alignment,*footprint,face_margin]
        passed=bool(np.isfinite(measurements).all() and abs(drift)<=50e-6 and alignment>=.95 and (footprint>=0).all() and face_margin>=0)
        details[finger]=dict(overlap_drift_m=drift,normal_alignment=alignment,footprint_margin_m=footprint.tolist(),same_face_margin_m=face_margin,passed=passed)
    return dict(passed=all(v['passed'] for v in details.values()),pads=details)

def patch_query(frames,waypoint,angle,interior=False):
    active=waypoint>=(13 if interior else 11)
    if not active:return dict(active=False,passed=True)
    from rubik_lab.diagnostics.whole_hand_path import side_patch_error
    pad=fks[active_side].pads(frames)['index'][0][0].numpy()
    result=side_patch_error(pad,angle,report['contact_face'],report['contact_lateral_max_m'],pr['contact_press_m'])
    tolerance=.006 if waypoint==11 else .004
    return dict(active=True,tolerance_m=tolerance,passed=result['error_m']<=tolerance,**result)

def transport_query(frames,source,waypoint,angle):
    if waypoint<=12:return dict(active=False,passed=True)
    matches=[s for s in report['sequences'] if (s['trial'],s['env'])==(source['trial'],source['env'])]
    if len(matches)!=1 or matches[0]['contact_transport_anchor_cube_m'] is None:raise ValueError('Missing transport anchor')
    anchor=np.asarray(matches[0]['contact_transport_anchor_cube_m'])
    rz=lambda a:np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])
    rotation=rz(angle);initial=rz(np.deg2rad(-source['error_deg']))
    tangent=np.array([0.,1.,0.]) if report['contact_face']=='B' else np.array([1.,0.,0.])
    pad=fks[active_side].pads(frames)['index'][0][0].numpy()
    target=rotation@initial.T@anchor;error=float((pad-target)@(rotation@tangent))
    return dict(active=True,error_m=error,passed=bool(np.isfinite(error) and abs(error)<=.0005))

rows=[];initial_cache={};previous_solutions={};initial_support_geometry={};initial_contact_cache={}
for j,record in enumerate(report['results']):
    key=(record['trial'],record['env']);indices=[i for i,r in enumerate(pr['results']) if (r['trial'],r['env'])==key]
    if len(indices)!=1:raise ValueError('Ambiguous source state')
    i=indices[0];source=pr['results'][i];frame=source['source_frame'];env=source['env']
    q0=tensor(z['U_initial_q'][i:i+1]);wp=tensor(z['U_wrist_pos'][i:i+1]);wr=tensor(z['U_wrist_rot'][i:i+1])
    pose=raw['poses'][frame,env];cube_R=quat_to_mat(tensor(pose[3:7]))
    other_hand=0 if other_side=='left' else 1;offset=7+7*other_hand
    lwp=cube_R.T@tensor(pose[offset:offset+3]-raw['env_origins'][env]-pose[:3]);lwr=cube_R.T@quat_to_mat(tensor(pose[offset+3:offset+7]));lq=tensor(raw['sensory'][frame,env,22*other_hand:22*(other_hand+1)])[None]
    left=fks[other_side].forward(lwp[None],lwr[None],lq)
    if key not in initial_cache:initial_cache[key]=query({other_side:left,active_side:fks[active_side].forward(wp,wr,q0)})
    x=tensor(solutions['solution'][j]);cols=report['variable_joint_indices'];q=q0.clone();q[:,cols]=x[:len(cols)]
    peak=raw['trace'][frame,env,14:16];dp=x[len(cols):len(cols)+3]*(.010-float(peak[0])-1e-6);dr=x[len(cols)+3:]*(np.deg2rad(10)-float(peak[1])-1e-6)
    right=fks[active_side].forward(wp+dp,axis_angle_to_mat(dr[None])@wr,q)
    current=query({other_side:left,active_side:right})
    initial_pairs={tuple(c['pair']) for c in initial_cache[key]}
    row=dict(trial=key[0],env=key[1],waypoint=record['waypoint'],initial_collisions=initial_cache[key],candidate_collisions=current,new_collision_pairs=[c for c in current if tuple(c['pair']) not in initial_pairs])
    if a.cube_clearance:
        if (source['move'],source['side']) not in (('U','right'),("U'",'left')):raise ValueError('Unsupported move/hand geometry')
        angle=audited_layer_angle(source,report,record['waypoint'])
        row['cube_clearance']=cube_query(right,angle)
        if 'contact_lateral_max_m' in report:row['contact_patch']=patch_query(right,record['waypoint'],angle)
        if report.get('contact_transport'):row['contact_transport']=transport_query(right,source,record['waypoint'],angle)
        if report.get('support_contact'):
            if key not in initial_contact_cache:
                initial_frames=fks[active_side].forward(wp,wr,q0)
                initial_contact_cache[key]=(initial_frames,cube_query(initial_frames,np.deg2rad(-source['error_deg'])))
            initial_frames,initial_cube=initial_contact_cache[key]
            row['support_contact']=support_query(right,initial_frames,row['cube_clearance'],initial_cube)
        if key not in initial_support_geometry:
            initial_support_geometry[key]=cube_query(fks[active_side].forward(wp,wr,q0),np.deg2rad(-source['error_deg']))['support_pad_signed_overlap_mm']
    if a.segment_samples:
        previous=previous_solutions.get(key)
        if previous is None:
            if record['waypoint']!=0:raise ValueError('Sequential audit must begin at waypoint0')
            prior=tensor(solutions['initial'][j]);prior_waypoint=0
        else:
            prior,prior_waypoint=previous
            if record['waypoint']!=prior_waypoint+1:raise ValueError('Sequential audit requires contiguous waypoints')
        samples=[]
        for fraction in np.arange(1,a.segment_samples+1)/(a.segment_samples+1):
            intermediate=prior+float(fraction)*(x-prior)
            iq=q0.clone();iq[:,cols]=intermediate[:len(cols)]
            ip=intermediate[len(cols):len(cols)+3]*(.010-float(peak[0])-1e-6)
            ir=intermediate[len(cols)+3:]*(np.deg2rad(10)-float(peak[1])-1e-6)
            intermediate_right=fks[active_side].forward(wp+ip,axis_angle_to_mat(ir[None])@wr,iq)
            collisions=query({other_side:left,active_side:intermediate_right})
            sample=dict(fraction=float(fraction),collisions=collisions)
            if a.cube_clearance:
                prior_angle=audited_layer_angle(source,report,None if previous is None else prior_waypoint)
                sample['cube_clearance']=cube_query(intermediate_right,prior_angle+float(fraction)*(angle-prior_angle))
                if 'contact_lateral_max_m' in report:sample['contact_patch']=patch_query(intermediate_right,record['waypoint'],prior_angle+float(fraction)*(angle-prior_angle),interior=True)
                if report.get('contact_transport'):sample['contact_transport']=transport_query(intermediate_right,source,record['waypoint'],prior_angle+float(fraction)*(angle-prior_angle))
                if report.get('support_contact'):
                    sample['support_contact']=support_query(intermediate_right,initial_frames,sample['cube_clearance'],initial_cube)
            samples.append(sample)
        row['segment_samples']=samples
        previous_solutions[key]=(x.clone(),record['waypoint'])
    rows.append(row);print(json.dumps(dict(trial=key[0],env=key[1],waypoint=record['waypoint'],initial_count=len(initial_pairs),candidate_count=len(current),new_count=len(row['new_collision_pairs']))),flush=True)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.with_suffix('.json').write_text(json.dumps(dict(active_side=active_side,contact_patch_audited=bool('contact_lateral_max_m' in report and a.cube_clearance),contact_lateral_max_m=report.get('contact_lateral_max_m'),contact_transport_audited=bool(report.get('contact_transport') and a.cube_clearance),support_contact_audited=bool(report.get('support_contact') and a.cube_clearance),initial_support_geometry=[dict(trial=k[0],env=k[1],support_pad_signed_overlap_mm=v) for k,v in initial_support_geometry.items()],support_geometry_scope="Signed cooked-pad/core and pad/top SAT overlaps, positive overlap and negative separating-axis gap; not contact force, deformation, or support-stability proof. Excluded from rigid .1mm scoring.",segment_samples=a.segment_samples,cube_clearance=a.cube_clearance,cube_scope='All active-hand rigid hulls vs both exact USD boxes plus index pad/core; independent NumPy SAT, not prefiltered by FCL. Optional linear joint/normalized wrist interpolation, with matching layer-angle interpolation, is sampled when segment_samples>0; no continuous collision proof.',cooked_hulls=str(a.cooked_hulls) if a.cooked_hulls else None,usd_convex=a.usd_convex,colliders=collider_metadata,authored_filters=[sorted(pair) for pair in sorted(usd_filters,key=lambda x:sorted(x))],geometry_caveat='USD mode uses authored hulls unless cooked_hulls is supplied, in which case it uses exported PhysX hulls. Neither models solver contact/rest offsets. SAT depth is computed independently only for FCL-positive pairs; not an audit of FCL false negatives. Default uses URDF triangle surfaces.',source=str(a.solutions),hpp_fcl_version=fcl.__version__,pair_count=len(pairs),scope='URDF collision-mesh surface intersections at recorded poses. Fixed clusters and directly connected rigid-body neighbors excluded; other active-hand self and all interhand pairs checked. Initial intersections reported, not silently ignored. Max depth is over at most8returnedcontacts, not a global penetration bound. No swept collision or dynamics claim.',results=rows),indent=2)+'\n')
