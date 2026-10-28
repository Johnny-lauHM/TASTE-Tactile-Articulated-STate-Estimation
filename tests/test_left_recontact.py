"""Reject invalid history and preserve motor-only intervention scope."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from rubik_lab.diagnostics.left_recontact import solve_recontact

def history():
    t=np.zeros((10,1,16));t[:,0,1]=np.arange(10);t[:,0,3]=2;t[:,0,4]=1;t[:,0,9:11]=1
    return dict(trace=t,contact=np.zeros((10,1,2,2,27)),sensory=np.zeros((10,1,148)),poses=np.zeros((10,1,21)),motor_targets=np.zeros((10,1,2,22)))

@pytest.mark.parametrize('change,message',[
    ('unfinished','completed third'),('gap','readiness/release'),('contact','readiness/release'),
    ('trial','readiness/release'),('nonfinite','Nonfinite'),('no_template','No observed')])
def test_reject_invalid_sources_before_solving(change,message):
    raw=history()
    if change=='unfinished':raw['trace'][-1,0,10]=0
    if change=='gap':raw['trace'][3,0,1]=1
    if change=='contact':raw['contact'][-2,0,0,0,9]=.6
    if change=='trial':raw['trace'][0,0,0]=1
    if change=='nonfinite':raw['sensory'][1,0,0]=np.nan
    with pytest.raises(ValueError,match=message):solve_recontact(raw,0,0)

def test_runtime_override_is_only_requested_index_and_cannot_credit_action():
    path=Path(__file__).resolve().parents[1]/'source/rubik_lab/rubik_lab/diagnostics/continuation.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TraceEnv')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_pre_physics_step')
    assign=next(n for n in method.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='recontact' for x in n.targets))
    guard=next(n for n in method.body if isinstance(n,ast.If) and ast.unparse(n.test)=='recontact is not None')
    base=torch.arange(44,dtype=torch.float32).reshape(2,22);cols=torch.tensor([5,6,7,8]);values=torch.tensor([.1,.2,.3,.4])
    e=SimpleNamespace(finger_tgt={'left':base.clone(),'right':base.clone()},fourth_recontact_target=(1,'left',cols,values),wrist_targets=torch.ones(2,7),physical_q=base.clone())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assign,guard],type_ignores=[])),str(path),'exec'),{'self':e})
    expected=base.clone();expected[1,cols]=values
    assert torch.equal(e.finger_tgt['left'],expected)
    assert torch.equal(e.finger_tgt['right'],base)
    assert torch.equal(e.physical_q,base)
    assert torch.equal(e.wrist_targets,torch.ones(2,7))
    guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='u.fourth_recontact_target is not None')
    full=torch.ones(2,dtype=torch.bool)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),str(path),'exec'),dict(u=e,torch=torch,s={'finger_tgt':torch.stack([e.finger_tgt['left'],e.finger_tgt['right']],1)},full=full))
    assert full.tolist()==[True,False]


@pytest.mark.parametrize('fault,passed',[(None,True),('missing_pad',False),('rigid_contact',False),('cube_drift',False),('early_stop',False)])
def test_component_scoring_uses_measured_contact_and_never_action_credit(tmp_path,fault,passed):
    import json
    from rubik_lab.diagnostics.left_recontact import summarize_recontact
    n=250;raw=history()
    for k,v in list(raw.items()):raw[k]=np.repeat(v[-1:],n,axis=0)
    t=raw['trace'];t[:,0,1]=np.arange(n);t[:10,0,3]=2;t[10:,0,3]=3
    raw['poses'][:,:,3]=1
    raw['wrist_targets']=np.zeros((n,1,2,7));raw['core_angular_speed']=np.zeros((n,1))
    raw['actuator_lower']=np.full((1,2,22),-1.);raw['actuator_upper']=np.ones((1,2,22))
    raw['contact'][:,0,0,:,5]=1
    raw['contact'][10:,0,1,0,10]=1
    raw['motor_targets'][10:,0,0,5]=np.minimum(np.arange(1,241)*.025,.1)
    prefix=tmp_path/'plan';source=tmp_path/'source';paths=tmp_path/'paths'
    keys=('trace','sensory','poses','motor_targets','wrist_targets','contact','core_angular_speed')
    np.savez(str(source)+'.npz',**{k:raw[k][:10] for k in keys})
    paths.with_suffix('.json').write_text(json.dumps(dict(source=str(source),results=[dict(source_frame=9,source_step=9)])))
    prefix.with_suffix('.json').write_text(json.dumps(dict(source=str(paths),bias_rad=[0.]*22)))
    initial=np.zeros((1,28));solution=initial.copy();solution[0,5]=.1
    np.savez(str(prefix)+'.npz',initial=initial,solution=solution)
    report=dict(control_hz=40,post_return_events=[dict(event='fourth_recontact_start',trial=0,env=0,step=10,source=str(prefix))])
    if fault=='missing_pad':raw['contact'][-1,0,1,0,10]=0
    if fault=='rigid_contact':raw['contact'][-1,0,0,0,9]=.6
    if fault=='cube_drift':raw['poses'][-1,0,0]=.003
    if fault=='early_stop':
        for k in keys:raw[k]=raw[k][:-1]
    result=summarize_recontact(report,raw)
    assert result['component_pass'] is passed
    assert result['full_action_pass'] is False


@pytest.mark.parametrize('press',[-.001,0,.002,float('nan')])
def test_unregistered_press_rejected(press):
    with pytest.raises(ValueError,match='preregistered'):solve_recontact(history(),0,0,press)


@pytest.mark.parametrize('shift',[-.005,.001,.01,float('nan')])
def test_unregistered_tangent_shift_rejected(shift):
    with pytest.raises(ValueError,match='tangent shift'):solve_recontact(history(),0,0,.001,shift)


@pytest.mark.parametrize('angle',[0.,.0842,-.6,1.5707963267948966])
def test_cube_tangent_preserves_face_depth_and_moves_toward_axis(angle):
    from scipy.spatial.transform import Rotation
    from rubik_lab.diagnostics.left_recontact import cube_face_centerward_tangent
    r=Rotation.from_rotvec([0,0,angle]);point=r.apply([.035,.025,.026]);normal=r.apply([-.84,.49,-.23])
    tangent,face=cube_face_centerward_tangent(point,normal,angle)
    assert np.allclose(face,r.apply([-1.,0,0]))
    assert abs(tangent@face)<1e-12
    assert abs(np.linalg.norm(tangent)-1)<1e-12
    shifted=point+.005*tangent
    assert abs((shifted-point)@face)<1e-12
    assert abs(np.cross(shifted,face)[2])<abs(np.cross(point,face)[2])

def test_cube_face_ambiguity_rejected():
    from rubik_lab.diagnostics.left_recontact import cube_face_centerward_tangent
    with pytest.raises(ValueError,match='Ambiguous'):cube_face_centerward_tangent([.035,.035,.025],[-1,-1,0],0)


def angle_auditor():
    p=Path(__file__).resolve().parents[1]/'source/rubik_lab/rubik_lab/diagnostics/hand_collision.py'
    fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='audited_layer_angle')
    scope={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p),'exec'),scope)
    return scope['audited_layer_angle']

def test_left_turn_starts_at_source_then_moves_positive():
    f=angle_auditor();source=dict(side='left',move="U'",error_deg=85.)
    report=dict(phase='turn_geometry',layer_angle_deltas_deg=[5.,10.],results=[{},{}])
    assert np.isclose(np.rad2deg(f(source,report)),-85)
    assert np.isclose(np.rad2deg(f(source,report,0)),-80)
    assert np.isclose(np.rad2deg(f(source,report,1)),-75)
    a=f(source,report);b=f(source,report,0)
    assert np.isclose(np.rad2deg(a+.5*(b-a)),-82.5)
    assert np.isclose(np.rad2deg(f(source,{},19)),-92)

@pytest.mark.parametrize('values',[[],[-5.],[float('nan')],[5.,4.],[91.]])
def test_malformed_left_angle_sequence_rejected(values):
    with pytest.raises(ValueError):
        angle_auditor()(dict(side='left',move="U'",error_deg=85.),dict(phase='turn_geometry',layer_angle_deltas_deg=values,results=[{} for _ in values]),0)


def turn_history():
    n=30;t=np.zeros((n,1,16));t[:,0,1]=np.arange(100,130);t[:,0,3]=3;t[:,0,4]=1
    c=np.zeros((n,1,2,2,27));c[:,0,1,0,10]=.1;c[:,0,0,1,5]=1
    return dict(trace=t,contact=c,sensory=np.zeros((n,1,148)),core_angular_speed=np.zeros((n,1)))

def test_turn_gate_uses_first_completed_causal_window():
    from rubik_lab.diagnostics.left_recontact import first_turn_source
    z=turn_history()
    assert first_turn_source(z,0,0,100)==19
    assert first_turn_source({k:v[:19] for k,v in z.items()},0,0,100) is None
    assert first_turn_source({k:v[:20] for k,v in z.items()},0,0,100)==19

@pytest.mark.parametrize('fault',['gap','rigid','wrong_stage','support','nonfinite','motion'])
def test_turn_gate_invalid_window_cannot_trigger(fault):
    from rubik_lab.diagnostics.left_recontact import first_turn_source
    z={k:v[:20].copy() for k,v in turn_history().items()}
    if fault=='gap':z['trace'][15,0,1]-=1
    if fault=='rigid':z['contact'][15,0,0,0,9]=.6
    if fault=='wrong_stage':z['trace'][15,0,3]=2
    if fault=='support':z['contact'][15,0,0,1,5]=0
    if fault=='nonfinite':z['sensory'][15,0,0]=np.nan
    if fault=='motion':z['trace'][15,0,12]=.016
    assert first_turn_source(z,0,0,100) is None


@pytest.mark.parametrize('count',[19,21,30])
def test_turn_solver_rejects_early_or_late_history_before_fk(count):
    from rubik_lab.diagnostics.left_recontact import solve_contact_turn
    raw={k:v[:count].copy() for k,v in turn_history().items()}
    with pytest.raises(ValueError,match='latest first eligible'):solve_contact_turn(raw,0,0,100)


@pytest.mark.parametrize('fault,passed',[(None,True),('no_turn',False),('partial_turn',False),('lost_contact',False),('motion',False),('early_end',False)])
@pytest.mark.parametrize('target',[5.,10.,30.])
@pytest.mark.parametrize('event_name',['contact_turn_start','second_contact_turn_start'])
@pytest.mark.parametrize('gain',[1.,3.])
def test_turn_readout_requires_measured_motion_and_contact(tmp_path,monkeypatch,fault,passed,target,event_name,gain):
    import json
    import rubik_lab.diagnostics.left_recontact as module
    n=241;t=np.zeros((n,1,16));t[:,0,1]=np.arange(n);t[:,0,3]=3;t[:,0,4]=1;t[:,0,5]=85-target;t[0,0,5]=85
    c=np.zeros((n,1,2,2,27));c[:,0,1,0,10]=.1;c[:,0,0,1,5]=1
    m=np.zeros((n,1,2,22));m[:,0,0,5]=np.minimum(np.arange(n)*.025,.08)
    raw=dict(trace=t,contact=c,motor_targets=m,sensory=np.zeros((n,1,148)),core_angular_speed=np.zeros((n,1)),poses=np.zeros((n,1,21)),wrist_targets=np.zeros((n,1,2,7)))
    prefix=tmp_path/'turn'
    prefix.with_suffix('.json').write_text(json.dumps(dict(bias_rad=[0.]*22,handoff_step=0,layer_angle_deltas_deg=[.25,.5,.75,1.,2.,3.,4.,5.]+([6.,7.,8.,9.,10.] if target==10 else [10.,15.,20.,25.,30.] if target==30 else []))))
    initial=np.zeros((1,28));count=8 if target==5 else 13;solution=np.zeros((count,28));solution[:,5]=np.linspace(.01,.08,count)
    np.savez(prefix.with_suffix('.npz'),initial=initial,solution=solution)
    original={k:v[:1,0].copy() for k,v in raw.items()}
    monkeypatch.setattr(module,'ContactTurnPath',lambda p:SimpleNamespace(source=dict(source_frame=0,source_step=0),original=original))
    report=dict(control_hz=40,post_return_events=[dict(event=event_name,trial=0,env=0,step=1,source=str(prefix))])
    if event_name=='second_contact_turn_start':
        report['post_return_events'].append(dict(event='contact_turn_start',source='not_second_stroke'))
        report['second_stroke_feedback']=True;report['second_stroke_feedback_gain']=gain
        raw['actuator_lower']=np.full((1,2,22),-1.);raw['actuator_upper']=np.ones((1,2,22))
        # All measured joints remain0: bounded gain1 hold ramps .08->.16 after .33s.
        raw['motor_targets'][14:,0,0,5]=np.minimum(.08+.025*np.arange(1,n-13),.08+min(gain*.08,.1))

    if fault=='no_turn':raw['trace'][1:,0,5]=85
    if fault=='partial_turn':raw['trace'][1:,0,5]=85-target/6
    if fault=='lost_contact':raw['contact'][-1,0,1,0,10]=0
    if fault=='motion':raw['trace'][-1,0,12]=.016
    if fault=='early_end':raw={k:v[:-1] if len(v)==n else v for k,v in raw.items()}
    result=module.summarize_contact_turn(report,raw,event_name=event_name)
    assert result['component_pass'] is passed
    assert result['full_action_pass'] is False
    if event_name=='second_contact_turn_start' and fault is None:
        # A changed previous measurement must invalidate unchanged delivered commands.
        raw['sensory'][20,0,5]=.07
        with pytest.raises(ValueError,match='motor delivery'):module.summarize_contact_turn(report,raw,event_name=event_name)



@pytest.mark.parametrize('depth,normal',[(.0004,False),(-.0004,True),(.001,True),(float('nan'),True)])
def test_surface_depth_cannot_use_pad_normal_or_unregistered_depth(depth,normal):
    with pytest.raises(ValueError,match='surface depth'):solve_recontact(history(),0,0,surface_normal=normal,surface_depth_m=depth)


@pytest.mark.parametrize("degrees",[None,[],[5.],[.25,.5,.75,1.,2.,3.,4.,5.,30.],[.25,.5,.75,1.,2.,3.,4.,5.,10.,15.,20.,25.,float("nan")]])
def test_unregistered_contact_turn_sequence_rejected(degrees):
    from rubik_lab.diagnostics.left_recontact import contact_turn_degrees
    with pytest.raises(ValueError,match="Unregistered"):contact_turn_degrees(dict(layer_angle_deltas_deg=degrees))


@pytest.mark.parametrize('reserve',[-.0001,.00005,.001,float('nan')])
def test_turn_unregistered_depth_reserve_rejected(reserve):
    from rubik_lab.diagnostics.left_recontact import solve_contact_turn
    with pytest.raises(ValueError,match='Unregistered turn depth reserve'):solve_contact_turn({},0,0,0,reserve)


@pytest.mark.parametrize('degrees',[0.,6.,11.,90.,float('nan')])
def test_unregistered_generated_extent_rejected(degrees):
    from rubik_lab.diagnostics.left_recontact import solve_contact_turn
    with pytest.raises(ValueError,match='Unregistered generated turn extent'):solve_contact_turn({},0,0,0,target_degrees=degrees)
