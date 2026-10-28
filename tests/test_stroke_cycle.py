import numpy as np
import pytest
from types import SimpleNamespace
from rubik_lab.diagnostics.stroke_cycle import first_stroke_lift_source

def history(n=241):
    t=np.zeros((n,1,16));t[:,0,1]=np.arange(n);t[:,0,3]=3;t[:,0,4]=1;t[:,0,5]=81.
    c=np.zeros((n,1,2,2,27));c[:,0,1,0,10]=.1;c[:,0,0,:,5]=1
    p=np.zeros((n,1,21));p[:,:,3]=1
    return dict(trace=t,contact=c,sensory=np.zeros((n,1,148)),core_angular_speed=np.zeros((n,1)),poses=p,wrist_targets=np.zeros((n,1,2,7)),motor_targets=np.zeros((n,1,2,22)))

def test_internal_boundary_does_not_require_quarter_completion():
    raw=history(8)
    assert first_stroke_lift_source(raw,0,0,0)==4
    assert first_stroke_lift_source({k:v[:4] for k,v in raw.items()},0,0,0) is None

@pytest.mark.parametrize('fault',['gap','rigid','support','motion','nonfinite','speed','stage'])
def test_internal_boundary_rejects_bad_window(fault):
    r=history(5)
    if fault=='gap':r['trace'][2,0,1]=1
    if fault=='rigid':r['contact'][2,0,0,0,9]=.6
    if fault=='support':r['contact'][2,0,0,1,5]=0
    if fault=='motion':r['trace'][2,0,12]=.016
    if fault=='nonfinite':r['sensory'][2,0,0]=np.nan
    if fault=='speed':r['sensory'][2,0,49]=1.1
    if fault=='stage':r['trace'][2,0,3]=2
    assert first_stroke_lift_source(r,0,0,0) is None

@pytest.mark.parametrize('fault,passed',[(None,True),('residual',False),('pad',False),('drift',False),('cube',False),('support',False),('early',False)])
def test_lift_score_uses_actual_release_and_original_deadline(monkeypatch,fault,passed):
    import rubik_lab.diagnostics.stroke_cycle as m
    r=history();r['contact'][1:,0,1,0,10]=0
    motor=np.zeros((2,22));motor[1,5]=.04
    r['motor_targets'][:,0,0,5]=np.minimum(np.arange(241)*.025,.04)
    path=SimpleNamespace(source=dict(env=0,trial=0,source_frame=0,source_step=0),handoff_step=0,motor=motor,duration=.04,verify_history=lambda h:None)
    monkeypatch.setattr(m,'StrokeLiftPath',lambda p,finished:path)
    if fault=='residual':r['contact'][-1,0,0,0,9]=.6
    if fault=='pad':r['contact'][-1,0,0,0,10]=.01
    if fault=='drift':r['trace'][10,0,5]+=1.1
    if fault=='cube':r['poses'][10,0,0]=.0021
    if fault=='support':r['contact'][-1,0,0,1,5]=0
    if fault=='early':r={k:v[:-1] for k,v in r.items()}
    report=dict(control_hz=40,post_return_events=[dict(event='stroke_lift_start',env=0,trial=0,step=1,source='unused',turn_targets_finished_step=0)])
    result=m.summarize_stroke_lift(report,r)
    assert result['component_pass'] is passed
    assert result['full_action_pass'] is False


def test_recontact_boundary_requires_first_four_released_frames():
    from rubik_lab.diagnostics.stroke_cycle import first_stroke_recontact_source
    r=history(8);r['contact'][:,0,1,0,10]=0
    assert first_stroke_recontact_source(r,0,0,0)==4
    r['contact'][2,0,0,0,10]=.006
    assert first_stroke_recontact_source({k:v[:5] for k,v in r.items()},0,0,0) is None

@pytest.mark.parametrize('fault,passed',[(None,True),('missing_top',False),('rigid',False),('drift',False),('early',False)])
def test_internal_recontact_requires_actual_top_contact(monkeypatch,fault,passed):
    import rubik_lab.diagnostics.stroke_cycle as m
    r=history();motor=np.zeros((3,22));motor[:,5]=[0,.04,.08]
    r['motor_targets'][:,0,0,5]=np.minimum(np.arange(241)*.025,.08)
    path=SimpleNamespace(source=dict(env=0,trial=0,source_frame=0,source_step=0),handoff_step=0,motor=motor,duration=.08,times=np.array([0,.04,.08]),verify_history=lambda h:None)
    monkeypatch.setattr(m,'StrokeRecontactPath',lambda p,finished:path)
    if fault=='missing_top':
        r['contact'][-1,0,1,0,10]=0;r['contact'][-1,0,0,0,10]=.1
    if fault=='rigid':r['contact'][-1,0,0,0,9]=.6
    if fault=='drift':r['trace'][10,0,5]+=1.1
    if fault=='early':r={k:v[:-1] for k,v in r.items()}
    report=dict(control_hz=40,post_return_events=[dict(event='stroke_recontact_start',env=0,trial=0,step=1,source='unused',lift_targets_finished_step=0)])
    result=m.summarize_stroke_recontact(report,r)
    assert result['component_pass'] is passed
    assert result['full_action_pass'] is False


@pytest.mark.parametrize('count',[4,6])
@pytest.mark.parametrize('recontact',[False,True])
def test_generator_rejects_early_or_late_source_before_kinematics(tmp_path,count,recontact):
    from rubik_lab.diagnostics.stroke_cycle import write_internal_candidate
    r=history(count)
    if recontact:r['contact'][:,0,1,0,10]=0
    with pytest.raises(ValueError,match='latest firsteligible'):
        write_internal_candidate(r,tmp_path/'invalid',0,0,0,prior_contact_step=0 if recontact else None)


def test_completed_drive_boundary_waits_for_actual_progress():
    r=history(14);r['trace'][7:,0,5]=71.5
    boundary=dict(source_step=0,feedback_start_step=5,target_degrees=10.)
    assert first_stroke_lift_source(r,0,0,3)==7
    assert first_stroke_lift_source(r,0,0,3,boundary)==10
    assert np.all(r['trace'][:7,0,5]==81.)
    r['trace'][8,0,5]=72.1
    assert first_stroke_lift_source(r,0,0,3,boundary)==12

@pytest.mark.parametrize('fault',['no_progress','overshoot','reverse','support','rigid','gap'])
def test_completed_drive_boundary_rejects_false_arrival(fault):
    r=history(9);r['trace'][5:,0,5]=71.
    if fault=='no_progress':r['trace'][:,0,5]=81.
    if fault=='overshoot':r['trace'][5:,0,5]=69.
    if fault=='reverse':r['trace'][5:,0,5]=91.
    if fault=='support':r['contact'][6,0,0,1,5]=0
    if fault=='rigid':r['contact'][6,0,0,0,9]=.6
    if fault=='gap':r['trace'][6,0,1]=5
    assert first_stroke_lift_source(r,0,0,3,dict(source_step=0,feedback_start_step=5,target_degrees=10.)) is None


def test_release_recontact_machine_preserves_causal_history_and_phase_order():
    from rubik_lab.diagnostics.stroke_cycle import ReleaseRecontactCycle
    r=history(20);r['trace'][7:,0,5]=71.5
    r['contact'][12:,0,1,0,10]=0
    machine=ReleaseRecontactCycle(0,3,5,10.)
    events=[];calls=[];verified=[]
    def prefix(n):return {k:v[:n] for k,v in r.items()}
    def plan(phase,finished,prior,boundary):
        calls.append((phase,finished,prior,boundary))
        step=10 if phase=='lift' else 15
        return SimpleNamespace(source=dict(trial=0,env=0,source_step=step),prefix=phase,duration=.025,
            verify_history=lambda raw:verified.append(int(raw['trace'][-1,0,1])),
            sample=lambda elapsed:np.full(4,1. if phase=='lift' else 2.))
    assert machine.command(prefix(10),0,0,10,.025,plan,events) is None
    assert not calls
    for step in range(11,16):
        assert np.array_equal(machine.command(prefix(step),0,0,step,.025,plan,events),np.ones(4))
    assert np.array_equal(machine.command(prefix(16),0,0,16,.025,plan,events),np.full(4,2.))
    assert [x[0] for x in calls]==['lift','recontact']
    assert calls[1][1:3]==(11,0)
    assert verified==[10,15]
    assert machine.phase=='recontact' and machine.finished_step==16
    assert [x['event'] for x in events]==['cycle_lift_start','cycle_lift_targets_finished','cycle_recontact_start','cycle_recontact_targets_finished']
    assert np.all(r['trace'][7:,0,5]==71.5)

@pytest.mark.parametrize('fault',['late','wrong_source','wrong_dt'])
def test_release_recontact_machine_rejects_noncausal_execution(fault):
    from rubik_lab.diagnostics.stroke_cycle import ReleaseRecontactCycle
    r=history(11 if fault!='late' else 12);r['trace'][7:,0,5]=71.
    machine=ReleaseRecontactCycle(0,3,5,10.)
    path=SimpleNamespace(source=dict(trial=0,env=0,source_step=9),verify_history=lambda raw:None)
    with pytest.raises(ValueError):
        machine.command(r,0,0,len(r['trace']),.05 if fault=='wrong_dt' else .025,lambda *args:path,[])


def test_completed_lift_feedback_is_delayed_bounded_and_proprioceptive():
    from rubik_lab.diagnostics.stroke_cycle import ReleaseRecontactCycle
    machine=ReleaseRecontactCycle(0,3,5,10.,lift_feedback=True)
    machine.phase='lift';machine.finished_step=1
    machine.path=SimpleNamespace(source=dict(source_step=0),duration=.05,physical_reference=np.zeros(4),motor=np.zeros((2,22)),sample=lambda elapsed:np.zeros(4))
    r=history(12);r['sensory'][:,0,5:9]=.02
    r['actuator_lower']=np.full((1,2,22),-1.);r['actuator_upper']=np.ones((1,2,22))
    events=[]
    before={k:v[:11] if k not in ('actuator_lower','actuator_upper') else v for k,v in r.items()}
    assert np.array_equal(machine.command(before,0,0,11,.025,None,events),np.zeros(4))
    assert not events
    assert np.allclose(machine.command(r,0,0,12,.025,None,events),-.025)
    assert events[0]['event']=='cycle_lift_feedback_start'
    r['sensory'][-1,0,5:9]=-.02
    assert np.allclose(machine.command(r,0,0,12,.025,None,events),.025)
    r['trace'][:,0,5]=123.
    assert np.allclose(machine.command(r,0,0,12,.025,None,events),.025)
    machine.phase='recontact'
    assert np.array_equal(machine.command(r,0,0,12,.025,None,events),np.zeros(4))


def test_recontact_feedback_scoring_rejects_changed_previous_observation(monkeypatch):
    import rubik_lab.diagnostics.stroke_cycle as m
    from rubik_lab.diagnostics.whole_hand_path import bounded_index_feedback
    r=history();r['actuator_lower']=np.full((1,2,22),-1.);r['actuator_upper']=np.ones((1,2,22))
    r['sensory'][:,0,5:9]=.01
    motor=np.zeros((3,22))
    path=SimpleNamespace(source=dict(env=0,trial=0,source_frame=0,source_step=0),handoff_step=0,motor=motor,duration=.05,times=np.array([0.,.025,.05]),physical_reference=np.zeros(4),verify_history=lambda h:None)
    monkeypatch.setattr(m,'StrokeRecontactPath',lambda p,finished:path)
    for k in range(1,241):
        if k*.025>=.3:
            r['motor_targets'][k,0,0,5:9]=bounded_index_feedback(np.zeros(4),r['sensory'][k-1,0,5:9],np.zeros(4),r['motor_targets'][k-1,0,0,5:9],np.full(4,-1.),np.ones(4),.025,gain=3.)[0]
    report=dict(control_hz=40,completed_recontact_feedback=True,post_return_events=[dict(event='cycle_recontact_start',env=0,trial=0,step=1,source='unused',lift_targets_finished_step=0)])
    result=m._summarize_internal(report,r,recontact=True,event_name='cycle_recontact_start')
    assert result['component_pass'] and not result['full_action_pass']
    r['sensory'][100,0,5:9]=-.01
    with pytest.raises(ValueError,match='motor delivery'):
        m._summarize_internal(report,r,recontact=True,event_name='cycle_recontact_start')


def test_early_contact_entry_preserves_ten_consecutive_stable_frames():
    from rubik_lab.diagnostics.left_recontact import first_turn_source
    r=history(30)
    assert first_turn_source(r,0,0,3)==22
    assert first_turn_source(r,0,0,3,entry_delay_frames=1)==13
    r['contact'][10,0,1,0,10]=0
    assert first_turn_source(r,0,0,3,entry_delay_frames=1)==20
    with pytest.raises(ValueError,match='entry delay'):
        first_turn_source(r,0,0,3,entry_delay_frames=0)
