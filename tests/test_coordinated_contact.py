"""Actual archived trajectory regression: no simulator/GPU required."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from rubik_lab.diagnostics.coordinated_contact import CoordinatedContactPath,validate_audit,HISTORY

PREFIX=Path('outputs/diagnostics/coordinated_contact_source435_normalreserve')
@pytest.fixture(scope='module')
def path():
    if not PREFIX.with_suffix('.npz').exists():pytest.skip('Local diagnostic source artifact required')
    return CoordinatedContactPath(PREFIX)

def test_original_clock_and_continuous_actuator_commands(path):
    assert path.source_step==435 and path.handoff_step==299
    assert path.duration==.65 and path.target_degrees==17
    start=path.sample(0)
    np.testing.assert_allclose(start['fingers'],path.history['motor_targets'][-1,0],rtol=0,atol=1e-15)
    np.testing.assert_array_equal(start['wrist_position'],path.history['wrist_targets'][-1,0,:3])
    samples=[path.sample(i/40) for i in range(28)]
    motor=np.asarray([v['fingers'] for v in samples])
    assert np.abs(np.diff(motor,axis=0)).max()<=.025+1e-12
    assert np.all(motor>=path.lower) and np.all(motor<=path.upper)
    assert all(np.array_equal(v['wrist_quaternion'],start['wrist_quaternion']) for v in samples)
    np.testing.assert_array_equal(samples[-1]['fingers'],path.motor[-1])

def test_history_rejects_changed_observation_and_future_frame(path):
    history={k:np.repeat(v[:,None],path.env+1,axis=1) for k,v in path.history.items()}
    path.verify_history(history)
    altered=dict(history);altered['sensory']=history['sensory'].copy();altered['sensory'][-1,path.env,0]+=1e-6
    with pytest.raises(ValueError,match='sensory'):path.verify_history(altered)
    altered={k:np.concatenate([v,v[-1:]],axis=0) for k,v in history.items()}
    with pytest.raises(ValueError,match='history'):path.verify_history(altered)

@pytest.mark.parametrize('kind',['support','cube','depth','rigid','missing'])
def test_independent_audit_rejection(path,kind):
    audit=json.loads(Path(str(PREFIX)+'_geometry_audit.json').read_text())
    sample=audit['results'][5]['segment_samples'][8]
    if kind=='support':sample['support_contact']['passed']=False
    elif kind=='cube':sample['cube_clearance']['pass_at_0p1mm']=False
    elif kind=='depth':sample['cube_clearance']['active_pad_top_signed_overlap_mm']=1.001
    elif kind=='rigid':sample['collisions'].append(dict(rigid_pair=True,sat_signed_overlap_m=.000101))
    else:audit['results'][5]['segment_samples'].pop()
    with pytest.raises(ValueError):validate_audit(audit,PREFIX,path.count)

@pytest.mark.parametrize('value',[-.01,float('nan'),float('inf')])
def test_invalid_time_rejected(path,value):
    with pytest.raises(ValueError):path.sample(value)

@pytest.fixture
def actual_raw():
    with np.load('outputs/diagnostics/whole_hand_coordinated_contact17.npz') as z:
        return {k:z[k].copy() for k in (*HISTORY,'env_origins','actuator_lower','actuator_upper')}

def test_actual_binding_keeps_changed_ft_but_does_not_weaken_history(path,actual_raw):
    evidence=path.validate_geometry_inputs(actual_raw)
    assert evidence['delivered_ft_difference_count']==2
    with pytest.raises(ValueError,match='sensory'):path.verify_history(actual_raw)

@pytest.mark.parametrize('key,index',[('sensory',(435,7,0)),('sensory',(435,7,44)),('poses',(435,7,0)),('motor_targets',(435,7,0,0)),('contact',(435,7,0,0,0)),('trace',(435,7,14)),('env_origins',(7,0)),('actuator_lower',(7,0,0))])
def test_binding_rejects_any_changed_consumed_input(path,actual_raw,key,index):
    actual_raw[key][index]+=1e-5
    with pytest.raises(ValueError,match='geometric'):path.validate_geometry_inputs(actual_raw)

def test_binding_preserves_actual_record_even_when_new_audit_rejects(path,actual_raw,tmp_path):
    called=[]
    def reject(geometry):
        called.append(geometry)
        raise ValueError('independent audit rejected')
    prefix=tmp_path/'bound'
    with pytest.raises(ValueError,match='independent audit rejected'):
        path.bind_to_actual(actual_raw,prefix,reject)
    assert len(called)==1
    with np.load(str(prefix)+'_raw.npz') as z:
        np.testing.assert_array_equal(z['sensory'],actual_raw['sensory'])
    with pytest.raises(FileExistsError):path.bind_to_actual(actual_raw,prefix,reject)
