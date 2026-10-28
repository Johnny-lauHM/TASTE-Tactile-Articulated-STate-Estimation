import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rubik_lab'))
from rubik_lab.dataset_protocol import observed_row_mask


def test_history_excludes_waits_that_reset_per_primitive_observers():
    uid=np.repeat([0,128],7)
    step=np.tile(np.arange(7),2)
    stage=np.tile([0,0,0,1,1,1,1],2)
    cfg={'scene':{'num_envs':128},'sim':{'dt':1/240},'tactile':{'fps':30},'decimation':6}
    report={'actual_startup_hold_steps':[0,1],'trial_sensor_phases':[0,2],'synchronize_each_primitive':True}
    mask=observed_row_mask(uid,step,stage,report,cfg)
    assert mask[:7].tolist()==[True,True,True,False,True,True,True]
    assert mask[7:].tolist()==[False,True,True,False,False,True,True]
    continuous=observed_row_mask(uid,step,stage,report,cfg,preserve_sequence=True)
    assert continuous[:7].all()
    assert continuous[7:].tolist()==[False,True,True,True,True,True,True]


def test_collector_initial_hold_is_removed_per_episode():
    uid=np.repeat([8,9],4);step=np.tile(np.arange(4),2)
    mask=observed_row_mask(uid,step,np.zeros(8,int),None,{},initial_hold=2)
    assert mask.tolist()==[False,False,True,True]*2


def test_collector_tail_does_not_count_as_complete_negative_episode():
    from rubik_lab.dataset_protocol import completed_collector_rows
    uid=np.array([0,1,2,0,1,2])
    terminal=np.array([0,0,0,1,0,0],bool)
    ready=np.array([0,0,0,0,1,0],bool)
    assert completed_collector_rows(uid,terminal,ready,False).tolist()==[True,False,False,True,False,False]
    assert completed_collector_rows(uid,terminal,ready,True).tolist()==[True,True,False,True,True,False]
