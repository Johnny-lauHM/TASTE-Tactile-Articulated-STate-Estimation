"""Match recorded histories to the observations actually seen by readiness models."""
import numpy as np


def observed_row_mask(uid,step,stage,report,cfg,preserve_sequence=False,initial_hold=0):
    uid,step,stage=map(np.asarray,(uid,step,stage))
    keep=np.ones(len(uid),dtype=bool)
    if not report:
        return step>=initial_hold
    n=int(cfg['scene']['num_envs'])
    trial=uid//n
    holds=report.get('actual_startup_hold_steps')
    if holds is not None:
        if len(uid) and trial.max()>=len(holds):raise ValueError('Raw evaluation UID does not match trial metadata')
        minimum=np.array([initial_hold if value is None else value for value in holds],dtype=int)[trial]
    else:
        minimum=np.full(len(uid),report.get('initial_settle_steps',initial_hold),dtype=int)
    keep &= (stage!=0)|(step>=minimum)
    if report.get('synchronize_each_primitive') and not preserve_sequence:
        phases=report.get('trial_sensor_phases')
        if phases is None:raise ValueError('Missing capture phase metadata')
        dt=cfg['sim']['dt'];period=max(1,round(1/(cfg['tactile']['fps']*dt)))
        stride=int(cfg['decimation'])
        for episode,segment in np.unique(np.stack([uid,stage],-1),axis=0):
            if segment==0:continue
            rows=np.flatnonzero((uid==episode)&(stage==segment))
            phase=(phases[int(episode)//n]+step[rows]*stride)%period
            starts=step[rows][phase==0]
            if len(starts):keep[rows] &= step[rows]>=starts.min()
            else:keep[rows]=False
    return keep


def completed_collector_rows(uid,terminal,primitive_done,terminate_on_success):
    """Exclude episodes truncated only by the collector's global step budget.

    A successful frame ends an episode only when the environment actually resets
    on success. Uncensored collection must reach timeout or physical termination.
    This prevents short unfinished negatives diluting event false-trigger rates.
    """
    uid=np.asarray(uid);ended=np.asarray(terminal,dtype=bool).copy()
    done=np.asarray(primitive_done,dtype=bool)
    if uid.shape!=ended.shape or uid.shape!=done.shape:raise ValueError('Collector fields must align')
    if terminate_on_success:ended |= done
    return np.isin(uid,np.unique(uid[ended]))
