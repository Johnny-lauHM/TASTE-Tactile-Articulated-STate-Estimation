"""Normalize sensor/label fields and retain specified original grasp identities.
Every primitive fragment receives a unique ID. Raw source files remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source/rubik_lab"))
from rubik_lab.dataset_protocol import observed_row_mask,completed_collector_rows
p=argparse.ArgumentParser()
p.add_argument('inputs',nargs='+')
p.add_argument('--source_grasps',required=True)
p.add_argument('--out',required=True)
p.add_argument('--preserve_sequence',action='store_true')
p.add_argument('--moves',default=None,help='optional comma-separated move IDs by notation; incompatible with continuous history')
a=p.parse_args()
if a.moves and a.preserve_sequence:raise ValueError('Move filtering cannot retain continuous histories')
if a.moves:
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rubik_lab'))
    from rubik_lab.moves import MOVE_INDEX
    move_ids=[MOVE_INDEX[m] for m in a.moves.split(',')]
else:move_ids=None
wanted=[int(v) for v in a.source_grasps.split(',')]
fields={'tactile_ft':np.float32,'finger_pos':np.float32,'finger_tgt':np.float32,'move_id':np.int64,
        'twist_error':np.float32,'primitive_done':bool,'cycle_ready':bool,'dropped':bool,
        'broken':bool,'cycle_failed':bool,'timeout':bool}
parts={key:[] for key in [*fields,'episode_uid','grasp_idx','step','source_file','source_episode_uid','sequence_index']}
next_uid=0
sources=[]
base=None
root=Path(__file__).resolve().parents[1]
for source_id,path in enumerate(a.inputs):
    with h5py.File(path,'r') as f:
        cfg=json.loads(f.attrs['config_json'])
        if cfg.get('cycle_contract_version')!=3:raise ValueError('Require consistent version-3 labels')
        if base is None:base=cfg
        for key in ('move_success_tol','move_success_vel','cycle_stable_s','cycle_stability_mode',
                    'cycle_stable_pos','cycle_stable_rot','cycle_finger_tol','cycle_finger_speed','cycle_release_force',
                    'cycle_wrist_pos','cycle_wrist_rot','cycle_cube_pos','cycle_cube_rot',
                    'cycle_cube_speed','cycle_core_ang_speed','home_min_pads','tactile_contact_threshold','cube_drop_height'):
            if key not in cfg or key not in base:raise ValueError('Missing readiness contract field: '+key)
            if cfg[key]!=base[key]:raise ValueError('Incompatible readiness contract: '+key)
        # Historical snapshots used the same hard-coded guard before it was saved explicitly.
        if cfg.get('move_core_ang_speed',1.0)!=base.get('move_core_ang_speed',1.0):
            raise ValueError('Incompatible readiness contract: move_core_ang_speed')
        cache=np.load(root/cfg['grasp_cache_path'])
        identity=cache['source_grasp_idx'] if 'source_grasp_idx' in cache else np.arange(len(cache['cube_pos']))
        gid=identity[f['grasp_idx'][:].astype(int)].astype(np.int64)
        uid=f['episode_uid'][:]
        stage=f['sequence_index'][:] if 'sequence_index' in f else np.zeros(len(uid),np.int64)
        report_path=Path(path).with_suffix(".json")
        report=json.loads(report_path.read_text()) if report_path.exists() else None
        if report and report.get("mode") not in ("fixed_quota_evaluation","smoke"):report=None
        observed=observed_row_mask(uid,f["step"][:],stage,report,cfg,a.preserve_sequence,int(f.attrs.get("initial_settle_steps",0)))
        complete_history=np.ones(len(uid),dtype=bool)
        if report is None and 'continue_after_success' in f.attrs:
            terminal=np.zeros(len(uid),dtype=bool)
            for key in ('timeout','dropped','broken','cycle_failed'):
                if key in f:terminal |= f[key][:].astype(bool)
            complete_history=completed_collector_rows(uid,terminal,f['primitive_done'][:],not bool(f.attrs['continue_after_success']))
        keep=np.isin(gid,wanted)&observed&complete_history
        if move_ids is not None:keep &= np.isin(f["move_id"][:],move_ids)
        selected=np.flatnonzero(keep)
        if not len(selected):continue
        fragment=np.stack([uid[selected],np.zeros(len(selected),np.int64) if a.preserve_sequence else stage[selected]],-1)
        unique,inverse=np.unique(fragment,axis=0,return_inverse=True)
        local_step=f['step'][:][selected].copy()
        for k in range(len(unique)):
            mask=inverse==k
            local_step[mask]-=local_step[mask].min()
        for key,dtype in fields.items():
            if key not in f:raise ValueError('Missing field: '+key)
            parts[key].append(f[key][:][selected].astype(dtype))
        parts['episode_uid'].append(inverse+next_uid)
        parts['grasp_idx'].append(gid[selected])
        parts['step'].append(local_step)
        parts['source_file'].append(np.full(len(selected),source_id,np.int64))
        parts['source_episode_uid'].append(uid[selected])
        parts['sequence_index'].append(stage[selected])
        next_uid+=len(unique)
        sources.append({'source_id':source_id,'file':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'rows':len(selected),'fragments':len(unique),'unobserved_wait_rows_removed':int((~observed).sum()),'censored_collector_tail_rows_excluded':int((~complete_history).sum()),'censored_collector_tail_episodes_excluded':int(len(np.unique(uid[~complete_history])))})
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
with h5py.File(a.out,'x') as out:
    out.attrs.update({'schema_version':3,'config_json':json.dumps(base),'sources_json':json.dumps(sources),
                      'source_grasps_json':json.dumps(wanted),'continuous_history':a.preserve_sequence,'moves_json':json.dumps(a.moves),'observation_protocol':'startup waits excluded; per-primitive models also exclude handoff clock waits','scope':'fixed original grasp identities; evaluate on fresh rollouts'})
    for key,values in parts.items():out[key]=np.concatenate(values)
    print('[build_detector] rows',len(out['step']),'episodes',next_uid,'instant-ready frames',int(out['cycle_ready'][:].sum()),'full-ready frames',int(out['primitive_done'][:].sum()))
