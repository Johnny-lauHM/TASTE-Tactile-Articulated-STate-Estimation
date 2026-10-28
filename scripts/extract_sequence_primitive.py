"""Extract one primitive's real sequence-start histories for independent training.
This edits dataset indexing only; physical sequence trajectories are untouched.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('input')
p.add_argument('--stage',type=int,required=True)
p.add_argument('--move',required=True)
p.add_argument('--out',required=True)
p.add_argument('--seconds',type=float,default=6.)
a=p.parse_args()
with h5py.File(a.input,'r') as source:
    keep=source['sequence_index'][:]==a.stage
    if not keep.any():raise ValueError('No rows for requested primitive stage')
    moves = [f+s for f in 'RLUDFB' for s in ('', "'", '2')] + ['x', "x'", 'x2']
    if a.move not in moves or not np.all(source['move_id'][:][keep] == moves.index(a.move)):
        raise ValueError('Requested move does not match recorded stage move IDs')
    config=json.loads(source.attrs['config_json'])
    original_config=json.dumps(config)
    config['episode_length_s']=a.seconds
    config['cycle_terminate_on_success']=True
    config['moves']=[a.move]
    uid=source['episode_uid'][:][keep]
    step=source['step'][:][keep].copy()
    for episode in np.unique(uid):
        ids=uid==episode
        step[ids]-=step[ids].min()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(a.out,'x') as output:
        output.attrs.update(dict(source.attrs))
        output.attrs['config_json']=json.dumps(config)
        output.attrs['source_config_json']=original_config
        output.attrs['source_dataset']=a.input
        output.attrs['source_sequence_stage']=a.stage
        output.attrs['extraction']='Actual mid-sequence starts; only local step numbering and standalone episode metadata changed'
        report=Path(a.input).with_suffix('.json')
        if report.exists():output.attrs['source_report_json']=report.read_text()
        for key in source:
            output[key]=step if key=='step' else source[key][:][keep]
        success=np.unique(uid[output['primitive_done'][:].astype(bool)])
        print('[extract] rows',sum(keep),'episodes',len(np.unique(uid)),'complete',len(success),'->',a.out)
