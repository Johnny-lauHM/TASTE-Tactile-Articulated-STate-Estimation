"""Merge compatible recorded trials while preserving disjoint episode IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('inputs', nargs='+')
p.add_argument('--out', required=True)
a = p.parse_args()
files = [h5py.File(path,'r') for path in a.inputs]
configs = [json.loads(f.attrs['config_json']) for f in files]
base = configs[0]
keys = set(files[0].keys())
physical_keys = ['grasp_cache_path','contact_model','notch_torque','detent_torque','twist_friction_torque',
                 'action_ema','residual_finger_range','cycle_contract_version','cycle_idle_lift',
                 'cycle_stable_s','cycle_stability_mode','cycle_stable_pos','cycle_stable_rot']
physical_keys += sorted(k for k in base if k.startswith("cycle_"))
physical_keys += ["move_success_tol", "move_success_vel", "home_min_pads", "flick_hold_press"]
for f, cfg in zip(files,configs):
    if set(f.keys()) != keys:
        raise ValueError('Dataset fields differ; do not silently discard fields')
    for key in physical_keys:
        if cfg.get(key) != base.get(key):
            raise ValueError('Incompatible physical/control configuration: '+key)
    for key in keys:
        if f[key].shape[1:] != files[0][key].shape[1:] or f[key].dtype != files[0][key].dtype:
            raise ValueError('Incompatible data shape: '+key)
lengths = [len(f['episode_uid']) for f in files]
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
with h5py.File(a.out,'x') as out:
    out.attrs.update({'schema_version':2,'task':'mixed compatible primitives','completion_label':'primitive_done',
                      'config_json':json.dumps(base),'sources_json':json.dumps(a.inputs),
                      'source_sha256_json':json.dumps([hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in a.inputs])})
    for key in sorted(keys):
        original = files[0][key]
        out.create_dataset(key,shape=(sum(lengths),*original.shape[1:]),dtype=original.dtype)
    out.create_dataset('source_id',shape=(sum(lengths),),dtype=np.int32)
    out.create_dataset('source_episode_uid',shape=(sum(lengths),),dtype=np.int64)
    offset = next_uid = 0
    for source,(f,n) in enumerate(zip(files,lengths)):
        rows = slice(offset,offset+n)
        for key in sorted(keys):
            out[key][rows] = f[key][:]
        uid = f['episode_uid'][:]
        unique,inverse = np.unique(uid,return_inverse=True)
        out['source_episode_uid'][rows] = uid
        out['episode_uid'][rows] = inverse+next_uid
        out['source_id'][rows] = source
        next_uid += len(unique)
        offset += n
print('[merge] rows',sum(lengths),'episodes',next_uid,'->',a.out)
for f in files:
    f.close()
