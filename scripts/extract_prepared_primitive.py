"""Remove a recorded initialization hold from teacher histories for turn-policy training.
Physics and observations are untouched; the executor must reproduce the initial hold.
"""
import argparse,json,hashlib
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser()
p.add_argument('input');p.add_argument('--out',required=True)
a=p.parse_args()
report=json.loads(Path(a.input).with_suffix('.json').read_text())
if report['sequence'] is not None:raise ValueError('Single primitive only')
if report.get('synchronize_tactile_clock'):raise ValueError('Use fixed-duration preparation for this extractor')
skip=report['initial_settle_steps']
if skip<=0:raise ValueError('Expected recorded nonzero initialization hold')
with h5py.File(a.input,'r') as source:
    step=source['step'][:];keep=step>=skip
    if not keep.any():raise ValueError('No issued turn actions')
    uid=source['episode_uid'][:][keep]
    local=step[keep]-skip
    for episode in np.unique(uid):
        observed=np.sort(local[uid==episode])
        if not np.array_equal(observed,np.arange(len(observed))):raise ValueError('Noncontiguous prepared trajectory')
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(a.out,'x') as out:
        out.attrs.update(dict(source.attrs))
        out.attrs['source_dataset']=a.input
        out.attrs['source_sha256']=hashlib.sha256(Path(a.input).read_bytes()).hexdigest()
        out.attrs['execution_protocol_json']=json.dumps({'initial_settle_steps':skip,'scope':'one-time ready hold after trial reset; real time and motion remain scored'})
        for key in source:out[key]=local if key=='step' else source[key][:][keep]
        print('[prepared_extract] rows',int(keep.sum()),'episodes',len(np.unique(uid)),'successes',len(np.unique(uid[out['primitive_done'][:].astype(bool)])))
