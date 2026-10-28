"""Extract actor-observed, per-primitive histories from a real no-reset sequence."""
import argparse,hashlib,json,sys
from pathlib import Path
import h5py,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rubik_lab'))
from rubik_lab.dataset_protocol import observed_row_mask
from rubik_lab.moves import MOVE_INDEX
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--move',required=True);p.add_argument('--out',required=True)
a=p.parse_args();out=Path(a.out)
if out.exists():raise FileExistsError(out)
report_path=Path(a.input).with_suffix('.json');report=json.loads(report_path.read_text())
with h5py.File(a.input,'r') as src:
 cfg=json.loads(src.attrs['config_json']);uid=src['episode_uid'][:];stage=src['sequence_index'][:];step=src['step'][:]
 observed=observed_row_mask(uid,step,stage,report,cfg,False)
 if 'command_issued' in src and not np.array_equal(observed,src['command_issued'][:].astype(bool)):raise ValueError('Observed protocol disagrees with recorded command mask')
 keep=observed&(src['move_id'][:]==MOVE_INDEX[a.move])
 keys,inverse=np.unique(np.stack([uid[keep],stage[keep]],-1),axis=0,return_inverse=True)
 local=step[keep].copy()
 for i in range(len(keys)):
  mask=inverse==i;local[mask]-=local[mask].min()
  if not np.array_equal(np.sort(local[mask]),np.arange(mask.sum())):raise ValueError('Noncontiguous actor history')
 out.parent.mkdir(parents=True,exist_ok=True)
 with h5py.File(out,'x') as dst:
  dst.attrs.update(dict(src.attrs));cfg['moves']=[a.move];cfg['episode_length_s']=report['per_primitive_timeout_seconds'];cfg['cycle_terminate_on_success']=True
  dst.attrs['config_json']=json.dumps(cfg);dst.attrs['source_dataset']=a.input;dst.attrs['source_sha256']=hashlib.sha256(Path(a.input).read_bytes()).hexdigest()
  dst.attrs['extraction']='No physical edits; remove only unobserved command waits, give each original trial/stage its own UID and local step'
  for key in src:
   value=src[key][:][keep]
   if key=='episode_uid':value=inverse
   elif key=='step':value=local
   dst.create_dataset(key,data=value,compression='lzf')
  dst['source_episode_uid']=uid[keep]
  good=np.unique(inverse[dst['primitive_done'][:].astype(bool)])
  summary={'source':a.input,'out':a.out,'move':a.move,'rows':int(keep.sum()),'fragments':len(keys),'successful_fragments':len(good),'stages':np.unique(stage[keep]).tolist(),'wait_rows_removed':int((~observed).sum()),'source_sha256':dst.attrs['source_sha256']}
Path(str(out)+'.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
