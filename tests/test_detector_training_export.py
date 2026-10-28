"""Exercise the actual training/export path with completed-window labels."""
import json
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
import torch


def test_trainer_exports_the_actual_label_semantics(tmp_path):
    data=tmp_path/'data.h5'
    with h5py.File(data,'w') as f:
        f['episode_uid']=np.repeat(np.arange(4),6)
        f['grasp_idx']=np.repeat(np.arange(4),6)
        f['step']=np.tile(np.arange(6),4)
        f['tactile_ft']=np.zeros((24,2,5,6),np.float32)
        f['finger_pos']=np.zeros((24,2,22),np.float32)
        f['finger_tgt']=np.zeros((24,2,22),np.float32)
        f['move_id']=np.full(24,6,np.int64)
        f['twist_error']=np.zeros(24,np.float32)
        f['cycle_ready']=np.ones(24,bool)
        f['primitive_done']=np.tile([False,False,True,True,True,False],4)
        f['dropped']=np.tile([False]*5+[True],4)
    out=tmp_path/'observer.pt'
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'scripts/train_sequence_detector.py'),str(data),
        '--out',str(out),'--device','cpu','--epochs','1','--label_key','primitive_done',
        '--hold_steps','1','--split_by','episode_uid'],cwd=root,check=True,capture_output=True,timeout=30)
    checkpoint=torch.load(out,weights_only=False,map_location='cpu')
    report=json.loads(Path(str(out)+'.json').read_text())
    assert checkpoint['label_key']==report['label_key']=='primitive_done'
    assert checkpoint['hold_steps']==1
    assert report['class_counts']==[6,9,3]
    assert Path(str(out)+'.sources/train_sequence_detector.py').is_file()
