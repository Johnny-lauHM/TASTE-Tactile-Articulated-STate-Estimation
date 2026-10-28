"""Run the vendor pressing trajectory without modifying its physics or controller.
Only import packaging, relative asset paths and compact recording are adapted.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument('--reference', default='/home/yuhang/workspace/third_party/sharpa-tacmap')
p.add_argument('--case', default='cylinder_D4_left_145_20250904193821_40_60')
p.add_argument('--out', default='outputs/diagnostics/press_reference')
p.add_argument('--seed', type=int, default=1200)
p.add_argument('--sensor-runtime', choices=['vendor','project'], default='vendor')
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
out = Path(a.out).resolve()
if out.with_suffix('.json').exists() or out.with_suffix('.npz').exists():
    raise FileExistsError(out)
app = AppLauncher(a).app
import numpy as np
import torch
reference = Path(a.reference).resolve()
spec = importlib.util.spec_from_file_location('sharpa_press_reference', reference/'__init__.py', submodule_search_locations=[str(reference)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from sharpa_press_reference.env_cfg import SharpaWaveEnvCfg
from sharpa_press_reference.env import SharpaWaveInhandRotateTactileAlignEnv
if a.sensor_runtime == 'project':
    import sharpa_press_reference.env as reference_env
    from rubik_lab.tacmap.sharpa_tacmap_vbts import SharpaTacmap
    reference_env.SharpaTacmap = SharpaTacmap
case_path = reference/'assets/test_case'/ (a.case+'.json')
case = json.loads(case_path.read_text())
cfg = SharpaWaveEnvCfg()
cfg.scene.num_envs = 1
cfg.seed = a.seed
cfg.sim.device = a.device
cfg.object_cfg.spawn.usd_path = cfg.object_cfg.spawn.usd_path.replace(cfg.presser_name,case['presser_name'])
cfg.presser_name = case['presser_name']
cfg.press_direction = case['presser_direction']
cfg.press_info = a.case
cfg.presser_init_pos[:3] = case['presser_init_pos']
direction = np.array(cfg.press_direction,dtype=float)
if np.linalg.norm(direction)>1e-6:
    cfg.presser_init_pos[:3] = (np.array(cfg.presser_init_pos[:3])-.004*direction/np.linalg.norm(direction)).tolist()
cfg.presser_init_pos[3:] = case['presser_init_rot']
cfg.action_target_pos = np.load(case_path.parent/case['action_target_pos_file'])
for sensor in cfg.vbts_sensor:
    for field in ('points_npy','normals_npy'):
        value=Path(getattr(sensor,field))
        if not value.is_absolute():setattr(sensor,field,str(reference/value))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
env = SharpaWaveInhandRotateTactileAlignEnv(cfg)
rows=[]
peak_image=None
peak=0
try:
    env.reset()
    # The vendor wrapper stops at the final sample, before the next auto-reset.
    max_steps=len(cfg.action_target_pos)*5
    for step in range(max_steps):
        target=int(env.target_pos_counter)
        obs,_,_,_,_=env.step(torch.zeros(1,cfg.action_space,device=env.device))
        deform=obs['vbts_deform'][0,1].detach().cpu().numpy()
        force=obs['tactile_forces'][0,1].detach().cpu().numpy()
        if not np.isfinite(deform).all() or not np.isfinite(force).all():raise ValueError('Nonfinite sensor output')
        nonzero=int(np.count_nonzero(deform))
        rows.append([step*env.step_dt,target,env.action_sequence_id,float(np.linalg.norm(force)),float(deform.max()),nonzero])
        if nonzero>peak:
            peak=nonzero;peak_image=deform.copy()
        if env.target_pos_counter==len(cfg.action_target_pos)-1 and env.action_sequence_id==4:break
    else:raise RuntimeError('Vendor final sample not reached before reset')
    trace=np.asarray(rows)
    result={'scope':'Official single-pad pressing smoke test, not cube skill acceptance or hardware calibration',
        'reference':str(reference),'case':str(case_path),'seed':a.seed,'sensor_runtime':a.sensor_runtime,'steps':len(rows),
        'physics':{'pad_stiffness':cfg.robot_cfg.spawn.compliant_contact_stiffness,'pad_damping':cfg.robot_cfg.spawn.compliant_contact_damping,'dt':cfg.sim.dt,'decimation':cfg.decimation},
        'adaptations':['package alias for hyphenated repository directory','absolute tactile asset paths','record scalar trace and peak image instead of GUI/full image stacks'],
        'force_peak_N':float(trace[:,3].max()),'deform_peak':float(trace[:,4].max()),'max_nonzero_pixels':int(trace[:,5].max()),
        'contact_frames_above_half_N':int((trace[:,3]>.5).sum()),'contact_frames_with_nonzero_image':int(((trace[:,3]>.5)&(trace[:,5]>0)).sum()),
        'trace_columns':['time_s','target_index','sequence_index','normal_force_N','deform_max','nonzero_pixels']}
    result['smoke_passed']=result['contact_frames_with_nonzero_image']>0
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'),trace=trace,peak_image=peak_image if peak_image is not None else np.zeros((240,240),dtype=np.uint8))
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
finally:
    env.close()
    app.close()
