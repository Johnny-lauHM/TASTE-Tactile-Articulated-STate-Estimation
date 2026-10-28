"""Measure missing tangential force without changing the actor's legacy inputs."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--num_envs',type=int,default=4);p.add_argument('--steps',type=int,default=40)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import json,hashlib
from pathlib import Path
import numpy as np
import torch,gymnasium as gym
import isaaclab.utils.math as mu
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import rubik_lab
from rubik_lab.blind_actor import BlindInference
from rubik_lab.primitive_config import restore_checkpoint_settings
out=Path(a.out)
if out.exists():raise FileExistsError(out)
source_hashes={}
archive=out.with_suffix('')/'sources';archive.mkdir(parents=True,exist_ok=False)
project=Path(__file__).resolve().parents[1]
for relative in ('source/rubik_lab/rubik_lab/contact_wrench.py','source/rubik_lab/rubik_lab/wave_se_tactile.py','scripts/verify_contact_wrench.py'):
 content=(project/relative).read_bytes();source_hashes[relative]=hashlib.sha256(content).hexdigest();(archive/Path(relative).name).write_bytes(content)
cfg=load_cfg_from_registry('Rubik-Cycle-U-Sharpa-v0','env_cfg_entry_point');restore_checkpoint_settings(cfg,a.checkpoint)
cfg.scene.num_envs=a.num_envs;cfg.seed=1111;cfg.grasp_cache_path=a.cache;cfg.cycle_terminate_on_success=False;cfg.gravity_curriculum=False
cfg.tactile.contact_cfg.track_friction_forces=True
env=gym.make('Rubik-Cycle-U-Sharpa-v0',cfg=cfg);u=env.unwrapped
policy=BlindInference(torch.load(a.checkpoint,map_location='cpu',weights_only=False),u.num_envs,u.device)
obs,_=env.reset();normal=[];friction=[];buffer_shapes={};steps=[];wrenches=[];force_errors=[]
with torch.inference_mode():
 for step in range(a.steps):
  obs,_,terminated,truncated,_=env.step(policy(obs));policy.reset((terminated|truncated).nonzero().flatten())
  nf=[];tf=[]
  for side in ('left','right'):
   nf.append(torch.cat([s.data.force_matrix_w for s in u.tactile.contact[side]],1).sum(2))
   tf.append(torch.cat([s.data.friction_forces_w for s in u.tactile.contact[side]],1).sum(2))
  normal.append(torch.stack(nf,1).cpu().numpy());friction.append(torch.stack(tf,1).cpu().numpy())
  wrench=u.tactile.raw_cube_wrench()
  expected_world=torch.stack(nf,1)+torch.stack(tf,1)
  expected_pad=[]
  for hi,side in enumerate(('left','right')):
   q=u.hands[side].data.body_quat_w[:,u.tactile.pad_body_ids[side]]
   expected_pad.append(mu.quat_apply_inverse(q.reshape(-1,4),expected_world[:,hi].reshape(-1,3)).reshape(u.num_envs,5,3))
  expected=torch.stack(expected_pad,1)
  error=(wrench[...,:3]-expected).abs().max()
  force_errors.append(float(error));wrenches.append(wrench.cpu().numpy())
  if not torch.isfinite(wrench).all():raise ValueError('Nonfinite reconstructed wrench')
  if step==20:
   sensor=u.tactile.contact['left'][1]
   for name,values in [('normal',sensor.contact_physx_view.get_contact_data(dt=cfg.sim.dt)),('friction',sensor.contact_physx_view.get_friction_data(dt=cfg.sim.dt))]:
    buffer_shapes[name]=[{'shape':list(x.shape),'dtype':str(x.dtype)} for x in values]
  if (step+1)%20==0:print('[friction]',step+1,flush=True)
n=np.stack(normal);f=np.stack(friction);summary={}
for h,side in enumerate(('left','right')):
 for finger in range(5):
  fn=np.linalg.norm(n[:,:,h,finger],axis=-1);ft=np.linalg.norm(f[:,:,h,finger],axis=-1);valid=fn>.2
  summary[side+'_'+str(finger)]={'normal_peak_N':float(fn.max()),'friction_peak_N':float(ft.max()),'friction_nonzero_fraction':float((ft>.05).mean()),'friction_over_normal_p50_p90':np.percentile(ft[valid]/fn[valid],[50,90]).tolist() if valid.any() else None}
np.savez_compressed(out.with_suffix('.npz'),normal_world=n,friction_world=f,cube_wrench_pad=np.stack(wrenches))
out.write_text(json.dumps({'scope':'Legacy actor inputs unchanged; friction tracked for diagnostic only. Filtered cube contacts; not a total hand wrench.','args':vars(a),'summary':summary,'source_sha256':source_hashes,'source_archive':str(archive),'buffer_shapes':buffer_shapes,'max_force_reconstruction_error_N':max(force_errors),'all_finite':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print('[friction] finished',json.dumps(summary),flush=True);env.close();app.close()
