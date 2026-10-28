"""Matched explicit-state, fixed-target diagnostic; never a skill evaluation."""
import argparse,json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--states',required=True)
p.add_argument('--out',required=True)
p.add_argument('--seconds',type=float,default=2.)
p.add_argument('--physics-hz',type=int,default=240)
p.add_argument('--forces-each-iteration',action='store_true')
p.add_argument('--friction-law',choices=['legacy','implicit-damping'],default='legacy')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();out=Path(a.out)
if out.with_suffix('.json').exists() or out.with_suffix('.npz').exists():raise FileExistsError(out)
if a.seconds<=0 or a.physics_hz<=0:raise ValueError('Positive duration and physics rate required')
app=AppLauncher(a).app
import numpy as np
import torch
from rubik_lab.tasks.bimanual_twist.cycle_cfg import CycleUEnvCfg
from rubik_lab.tasks.bimanual_twist.cycle_env import PrimitiveCycleEnv
from rubik_lab.primitive_config import restore_checkpoint_settings
from rubik_lab.passive_friction import friction_damping
raw=np.load(a.states);meta=json.loads(Path(a.states).with_suffix('.json').read_text())
index=raw['terminal_state_index'];n=len(index)
if n==0:raise ValueError('No captured states')
cfg=CycleUEnvCfg();restore_checkpoint_settings(cfg,meta['models'][0])
cfg.scene.num_envs=n;cfg.seed=1240;cfg.moves=('U',"U'")
cfg.grasp_cache_path=meta['cache'];cfg.gravity_curriculum=False
cfg.sim.dt=1/a.physics_hz;cfg.decimation=1;cfg.sim.render_interval=1
cfg.sim.physx.enable_external_forces_every_iteration=a.forces_each_iteration
if cfg.detent_torque or cfg.notch_torque:raise ValueError('Expected unchanged zero detent/notch')
u=PrimitiveCycleEnv(cfg)
try:
    u.reset(seed=1240)
    state={'articulation':{},'rigid_object':{},'deformable_object':{},'gripper':{}}
    targets={}
    fields=('root_pose','root_velocity','joint_position','joint_velocity')
    target_fields=('joint_pos_target','joint_vel_target','joint_effort_target')
    for name,asset in u.scene.articulations.items():
        if list(asset.joint_names)!=meta['terminal_state_joint_names'][name]:raise ValueError('Joint ordering mismatch')
        state['articulation'][name]={field:torch.as_tensor(raw[f'terminal_state__{name}__{field}'],device=u.device) for field in fields}
        targets[name]={field:torch.as_tensor(raw[f'terminal_state__{name}__{field}'],device=u.device) for field in target_fields}
        if any(not torch.isfinite(x).all() for x in [*state['articulation'][name].values(),*targets[name].values()]):raise ValueError('Nonfinite capture')
    u.scene.reset_to(state,is_relative=True)
    # reset_to replaces both position AND velocity targets. Restore both, plus
    # feedforward effort: retaining captured nonzero velocities is not a target.
    for name,asset in u.scene.articulations.items():
        asset.set_joint_position_target(targets[name]['joint_pos_target'])
        asset.set_joint_velocity_target(targets[name]['joint_vel_target'])
        asset.set_joint_effort_target(targets[name]['joint_effort_target'])
    u.scene.write_data_to_sim();u.sim.forward();u.scene.update(0.)
    restored=u.scene.get_state(is_relative=True)['articulation'];errors={}
    for name,asset in u.scene.articulations.items():
        for field in fields:
            error=float((restored[name][field]-state['articulation'][name][field]).abs().max())
            errors[f'{name}/{field}']=error
            if error>2e-6:raise RuntimeError(f'Explicit state restore mismatch: {name}/{field}={error}')
        for field in target_fields:
            error=float((getattr(asset.data,field)-targets[name][field]).abs().max())
            errors[f'{name}/{field}']=error
            if error>1e-7:raise RuntimeError(f'Actuator target restore mismatch: {name}/{field}')
    base_damping=u.cube.data.joint_damping[:,u.twist_id:u.twist_id+1].clone()
    steps=round(a.seconds*a.physics_hz);trace=[]
    for step in range(steps+1):
        q=u.cube.data.joint_pos[:,u.twist_id]
        v=u.cube.data.joint_vel[:,u.twist_id]
        force=u.cube_contacts.forces()[:,1].sum((-1,-2))
        trace.append(torch.stack([q,v,force,u.cube.data.body_ang_vel_w[:,u.core_id].norm(dim=-1)],-1).cpu().numpy())
        if step==steps:break
        if a.friction_law=='implicit-damping':
            damping=base_damping+friction_damping(v[:,None],cfg.twist_friction_torque,.05)
            u.cube.write_joint_damping_to_sim(damping,joint_ids=[u.twist_id])
            tau=torch.zeros_like(v[:,None])
        else:
            tau=-cfg.twist_friction_torque*torch.tanh(v[:,None]/.05)
        u.cube.set_joint_effort_target(tau,joint_ids=[u.twist_id])
        u.scene.write_data_to_sim();u.sim.step(render=False);u.scene.update(cfg.sim.dt)
    trace=np.asarray(trace)
    warm=[]
    source_interval=a.physics_hz/meta["physics_hz"]
    compare_step=int(source_interval) if source_interval.is_integer() else None
    source=raw['release_substeps']
    for i,(trial,env,physics_step) in enumerate(index):
        rows=source[(source[:,0]==trial)&(source[:,2]==env)&(source[:,1]==physics_step+1)]
        warm.append(None if len(rows)!=1 or compare_step is None or compare_step>=len(trace) else {'angle_difference_rad':float(trace[compare_step,i,0]-rows[0,3]),'speed_difference_rad_s':float(trace[compare_step,i,1]-rows[0,4])})
    report={'scope':'Matched explicit-state cold restore, fixed actuator targets; no policy/no skill acceptance. Solver contact/constraint warmstart not exported.',
            'states':str(a.states),'count':n,'physics_hz':a.physics_hz,'seconds':a.seconds,
            'forces_each_iteration':a.forces_each_iteration,'friction_law':a.friction_law,'friction_torque':cfg.twist_friction_torque,
            'warm_comparison_time_s':1/meta['physics_hz'],'restore_max_errors':errors,'first_step_vs_source_warm':warm,
            'trace_columns':['joint_angle_rad','joint_speed_rad_s','top_total_normal_force_N','core_angular_speed_rad_s']}
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'),trace=trace,state_index=index)
    out.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'count':n,'restore_max_error':max(errors.values())}),flush=True)
finally:
    u.close();app.close()
