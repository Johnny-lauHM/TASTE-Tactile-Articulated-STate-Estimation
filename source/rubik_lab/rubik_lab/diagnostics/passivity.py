"""Isolated passive-joint spin-down: diagnose numerical friction, never skill acceptance."""
import argparse,json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--physics-hz',type=int,default=240);p.add_argument('--out',required=True);p.add_argument('--friction-law',choices=['legacy','impulse-cap','implicit-damping'],default='legacy');p.add_argument('--external-torque',type=float,default=0.0);p.add_argument('--forces-each-iteration',action='store_true')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();out=Path(a.out)
if out.with_suffix('.json').exists() or out.with_suffix('.npz').exists():raise FileExistsError(out)
app=AppLauncher(a).app
import numpy as np
import torch
from isaaclab.sim import SimulationCfg,SimulationContext,PhysxCfg
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from isaaclab.utils import configclass
from rubik_lab.assets import rubik_cube_cfg
@configclass
class SceneCfg(InteractiveSceneCfg):
    cube=rubik_cube_cfg('rigid')
    cube.spawn.articulation_props.fix_root_link=None
    cube.spawn.rigid_props.disable_gravity=True
dt=1/a.physics_hz
sim=SimulationContext(SimulationCfg(dt=dt,device='cuda:0',gravity=(0.0,0.0,0.0),physx=PhysxCfg(enable_external_forces_every_iteration=a.forces_each_iteration)))
scene=InteractiveScene(SceneCfg(num_envs=8,env_spacing=1.0,replicate_physics=True))
from pxr import UsdPhysics,Sdf,Gf
# Asset articulation root is an Xform, so anchor its core body explicitly.
for i,origin in enumerate(scene.env_origins.cpu().tolist()):
    joint_prim=UsdPhysics.FixedJoint.Define(sim.stage,f'/World/envs/env_{i}/cube/diagnostic_fixed_core')
    joint_prim.CreateBody1Rel().SetTargets([Sdf.Path(f'/World/envs/env_{i}/cube/core')])
    joint_prim.CreateLocalPos0Attr(Gf.Vec3f(*origin))
    joint_prim.CreateLocalPos1Attr(Gf.Vec3f(0.0,0.0,0.0))
sim.reset();scene.reset()
cube=scene['cube'];joint=cube.find_joints('twist_joint')[0][0]
initial=torch.tensor([0.001,0.01,0.05,0.1,0.5,1.,5.,10.],device='cuda:0')
q=cube.data.default_joint_pos.clone();v=torch.zeros_like(q);v[:,joint]=initial
cube.write_joint_state_to_sim(q,v);scene.write_data_to_sim();scene.update(dt)
from rubik_lab.passive_friction import bounded_friction_torque,friction_damping
base_damping=cube.data.joint_damping[:,joint].clone()
inertia_bound=cube.data.joint_armature[:,joint].clone()
if (inertia_bound<=0).any():raise ValueError("Positive joint armature required for passivity bound")
trace=[]
try:
    for step in range(a.physics_hz):
        before=cube.data.joint_vel[:,joint].clone()
        damping=base_damping
        if a.friction_law=="legacy":tau=-0.01*torch.tanh(before/0.05)
        elif a.friction_law=="impulse-cap":tau=bounded_friction_torque(before,0.01,0.05,dt,inertia_bound)
        else:
            damping=base_damping+friction_damping(before,0.01,0.05)
            cube.write_joint_damping_to_sim(damping[:,None],joint_ids=[joint])
            tau=torch.zeros_like(before)
        tau=tau+a.external_torque
        cube.set_joint_effort_target(tau[:,None],joint_ids=[joint]);scene.write_data_to_sim();sim.step(render=False);scene.update(dt)
        after=cube.data.joint_vel[:,joint].clone()
        trace.append(torch.stack([before,tau,after,cube.data.joint_pos[:,joint],damping],-1).cpu().numpy())
    t=np.asarray(trace);tail=t[-max(1,a.physics_hz//5):,:,2]
    result={'scope':'Isolated fixed-core cube, no hands/gravity/contact; numerical diagnostic, not production physics or skill acceptance. Same USD twist damping/armature; timestep or explicitly selected numerical friction impulse bound varied.','forces_each_iteration':a.forces_each_iteration,'external_torque_Nm':a.external_torque,'base_damping':base_damping.cpu().tolist(),'friction_law':a.friction_law,'inertia_lower_bound':inertia_bound.cpu().tolist(),'physics_hz':a.physics_hz,'duration_s':1.,'law':{'legacy':'explicit -0.01*tanh(omega/0.05)','impulse-cap':'explicit friction bounded byarmature*speed/dt','implicit-damping':'native implicitjoint damping +=0.01*tanh(abs(omega)/0.05)/abs(omega),limit0.2atrest'}[a.friction_law],'initial_omega_rad_s':initial.cpu().tolist(),'tail_rms_speed':np.sqrt((tail**2).mean(0)).tolist(),'tail_max_abs_speed':abs(tail).max(0).tolist(),'speed_magnitude_growth_steps':(abs(t[:,:,2])>abs(t[:,:,0])+1e-5).sum(0).tolist(),'sign_reversal_steps':(t[:,:,0]*t[:,:,2]<0).sum(0).tolist(),'trace_columns':['omega_before','friction_torque_Nm','omega_after','angle_rad','implicit_joint_damping'],'finite':bool(np.isfinite(t).all())}
    result['generalized_mass_matrix']=cube.root_physx_view.get_generalized_mass_matrices().cpu().tolist()
    fd=np.diff(t[:,:,3],axis=0)/dt
    result['tail_mean_angle_difference_speed']=fd[-max(1,a.physics_hz//5):].mean(0).tolist()
    result['tail_rms_angle_difference_speed']=np.sqrt((fd[-max(1,a.physics_hz//5):]**2).mean(0)).tolist()
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),trace=t)
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except BaseException:
    import traceback;traceback.print_exc();raise
finally:
    del cube
    del scene
    sim.clear_all_callbacks()
    sim.clear_instance()
    app.close()
