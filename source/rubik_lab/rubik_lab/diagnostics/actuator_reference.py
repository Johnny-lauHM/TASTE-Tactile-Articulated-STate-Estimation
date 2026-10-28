"""Sharpa actuator step response without cube contacts; no skill acceptance."""
import argparse,json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--physics-hz',type=int,required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
if a.out.with_suffix('.json').exists() or a.out.with_suffix('.npz').exists():raise FileExistsError(a.out)
app=AppLauncher(a).app
import numpy as np
import torch
from pxr import UsdGeom,Gf
from isaaclab.sim import SimulationCfg,SimulationContext,PhysxCfg
from isaaclab.assets import Articulation
from rubik_lab.assets import sharpa_hand_cfg
from rubik_lab.constants import FINGER_JOINTS,FLOAT_BASE_JOINTS
settings=json.loads(Path('logs/rsl_rl/rubik_primitive_cycle/U_blind_full/env.json').read_text())
c=np.load('cache/g70s_cycle_demo1_v3.npz')
dt=1/a.physics_hz
sim=SimulationContext(SimulationCfg(dt=dt,device=a.device,gravity=(0.,0.,0.),physx=PhysxCfg(enable_external_forces_every_iteration=False)))
for i in range(4):
    prim=UsdGeom.Xform.Define(sim.stage,f'/World/bench_{i}');prim.AddTranslateOp().Set(Gf.Vec3d(i*.4,0,0))
cfg=sharpa_hand_cfg('right',contact_model=settings['contact_model'],wrist_stiffness=settings.get('wrist_stiffness'))
cfg.prim_path='/World/bench_.*/hand';cfg.spawn.collision_props.collision_enabled=False
cfg.spawn.articulation_props.enabled_self_collisions=False
cfg.init_state.joint_pos={**{'right_'+j:0. for j in FLOAT_BASE_JOINTS},**{'right_'+j:float(v) for j,v in zip(FINGER_JOINTS,c['right_fingers'][0])}}
hand=Articulation(cfg);sim.reset();hand.reset()
base=hand.data.default_joint_pos.clone();hand.write_joint_state_to_sim(base,torch.zeros_like(base))
names=['right_index_MCP_FE','right_index_MCP_AA','right_index_PIP','right_index_DIP']
joints=torch.tensor(hand.find_joints(names,preserve_order=True)[0],device=sim.device)
rows=torch.arange(4,device=sim.device)
limits=hand.data.joint_pos_limits[rows,joints]
pulse=torch.where(base[rows,joints]+.05<=limits[:,1],.05,-.05)
if bool(((base[rows,joints]+pulse<limits[:,0])|(base[rows,joints]+pulse>limits[:,1])).any()):raise ValueError('Pulse would hit physical joint limit')
trace=[]
try:
    for step in range(round(1.2*a.physics_hz)):
        time=step*dt;target=base.clone()
        if .2-1e-9<=time<.7-1e-9:target[rows,joints]+=pulse
        hand.set_joint_position_target(target);hand.write_data_to_sim();sim.step(render=False);hand.update(dt)
        trace.append(torch.stack([target,hand.data.joint_pos,hand.data.joint_vel,hand.data.computed_torque,hand.data.applied_torque,hand._joint_effort_target_sim],1).cpu().numpy())
    raw=np.asarray(trace);ids=joints.cpu().numpy();act=hand.actuators['fingers'];tail=raw[-round(.2*a.physics_hz):]
    error=np.abs(tail[:,np.arange(4),1,ids]-base.cpu().numpy()[np.arange(4),ids])
    speed=np.abs(tail[:,np.arange(4),2,ids])
    result=dict(scope='Four full Sharpa hands, no collision/gravity; each pulses one index joint by0.05rad inward from physical limit from0.2to0.7s. Not a cube skill.',physics_hz=a.physics_hz,control_hz=40,external_forces_every_iteration=False,solver='TGS',joint_names=hand.joint_names,excited_joints=names,pulse_rad=pulse.cpu().tolist(),finite=bool(np.isfinite(raw).all()),tail_max_error_deg=np.rad2deg(error.max(0)).tolist(),tail_max_speed_rad_s=speed.max(0).tolist(),actual_finger_stiffness=act.stiffness.cpu().tolist(),actual_finger_damping=act.damping.cpu().tolist(),actual_finger_effort_limit=act.effort_limit.cpu().tolist(),actual_armature=hand.root_physx_view.get_dof_armatures().cpu().tolist(),trace_layout='step,hand,field(target/q/qd/computed_tau/applied_tau/physx_effort_target),joint',actual_mass_kg=hand.root_physx_view.get_masses().cpu().tolist())
    a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out.with_suffix('.npz'),trace=raw,initial_q=base.cpu().numpy());a.out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['finite','tail_max_error_deg','tail_max_speed_rad_s']}),flush=True)
finally:
    sim.clear_all_callbacks();sim.clear_instance();app.close()
