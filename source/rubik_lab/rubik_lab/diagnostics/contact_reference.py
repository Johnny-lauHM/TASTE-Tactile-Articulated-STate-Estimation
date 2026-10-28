"""Zero-damping rigid boxes on a static plane: contact measurement oracle, no skills."""
import argparse,json,math
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--physics-hz',type=int,default=240)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
if a.out.with_suffix('.json').exists() or a.out.with_suffix('.npz').exists():raise FileExistsError(a.out)
app=AppLauncher(a).app
import numpy as np
import torch
from pxr import Gf,UsdGeom,UsdPhysics,PhysxSchema
from isaaclab.assets import RigidObject,RigidObjectCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.sim import SimulationContext,SimulationCfg,PhysxCfg,RigidBodyMaterialCfg
from isaaclab.sim.utils import bind_physics_material
from rubik_lab.contact_wrench import point_contact_wrench
dt=1/a.physics_hz
sim=SimulationContext(SimulationCfg(dt=dt,device=a.device,physx=PhysxCfg(enable_external_forces_every_iteration=True)))
mat=RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.8,restitution=0.)
mat.func('/World/material',mat)
ground=UsdGeom.Cube.Define(sim.stage,'/World/ground');ground.CreateSizeAttr(1.)
ground.AddTranslateOp().Set(Gf.Vec3d(0,0,-.01));ground.AddScaleOp().Set(Gf.Vec3f(2,2,.02))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim());bind_physics_material('/World/ground','/World/material')
for i in range(2):
    path=f'/World/box_{i}';box=UsdGeom.Cube.Define(sim.stage,path);box.CreateSizeAttr(.07)
    box.AddTranslateOp().Set(Gf.Vec3d(i*.4,0,.06))
    box.AddOrientOp().Set(Gf.Quatf(math.cos(math.pi/24),0,math.sin(math.pi/24),0))
    UsdPhysics.RigidBodyAPI.Apply(box.GetPrim());UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    mass=UsdPhysics.MassAPI.Apply(box.GetPrim());mass.CreateMassAttr(.148);mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.148*.07**2/6))
    phys=PhysxSchema.PhysxRigidBodyAPI.Apply(box.GetPrim());phys.CreateLinearDampingAttr(0.);phys.CreateAngularDampingAttr(0.)
    phys.CreateSolverPositionIterationCountAttr(8);phys.CreateSolverVelocityIterationCountAttr(0)
    PhysxSchema.PhysxContactReportAPI.Apply(box.GetPrim()).CreateThresholdAttr(0.)
    collision=PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim());collision.CreateContactOffsetAttr(.002);collision.CreateRestOffsetAttr(0.)
    bind_physics_material(path,'/World/material')
asset=RigidObject(RigidObjectCfg(prim_path='/World/box_.*'))
sensor=ContactSensor(ContactSensorCfg(prim_path='/World/box_.*',filter_prim_paths_expr=['/World/ground'],max_contact_data_count_per_prim=64))
sim.reset();asset.reset();sensor.reset()
vel=torch.tensor([[.25,.05,0.,0.,1.,.5],[-.15,.1,0.,.5,-.8,1.]],device=sim.device)
asset.write_root_velocity_to_sim(vel)
states=[asset.data.root_state_w[:,None].cpu().numpy().copy()];wrenches=[np.zeros((2,1,2,6),np.float32)]
try:
    for tick in range(a.physics_hz):
        sim.step(render=False);asset.update(dt);sensor.update(dt,force_recompute=True)
        view=sensor.contact_physx_view;origin=asset.data.root_pos_w[:,None,:]
        mag,pts,norm,_,counts,starts=view.get_contact_data(dt=dt)
        normal=point_contact_wrench(mag.reshape(-1,1)*norm,pts,counts,starts,origin).sum(-2)
        force,pts,counts,starts=view.get_friction_data(dt=dt)
        friction=point_contact_wrench(force,pts,counts,starts,origin).sum(-2)
        states.append(asset.data.root_state_w[:,None].cpu().numpy().copy())
        wrenches.append(torch.stack([normal,friction],1)[:,None].cpu().numpy())
    raw=np.asarray(states);N=len(raw);view=asset.root_physx_view
    result={'physics_fixture':True,'scope':'Two independent70mm148g rigid boxes slide/tumble onstaticplane; zero damping, no joints/hands, not cube skill acceptance.','configuration_sources':{'physics_dt':dt},'substep_schema':{'body_names':['box'],'reference_body':'box','state':'world actor pose+COM velocities; samples aftereachphysicsstep, initialstateincluded'},'cube_runtime_mass_properties':{'mass_kg':view.get_masses().cpu().numpy().reshape(2,1).tolist(),'inertia':view.get_inertias().cpu().numpy().reshape(2,1,9).tolist(),'com_local_pose':view.get_coms().cpu().numpy().reshape(2,1,7).tolist()},'physics_hz':a.physics_hz,'gravity':[0,0,-9.81],'damping':0,'solver':'TGS','external_forces_every_iteration':True,'finite':bool(np.isfinite(raw).all()),'initial_velocity':vel.cpu().tolist()}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out.with_suffix('.npz'),trace=[],states=[],planner_state=[],contacts=[],substep_index=np.stack([np.zeros(N,dtype=int),np.arange(N)],-1),substep_active=np.ones((N,2),bool),substep_body_state=raw,substep_wrenches=np.asarray(wrenches))
    a.out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'finite':result['finite'],'frames':N}),flush=True)
finally:
    sim.clear_all_callbacks();sim.clear_instance();app.close()
