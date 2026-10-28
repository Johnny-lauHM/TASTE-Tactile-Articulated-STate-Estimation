"""Diagnostic only: compare actual pad contacts against the TacMap sensing surface."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--cache',required=True)
p.add_argument('--out',required=True)
p.add_argument('--num_envs',type=int,default=4)
p.add_argument('--steps',type=int,default=64)
p.add_argument('--seed',type=int,default=1102)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();app=AppLauncher(a).app
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import isaaclab.utils.math as mu
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import rubik_lab
from rubik_lab.blind_actor import BlindInference
from rubik_lab.primitive_config import restore_checkpoint_settings
from rubik_lab.wave_se_tactile import MAP_DIR
out=Path(a.out)
if out.exists():raise FileExistsError(out)
cfg=load_cfg_from_registry('Rubik-Cycle-U-Sharpa-v0','env_cfg_entry_point')
restore_checkpoint_settings(cfg,a.checkpoint)
cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.grasp_cache_path=a.cache
cfg.tactile.image=True;cfg.cycle_terminate_on_success=False;cfg.gravity_curriculum=False
env=gym.make('Rubik-Cycle-U-Sharpa-v0',cfg=cfg);u=env.unwrapped;u.record=True
policy=BlindInference(torch.load(a.checkpoint,weights_only=False,map_location='cpu'),u.num_envs,u.device)
obs,_=env.reset();rows=[]
maps={}
for kind,suffix in [('TH','TH'),('4F','4F')]:
    ending='' if kind=='TH' else '_origin'
    points=torch.tensor(np.load(Path(MAP_DIR)/('tactileSensor_map_'+suffix+'_point'+ending+'.npy'))[::4,::4].reshape(-1,3)*.001,device=u.device,dtype=torch.float32)
    normals=torch.tensor(np.load(Path(MAP_DIR)/('tactileSensor_map_'+suffix+'_normal'+ending+'.npy'))[::4,::4].reshape(-1,3),device=u.device,dtype=torch.float32)
    maps[kind]=(points,torch.nn.functional.normalize(normals,dim=-1))
mesh_diagnostics=[]
with torch.inference_mode():
    for step in range(a.steps):
        obs,_,terminated,truncated,_=env.step(policy(obs))
        ended=terminated|truncated
        policy.reset(ended.nonzero().flatten())
        # Both fields below query current post-step sensor geometry, bypassing
        # delivered sample delay for diagnosis only. Actor input is unchanged.
        images=u.tactile.raw_image()
        if True:
            from isaaclab.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster
            from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
            image_sensor=u.tactile.images["left"][0]
            for mi,(view,target_cfg) in enumerate(zip(image_sensor._mesh_views,image_sensor._raycast_targets_cfg)):
                body_name="core" if "/core/" in target_cfg.prim_expr else "top"
                body_id=u.cube.find_bodies(body_name)[0][0]
                position=u.cube.data.body_pos_w[:,body_id]
                orientation=u.cube.data.body_quat_w[:,body_id]
                dp,dq=MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                expected_p,expected_q=mu.combine_frame_transforms(position,orientation,dp,dq)
                actual_p=image_sensor._mesh_positions_w[:,mi]
                actual_q=image_sensor._mesh_orientations_w[:,mi]
                mesh_diagnostics.append({"step":step,"target":target_cfg.prim_expr,"local_offset_m":dp[0].cpu().tolist(),
                    "position_error_mm":((actual_p-expected_p).norm(dim=-1)*1000).cpu().tolist(),
                    "orientation_error_deg":torch.rad2deg(mu.quat_error_magnitude(actual_q,expected_q)).cpu().tolist(),
                    "actual_mesh_position_m":actual_p[0].cpu().tolist(),"expected_mesh_position_m":expected_p[0].cpu().tolist()})
        for h,side in enumerate(('left','right')):
            hand=u.hands[side].data
            for finger,sensor in enumerate(u.tactile.contact[side]):
                body=u.tactile.pad_body_ids[side][finger]
                position=hand.body_pos_w[:,body];quat=hand.body_quat_w[:,body]
                contact=sensor.data.contact_pos_w[:,0]
                force=sensor.data.force_matrix_w[:,0].norm(dim=-1)
                local=mu.quat_apply_inverse(quat[:,None].expand(-1,2,-1).reshape(-1,4),
                    (contact-position[:,None]).reshape(-1,3)).reshape(-1,2,3)
                valid=(force>.2)&torch.isfinite(local).all(-1)&~ended[:,None]
                ids,target=valid.nonzero(as_tuple=True)
                if not len(ids):continue
                points,normals=maps['TH' if finger==0 else '4F']
                query=local[ids,target]
                distance,nearest=torch.cdist(query,points).min(-1)
                signed=((query-points[nearest])*normals[nearest]).sum(-1)
                maximum=images[ids,h,finger].flatten(1).max(-1).values
                values=torch.stack([ids.float(),target.float(),force[ids,target],distance*1000,signed*1000,maximum.float()],-1).cpu().tolist()
                for value in values:rows.append({'step':step,'hand':side,'finger':finger,'env':int(value[0]),'cube_body':int(value[1]),'force_N':value[2],'nearest_surface_mm':value[3],'signed_normal_mm':value[4],'image_max':int(value[5])})
        if (step+1)%20==0:print('[surface_diagnostic]',step+1,'rows',len(rows),flush=True)
summary={}
for side in ('left','right'):
    for finger in range(5):
        selected=[r for r in rows if r['hand']==side and r['finger']==finger]
        if not selected:continue
        summary[side+'_'+str(finger)]={'contacts':len(selected),'zero_image_fraction':float(np.mean([r['image_max']==0 for r in selected])),
            'nearest_surface_mm_p50_p90':np.percentile([r['nearest_surface_mm'] for r in selected],[50,90]).tolist(),
            'signed_normal_mm_p50_p90':np.percentile([r['signed_normal_mm'] for r in selected],[50,90]).tolist()}
report={'scope':'Current contact geometry diagnostic, not an autonomous success test. No sensing changes used by actor.',
        'args':vars(a),'mesh_pose_error_mm_max':max(max(row['position_error_mm']) for row in mesh_diagnostics),
        'mesh_pose_angle_error_deg_max':max(max(row['orientation_error_deg']) for row in mesh_diagnostics),
        'mesh_diagnostics':mesh_diagnostics,'summary':summary,'rows':rows,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
print('[surface_diagnostic] finished',out,json.dumps(summary),flush=True)
env.close();app.close()
