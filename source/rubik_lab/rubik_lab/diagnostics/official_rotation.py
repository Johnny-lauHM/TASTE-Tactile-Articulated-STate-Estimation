"""Run official rotation project's zero-action grasp baseline without its file cleanup."""
import argparse,json,sys
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/diagnostics/rotation_grasp_reference');p.add_argument('--reference',default='/home/yuhang/workspace/third_party/sharpa-rl-lab');p.add_argument('--seed',type=int,default=1220)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();out=Path(a.out)
if out.with_suffix('.json').exists() or out.with_suffix('.npz').exists():raise FileExistsError(out)
app=AppLauncher(a).app
import numpy as np
import torch
sys.path.insert(0,a.reference)
from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_grasp_env_cfg import SharpaWaveEnvCfg
from rl_isaaclab.tasks.inhand_rotate.sharpa_wave_grasp_env import SharpaWaveInhandRotateGraspEnv
class RecordedReference(SharpaWaveInhandRotateGraspEnv):
    def _get_rewards(self):
        result=super()._get_rewards()
        force=torch.cat([self._contact_sensor[i].data.force_matrix_w[:,0,0,:].unsqueeze(1) for i in range(10)],1)
        self.pre_reset_trace=torch.cat([self.object_pos,self.object_rot,self.hand_dof_pos,force.flatten(1),self.episode_length_buf[:,None]],-1).clone()
        return result
cfg=SharpaWaveEnvCfg();cfg.scene.num_envs=8;cfg.seed=a.seed
u=RecordedReference(cfg);traces=[];dones=[];gravity=[]
try:
    u.reset(seed=a.seed)
    for step in range(240):
        with torch.no_grad():obs,_,term,trunc,_=u.step(torch.zeros(8,22,device=u.device))
        traces.append(u.pre_reset_trace.cpu().numpy());dones.append(torch.stack([term,trunc],-1).cpu().numpy());gravity.append(list(u.physics_sim_view.get_gravity()))
        if (step+1)%40==0:print(f'[rotation-reference] step{step+1}/240',flush=True)
    t=np.asarray(traces);d=np.asarray(dones);g=np.asarray(gravity)
    force=np.linalg.norm(t[:,:,29:59].reshape(240,8,10,3),axis=-1)
    result={'scope':'Official rotation project grasp-generation baseline, zero actions and native six-direction gravity schedule. No trained rotation policy or cube skill claimed.','source':a.reference,'seed':a.seed,'envs':8,'steps':240,'control_dt':u.step_dt,'changes':'Only environment count,seed,pre-reset recording and bounded runtime. Native asset,physics,controller andreset behavior retained. gen_grasp.py destructive outputs cleanup is not called.','finite':bool(np.isfinite(t).all()),'contact_force_peak_N':float(force.max()),'frames_at_least3contacts_above0_5N':int((np.sum(force>0.5,axis=-1)>=3).sum()),'termination_events':int(d[:,:,0].sum()),'timeout_events':int(d[:,:,1].sum()),'saved_grasps':sum(len(x) for x in u.saved_grasping_states),'trace_columns':{'object_xyz_quaternion':[0,7],'joint_positions':[7,29],'ten_contact_force_vectors':[29,59],'episode_step':[59,60]},'limitations':['Single right hand holding a cylinder, not bimanual layer rotation.','Official grasp condition mutates reset_buf during reward; effective termination order should be audited before reusing its cache acceptance logic.','No physical calibration or grasp quality acceptance claim.']}
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),trace=t,dones=d,gravity=g)
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except BaseException:
    import traceback;traceback.print_exc();raise
finally:u.close();app.close()
