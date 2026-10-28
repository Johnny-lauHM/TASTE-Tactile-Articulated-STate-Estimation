"""Matched commanded hand posture, intervened layer angle, no angle-fed control.
Artificial reset states diagnose information availability; not execution success.
"""
import argparse,json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/diagnostics/sensory_probe');p.add_argument('--seed',type=int,default=1202);p.add_argument('--probe-rad',type=float,default=0.0);p.add_argument('--probe-force-limit',type=float,default=0.0);p.add_argument('--wrench-mode',choices=['legacy','cube-point'],default='legacy')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();out=Path(a.out)
if out.with_suffix('.json').exists() or out.with_suffix('.npz').exists():raise FileExistsError(out)
app=AppLauncher(a).app
import numpy as np
import torch
from rubik_lab.tasks.bimanual_twist.cycle_cfg import CycleUEnvCfg
from rubik_lab.tasks.bimanual_twist.cycle_env import PrimitiveCycleEnv
from rubik_lab.primitive_config import restore_checkpoint_settings
from rubik_lab.moves import MOVE_INDEX
cfg=CycleUEnvCfg();restore_checkpoint_settings(cfg,'logs/rsl_rl/rubik_primitive_cycle/U_blind_full/model.pt')
cfg.scene.num_envs=64;cfg.seed=a.seed;cfg.grasp_cache_path='cache/g70s_cycle_demo1_v3.npz';cfg.cycle_terminate_on_success=False;cfg.gravity_curriculum=False
if a.wrench_mode=="cube-point":cfg.tactile.contact_cfg.track_friction_forces=True
u=PrimitiveCycleEnv(cfg);u.record=True
if a.wrench_mode=="cube-point":u.tactile.raw_ft=u.tactile.raw_cube_wrench
histories=[];labels=[];validity=[];offsets=[]
try:
    for trial in range(8):
        u.forced_move_id=torch.full((64,),MOVE_INDEX['U'],device=u.device,dtype=torch.long)
        u.reset(seed=a.seed+trial)
        rng=np.random.default_rng(a.seed+trial)
        offset=np.tile([-10.,-7.,-4.,-1.,1.,4.,7.,10.],8);rng.shuffle(offset)
        angle=u.move_target[u.move_id]+torch.deg2rad(torch.as_tensor(offset,device=u.device,dtype=torch.float32))
        # This write occurs only at diagnostic initialization, never inside a skill.
        u.cube.write_joint_state_to_sim(angle[:,None],torch.zeros_like(angle[:,None]),joint_ids=[u.twist_id])
        ready=u._idle_offsets()[u.flick_cols[u.move_id]]/cfg.residual_finger_range
        active=torch.ones(64,dtype=torch.bool,device=u.device)
        sensory=[];truth=[];valid=[]
        probe_offset=torch.zeros(64,device=u.device)
        for step in range(32):
            probe_fraction=min(max((step-7)/16,0.0),1.0)
            flex=u.finger_flex["index"][u.flick_cols[u.move_id]]
            desired=a.probe_rad*probe_fraction
            if a.probe_force_limit>0:
                rows=torch.arange(64,device=u.device)
                measured=u.tactile.ft[rows,u.flick_hand[u.move_id],u.flick_pad[u.move_id],:3].norm(dim=-1)
                advance=torch.minimum(probe_offset+a.probe_rad/16,torch.full_like(probe_offset,desired))
                probe_offset=torch.where(measured>a.probe_force_limit*1.2,(probe_offset-0.005).clamp(min=0),torch.where(measured<a.probe_force_limit*0.8,advance,probe_offset))
                amount=probe_offset[:,None]
            else: amount=desired
            command=ready+amount*flex/cfg.residual_finger_range
            _,_,terminated,truncated,_=u.step(command.clamp(-1,1))
            s=u.snapshot
            q=s['finger_pos'].reshape(64,-1)
            qd=torch.cat([u.hands[side].data.joint_vel[:,u.finger_ids[side]] for side in ('left','right')],-1)
            ft=s['tactile_ft'].reshape(64,-1)
            sensory.append(torch.cat([q,qd,ft],-1).cpu().numpy())
            truth.append(torch.stack([torch.rad2deg(s['twist_error']),s['twist_velocity'],s['primitive_done'].float()],-1).cpu().numpy())
            active &= ~(terminated.bool()|truncated.bool())
            valid.append(active.cpu().numpy().copy())
        histories.append(np.stack(sensory,1));labels.append(np.stack(truth,1));validity.append(np.stack(valid,1));offsets.append(offset)
        print(f'[sensory] round{trial+1}/8 completed',flush=True)
    h=np.asarray(histories);y=np.asarray(labels);v=np.asarray(validity);offs=np.asarray(offsets)
    # Summaries of the last8 control samples; normalization uses fitting rounds only.
    def design(x):
        window=x[:,:,-8:,:]
        return np.concatenate([window.mean(2),window.std(2),window[:,:,-1]-window[:,:,0]],-1)
    good=v[:,:,-8:].all(-1)&np.isfinite(h).all(axis=(-1,-2))
    target=y[:,:,-1,0]
    metrics={}
    for name,cols in [('proprio',slice(0,88)),('ft',slice(88,148)),('combined',slice(0,148))]:
        x=design(h[:,:,:,cols])
        train=good[:4];test=good[4:]
        xx=x[:4][train];xt=x[4:][test];yy=target[:4][train];yt=target[4:][test]
        if len(xx)<10 or len(xt)<10:metrics[name]={'status':'insufficient_valid_histories'};continue
        mean=xx.mean(0);std=xx.std(0).clip(min=1e-5)
        xx=np.c_[(xx-mean)/std,np.ones(len(xx))];xt=np.c_[(xt-mean)/std,np.ones(len(xt))]
        penalty=np.eye(xx.shape[1]);penalty[-1,-1]=0
        beta=np.linalg.solve(xx.T@xx+penalty,xx.T@yy);pred=xt@beta
        true=np.abs(yt)<=5;guess=np.abs(pred)<=5
        balanced=np.mean([(guess[true]==True).mean(),(guess[~true]==False).mean()]) if true.any() and (~true).any() else None
        metrics[name]={'train_n':len(xx),'heldout_n':len(xt),'angle_MAE_deg':float(np.abs(pred-yt).mean()),'angle_p95_abs_error_deg':float(np.percentile(np.abs(pred-yt),95)),'within5deg_balanced_accuracy':None if balanced is None else float(balanced),'heldout_within5deg_count':int(true.sum())}
    result={'scope':'Diagnostic initial layer-angle intervention at matched commanded finger posture, no angle-fed actions. Not a policy or full-readiness acceptance test.','seed':a.seed,'probe_rad':a.probe_rad,'probe_force_limit_N':a.probe_force_limit,'wrench_mode':a.wrench_mode,'shape':list(h.shape),'feature_columns':{'q':[0,44],'qd':[44,88],'delivered_ft':[88,148]},'truth_columns':['angle_error_deg','layer_speed','full_ready'],'valid_episodes':int(good.sum()),'all_episodes':512,'heldout_rounds':[4,5,6,7],'readout':'ridge penalty1, fixed last8samples mean/std/delta; train-only channel normalization','metrics':metrics,'limitations':['Only artificial ready-state distribution and selected canonical grasp.','Good angle readout does not establish full physical completion detection.','Poor linear readout does not prove information-theoretic unobservability.','Histories with any physical termination before or at endpoint are excluded and their count reported; raw validity masks retained.']}
    # Explicitly invalidate terminal endpoints as well, rather than rely on reset-frame qd.
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'),sensory=h,truth=y,valid=v,initial_offsets_deg=offs)
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except BaseException:
    import traceback;traceback.print_exc();raise
finally:
    u.close();app.close()
