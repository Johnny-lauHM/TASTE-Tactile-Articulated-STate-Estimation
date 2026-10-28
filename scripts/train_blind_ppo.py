"""Refine a sensor-only GRU using recurrent PPO and successful-demo rehearsal.
Only the critic and simulation reward receive privileged cube state.
"""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--task',default='Rubik-Cycle-U-Sharpa-v0')
p.add_argument('--cache',required=True)
p.add_argument('--out',required=True)
p.add_argument('--demo_data')
p.add_argument('--prefix-checkpoint')
p.add_argument('--prefix-move',choices=['U',"U'"],default='U')
p.add_argument('--prefix-hold-steps',type=int,default=4)
p.add_argument('--dry-run',action='store_true')
p.add_argument('--demo_grasp',type=int,default=None)
p.add_argument('--num_envs',type=int,default=256)
p.add_argument('--iterations',type=int,default=500)
p.add_argument('--horizon',type=int,default=32)
p.add_argument('--epochs',type=int,default=2)
p.add_argument('--batch_envs',type=int,default=64)
p.add_argument('--learning_rate',type=float,default=2e-6)
p.add_argument('--noise_std',type=float,default=.03)
p.add_argument('--demo_weight',type=float,default=5.)
p.add_argument('--pad_progress_reward',type=float,default=0.)
p.add_argument('--seed',type=int,default=1071)
p.add_argument('--save_interval',type=int,default=25)
p.add_argument('--initial_settle_steps',type=int,default=None,help='override initialization ready-hold; physical time and motion count, actor history cleared during hold')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.pad_progress_reward<0:p.error("pad_progress_reward must be nonnegative")
if a.initial_settle_steps is not None and a.initial_settle_steps<0:p.error("initial_settle_steps must be nonnegative")
if not a.dry_run and not a.demo_data:p.error('Training requires demo_data')
if a.prefix_hold_steps<0:p.error('Negative prefix hold')
app=AppLauncher(a).app
import hashlib,json,signal,time
from pathlib import Path
import gymnasium as gym
import h5py,numpy as np,torch
from torch import nn
from torch.distributions import Normal
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import rubik_lab
from rubik_lab.blind_actor import BlindPrimitiveActor,student_features
from rubik_lab.primitive_config import restore_checkpoint_settings,validate_sequence_policy_settings
from rubik_lab.recurrent_training import generalized_advantage,recurrent_actions,continuation_advantage
from rubik_lab.continuous_prefix import advance_preserving_state
from rubik_lab.blind_actor import BlindInference
from rubik_lab.primitive_execution import apply_ready_hold

torch.set_num_threads(4)
torch.manual_seed(a.seed)
rng=np.random.default_rng(a.seed)
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
cfg=load_cfg_from_registry(a.task,'env_cfg_entry_point')
restore_checkpoint_settings(cfg,a.checkpoint)
cfg.grasp_cache_path=a.cache
cfg.scene.num_envs=a.num_envs
cfg.seed=a.seed
cfg.gravity_curriculum=False
cfg.cycle_terminate_on_success=True
cfg.rew_pad_progress=a.pad_progress_reward
(out/'args.json').write_text(json.dumps(vars(a),indent=2))
if not a.dry_run:
    root=Path(__file__).resolve().parents[1]
    archive=out/'sources';archive.mkdir()
    manifest={}
    for path in [Path(__file__),*sorted((root/'source/rubik_lab/rubik_lab').rglob('*.py'))]:
        content=path.read_bytes();name=str(path.relative_to(root))
        destination=archive/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(content)
        manifest[name]=hashlib.sha256(content).hexdigest()
    manifest['numerics']={'torch':torch.__version__,'cuda':torch.version.cuda,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
    manifest['initial_checkpoint']=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    manifest['demonstrations']=hashlib.sha256(Path(a.demo_data).read_bytes()).hexdigest()
    manifest['grasp_cache']=hashlib.sha256(Path(a.cache).read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if a.prefix_checkpoint:
    from rubik_lab.tasks.bimanual_twist.cycle_env import ContinuousPrefixTrainingEnv
    if len(cfg.moves)!=1:raise ValueError('Prefix training requires a single target move')
    training_move=cfg.moves[0]
    validate_sequence_policy_settings(cfg,a.prefix_checkpoint)
    env=ContinuousPrefixTrainingEnv(cfg,prefix_move=a.prefix_move);u=env
else:
    env=gym.make(a.task,cfg=cfg);u=env.unwrapped
u.record=True
metadata=cfg.to_dict()
if a.prefix_checkpoint:metadata['episode_length_s']=u.primitive_seconds
(out/'env.json').write_text(json.dumps(metadata,indent=2,default=str))
dev=u.device
payload=torch.load(a.checkpoint,map_location=dev,weights_only=False)
if payload.get('format')!='blind_primitive_gru_v1':raise ValueError('Expected sensor GRU warm start')
execution_protocol=dict(payload.get('execution_protocol',{}))
if a.prefix_checkpoint:execution_protocol['training_prefix']={'checkpoint':a.prefix_checkpoint,'move':a.prefix_move,'hold_steps':a.prefix_hold_steps,'handoff':'privileged_full_ready','physical_reset_at_handoff':False,'primitive_seconds':u.primitive_seconds}
if a.initial_settle_steps is not None:
    execution_protocol['initial_settle_steps']=a.initial_settle_steps
initial_settle_steps=int(execution_protocol.get('initial_settle_steps',0))
if initial_settle_steps<0:raise ValueError('Invalid initialization hold')
actor=BlindPrimitiveActor(payload['hidden']).to(dev);actor.load_state_dict(payload['model'])
critic=nn.Sequential(nn.Linear(250,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1)).to(dev)
actor_optimizer=torch.optim.Adam(actor.parameters(),lr=a.learning_rate)
critic_optimizer=torch.optim.Adam(critic.parameters(),lr=3e-4)
demos=[]
if not a.dry_run:
    with h5py.File(a.demo_data,'r') as f:
        uid,gid,step=f['episode_uid'][:],f['grasp_idx'][:],f['step'][:]
        good=np.unique(uid[f['primitive_done'][:].astype(bool)])
        for key in ('dropped','broken','cycle_failed'):
            if key in f:good=np.setdiff1d(good,uid[f[key][:].astype(bool)])
        if a.demo_grasp is not None:good=good[np.isin(good,uid[gid==a.demo_grasp])]
        features=student_features(torch.from_numpy(f['policy_obs'][:]))
        actions=torch.from_numpy(f['policy_action'][:])
    demos=[]
    for episode in good:
        indices=np.flatnonzero(uid==episode);indices=indices[np.argsort(step[indices])]
        if not np.array_equal(step[indices],np.arange(len(indices))):raise ValueError('Demonstrations require full histories')
        demos.append((features[indices].to(dev),actions[indices].to(dev)))
    if not demos:raise ValueError('No successful demonstrations in requested grasp scope')
print('[blind_ppo] demonstrations',len(demos),'actor sensor-only; critic privileged',flush=True)

def rehearsal():
    selected=[demos[i] for i in rng.integers(len(demos),size=min(8,len(demos)))]
    lengths=torch.tensor([len(x) for x,y in selected],device=dev)
    x=nn.utils.rnn.pad_sequence([x for x,y in selected],batch_first=True)
    target=nn.utils.rnn.pad_sequence([y for x,y in selected],batch_first=True)
    predicted,_=actor(x)
    mask=torch.arange(x.shape[1],device=dev)[None]<lengths[:,None]
    return (((predicted-target).square().mean(-1)*mask).sum(-1)/lengths).mean()

iteration=-1
def save(name):
    if a.dry_run:return
    torch.save({'format':'blind_primitive_gru_v1','hidden':actor.hidden,'model':actor.state_dict(),
        'feature_layout':'cycle_250_to_sensor_173_v1','iteration':iteration,
        'critic':critic.state_dict(),'actor_optimizer':actor_optimizer.state_dict(),
        'critic_optimizer':critic_optimizer.state_dict(),'noise_std':a.noise_std,'execution_protocol':execution_protocol},out/name)
def interrupt(signum,frame):raise KeyboardInterrupt
signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
prefix_actor=BlindInference(torch.load(a.prefix_checkpoint,map_location='cpu',weights_only=False),a.num_envs,dev) if a.prefix_checkpoint else None
dry_weights={k:v.detach().clone() for k,v in actor.state_dict().items()} if a.dry_run else None
dry_critic={k:v.detach().clone() for k,v in critic.state_dict().items()} if a.dry_run else None
dry_rollouts=[];dry_events=[];handoff_count=0;prefix_failures=0
obs,_=env.reset()
hidden=torch.zeros(1,a.num_envs,actor.hidden,device=dev)
reset_before=torch.ones(a.num_envs,dtype=torch.bool,device=dev)
started=time.time()
save('model_initial.pt')
try:
    for iteration in range(a.iterations):
        start_hidden=hidden.detach().clone()
        storage={key:[] for key in ('obs','features','action','log_prob','value','reward','done','reset','executed')}
        completed=successful=0
        with torch.no_grad():
            for t in range(a.horizon):
                privileged=obs['policy']
                sensory=student_features(privileged)
                mean,hidden=actor(sensory[:,None],hidden)
                distribution=Normal(mean[:,0],a.noise_std)
                action=mean[:,0] if a.dry_run else distribution.sample()
                value=critic(privileged).squeeze(-1)
                learning_phase=(u.cycle_sequence_index>0).clone() if prefix_actor else torch.ones(a.num_envs,dtype=torch.bool,device=dev)
                settling=(~learning_phase)&(u.episode_length_buf<a.prefix_hold_steps) if prefix_actor else u.episode_length_buf<initial_settle_steps
                command=action.clamp(-1,1)
                if prefix_actor is not None:
                    prefix_command=prefix_actor(obs)
                    ready=u._idle_offsets()[u.flick_cols[u.move_id]]/cfg.residual_finger_range
                    prefix_command=apply_ready_hold(prefix_command,ready,settling,[prefix_actor])
                    command=torch.where(learning_phase[:,None],command,prefix_command)
                elif initial_settle_steps:
                    ready=u._idle_offsets()[u.flick_cols[u.move_id]]/cfg.residual_finger_range
                    command=torch.where(settling[:,None],ready,command)
                next_obs,reward,terminated,truncated,extras=env.step(command)
                done=terminated|truncated
                if prefix_actor is not None and (u.snapshot['cycle_step']>u.primitive_step_limit).any():raise RuntimeError('Primitive deadline exceeded')
                handed_off=torch.zeros_like(done)
                if prefix_actor is not None:
                    handed_off=(~learning_phase)&u.snapshot['primitive_done']&~done
                    ids=handed_off.nonzero().flatten()
                    if len(ids):
                        advance_preserving_state(u,ids,training_move);handoff_count+=len(ids)
                        next_obs=u._get_observations()
                    prefix_failures+=int(((~learning_phase)&done).sum())
                    prefix_actor.reset(done.nonzero().flatten())
                if a.dry_run:
                    dry_events.append(torch.stack([learning_phase,handed_off,done,u.snapshot['cycle_step'],u.snapshot['primitive_done'],u.snapshot['twist_error'],u.snapshot['cycle_failed']],-1).cpu().numpy())
                successful+=int((u.snapshot['primitive_done']&done&learning_phase).sum())
                completed+=int((done&learning_phase).sum())
                values=(privileged,sensory,action,distribution.log_prob(action).sum(-1),value,reward*.1,done,reset_before,learning_phase&~settling)
                for key,val in zip(storage,values):storage[key].append(val.detach().clone())
                hidden[:,done|settling|handed_off|~learning_phase]=0
                reset_before=(done|settling|handed_off|~learning_phase).clone()
                obs=next_obs
            last_value=critic(obs['policy']).squeeze(-1)
        data={key:torch.stack(value) for key,value in storage.items()}
        if prefix_actor is not None:
            advantage,returns=continuation_advantage(data['reward'],data['value'],data['done'],last_value,data['executed'],u.cycle_sequence_index>0)
            if (returns[~data['executed']]!=0).any():raise RuntimeError('Prefix return leakage')
        else:advantage,returns=generalized_advantage(data['reward'],data['value'],data['done'],last_value)
        if a.dry_run:dry_rollouts.append({k:v.detach().cpu().numpy() for k,v in data.items()})
        executed_advantage=advantage[data['executed']]
        if len(executed_advantage)>1:
            advantage=(advantage-executed_advantage.mean())/(executed_advantage.std()+1e-8)
        metrics=[]
        for epoch in range(1 if a.dry_run else a.epochs):
            for indices in torch.randperm(a.num_envs,device=dev).split(a.batch_envs):
                predicted,_=recurrent_actions(actor,data['features'][:,indices],start_hidden[:,indices],data['reset'][:,indices])
                distribution=Normal(predicted,a.noise_std)
                log_prob=distribution.log_prob(data['action'][:,indices]).sum(-1)
                log_ratio=log_prob-data['log_prob'][:,indices]
                ratio=log_ratio.exp()
                executed=data['executed'][:,indices].float()
                denominator=executed.sum().clamp(min=1)
                policy_loss=-(torch.minimum(ratio*advantage[:,indices],ratio.clamp(.8,1.2)*advantage[:,indices])*executed).sum()/denominator
                demo_loss=torch.zeros((),device=dev) if a.dry_run else rehearsal()
                actor_loss=policy_loss+a.demo_weight*demo_loss
                approximate_kl=((((ratio-1)-log_ratio)*executed).sum()/denominator).detach()
                if epoch==0 and len(metrics)==0 and float(log_ratio.abs().max())>.01:
                    raise RuntimeError('Recurrent replay differs from rollout before first update')
                if not torch.isfinite(actor_loss):raise RuntimeError('Non-finite actor loss')
                # Avoid drifting far from the sampled rollout during repeated updates.
                if not a.dry_run and data['executed'][:,indices].any() and float(approximate_kl)<.02:
                    actor_optimizer.zero_grad();actor_loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1.);actor_optimizer.step()
                prediction=critic(data['obs'][:,indices]).squeeze(-1)
                value_error=(prediction-returns[:,indices]).square()
                value_loss=(value_error*executed).sum()/denominator if prefix_actor else value_error.mean()
                if not a.dry_run and (prefix_actor is None or data['executed'][:,indices].any()):
                    critic_optimizer.zero_grad();value_loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.);critic_optimizer.step()
                metrics.append([float(policy_loss.detach()),float(demo_loss.detach()),float(value_loss.detach()),float(approximate_kl)])
        # Hidden is the rollout policy's state; standard truncated-BPTT boundary.
        hidden=hidden.detach()
        row={'iteration':iteration,'steps':(iteration+1)*a.horizon*a.num_envs,
            'executed_turn_samples':int(data['executed'].sum()),'prefix_handoffs_total':handoff_count,'prefix_failures_total':prefix_failures,'dry_run':a.dry_run,'initial_settle_steps':initial_settle_steps,
            'completed':completed,'successes':successful,'success_rate':successful/completed if completed else None,
            'reward_mean':float(data['reward'].mean())*10,'policy_loss':float(np.mean(metrics,axis=0)[0]),
            'demo_mse':float(np.mean(metrics,axis=0)[1]),'critic_mse':float(np.mean(metrics,axis=0)[2]),
            'approx_kl':float(np.mean(metrics,axis=0)[3]),'wall_seconds':time.time()-started}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('[blind_ppo]',json.dumps(row),flush=True)
        if iteration%a.save_interval==0:save('model_'+str(iteration)+'.pt')
    if a.dry_run:
        if any(not torch.equal(v,dry_weights[k]) for k,v in actor.state_dict().items()):raise RuntimeError('Dry run changed actor weights')
        if any(not torch.equal(v,dry_critic[k]) for k,v in critic.state_dict().items()):raise RuntimeError('Dry run changed critic weights')
        np.savez_compressed(out/'dry_rollout.npz',events=np.asarray(dry_events),**{k:np.concatenate([r[k] for r in dry_rollouts],axis=0) for k in dry_rollouts[0]})
        (out/'dry_summary.json').write_text(json.dumps({'weights_unchanged':True,'critic_weights_unchanged':True,'event_columns':['training_phase','handoff','done','cycle_step','primitive_done','angle_error_rad','cycle_failed'],'optimizer_steps':0,'handoffs':handoff_count,'prefix_failures':prefix_failures,'rollout_steps':a.iterations*a.horizon,'scope':'Training plumbing only; not acceptance.'},indent=2)+'\n')
    save('model_final.pt')
    print('[blind_ppo] finished',out,flush=True)
except KeyboardInterrupt:
    save('model_interrupted.pt')
    print('[blind_ppo] interrupted checkpoint saved',flush=True)
finally:
    env.close();app.close()
