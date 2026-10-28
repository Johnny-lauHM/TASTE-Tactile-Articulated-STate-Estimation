"""Unselected, fixed-quota evaluation of legacy or complete-cycle primitives.

Examples:
  python scripts/eval_primitive.py --task Rubik-Cycle-U-Sharpa-v0 --checkpoint PATH --headless
  python scripts/eval_primitive.py --task Rubik-Cycle-U-Sharpa-v0 --checkpoint U.pt --checkpoint_up Up.pt --sequence "U,U'" --headless
Every env contributes exactly --episodes trials. Sequence transitions do not
reset physics. Default triggers use privileged acceptance labels. --detector uses
sensor predictions; ground truth then scores and aborts premature requests.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", default="Rubik-Cycle-U-Sharpa-v0")
p.add_argument("--checkpoint")
p.add_argument("--checkpoint_up")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--episodes", type=int, default=4)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default="outputs/primitive_cycle/evaluation.json")
p.add_argument("--sequence", default=None)
p.add_argument("--stochastic", action="store_true")
p.add_argument("--synchronize_each_primitive", action="store_true", help="synchronize the first commanded action of every primitive, preserving physics during handoff wait")
p.add_argument("--synchronize_tactile_clock", action="store_true", help="start each trial turn at a sensor capture-clock boundary; physical wait counts toward deadline")
p.add_argument("--command_slew_rad_s",type=float,default=None,help="optional maximum finger-command speed, rad/s; changes controller, not acceptance")
p.add_argument("--initial_settle_steps", type=int, default=0, help="real ready-hold steps after each trial reset, included in timeout and motion scoring")
p.add_argument("--probe_return_moves", default=None, help="restrict scripted return to comma-separated moves in a mixed diagnostic sequence")
p.add_argument("--probe_offsets", default=None, help="diagnostic comma-separated turn overshoot offsets in degrees before scripted return")
p.add_argument("--probe_abduct_return", action="store_true", help="privileged teacher diagnostic: side-release, extend, then return abduction")
p.add_argument("--abduction_offsets", default="-0.3,-0.15,0,0.15,0.3")
p.add_argument("--probe_return", action="store_true", help="diagnostic only: use privileged latch to command finger return")
p.add_argument("--probe_lift_return", action="store_true", help="diagnostic IK lift-first return; privileged state")
p.add_argument("--smoke_steps", type=int, default=0)
p.add_argument("--notch", type=float, default=None, help="override checkpoint setting; legacy default is zero")
p.add_argument("--cache")
p.add_argument("--detector", default=None, help="streaming tactile readiness checkpoint; gate transitions using predictions")
p.add_argument("--detector_up",default=None,help="optional independent Up observer; per-primitive histories only")
p.add_argument("--detector_up_threshold",type=float,default=0.95)
p.add_argument("--detector_threshold", type=float, default=0.95)
p.add_argument("--detector_hold_steps", type=int, default=None)
p.add_argument("--terminal_cache_stage",type=int,default=None,help="sequence stage whose successful end states are saved for training resets; defaults to last")
p.add_argument("--terminal_cache", default=None, help="collect successful end states for next-primitive training initialization")
p.add_argument("--video", default=None, help="unselected first trial, env 0, real time")
p.add_argument("--dataset", default=None, help="optional tactile and teacher-action HDF5 from these unselected trials")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.terminal_cache_stage is not None and not a.terminal_cache:
    p.error("terminal_cache_stage requires terminal_cache")
if a.detector_up and (not a.detector or not a.sequence):
    p.error("detector_up requires primary detector and a sequence")
if a.synchronize_each_primitive:
    a.synchronize_tactile_clock = True
if a.initial_settle_steps < 0:
    p.error("initial_settle_steps must be nonnegative")
if a.num_envs < 1 or a.episodes < 1:
    p.error("num_envs and episodes must be positive")
if a.video:
    if a.num_envs > 16:
        p.error("Video is limited to <=16 environments; use a separate statistics run")
    a.enable_cameras = True
app = AppLauncher(a).app

import hashlib
import io
import json
import math
from pathlib import Path
import time
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import rubik_lab
from rubik_lab.moves import ALL_MOVES, MOVE_INDEX
from rubik_lab.evaluation_metrics import summarize_stages
from rubik_lab.primitive_execution import apply_ready_hold

# Capture source provenance before the rollout; files may change while long jobs run.
project_root = Path(__file__).resolve().parents[1]
source_paths = [Path(__file__), *sorted((project_root/'source/rubik_lab/rubik_lab').rglob('*.py'))]
source_bytes = {str(path.relative_to(project_root)): path.read_bytes() for path in source_paths}
source_hashes = {name:hashlib.sha256(content).hexdigest() for name,content in source_bytes.items()}
for destination in (a.out,a.dataset,a.video,a.terminal_cache):
    if destination and Path(destination).exists():
        raise FileExistsError('Refusing to overwrite evaluation artifact: '+destination)
source_archive = Path(a.out).with_suffix('') / 'sources'
source_archive.mkdir(parents=True,exist_ok=False)
for name,content in source_bytes.items():
    destination=source_archive/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(content)

cfg = load_cfg_from_registry(a.task, "env_cfg_entry_point")
agent = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
from rubik_lab.primitive_config import restore_checkpoint_settings, validate_sequence_policy_settings
restored_config = restore_checkpoint_settings(cfg, a.checkpoint)
cfg.scene.num_envs, cfg.seed = a.num_envs, a.seed
cfg.gravity_curriculum = False
# Standalone evaluations share termination semantics regardless of training wrapper.
# Sequence, detector and cache modes explicitly disable this below.
if hasattr(cfg,"cycle_terminate_on_success"):
    cfg.cycle_terminate_on_success = True
if a.notch is not None or restored_config is None:
    cfg.notch_torque = a.notch if a.notch is not None else 0.0
if a.cache:
    cfg.grasp_cache_path = a.cache
if a.command_slew_rad_s is not None:
    if a.command_slew_rad_s<0 or not hasattr(cfg,"cycle_command_slew_rad_s"):raise ValueError("Invalid command slew override")
    cfg.cycle_command_slew_rad_s=a.command_slew_rad_s
primitive_seconds = cfg.episode_length_s
sequence = a.sequence.split(",") if a.sequence else None
if sequence:
    if not hasattr(cfg, "cycle_stable_s"):
        raise ValueError("Sequences require a complete-cycle task")
    if len({m[0] for m in sequence}) != 1 or any(m not in cfg.flick_moves for m in sequence):
        raise ValueError("Only supported same-face moves may be chained")
    cfg.cycle_terminate_on_success = False
    cfg.episode_length_s *= len(sequence)
if a.detector:
    if not hasattr(cfg, "cycle_stable_s"):
        raise ValueError("Tactile-trigger evaluation requires cycle task")
    cfg.cycle_terminate_on_success = False
if a.terminal_cache:
    if not hasattr(cfg, "cycle_stable_s"):
        raise ValueError("Terminal cache collection requires a complete-cycle task")
    cache_stage = a.terminal_cache_stage if a.terminal_cache_stage is not None else (len(sequence)-1 if sequence else 0)
    if cache_stage<0 or cache_stage>=(len(sequence) if sequence else 1):raise ValueError("Terminal cache stage outside sequence")
    cfg.cycle_terminate_on_success = False
if a.video:
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (-0.40, -0.30, 0.46)
    cfg.viewer.lookat = (0.0, 0.0, 0.30)
    cfg.viewer.resolution = (960, 640)
# Top-level USD hashes do not include their external mesh/reference dependencies.
input_artifact_sha256={}
for name,path in {'grasp_cache':cfg.grasp_cache_path,'cube_usd':cfg.cube_cfg.spawn.usd_path,
                  'left_hand_usd':cfg.left_hand_cfg.spawn.usd_path,'right_hand_usd':cfg.right_hand_cfg.spawn.usd_path}.items():
    if path:
        input_artifact_sha256[name]={'path':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
env = gym.make(a.task, cfg=cfg, render_mode="rgb_array" if a.video else None)
u = env.unwrapped
u.record = True
wenv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)


loaded_checkpoint_hashes = {}
loaded_checkpoint_snapshots = {}

def load_payload(path):
    content=Path(path).read_bytes()
    payload=torch.load(io.BytesIO(content),map_location="cpu",weights_only=False)
    digest=hashlib.sha256(content).hexdigest()
    loaded_checkpoint_hashes[str(path)]=digest
    snapshot=source_archive.parent/"checkpoints"/(digest+".pt")
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists():snapshot.write_bytes(content)
    loaded_checkpoint_snapshots[str(path)]=str(snapshot)
    return payload

def load_policy(path):
    if path is None:
        return lambda obs: torch.zeros((u.num_envs, cfg.action_space), device=u.device)
    payload = load_payload(path)
    if payload.get("format") == "sensor_prototype_trajectory_v1":
        from rubik_lab.blind_actor import TrajectoryInference
        return TrajectoryInference(payload,u.num_envs,u.device)
    if payload.get("format") == "blind_primitive_gru_v1":
        from rubik_lab.blind_actor import BlindInference
        return BlindInference(payload, u.num_envs, u.device)
    runner = OnPolicyRunner(wenv, agent.to_dict(), log_dir=None, device=u.device)
    runner.load(loaded_checkpoint_snapshots[str(path)], load_optimizer=False)
    if a.stochastic:
        runner.alg.policy.eval()
        return lambda obs: runner.alg.policy.act(obs).detach()
    return runner.get_inference_policy(device=u.device)

policy = load_policy(a.checkpoint)
execution_protocol = getattr(policy,'execution_protocol',{})
a.initial_settle_steps = max(a.initial_settle_steps,int(execution_protocol.get('initial_settle_steps',0)))
secondary_config = validate_sequence_policy_settings(cfg,a.checkpoint_up) if sequence else None
up_policy = load_policy(a.checkpoint_up) if a.checkpoint_up else policy
detector = None
if a.detector:
    from rubik_lab.detector.streaming import StreamingReadiness
    detector = StreamingReadiness(load_payload(a.detector),u.num_envs,u.device,a.detector_threshold)
    if a.detector_up:
        from rubik_lab.detector.streaming import PerMoveReadiness
        if set(sequence)-{"U","U'"}:raise ValueError("Separate detector routing supports U/Up only")
        detector=PerMoveReadiness({MOVE_INDEX["U"]:detector,
            MOVE_INDEX["U'"]:StreamingReadiness(load_payload(a.detector_up),u.num_envs,u.device,a.detector_up_threshold)})
    minimum_detector_hold = math.ceil(cfg.cycle_stable_s/u.step_dt) if detector.label_key=="cycle_ready" else 1
    if a.detector_hold_steps is not None:
        detector.hold_steps = a.detector_hold_steps
    if detector.hold_steps < minimum_detector_hold:
        raise ValueError("Instantaneous-readiness observers must retain the full dwell; completed-window observers require at least one prediction")
planner = None
if a.probe_lift_return:
    if sequence:
        raise ValueError("Lift probe is only a single-move diagnostic")
    from rubik_lab.flick import FlickPlanner, to_cube_frame
    from rubik_lab.hand_kinematics import quat_to_mat
    planner = FlickPlanner(u.device)
offset_values = [float(v) for v in a.probe_offsets.split(",")] if a.probe_offsets else None
if offset_values and (sequence or a.probe_return or a.probe_lift_return or a.num_envs % len(offset_values)):
    raise ValueError("Offset sweep requires single move, no other probe, num_envs divisible by offsets")
probe_offsets = torch.tensor(offset_values,device=u.device).repeat(a.num_envs//len(offset_values)) if offset_values else torch.zeros(a.num_envs,device=u.device)
if offset_values:
    u.forced_grasp_idx = (torch.arange(u.num_envs,device=u.device)//len(offset_values)) % len(u.grasp_cache["cube_pos"])
    u.forced_move_id = torch.full((u.num_envs,),MOVE_INDEX[cfg.moves[0]],device=u.device,dtype=torch.long)
abduction_values = [float(v) for v in a.abduction_offsets.split(',')] if a.probe_abduct_return else None
if abduction_values and (sequence or offset_values or a.probe_return or a.probe_lift_return or a.num_envs % len(abduction_values)):
    raise ValueError('Abduction diagnostic requires single move, no other probe, and balanced environment count')
abduction_offsets = torch.tensor(abduction_values,device=u.device).repeat(a.num_envs//len(abduction_values)) if abduction_values else torch.zeros(a.num_envs,device=u.device)
terminal_states = []
records = []
data_rows = {}
writer = None
if a.video:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    Path(a.video).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(a.video, fps=round(1 / u.step_dt))
t0 = time.time()
trial_sensor_phases = []
turn_start_sensor_phases = []
actual_startup_hold_steps = []
trials = 1 if a.smoke_steps else a.episodes
for trial in range(trials):
    if sequence:
        u.forced_move_id = torch.full((u.num_envs,), MOVE_INDEX[sequence[0]], device=u.device, dtype=torch.long)
    env.reset(seed=a.seed + trial)
    trial_sensor_phases.append(int(u.tactile.step_count % u.tactile.capture_every))
    if detector is not None:
        detector.reset()
    for actor in (policy, up_policy):
        if hasattr(actor, "reset"):
            actor.reset()
    obs = wenv.get_observations()
    active = torch.ones(u.num_envs, dtype=torch.bool, device=u.device)
    reached = torch.zeros_like(active)
    stage = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
    peak_wrist_pos = torch.zeros(u.num_envs, device=u.device)
    peak_wrist_rot = torch.zeros_like(peak_wrist_pos)
    start_moves = u.move_id.clone()
    probe_age = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
    probe_returned = torch.zeros(u.num_envs,dtype=torch.bool,device=u.device)
    probe_lift = torch.zeros(u.num_envs, 4, device=u.device)
    probe_across = torch.zeros_like(probe_lift)
    probe_abduct = torch.zeros_like(probe_lift)
    primitive_records = [[] for _ in range(u.num_envs)]
    primitive_pending = torch.zeros(u.num_envs,dtype=torch.bool,device=u.device)
    primitive_start_phase = torch.full((u.num_envs,),-1,dtype=torch.long,device=u.device)
    primitive_wait_steps = torch.zeros(u.num_envs,dtype=torch.long,device=u.device)
    if writer is not None and trial == 0:
        # Warm the renderer without stepping physics; the first RGB frame may
        # otherwise be black and conceal the beginning of a fast primitive.
        for render_attempt in range(30):
            initial_frame = env.render()
        if initial_frame is None or np.asarray(initial_frame).mean() <= 1:
            raise RuntimeError("Renderer did not produce a visible initial frame")
    max_steps = a.smoke_steps or (u.max_episode_length + 1)
    startup_finished = False
    turn_start_sensor_phases.append(None)
    actual_startup_hold_steps.append(None)
    for step in range(max_steps):
        phase = int(u.tactile.step_count % u.tactile.capture_every)
        if not startup_finished and step >= a.initial_settle_steps and (not a.synchronize_tactile_clock or phase == 0):
            startup_finished = True
            turn_start_sensor_phases[-1] = phase
            actual_startup_hold_steps[-1] = step
        holding_startup = not startup_finished
        if phase == 0:
            primitive_pending.zero_()
        waiting_for_command = primitive_pending | holding_startup
        starting = (primitive_start_phase < 0) & ~waiting_for_command & active
        primitive_start_phase[starting] = phase
        primitive_wait_steps[waiting_for_command & active] += 1
        if writer is not None and trial == 0 and active[0]:
            # Pre-step rendering preserves the actual physical state even if the
            # subsequent step auto-resets. No repeated final frame or slow motion.
            frame = Image.fromarray(env.render()[..., :3])
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0,0,960,34),fill=(0,0,0))
            draw.text((12,12),f"t={step*u.step_dt:.3f}s  move={ALL_MOVES[int(u.move_id[0])]}  stage={int(stage[0])+1}  trigger={'tactile' if detector is not None else 'ground truth'}; ground-truth scoring",fill=(255,255,255))
            writer.append_data(np.asarray(frame))
        with torch.inference_mode():
            teacher_obs = obs
            if offset_values:
                _,_,angle,velocity = u._cube_state()
                shift = torch.sign(u.move_target[u.move_id])*torch.deg2rad(probe_offsets)
                compensated = angle-shift
                teacher_obs = obs.clone()
                teacher_obs['policy'] = obs['policy'].clone()
                teacher_obs['policy'][:,211] = compensated.sin()
                teacher_obs['policy'][:,212] = compensated.cos()
                teacher_obs['policy'][:,242] = 0.0
                error = torch.atan2((u.move_target[u.move_id]-compensated).sin(),(u.move_target[u.move_id]-compensated).cos())
                core_speed = u.cube.data.body_ang_vel_w[:,u.core_id].norm(dim=-1)
                probe_returned |= (error.abs()<cfg.move_success_tol) & (velocity.abs()<cfg.move_success_vel) & (core_speed<1.0)
            actions = policy(teacher_obs)
            if offset_values:
                ready_action = u._idle_offsets()[u.flick_cols[u.move_id]]/cfg.residual_finger_range
                actions = torch.where(probe_returned[:,None],ready_action,actions)
                probe_age[probe_returned] += 1
            if sequence and a.checkpoint_up:
                other = up_policy(obs)
                actions = torch.where((u.move_id == MOVE_INDEX["U'"])[:, None], other, actions)
        probe_move_mask = torch.ones(u.num_envs,dtype=torch.bool,device=u.device)
        if a.probe_return_moves:
            probe_move_mask = torch.zeros_like(probe_move_mask)
            for move in a.probe_return_moves.split(","):
                probe_move_mask |= u.move_id == MOVE_INDEX[move]
        if a.probe_abduct_return:
            begin = u.layer_done_latch & (probe_age == 0) & active
            rows = torch.arange(u.num_envs,device=u.device)
            h,cols = u.flick_hand[u.move_id],u.flick_cols[u.move_id]
            q = torch.stack([u.hands[side].data.joint_pos[:,u.finger_ids[side]] for side in ('left','right')],1)
            grasp = torch.stack([u.finger_grasp[side] for side in ('left','right')],1)
            current = (q-grasp)[rows[:,None],h[:,None],cols]/cfg.residual_finger_range
            probe_abduct[begin] = current[begin]
            # Index columns are MCP_FE, MCP_AA, PIP, DIP.
            probe_abduct[begin,1] += abduction_offsets[begin]/cfg.residual_finger_range
            returning = u.layer_done_latch & active
            probe_age[returning] += 1
            ready_action = u._idle_offsets()[cols]/cfg.residual_finger_range
            extending = ready_action.clone()
            extending[:,1] = probe_abduct[:,1]
            command = torch.where((probe_age <= 4)[:,None],probe_abduct,
                torch.where((probe_age <= 12)[:,None],extending,ready_action))
            actions = torch.where(returning[:,None],command,actions)
        elif a.probe_lift_return:
            begin = u.layer_done_latch & (probe_age == 0) & active & probe_move_mask
            for move, side in (("U", "right"), ("U'", "left")):
                ids = (begin & (u.move_id == MOVE_INDEX[move])).nonzero().flatten()
                if not len(ids):
                    continue
                wp, wr, q, _ = to_cube_frame(u, side)
                wp, wr, q = wp[ids], wr[ids], q[ids]
                fk = planner.fk[side]
                cols = u.flick_cols[u.move_id[ids]][0]
                pad = fk.pads(fk.forward(wp, wr, q))["index"][0]
                ready_q = (u.finger_grasp[side][ids] + u._idle_offsets()).clamp(
                    min=u.finger_lower[side][ids], max=u.finger_upper[side][ids])
                ready_pad = fk.pads(fk.forward(wp, wr, ready_q))["index"][0]
                height = torch.maximum(pad[:, 2], ready_pad[:, 2]) + 0.012
                lift_target = pad.clone()
                lift_target[:, 2] = height
                across_target = ready_pad.clone()
                across_target[:, 2] = height
                with torch.enable_grad():
                    lift_q = planner.track(move, wp, wr, q, q, lift_target, iters=50)
                    across_q = planner.track(move, wp, wr, q, lift_q, across_target, iters=50)
                probe_lift[ids] = (lift_q[:, cols] - u.finger_grasp[side][ids][:, cols]) / cfg.residual_finger_range
                probe_across[ids] = (across_q[:, cols] - u.finger_grasp[side][ids][:, cols]) / cfg.residual_finger_range
            returning = u.layer_done_latch & active & probe_move_mask
            probe_age[returning] += 1
            ready_action = u._idle_offsets()[u.flick_cols[u.move_id]] / cfg.residual_finger_range
            command = torch.where((probe_age < 16)[:, None], probe_lift,
                torch.where((probe_age < 32)[:, None], probe_across, ready_action))
            actions = torch.where(returning[:, None], command, actions)
        elif a.probe_return:
            if not hasattr(u, "_idle_offsets"):
                raise ValueError("Return probe requires cycle task")
            ready_action = u._idle_offsets()[u.flick_cols[u.move_id]] / cfg.residual_finger_range
            returning = u.layer_done_latch & probe_move_mask
            probe_age[returning] += 1
            actions = torch.where(returning[:, None], ready_action, actions)
        if waiting_for_command.any():
            ready_action = u._idle_offsets()[u.flick_cols[u.move_id]] / cfg.residual_finger_range
            actions = apply_ready_hold(actions,ready_action,waiting_for_command,(policy,up_policy))
        pre_obs = obs["policy"].detach().clone() if a.dataset else None
        obs, _, dones, _ = wenv.step(actions)
        snap = u.snapshot
        detected = None
        if detector is not None:
            if holding_startup:
                # No turn has been issued during startup hold. Completion cannot
                # acknowledge an unissued move; arm from its first real action.
                detector.reset()
                detected = {'ready_probability':torch.zeros(u.num_envs,device=u.device),
                    'failed_probability':torch.zeros(u.num_envs,device=u.device),
                    'done':torch.zeros(u.num_envs,dtype=torch.bool,device=u.device),
                    'stable_steps':torch.zeros(u.num_envs,dtype=torch.long,device=u.device)}
                if detector.angle_gate_deg is not None:
                    detected['angle_error_deg']=torch.full((u.num_envs,),float('nan'),device=u.device)
            else:
                detected = detector.step(**{key:snap[key] for key in ('tactile_ft','finger_pos','finger_tgt','move_id')})
            if primitive_pending.any():
                detector.advance(primitive_pending.nonzero().flatten())
                detected['done'] = detected['done'] & ~primitive_pending
                detected['stable_steps'] = torch.where(primitive_pending,0,detected['stable_steps'])
            detector.reset(dones.bool().nonzero().flatten())
        for actor in (policy, up_policy):
            if hasattr(actor, "reset"):
                actor.reset(dones.bool().nonzero().flatten())
        if a.dataset:
            # Only the first, explicitly counted episode of each env/trial is
            # collected; auto-reset background episodes are excluded.
            ids = active.nonzero().flatten()
            values = {k: v[ids].detach().cpu().numpy() for k, v in snap.items()}
            values.update({"command_issued": (~waiting_for_command[ids]).cpu().numpy(),
                "episode_uid": (trial*u.num_envs+ids).cpu().numpy(),
                "env_id": ids.cpu().numpy(), "step": np.full(len(ids), step, np.int64),
                "policy_obs": pre_obs[ids].cpu().numpy(),
                "policy_action": actions[ids].detach().clamp(-1,1).cpu().numpy(),
                "teacher_stage": probe_age[ids].cpu().numpy(),
                "return_offset_deg": probe_offsets[ids].cpu().numpy(),
                "abduction_offset_rad": abduction_offsets[ids].cpu().numpy()})
            if detected is not None:
                values.update({"detector_"+k:v[ids].detach().cpu().numpy() for k,v in detected.items()})
            for key, value in values.items():
                data_rows.setdefault(key, []).append(value)
        reached |= snap["layer_done_latch"] & active
        peak_wrist_pos = torch.maximum(peak_wrist_pos, snap["wrist_pos_error"].amax(-1))
        peak_wrist_rot = torch.maximum(peak_wrist_rot, snap["wrist_rot_error"].amax(-1))
        truth_full = snap["primitive_done"] & active
        full = (detected["done"] & active) if detected is not None else truth_full
        invalid = snap["dropped"] | snap["broken"] | snap.get("cycle_failed", torch.zeros_like(active))
        premature = full & ~truth_full & ~invalid
        success = full & truth_full & ~invalid
        primitive_timeout = ((snap.get("cycle_step", torch.zeros_like(stage)) >= math.ceil(primitive_seconds/u.step_dt)) & active & ~full) if sequence else torch.zeros_like(active)
        finished = (dones.bool() & active) | premature | primitive_timeout
        if detector is not None and not sequence:
            finished |= success
        if a.terminal_cache:
            collect = success & ~dones.bool() & (stage == cache_stage)
            ids = collect.nonzero().flatten()
            if len(ids):
                source = u.episode_grasp_idx[ids]
                state = {k: v[source].detach().cpu().numpy().copy() for k,v in u.grasp_cache.items()}
                pos, quat, angle, _ = u._cube_state()
                state['cube_pos'] = pos[ids].cpu().numpy()
                state['cube_quat'] = quat[ids].cpu().numpy()
                # Training cache only: subtract the ideal completed quarter turn.
                # The residual alignment error is retained; a square layer has
                # the same geometry every 90 deg. Live sequence never writes this.
                state['cube_twist'] = (angle[ids]-u.move_target[u.move_id[ids]]).cpu().numpy()
                state['prior_move_id'] = u.move_id[ids].cpu().numpy()
                state['cycle_ready'] = np.ones(len(ids),np.float32)
                for side in ('left','right'):
                    p_actual, q_actual = u.ctrl[side].ee_pose_w()
                    state[side+'_ee_pos_act'] = (p_actual[ids]-u.scene.env_origins[ids]).cpu().numpy()
                    state[side+'_ee_quat_act'] = q_actual[ids].cpu().numpy()
                    state[side+'_fingers_act'] = u.hands[side].data.joint_pos[ids][:,u.finger_ids[side]].cpu().numpy()
                    state[side+'_joint_pos_act'] = u.hands[side].data.joint_pos[ids].cpu().numpy()
                state['cells'] = u.hand_cells()[ids].cpu().numpy()
                state['link_touch'] = (u.cube_contacts.forces().sum(1)>0.05)[ids].cpu().numpy()
                terminal_states.append(state)
            if not sequence:finished |= success
        if sequence:
            for idx in (full & truth_full & ~invalid).nonzero().flatten().tolist():
                primitive_records[idx].append({"move": ALL_MOVES[int(snap["move_id"][idx])],
                    "seconds": int(snap["cycle_step"][idx]) * u.step_dt,
                    "start_sensor_phase": int(primitive_start_phase[idx]), "pre_command_wait_steps": int(primitive_wait_steps[idx]),
                    "error_deg": math.degrees(float(snap["twist_error"][idx]))})
            final_stage = stage == len(sequence) - 1
            finished |= full & final_stage
            success &= final_stage
            continue_mask = full & truth_full & ~final_stage & ~invalid & ~dones.bool() & ~primitive_timeout
            for k in range(len(sequence) - 1):
                ids = (continue_mask & (stage == k)).nonzero().flatten()
                if len(ids):
                    u.advance_cycle(ids, sequence[k + 1])
                    primitive_pending[ids] = a.synchronize_each_primitive
                    primitive_start_phase[ids] = -1
                    primitive_wait_steps[ids] = 0
                    if detector is not None:
                        detector.advance(ids)
                    for actor in (policy, up_policy):
                        if hasattr(actor, "reset"):
                            actor.reset(ids)
            stage[continue_mask] += 1
            # Goal/latch changed, but no physical reset or new grasp reference.
            obs = wenv.get_observations()
        for idx in finished.nonzero().flatten().tolist():
            reason = "success" if success[idx] else ("premature_trigger" if premature[idx] else ("broken" if snap["broken"][idx] else (
                "drop" if snap["dropped"][idx] else ("motion_limit" if snap.get("cycle_failed", invalid)[idx] else "timeout"))))
            records.append({"trial": trial, "env": idx, "move": ALL_MOVES[int(start_moves[idx])],
                "grasp_idx": int(snap["grasp_idx"][idx]), "return_offset_deg": float(probe_offsets[idx]), "abduction_offset_rad": float(abduction_offsets[idx]), "success": bool(success[idx]), "reason": reason,
                "layer_reached": bool(reached[idx]), "seconds": (step + 1) * u.step_dt,
                "error_deg": math.degrees(float(snap["twist_error"][idx])),
                "peak_cube_mm": float(snap["peak_cube_pos"][idx]) * 1000,
                "peak_cube_deg": math.degrees(float(snap["peak_cube_rot"][idx])),
                "peak_wrist_mm": float(peak_wrist_pos[idx]) * 1000,
                "peak_wrist_deg": math.degrees(float(peak_wrist_rot[idx])),
                "primitives": primitive_records[idx],
                **({"finger_ready_error": float(snap["finger_ready_error"][idx]),
                    "release_force": float(snap["release_force"][idx])} if "finger_ready_error" in snap else {})})
        active[finished] = False
        if (step + 1) % 100 == 0:
            print(f"[eval] trial={trial} step={step+1} active={int(active.sum())} reached={int(reached.sum())}", flush=True)
        if not active.any():
            break
    if not a.smoke_steps and active.any():
        raise RuntimeError("Evaluation ended without each environment contributing one outcome")
    print(f"[eval] trial {trial + 1}/{trials}, completed {len(records)} outcomes", flush=True)


def summarize(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    successes = sum(r["success"] for r in rows)
    rate = successes / n
    z = 1.96
    den = 1 + z*z/n
    center = (rate + z*z/(2*n))/den
    half = z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n))/den
    return {"n": n, "success_rate": rate, "wilson95": [center-half, center+half],
        "layer_reached_rate": sum(r["layer_reached"] for r in rows)/n,
        "reasons": {reason: sum(r["reason"] == reason for r in rows) for reason in sorted({r["reason"] for r in rows})},
        "median_seconds_success": float(np.median([r["seconds"] for r in rows if r["success"]])) if successes else None,
        "strict_calm_success_rate": sum(r["success"] and r["peak_cube_mm"] <= 5 and r["peak_cube_deg"] <= 5 for r in rows)/n,
        "p90_peak_cube_mm": float(np.percentile([r["peak_cube_mm"] for r in rows], 90)),
        "p90_peak_cube_deg": float(np.percentile([r["peak_cube_deg"] for r in rows], 90)),
        "p90_peak_wrist_mm": float(np.percentile([r["peak_wrist_mm"] for r in rows],90)),
        "p90_peak_wrist_deg": float(np.percentile([r["peak_wrist_deg"] for r in rows],90)),
        "max_peak_wrist_mm": max(r["peak_wrist_mm"] for r in rows),
        "max_peak_wrist_deg": max(r["peak_wrist_deg"] for r in rows)}

if a.terminal_cache and terminal_states:
    Path(a.terminal_cache).parent.mkdir(parents=True, exist_ok=True)
    with open(a.terminal_cache,'xb') as cachefile:
        np.savez_compressed(cachefile, **{key:np.concatenate([row[key] for row in terminal_states]) for key in terminal_states[0]})
    print('[eval] terminal cache', a.terminal_cache, flush=True)
if writer is not None:
    writer.close()
if a.dataset:
    import h5py
    Path(a.dataset).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.dataset, "w") as datafile:
        datafile.attrs.update({"schema_version": 2, "tactile_ft_scope": "legacy normal-only contact force and approximate moment; friction omitted", "task": a.task, "completion_label": "primitive_done",
            "teacher_mode": "abduction_release" if abduction_values else ("angle_compensation_sweep" if offset_values else ("lift_first" if a.probe_lift_return else ("direct_return" if a.probe_return else "policy"))),
            "config_json": json.dumps(cfg.to_dict(), default=str), "checkpoint": a.checkpoint or "zero",
            "checkpoint_sha256": loaded_checkpoint_hashes[str(a.checkpoint)] if a.checkpoint else "zero"})
        for key, parts in data_rows.items():
            datafile[key] = np.concatenate(parts)
    print(f"[eval] dataset {a.dataset}", flush=True)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
report = {"cycle_terminate_on_success":getattr(cfg,"cycle_terminate_on_success",None),"numerics": {"torch":torch.__version__,"cuda":torch.version.cuda,"matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_allow_tf32":torch.backends.cudnn.allow_tf32}, "tactile_ft_scope": "legacy normal-only contact force and approximate moment; friction omitted", "task": a.task, "seed": a.seed, "sequence": sequence, "trial_sensor_phases": trial_sensor_phases, "trigger": "tactile_detector" if detector is not None else "privileged_ground_truth",
    "synchronize_each_primitive": a.synchronize_each_primitive,
    "synchronize_tactile_clock": a.synchronize_tactile_clock, "turn_start_sensor_phases": turn_start_sensor_phases, "actual_startup_hold_steps": actual_startup_hold_steps,
    "per_primitive_timeout_seconds": primitive_seconds, "initial_settle_steps": a.initial_settle_steps,
    "detector_preserves_history_between_moves": detector.preserve_history_between_moves if detector is not None else None,
    "detector_arming": "first commanded turn action after startup hold",
    "detector_label_key": detector.label_key if detector is not None else None,
    "detector": a.detector, "detector_hold_steps": detector.hold_steps if detector is not None else None,
    "detector_sha256": loaded_checkpoint_hashes[str(a.detector)] if a.detector else None,
    "detector_threshold": a.detector_threshold if detector is not None else None,
    "detector_up": a.detector_up, "detector_up_threshold": a.detector_up_threshold if a.detector_up else None,
    "detector_up_sha256": loaded_checkpoint_hashes[str(a.detector_up)] if a.detector_up else None,
    "scoring_abort_on_premature_trigger": detector is not None, "scripted_return_probe": "abduction_release" if abduction_values else ("angle_compensation_sweep" if offset_values else ("lift_first" if a.probe_lift_return else a.probe_return)),
    "mode": "smoke" if a.smoke_steps else "fixed_quota_evaluation", "wall_seconds": time.time()-t0,
    "checkpoint": a.checkpoint, "checkpoint_up": a.checkpoint_up, "stochastic": a.stochastic,
    "actor_execution_protocol": execution_protocol,
    "actor_observation_scope": getattr(policy,"observation_scope","privileged"),
    "actor_observation_scope_up": getattr(up_policy,"observation_scope","privileged") if sequence else None,
    "probe_return_moves": a.probe_return_moves,
    "checkpoint_up_sha256": loaded_checkpoint_hashes[str(a.checkpoint_up)] if a.checkpoint_up else None,
    "checkpoint_sha256": loaded_checkpoint_hashes[str(a.checkpoint)] if a.checkpoint else None,
    "secondary_checkpoint_config": secondary_config, "secondary_control_settings_checked": bool(secondary_config) if a.checkpoint_up else None,
    "restored_config": restored_config, "config": cfg.to_dict(),
    "terminal_cache_stage": cache_stage if a.terminal_cache else None,
    "terminal_cache": a.terminal_cache if terminal_states else None, "terminal_cache_requested": a.terminal_cache, "terminal_cache_count": sum(len(row["cube_pos"]) for row in terminal_states), "terminal_cache_initial_velocities": "zeroed on future reset" if a.terminal_cache else None,
    "source_sha256": source_hashes, "input_artifact_sha256": input_artifact_sha256,
    "asset_hash_scope": "Top-level USD only; referenced mesh dependency files not captured by these hashes",
    "source_archive": str(source_archive),
    "checkpoint_snapshots": loaded_checkpoint_snapshots,
    "evaluator_sha256": source_hashes['scripts/eval_primitive.py'],
    "blind_actor_sha256": source_hashes['source/rubik_lab/rubik_lab/blind_actor.py'],
    "summary": summarize(records),
    "interval_scope": "Wilson binomial interval over counted trials; repeated environments and selected grasp do not establish independent real-world generalization",
    "per_trial": {str(trial):summarize([r for r in records if r["trial"]==trial]) for trial in sorted({r["trial"] for r in records})},
    "per_abduction_offset_rad": {str(value):summarize([r for r in records if abs(r["abduction_offset_rad"]-value)<1e-6]) for value in abduction_values} if abduction_values else None,
    "per_return_offset": {str(value):summarize([r for r in records if r["return_offset_deg"]==value]) for value in offset_values} if offset_values else None,
    "per_primitive_stage": summarize_stages(records,sequence),
    "per_move_scope": "whole trial grouped by its starting move",
    "per_move": {m: summarize([r for r in records if r["move"] == m]) for m in sorted({r["move"] for r in records})},
    "episodes": records}
out.write_text(json.dumps(report, indent=2, default=str)+"\n")
print("[eval] " + json.dumps(report["summary"]), flush=True)
print(f"[eval] report {out}", flush=True)
env.close()
app.close()
