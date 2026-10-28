"""Collect Wave SE tactile + proprioception + completion labels for training a move-completion detector.

Writes one HDF5 file with per-step rows (env-major chunks appended over time):
    tactile_ft (2,5,6) f16 | finger_pos/finger_tgt (2,22) f16 | ee_pose (2,7) f32 | move_id i8 | twist_error f32
    move_done / layer_done_latch / primitive_done / dropped bool | tactile_layer_force (2,5,2) f16 | episode_step i32
    env_id i32 | episode_uid i64 | step i64 | [tactile_image (2,5,60,60) u8, only with --images]
Episodes can be rebuilt by grouping on episode_uid and sorting by step.

Action sources:
    --policy random    : random relative actions (pipeline test)
    --policy scripted  : R/L moves by a scripted wrist roll of the own-side hand (needs ee_action_mode=relative)
    --checkpoint PATH  : an rsl_rl policy (model_*.pt) trained on the same task config

    python scripts/collect_tactile_data.py --headless --num_envs 64 --steps 400 --images --out data/tactile_test.h5
"""

import argparse
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400, help="policy steps to record")
parser.add_argument("--images", action="store_true", help="also record the 60x60 TacMap images (slower, bigger)")
parser.add_argument("--policy", choices=["random", "scripted"], default="scripted")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--moves", default=None)
parser.add_argument("--task", default="Rubik-Cycle-U-Sharpa-v0")
parser.add_argument("--cache", default=None)
parser.add_argument("--notch", type=float, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--initial_settle_steps", type=int, default=0)
parser.add_argument("--out", default="data/tactile_dataset.h5")
parser.add_argument("--continue_after_success", action="store_true", help="observe policy after readiness until failure or timeout; uncensored detector data")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.initial_settle_steps < 0:
    parser.error("initial_settle_steps must be nonnegative")
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.moves import ALL_MOVES, FACE_NORMALS, move_angle, move_face  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import CUBE_CENTER, BimanualTwistEnvCfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
from rubik_lab.primitive_config import restore_checkpoint_settings
restored_config = restore_checkpoint_settings(cfg, args.checkpoint)
cfg.seed = args.seed
if args.notch is not None or restored_config is None:
    cfg.notch_torque = args.notch if args.notch is not None else 0.0
if args.cache:
    cfg.grasp_cache_path = args.cache
cfg.gravity_curriculum = False  # real gravity
cfg.scene.num_envs = args.num_envs
cfg.tactile.image = args.images
if args.continue_after_success:
    if not hasattr(cfg,"cycle_terminate_on_success"):
        raise ValueError("Post-success collection requires complete-cycle task")
    cfg.cycle_terminate_on_success = False
if args.moves:
    cfg.moves = tuple(args.moves.split(","))
env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
u.record = True
n, dev = u.num_envs, u.device
obs, _ = env.reset()

policy = None
if args.checkpoint:
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from rubik_lab.tasks.bimanual_twist.agents.rsl_rl_ppo_cfg import BimanualTwistPPORunnerCfg

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    payload = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if payload.get('format')=='sensor_prototype_trajectory_v1':
        from rubik_lab.blind_actor import TrajectoryInference
        policy = TrajectoryInference(payload,n,dev)
    elif payload.get('format')=='blind_primitive_gru_v1':
        from rubik_lab.blind_actor import BlindInference
        policy = BlindInference(payload,n,dev)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=dev)
        runner.load(args.checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=dev)
    args.initial_settle_steps=max(args.initial_settle_steps,int(getattr(policy,'execution_protocol',{}).get('initial_settle_steps',0)))
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

per_hand = u.per_hand
center = torch.tensor(CUBE_CENTER, device=dev)
roll_steps = int(1.5 / u.step_dt)


def scripted_action():
    """the twisting hand rolls its wrist about the face axis over 1.5 s, then holds (others zero)."""
    a = torch.zeros(n, cfg.action_space, device=dev)
    for i, side in enumerate(("left", "right")):
        mask = u.wrist_mask[:, i] > 0
        if not mask.any():
            continue
        axis = torch.tensor([FACE_NORMALS[move_face(ALL_MOVES[m])] for m in u.move_id.tolist()], device=dev)
        ang = torch.tensor([move_angle(ALL_MOVES[m]) for m in u.move_id.tolist()], device=dev)
        if cfg.ee_action_mode == "residual":  # offset from the grasp: ramp the roll up to the move angle, then hold
            frac = (u.episode_length_buf.float() / roll_steps).clamp(max=1.0)
            a[:, i * per_hand + 3 : i * per_hand + 6] = torch.where(
                mask.unsqueeze(-1), axis * (ang * frac / cfg.residual_rot_range).unsqueeze(-1), 0.0
            )
        else:  # relative: constant roll rate for roll_steps
            rate = ang / roll_steps / cfg.ee_rot_scale  # action units per step
            active = mask & (u.episode_length_buf < roll_steps)
            a[:, i * per_hand + 3 : i * per_hand + 6] = torch.where(active.unsqueeze(-1), axis * rate.unsqueeze(-1), 0.0)
    return a.clamp(-1, 1)


os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
path = os.path.join(ROOT, args.out)
h5 = h5py.File(path, "x")
import json,hashlib
h5.attrs["tactile_ft_scope"] = "legacy normal contact forces only; cube mean-point normal-force moments; friction omitted"
h5.attrs["tactile_geometry_version"] = cfg.tactile.geometry_version
h5.attrs["tactile_image_scope"] = "geometric penetration proxy; not calibrated soft-pad deformation"
geometry_path=os.path.join(ROOT,"source/rubik_lab/rubik_lab/tacmap/sharpa_tacmap_vbts.py")
with open(geometry_path,"rb") as source:h5.attrs["tactile_geometry_source_sha256"]=hashlib.sha256(source.read()).hexdigest()
h5.attrs["schema_version"] = 2
h5.attrs["task"] = args.task
h5.attrs["seed"] = args.seed
h5.attrs["config_json"] = json.dumps(cfg.to_dict(), default=str)
h5.attrs["completion_label"] = "primitive_done"
h5.attrs["continue_after_success"] = args.continue_after_success
h5.attrs["actor_observation_scope"] = getattr(policy,"observation_scope","privileged_or_scripted")
h5.attrs["initial_settle_steps"] = args.initial_settle_steps
h5.attrs.update({"moves": ",".join(cfg.moves), "policy": args.checkpoint or args.policy, "step_dt": u.step_dt,
                 "all_moves": ",".join(ALL_MOVES), "tactile": "Wave SE shaped simulation proxy: geometric TacMap + normal-only approximate wrench, 30fps, 20ms"})  # fmt: skip
DT = {"tactile_ft": np.float16, "finger_pos": np.float16, "finger_tgt": np.float16, "tactile_layer_force": np.float16,
      "ee_pose": np.float32, "twist_error": np.float32, "move_id": np.int8, "episode_step": np.int32,
      "move_done": bool, "layer_done_latch": bool, "primitive_done": bool, "dropped": bool, "tactile_image": np.uint8}  # fmt: skip
dsets = {}


def append(name, arr):
    arr = np.asarray(arr)
    if name not in dsets:
        kw = dict(compression="gzip", compression_opts=4) if name == "tactile_image" else {}
        dsets[name] = h5.create_dataset(name, shape=(0, *arr.shape[1:]), maxshape=(None, *arr.shape[1:]), dtype=arr.dtype,
                                        chunks=(max(1, min(256, len(arr))), *arr.shape[1:]), **kw)  # fmt: skip
    d = dsets[name]
    d.resize(d.shape[0] + len(arr), axis=0)
    d[-len(arr):] = arr


episode_uid = torch.arange(n, device=dev, dtype=torch.long)
next_uid = n
t0 = time.time()
n_done = n_drop = n_eps = 0
for k in range(args.steps):
    with torch.inference_mode():
        if policy is not None:
            act = policy(obs)
        elif args.policy == "scripted":
            # residual mode with the env's nominal wrist-roll prior: zero action = follow the prior
            act = torch.zeros(n, cfg.action_space, device=dev) if cfg.ee_action_mode == "residual" and cfg.wrist_prior else scripted_action()
        else:
            act = 0.3 * torch.randn(n, cfg.action_space, device=dev)
        if args.initial_settle_steps:
            settling = u.episode_length_buf < args.initial_settle_steps
            ready = u._idle_offsets()[u.flick_cols[u.move_id]] / cfg.residual_finger_range
            act = torch.where(settling[:,None],ready,act)
            if hasattr(policy,'reset'):policy.reset(settling.nonzero().flatten())
        out = env.step(act)
    obs = out[0]
    snap = u.snapshot
    rows = {k2: v.cpu().numpy().astype(DT.get(k2, v.cpu().numpy().dtype)) for k2, v in snap.items()}
    rows["env_id"] = np.arange(n, dtype=np.int32)
    rows["episode_uid"] = episode_uid.cpu().numpy()
    rows["step"] = np.full(n, k, dtype=np.int64)
    for name, arr in rows.items():
        append(name, arr)
    # Use actual reset flags, including numerical failures and cycle violations.
    ended = out[2].bool() if policy is not None else (out[2] | out[3])
    n_done += int(snap["primitive_done"].sum()); n_drop += int(snap["dropped"].sum()); n_eps += int(ended.sum())  # noqa: E702
    if ended.any():
        ids = ended.nonzero().flatten()
        if hasattr(policy,"reset"):
            policy.reset(ids)
        episode_uid[ids] = torch.arange(next_uid, next_uid + len(ids), device=dev)
        next_uid += len(ids)
    if (k + 1) % 50 == 0:
        print(f"[collect] step {k + 1}/{args.steps}  rows={dsets['move_id'].shape[0]}  episodes ended={n_eps} "
              f"(primitive_done frames {n_done}, dropped frames {n_drop})  move_done rows={int(np.sum(dsets['move_done'][:]))}  "
              f"{(k + 1) / (time.time() - t0):.1f} steps/s", flush=True)  # fmt: skip
h5.close()
print(f"[collect] wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)
env.close()
app.close()
