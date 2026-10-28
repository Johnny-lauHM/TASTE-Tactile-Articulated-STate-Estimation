"""Generate edge grasps (human-like: each hand pinches only its own outer column with the fingertips) for resets.

Per env a random closing pose is sampled. The cube is held in place while both palms squeeze in and the fingers close,
then it is released under gravity. Grasps that keep the cube still are saved, with both targets and actual states:
    cache/home_grip.npz : {left,right}_{ee_pos, ee_quat, fingers}[_act], cube_pos, cube_quat (logical, env frame)

    python scripts/gen_grasp_cache.py --headless --num_envs 2048
    python scripts/gen_grasp_cache.py --num_envs 16          # watch it in the GUI (nothing is saved without --save)
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--hold_s", type=float, default=1.0, help="cube held in place while closing")
parser.add_argument("--release_s", type=float, default=1.5, help="free fall test after release")
parser.add_argument("--squeeze", type=float, nargs=2, default=(0.0, 0.004), help="palm push into the cube [m] range")
parser.add_argument("--rot_noise_deg", type=float, default=6.0, help="random hand rotation about the cube center")
parser.add_argument("--side_noise", type=float, default=0.005, help="random shift across the palm [m]")
parser.add_argument("--max_squeeze_offset", type=float, default=1.0, help="keep grasps with |EE target - actual| below [m]")
parser.add_argument("--out", default="cache/home_grip.npz")
parser.add_argument("--save", action="store_true", help="save even when the GUI is used")
parser.add_argument("--tilt_range", type=float, nargs=2, default=(20.0, 85.0), help="wrist tilt [deg] (fingers up -> back)")
parser.add_argument("--gap_range", type=float, nargs=2, default=(0.005, 0.035), help="palm-to-cube gap [m]")
parser.add_argument("--along_range", type=float, nargs=2, default=(0.17, 0.23), help="cube center along the fingers [m]")
parser.add_argument("--min_tips", type=int, default=2, help="fingertip links in contact per hand")
parser.add_argument("--allow_mp", action="store_true", help="also allow middle phalanges to touch")
parser.add_argument("--no_edge_filter", action="store_true", help="keep all stable grasps (old behaviour)")
parser.add_argument("--require_thumb", action="store_true", help="each hand must touch with its thumb tip (human pinch)")
parser.add_argument("--synth", default=None, help="start from kinematically synthesized grasps (scripts/synth_grasps.py)")
parser.add_argument("--press", type=float, nargs=2, default=(0.05, 0.25), help="synth: extra flexion of the contact fingers [rad]")
parser.add_argument("--synth_noise", type=float, default=0.002, help="synth: wrist position noise [m]")
parser.add_argument("--save_frames", action="store_true", help="PNGs of env 0 to logs/grasp_frames (needs --enable_cameras)")
parser.add_argument("--region_left", default="edge", choices=["edge", "center", "wide"])
parser.add_argument("--region_right", default="edge", choices=["edge", "center", "wide"])
parser.add_argument("--moves", default="R,L,U,D,F,B", help="internal twist axes to generate grasps for (faces)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.assets import FINGER_JOINTS  # noqa: E402
from rubik_lab.moves import ALL_MOVES  # noqa: E402
from rubik_lab.cube_contacts import FINGERTIP_LINK, LINKS, face_layer_mask  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import CUBE_CENTER, BimanualTwistEnvCfg, home_grip_pose  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIDES = ("left", "right")

cfg = BimanualTwistEnvCfg()
cfg.gravity_curriculum = False  # real gravity
cfg.scene.num_envs = args.num_envs
cfg.ee_action_mode = "absolute"
cfg.grasp_cache_path = ""
cfg.moves = tuple(args.moves.split(","))  # grasps per internal twist axis (env picks a matching one); default: every axis
cfg.episode_length_s = 1e6
cfg.ee_init_pos_noise = 0.0
cfg.finger_init_noise = 0.0
cfg.cube_init_pos_noise = 0.0
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-0.30, -0.25, 0.42)  # front-right, sees the thumbs
cfg.viewer.lookat = CUBE_CENTER
env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg, render_mode="rgb_array" if args.save_frames else None)
u = env.unwrapped
n, dev = u.num_envs, u.device
env.reset()


def rand(lo, hi, shape=(n,)):
    return lo + (hi - lo) * torch.rand(shape, device=dev)


# random closing pose per env and hand (rad); ranges cover light pinches to full wraps
RANGES = {
    "thumb_CMC_FE": (0.4, 1.5), "thumb_CMC_AA": (-0.3, 0.3), "thumb_MCP_FE": (0.0, 0.9), "thumb_MCP_AA": (-0.2, 0.2),
    "thumb_IP": (0.1, 1.0), "MCP_FE": (0.3, 1.3), "MCP_AA": (-0.15, 0.15), "PIP": (0.2, 1.3), "DIP": (0.1, 1.0),
    "pinky_CMC": (0.0, 0.2),
}  # fmt: skip
grip = {}
for side in SIDES:
    q = torch.zeros(n, len(FINGER_JOINTS), device=dev)
    for i, j in enumerate(FINGER_JOINTS):
        key = j if j in RANGES else j.split("_", 1)[1]
        q[:, i] = rand(*RANGES[key])
    grip[side] = q

# EE targets (edge grasp search): per env a random side grip (wrist tilt, palm gap, cube depth along the fingers) from
# home_grip_pose, a small random rotation about the cube center and tangential offsets, plus a small squeeze.
center = torch.tensor(CUBE_CENTER, device=dev)
grip_params = torch.stack([rand(*args.tilt_range), rand(*args.gap_range), rand(*args.along_range)], -1)  # (n, 3)
ee_tgt = {}
for side in SIDES:
    p0 = torch.tensor(u.home[side][0], device=dev).repeat(n, 1)
    q0 = torch.tensor(u.home[side][1], device=dev).repeat(n, 1)
    poses = [home_grip_pose(side, *map(float, gp)) for gp in grip_params.cpu().numpy()]
    ph = torch.tensor([pp for pp, _ in poses], device=dev)
    qh = torch.tensor([qq for _, qq in poses], device=dev)
    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=dev), dim=-1)
    dq = math_utils.quat_from_angle_axis(rand(0.0, math.radians(args.rot_noise_deg)), axis)
    q = math_utils.quat_mul(dq, qh)
    p = center + math_utils.quat_apply(dq, ph - center)
    palm_n = math_utils.quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
    fingers_dir = math_utils.quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
    side_dir = torch.cross(palm_n, fingers_dir, dim=-1)
    p = p + side_dir * rand(-args.side_noise, args.side_noise).unsqueeze(-1)
    p = p + palm_n * rand(*args.squeeze).unsqueeze(-1)
    ee_tgt[side] = (p, q, p0, q0)


def slerp_from(q0, q1, s):
    aa = math_utils.axis_angle_from_quat(math_utils.quat_mul(q1, math_utils.quat_inv(q0)))
    ang = torch.linalg.norm(aa, dim=-1)
    return math_utils.quat_mul(math_utils.quat_from_angle_axis(ang * s, aa), q0)


finger_base = {s: torch.zeros(n, len(FINGER_JOINTS), device=dev) for s in SIDES}
base_pose = {s: (torch.tensor(u.home[s][0], device=dev).repeat(n, 1), torch.tensor(u.home[s][1], device=dev).repeat(n, 1)) for s in SIDES}
if args.synth:
    # place each hand directly at a synthesized grasp (base at the wrist pose -> virtual joints ~0), fingers at the
    # synthesized joints; then the contact fingers (thumb/index/middle) press a bit further to create grip force
    syn = np.load(os.path.join(ROOT, args.synth))
    k_syn = len(syn["loss"])
    sel = torch.arange(n, device=dev) % k_syn
    center_t = torch.tensor(CUBE_CENTER, device=dev)
    press_cols = [i for i, j in enumerate(FINGER_JOINTS) if j.split("_")[0] in ("thumb", "index", "middle") and ("FE" in j or j.endswith(("PIP", "DIP", "IP")))]
    for side in SIDES:
        wp = torch.tensor(syn[f"{side}_wrist_pos"], device=dev)[sel] + center_t + args.synth_noise * (2 * torch.rand(n, 3, device=dev) - 1)
        wq = torch.tensor(syn[f"{side}_wrist_quat"], device=dev)[sel]
        q = torch.tensor(syn[f"{side}_fingers"], device=dev)[sel]
        base_pose[side] = (wp, wq)
        u.ctrl[side].set_base_pose(wp + u.scene.env_origins, wq, slice(None))
        jp = u.hands[side].data.default_joint_pos.clone()
        jp[:, u.finger_ids[side]] = q
        u.hands[side].write_joint_state_to_sim(jp, torch.zeros_like(jp))
        u.hands[side].set_joint_position_target(jp)
        finger_base[side] = q
        delta = torch.zeros_like(q)
        delta[:, press_cols] = rand(*args.press, shape=(n, len(press_cols)))
        grip[side] = delta
        ee_tgt[side] = (wp, wq, wp, wq)
    grip_params = torch.stack([sel.float(), torch.zeros(n, device=dev), torch.zeros(n, device=dev)], -1)  # synth index
    pair_id = torch.tensor(syn["ref_id"], device=dev)[sel] if "ref_id" in syn else torch.full((n,), -1.0, device=dev)


def action(s_pose, s_fingers):
    """move the open hand from the default home to its sampled grip pose, then close the fingers."""
    acts = []
    for side in SIDES:
        p, q, p0, q0 = ee_tgt[side]
        lo, hi = u.finger_lower[side], u.finger_upper[side]
        f = finger_base[side] + s_fingers * grip[side]
        acts.append(torch.cat([p0 + s_pose * (p - p0), slerp_from(q0, q, s_pose), 2 * (f - lo) / (hi - lo) - 1], -1))
    return torch.cat(acts, -1)


cube_root = u.cube.data.root_state_w.clone()
twist0 = torch.zeros(n, 1, device=dev)
failed = torch.zeros(n, dtype=torch.bool, device=dev)
hold_steps, release_steps = int(args.hold_s / u.step_dt), int(args.release_s / u.step_dt)
frame_dir = os.path.join(ROOT, "logs/grasp_frames")
if args.save_frames:
    import imageio.v2 as imageio

    os.makedirs(frame_dir, exist_ok=True)
for k in range(hold_steps + release_steps):
    s_pose = 1.0 if args.synth else min(1.0, (k + 1) / (0.4 * hold_steps))
    s_fingers = min(1.0, (k + 1) / (0.5 * hold_steps)) if args.synth else min(1.0, max(0.0, ((k + 1) - 0.4 * hold_steps) / (0.4 * hold_steps)))
    if k < hold_steps:  # keep the cube where it is while the hands close
        u.cube.write_root_state_to_sim(cube_root)
        u.cube.write_joint_state_to_sim(twist0, torch.zeros_like(twist0), joint_ids=[u.twist_id])
    _, _, terminated, truncated, _ = env.step(action(s_pose, s_fingers))
    failed |= terminated | truncated
    if args.save_frames and k % 3 == 0:
        imageio.imwrite(os.path.join(frame_dir, f"{k:04d}.png"), env.render())
    if k == hold_steps + release_steps // 2 - 1:  # settled-state reference, half way through the release test
        core_mid, quat_mid, _, _ = u._cube_state()

# stable = no motion during the second half of the release, still near the nominal pose, layer not turned
core, quat, twist, _ = u._cube_state()
creep = torch.linalg.norm(core - core_mid, dim=-1)
creep_rot = math_utils.quat_error_magnitude(quat, quat_mid)
offset = torch.linalg.norm(core - torch.tensor(CUBE_CENTER, device=dev), dim=-1)
tilt = math_utils.quat_error_magnitude(quat, torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(n, 4))
f = u.extras["tactile_layer_force"].sum(-1)  # (n, 2, 5)
tips_in_contact = (f > cfg.tactile_contact_threshold).sum(-1)  # (n, 2)
ok = (
    ~failed
    & (creep < 0.002) & (creep_rot < math.radians(2))
    & (offset < 0.025) & (tilt < math.radians(25))
    & (twist.abs() < math.radians(5))
)  # fmt: skip
stable = ok.clone()
for side in SIDES:  # a large target-actual offset means a stiff squeeze: unstable to restore, and the layer can't turn
    p_act = u.ctrl[side].ee_pose_w()[0] - u.scene.env_origins
    ok &= torch.linalg.norm(ee_tgt[side][0] - p_act, dim=-1) < args.max_squeeze_offset

# edge grasp: (1) only fingertip links (distal phalanx / elastomer) touch the cube, (2) >= min_tips of them per hand,
# (3) each hand only touches its own outer column (left: L layer, right: R layer)
touch = u.cube_contacts.forces().sum(1) > 0.05  # (n, 2 hands, 27 links)
allowed = FINGERTIP_LINK.to(dev) | (torch.tensor(["_MP" in l for l in LINKS], device=dev) if args.allow_mp else False)
nonpad = (touch & ~allowed).any(-1).any(-1)
ntips = (touch & allowed).sum(-1)  # (n, 2)
cells = u.hand_cells()  # (n, 2, 27)
mid_col = torch.tensor([(i // 3) % 3 == 1 for i in range(27)], device=dev)  # middle column (M slice)
allow_l = (mid_col | face_layer_mask("L", dev)) if args.region_left in ("center", "wide") else face_layer_mask("L", dev)  # holding hand: L + M
allow_r = (mid_col | face_layer_mask("R", dev)) if args.region_right in ("center", "wide") else face_layer_mask("R", dev)  # holding hand: R + M
outside = (cells[:, 0] & ~allow_l).any(-1) | (cells[:, 1] & ~allow_r).any(-1)
c1 = ok & ~nonpad
c2 = c1 & (ntips >= args.min_tips).all(-1)
c3 = c2 & ~outside
thumb = (touch[..., LINKS.index("thumb_DP")] | touch[..., LINKS.index("thumb_elastomer")]).all(-1)  # both hands
c4 = c3 & thumb
print(f"[gen_grasp] funnel: stable {int(stable.sum())} -> squeeze ok {int(ok.sum())} -> fingertip links only {int(c1.sum())} "
      f"-> >={args.min_tips} tips/hand {int(c2.sum())} -> own outer column only {int(c3.sum())} -> with both thumbs {int(c4.sum())}   (of {n})", flush=True)  # fmt: skip
face = torch.tensor(["RLUDFB".index(ALL_MOVES[m][0]) for m in u.move_id.tolist()], device=dev)
for name, msk in (("stable", stable), ("fingertip only", c1), ("own column", c3)):
    print(f"[gen_grasp]   by internal axis, {name:14s}: " + " ".join(f"{f}={int((msk & (face == i)).sum())}/{int((face == i).sum())}" for i, f in enumerate("RLUDFB")), flush=True)
if not args.no_edge_filter:
    ok = c4 if args.require_thumb else c3
print(f"[gen_grasp] {ok.sum().item()}/{n} stable grasps | dropped {failed.sum().item()} | creep(med)={creep.median() * 1000:.1f} mm "
      f"offset(med)={offset.median() * 1000:.1f} mm tilt(med)={math.degrees(tilt.median()):.1f} deg | "
      f"pads in contact, all/stable (mean L,R)={tips_in_contact.float().mean(0).tolist()} / "
      f"{tips_in_contact[ok].float().mean(0).tolist() if ok.any() else '-'}", flush=True)  # fmt: skip

if ok.any() and (args.headless or args.save):
    out = {"cube_pos": core[ok], "cube_quat": quat[ok], "cube_twist": twist[ok], "move_id": u.move_id[ok].float(),
           "cells": cells[ok].float(), "link_touch": touch[ok].float(), "grip_params": grip_params[ok]}  # fmt: skip
    for side in SIDES:
        hand = u.hands[side]
        p_act, q_act = u.ctrl[side].ee_pose_w()
        out[f"{side}_ee_pos"] = ee_tgt[side][0][ok]
        out[f"{side}_ee_quat"] = ee_tgt[side][1][ok]
        out[f"{side}_ee_pos_act"] = (p_act - u.scene.env_origins)[ok]
        out[f"{side}_ee_quat_act"] = q_act[ok]
        out[f"{side}_fingers"] = (finger_base[side] + grip[side])[ok]
        out[f"{side}_base_pos"] = base_pose[side][0][ok]
        out[f"{side}_base_quat"] = base_pose[side][1][ok]
        out[f"{side}_fingers_act"] = hand.data.joint_pos[:, u.finger_ids[side]][ok]
        out[f"{side}_joint_pos_act"] = hand.data.joint_pos[ok]  # all 28 joints, exact restore
    if args.synth:
        out["pair_id"] = pair_id[ok]  # index of the reference grasp this one was synthesized from (paired regrip)
    out["region"] = torch.tensor([[float(args.region_left == "center"), float(args.region_right == "center")]], device=dev).expand(int(ok.sum()), 2)
    path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{k: v.cpu().numpy() for k, v in out.items()})
    print(f"[gen_grasp] saved {ok.sum().item()} grasps to {path}", flush=True)

env.close()
app.close()
