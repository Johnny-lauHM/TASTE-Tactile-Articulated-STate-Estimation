"""Render one env of a recorded chained execution (scripts/solve/execute.py --record) to an mp4: the recorded hand
joint states and cube pose / layer angle are written frame by frame (no physics), stickers repainted from the
logical cube state. Blind runs (recorded with --estimator) render a dashboard next to the scene: the 10 pads' 60x60
TacMap images and forces, the skill state machine (plan, completion probability / hold, trigger) and the estimated cube
state against the simulator's.

    python scripts/solve/replay.py --headless --record outputs/solve/full8/record.pt --env 3 --out outputs/solve/full8/env3.mp4
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--record", required=True)
p.add_argument("--env", type=int, required=True)
p.add_argument("--out", required=True)
p.add_argument("--fps", type=int, default=30)
p.add_argument("--no_dashboard", action="store_true")
p.add_argument("--until", type=float, default=0.0, help="s of sim time to render (the env's done time + a margin); 0: until its state stops changing")
p.add_argument("--eye", default="", help="camera 'ex,ey,ez;lx,ly,lz' (env frame)")
p.add_argument("--arm_cfg", default="", help="fr3 layout,urdf_dir,usd_dir of the recorded run (records that do not carry them)")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app
import sys as _sys, traceback as _tb
def _die(t, v, b):
    _tb.print_exception(t, v, b); _sys.stdout.flush(); os._exit(1)
_sys.excepthook = _die
import torch
import isaaclab.utils.math as mu
from rubik_lab.solve.cube_logic import Cube
from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.visual import StickerPainter

rec = torch.load(a.record, weights_only=False)
e = a.env
cfg = SolveEnvCfg()
cfg.scene.num_envs = 1
cfg.ready_cache = rec["ready_cache"]
cfg.episode_length_s = 1e6
cfg.skills = ("hold",)
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.arm = rec.get("arm", "float")
# the arm assets / layout of the recorded run (records before this field: pass them with --arm_cfg layout,urdf_dir,usd_dir)
for k_, v_ in rec.get("arm_cfg", {}).items():
    setattr(cfg, k_, v_)
if a.arm_cfg:
    cfg.fr3_layout, cfg.fr3_urdf_dir, cfg.fr3_usd_dir = a.arm_cfg.split(",")
cfg.viewer.eye = (-0.30, -0.28, 0.42)
cfg.viewer.lookat = (0.0, 0.0, 0.3)
if cfg.arm == "fr3":  # both arms in view: from the cube's far side (the bases stand behind the thumbs, -x)
    cfg.viewer.eye = (0.62, -0.12, 0.80)  # above, in front of the cube, facing the two arm bases
    cfg.viewer.lookat = (-0.08, 0.0, 0.12)
if a.eye:
    cfg.viewer.eye, cfg.viewer.lookat = (tuple(map(float, v.split(","))) for v in a.eye.split(";"))
cfg.viewer.resolution = (640, 480)
env = SolveEnv(cfg, render_mode="rgb_array")
env.reset()
dev = env.device
ids = torch.zeros(1, dtype=torch.long, device=dev)
idx = rec["ready_idx"][e : e + 1].to(dev)
env.ready_idx[ids] = idx
for h in range(2):
    env._set_base(ids, h, idx)
painter = StickerPainter()
blind = None if a.no_dashboard else rec.get("blind")
if blind is not None:
    import cv2
    import numpy as np

    from rubik_lab.constants import FINGERTIPS
    from rubik_lab.solve.estimator import rot6_to_matrix

    B = {k: v[e] for k, v in blind.items()}
    SK, plan = rec["skills"], rec["plans"][e]["skills"]
    hold_n, thr = max(1, int(rec["done_hold"])), float(rec["done_threshold"])
    est, tru = B["est"].float(), B["truth"].float()
    pos_err = (est[:, 0:3] - tru[:, 0:3]).norm(dim=-1) * 50.0  # x20 m -> mm
    Ra, Rb = rot6_to_matrix(est[:, 3:9]), rot6_to_matrix(tru[:, 3:9])
    rot_err = torch.rad2deg(torch.acos((((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
    lay_est, lay_tru = torch.rad2deg(est[:, 9]), torch.rad2deg(tru[:, 9])
    ang = lambda R: torch.rad2deg(torch.acos(((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))  # noqa: E731
    rot_est, rot_tru = ang(Ra), ang(Rb)  # off the skill's goal orientation
    W, H, SW = 1280, 720, 512  # canvas, side panel width
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    WHITE, GREY, GREEN, RED, CYAN, YEL = (235, 235, 235), (140, 140, 140), (90, 220, 110), (80, 90, 240), (230, 200, 80), (60, 210, 240)

    def txt(img, s_, xy, sc=0.5, col=WHITE, th=1):
        cv2.putText(img, s_, xy, FONT, sc, col, th, cv2.LINE_AA)

    def dashboard(frame, f):
        """(h, w, 3) RGB scene -> (720, 1280, 3) RGB: scene + layer plot (left), tactile / state machine / estimate (right)."""
        c = np.full((H, W, 3), 24, np.uint8)  # drawn in BGR, converted at the end
        c[:576, :768] = cv2.resize(frame[..., ::-1], (768, 576))
        txt(c, "BLIND: no vision, tactile + proprioception only", (12, 26), 0.6, YEL, 1)
        # --- layer angle (error to the target) estimate vs simulator, last 6 s
        x0, y0, pw, ph = 50, 590, 700, 115
        cv2.rectangle(c, (x0, y0), (x0 + pw, y0 + ph), (60, 60, 60), 1)
        span = int(6.0 / rec["step_dt"])
        lo = max(0, f - span)
        for v_ in (0, -45, -90, 45, 90):
            yy = int(y0 + ph / 2 - v_ / 100 * ph / 2)
            cv2.line(c, (x0, yy), (x0 + pw, yy), (45, 45, 45), 1)
            txt(c, f"{v_:+d}" if v_ else "0", (x0 - 40, yy + 4), 0.35, GREY)
        for series, col in ((lay_tru, GREEN), (lay_est, CYAN)):
            seg = series[lo : f + 1].clamp(-100, 100).numpy()
            pts = np.stack([x0 + (np.arange(len(seg)) + span - (f - lo)) * pw / span, y0 + ph / 2 - seg / 100 * ph / 2], -1)
            cv2.polylines(c, [pts.astype(np.int32)], False, col, 2, cv2.LINE_AA)
        txt(c, "layer angle to target [deg], last 6 s:", (x0, y0 - 4), 0.42, WHITE)
        txt(c, "simulator", (x0 + 330, y0 - 4), 0.42, GREEN)
        txt(c, "tactile estimate", (x0 + 420, y0 - 4), 0.42, CYAN)
        # --- tactile images: 2 hands x 5 pads, 60x60 TacMap (pooled 30x30 in the record) + pad force
        X = 768 + 12
        txt(c, "Tactile (Sharpa TacMap 60x60 per fingertip)", (X, 24), 0.5, YEL)
        img = B.get("tac_img")
        for h_, side in enumerate(("left", "right")):
            yy = 40 + h_ * 132
            txt(c, side, (X, yy + 12), 0.45, GREY)
            for k_, tip in enumerate(FINGERTIPS):
                xx = X + 44 + k_ * 92
                tile = np.zeros((84, 84), np.uint8) if img is None else cv2.resize(img[f, h_ * 5 + k_].numpy(), (84, 84), interpolation=cv2.INTER_NEAREST)
                c[yy : yy + 84, xx : xx + 84] = cv2.applyColorMap(tile, cv2.COLORMAP_INFERNO)
                cv2.rectangle(c, (xx, yy), (xx + 84, yy + 84), (70, 70, 70), 1)
                fz = float(B["force"][f, h_, k_])
                txt(c, f"{tip[:5]} {fz:4.1f}N", (xx, yy + 100), 0.37, WHITE if fz > 0.05 else GREY)
        # --- state machine: plan strip, current skill, completion probability, hold counter, trigger
        Y = 312
        txt(c, "State machine", (X, Y), 0.5, YEL)
        cur = int(B["ptr"][f])
        for j in range(-2, 5):
            i_ = cur + j
            if not 0 <= i_ < len(plan):
                continue
            bx = X + (j + 2) * 70
            col = YEL if j == 0 else (GREEN if j < 0 else GREY)
            cv2.rectangle(c, (bx, Y + 10), (bx + 64, Y + 36), col, 2 if j == 0 else 1)
            txt(c, str(plan[i_])[:8], (bx + 4, Y + 28), 0.4, col)
        txt(c, f"skill {min(cur + 1, len(plan))}/{len(plan)}", (X, Y + 56), 0.45)
        sk = int(B["skill"][f])
        t_ = float(B["t_skill"][f])
        dur = float(rec["k_duration"][sk])
        phase = "EXECUTE" if t_ < dur else "WAIT FOR DONE"
        txt(c, f"{SK[sk]}  t={t_:4.2f}s / {dur:.2f}s", (X + 100, Y + 56), 0.42, CYAN)
        txt(c, phase, (X + 330, Y + 56), 0.42, WHITE if t_ < dur else YEL)
        pd = float(B["done"][f])
        held = int(B["above"][f])
        bx, by = X, Y + 70
        cv2.rectangle(c, (bx, by), (bx + 300, by + 16), (70, 70, 70), 1)
        cv2.rectangle(c, (bx, by), (bx + int(300 * pd), by + 16), GREEN if pd > thr else (120, 120, 120), -1)
        cv2.line(c, (bx + int(300 * thr), by - 3), (bx + int(300 * thr), by + 19), WHITE, 1)
        txt(c, f"P(done) {pd:.2f}", (bx + 310, by + 13), 0.45)
        for j in range(hold_n):
            cx = bx + 6 + j * 16
            cv2.circle(c, (cx, by + 34), 5, GREEN if j < held else (70, 70, 70), -1)
        txt(c, f"hold {min(held, hold_n)}/{hold_n}", (bx + hold_n * 16 + 10, by + 39), 0.42)
        nxt = f + 1 < len(B["ptr"]) and int(B["ptr"][f + 1]) > cur
        if nxt:
            txt(c, "TRIGGER -> next skill", (bx + 310, by + 39), 0.5, GREEN, 2)
        # --- estimated cube state vs simulator
        Y2 = Y + 150
        txt(c, "Cube state: tactile estimate vs simulator", (X, Y2), 0.5, YEL)
        rows = [("pos [mm]", f"{est[f, 0] * 50:+.1f} {est[f, 1] * 50:+.1f} {est[f, 2] * 50:+.1f}",
                 f"{tru[f, 0] * 50:+.1f} {tru[f, 1] * 50:+.1f} {tru[f, 2] * 50:+.1f}", f"{float(pos_err[f]):.1f} mm"),
                ("tilt", f"{float(rot_est[f]):5.1f} deg", f"{float(rot_tru[f]):5.1f} deg", f"{float(rot_err[f]):.1f} deg"),
                ("layer", f"{float(lay_est[f]):+6.1f} deg", f"{float(lay_tru[f]):+6.1f} deg", f"{abs(float(lay_est[f] - lay_tru[f])):.1f} deg")]
        txt(c, "estimate", (X + 80, Y2 + 24), 0.42, CYAN)
        txt(c, "simulator", (X + 240, Y2 + 24), 0.42, GREEN)
        txt(c, "error", (X + 405, Y2 + 24), 0.42, GREY)
        for r_, (name, a_, b_, err) in enumerate(rows):
            yy = Y2 + 48 + r_ * 22
            txt(c, name, (X, yy), 0.42, GREY)
            txt(c, a_, (X + 80, yy), 0.4, CYAN)
            txt(c, b_, (X + 240, yy), 0.4, GREEN)
            txt(c, err, (X + 405, yy), 0.4)
        txt(c, f"t = {f * rec['step_dt']:6.1f} s   moves done {int(n_all[f])}", (X, H - 16), 0.45, GREY)
        return np.ascontiguousarray(c[..., ::-1])
R_ready = mu.matrix_from_quat(env.ready["cube_quat"][idx])[0]
start, performed = rec["plans"][e]["start"], rec["performed"][e]
q_all, c_all, n_all = rec["hand_q"][e], rec["cube"][e], rec["n_performed"][e]
# frames until the env stopped (its state stays constant afterwards)
last = len(q_all)
for f in range(len(q_all) - 1, 0, -1):
    if not torch.equal(c_all[f], c_all[f - 1]) or not torch.equal(q_all[f], q_all[f - 1]):
        last = f + 1
        break
if a.until > 0:
    last = min(last, int(a.until / rec["step_dt"]))
for _ in range(5):  # renderer warm-up (the first frames come out black)
    env.render()
import imageio.v2 as imageio
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
writer = imageio.get_writer(a.out, fps=a.fps)  # streamed: a dashboard video does not fit in memory
root = env.cube.data.default_root_state[ids].clone()
for f in range(last):
    for h in range(2):
        env._write_hand(ids, h, q_all[f, h].float().unsqueeze(0).to(dev))
    c = c_all[f].to(dev)
    root[:, :7] = c[:7].unsqueeze(0)
    root[:, :3] += env.scene.env_origins[ids]
    root[:, 7:] = 0.0
    env.cube.write_root_state_to_sim(root, env_ids=ids)
    tw = c[7:8].view(1, 1)
    env.cube.write_joint_state_to_sim(tw, torch.zeros_like(tw), joint_ids=[env.twist_id], env_ids=ids)
    # the stickers ride on the two bodies, so repaint only where the colors change: a finished skill (logical state) or
    # a face re-binding (the core frame jumps by a cube symmetry, the twist snaps back); repainting every frame would
    # snap the mapping to the nearest symmetry halfway through an x / roll and show a wrong pattern until it ends
    # (a finished turn whose layer has not snapped back yet, e.g. the last one, already shows on the turned top body)
    if f == 0 or (n_all[f] != n_all[f - 1] and c_all[f, 7].abs() < 0.8) or (c_all[f, 7] - c_all[f - 1, 7]).abs() > 0.8 or (
        2 * torch.acos((c_all[f, 3:7].float() * c_all[f - 1, 3:7].float()).sum().abs().clamp(max=1.0)) > 0.5):
        R_core = mu.matrix_from_quat(c[3:7].unsqueeze(0))[0]
        painter.recolor((R_ready.T @ R_core).cpu().numpy(), Cube(start).apply(performed[: int(n_all[f])]).s)
    frame = env.render()[..., :3]
    writer.append_data(frame if blind is None else dashboard(frame, f))
    if f % 300 == 0:
        print(f"[replay] frame {f}/{last}", flush=True)
writer.close()
print(f"[replay] {last} frames ({last * rec['step_dt']:.0f} s sim) -> {a.out}, solved {Cube(start).apply(performed).solved()}", flush=True)
os._exit(0)
