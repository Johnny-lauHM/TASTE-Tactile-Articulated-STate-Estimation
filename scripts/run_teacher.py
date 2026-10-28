"""Run the scripted teacher (rubik_lab.teacher) and report per-move success; optionally record a video of env 0.
    python scripts/run_teacher.py --headless --num_envs 256 --moves "R,R',L,L'" [--enable_cameras --video]
"""

import argparse
import os
from collections import Counter, defaultdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--moves", default="R,R',L,L'")
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--video", action="store_true")
parser.add_argument("--cache", default=None)
parser.add_argument("--flick", default="", help="moves executed as finger tricks, e.g. \"U,U',D,D'\"")
parser.add_argument("--trace", action="store_true", help="print env 0 phase changes")
parser.add_argument("--out", default="logs/demo/teacher.mp4")
parser.add_argument("--eye", type=float, nargs=3, default=None, help="camera offset from the cube centre [m]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402

import rubik_lab  # noqa: E402, F401
from rubik_lab.moves import ALL_MOVES  # noqa: E402
from rubik_lab.tasks.bimanual_twist.env_cfg import CUBE_CENTER, BimanualTwistEnvCfg  # noqa: E402
from rubik_lab.flick import FLICKS, to_cube_frame  # noqa: E402
from rubik_lab.hand_kinematics import quat_to_mat  # noqa: E402
from rubik_lab.constants import FINGER_JOINTS  # noqa: E402
from rubik_lab.cube_contacts import LINKS as CC_LINKS  # noqa: E402
from rubik_lab.teacher import F_DONE, PHASES, ScriptedTeacher  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = BimanualTwistEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.moves = tuple(args.moves.split(","))
cfg.gravity_curriculum = False
cfg.wrist_prior = False  # the teacher commands the roll itself
if args.cache:
    cfg.grasp_cache_path = args.cache
if args.flick:
    cfg.flick_moves = tuple(args.flick.split(","))
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (-0.42, -0.25, 0.42) if args.eye is None else tuple(float(c) + float(o) for c, o in zip(CUBE_CENTER, args.eye))
cfg.viewer.lookat = CUBE_CENTER
env = gym.make("Rubik-Twist-Bimanual-Sharpa-v0", cfg=cfg, render_mode="rgb_array" if args.video else None)
u = env.unwrapped
u.record = True
u.trace_debug = args.trace
env.reset()
teacher = ScriptedTeacher(u)
teacher.reset(torch.arange(u.num_envs, device=u.device))
frames = []
results = defaultdict(Counter)  # move -> outcome counts
fail_phase = defaultdict(Counter)
durations = defaultdict(list)
timeout_why = defaultdict(Counter)
errlog = defaultdict(list)
max_phase = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
import time as _time
_t0 = _time.time()
for k in range(args.steps):
    if k % 50 == 0 and k:
        print(f"[progress] step {k}/{args.steps} {(_time.time() - _t0) / k:.2f}s/step ended={sum(sum(c.values()) for c in results.values())}", flush=True)
    prev_phase = int(teacher.phase[0])
    act = teacher.act()
    env.step(act)
    if args.trace and (int(teacher.phase[0]) != prev_phase or k % 20 == 0 or PHASES[int(teacher.phase[0])] in ("f_push", "f_release")):
        pads = (u.cube_contacts.forces().sum(1)[0] > 0.05).sum(-1).tolist()
        cz = u._cube_state()[0][0, 2].item()
        tw = np.degrees(float(u._cube_state()[2][0])); wv = float(u.cube.data.body_ang_vel_w[0, u.core_id].norm())
        rot_act = [np.degrees(float(torch.linalg.norm(u._ee_home_error(sd)[1][0]))) for sd in ("left", "right")]
        rot_tgt = [np.degrees(float(mu.quat_error_magnitude(u.ee_quat_tgt[sd][0:1], u.ee_home_quat[sd][0:1]))) for sd in ("left", "right")]
        if hasattr(teacher, "f_kmax"):
            rd = (teacher.f_kmax - teacher.planner.n_app).clamp(min=0).float() * 5
            fk_dbg = f" plan_reach={rd[0].item():.0f}deg (env-mean {rd.mean().item():.0f}) prog={teacher.f_prog[0].item():.0f}"
            mname = ALL_MOVES[int(u.move_id[0])]
            if mname in FLICKS:
                fs, ff, fcf = FLICKS[mname][:3]
                hh = 0 if fs == "left" else 1
                cols = [i for i, j in enumerate(FINGER_JOINTS) if j.startswith(ff)]
                qa = u.hands[fs].data.joint_pos[0, u.finger_ids[fs]][cols]
                qt = u.finger_tgt[fs][0][cols]
                pc, Rcc, fq, _ = to_cube_frame(u, fs, mode="actual")
                with torch.no_grad():
                    padc, _ = teacher.planner.fk[fs].pads(teacher.planner.fk[fs].forward(pc[0:1], Rcc[0:1], fq[0:1]))[ff]
                padc = padc[0].cpu().numpy()
                ftop = u.cube_contacts.forces()[0, 1]  # (2 hands, 27 links) on the moving layer
                fin = [li for li, ln in enumerate(CC_LINKS) if ln.startswith(ff)]
                kk = min(int(teacher.f_prog[0]), int(teacher.f_kmax[0]))
                plan = teacher.f_pad[0, kk].cpu().numpy() if hasattr(teacher, "f_pad") else np.zeros(3)
                # pad of the commanded configuration (wrist target + finger target) in the cube frame
                fk_dbg += f" plan=({plan[0] * 1000:+.0f},{plan[1] * 1000:+.0f},{plan[2] * 1000:+.0f})"
                if PHASES[int(teacher.phase[0])] == "f_push" and int(teacher.phase[0]) != prev_phase:
                    names = ["thumb", "index", "middle", "ring", "pinky"]
                    lifted = {sd: [f for f in names if (teacher.f_lift[0, hi] * teacher.finger_cols[f]).sum() > 0] for hi, sd in enumerate(("left", "right"))}
                    ftop0 = u.cube_contacts.forces()[0, 1]
                    on_layer = {sd: [f for f in names if ftop0[hi, [li for li, ln in enumerate(CC_LINKS) if ln.startswith(f)]].sum() > 0.05] for hi, sd in enumerate(("left", "right"))}
                    jn = [FINGER_JOINTS[i].split("_", 1)[1] for i in cols]
                    print(f"  [plan] on-layer {on_layer} lifted {lifted} | {ff} joints tgt-act " + " ".join(f"{n}:{t:+.2f}/{a_:+.2f}" for n, t, a_ in zip(jn, qt.tolist(), qa.tolist())) + f" lower {[round(float(x), 2) for x in u.finger_lower[fs][0][cols]]} upper {[round(float(x), 2) for x in u.finger_upper[fs][0][cols]]}", flush=True)
                if PHASES[int(teacher.phase[0])] == "f_push" and int(teacher.f_prog[0]) % 3 == 0:
                    print(f"  [push] {ff} joints tgt-act " + " ".join(f"{t:+.2f}/{a_:+.2f}" for t, a_ in zip(qt.tolist(), qa.tolist())) + f" cube d=({(u._cube_state()[0][0] - u.cube_center[0]).cpu().numpy().round(3)}) wrist dp={(teacher.f_dp[0, min(int(teacher.f_prog[0]), teacher.f_dp.shape[1] - 1)] * 1000).cpu().numpy().round(1)}mm", flush=True)
                fk_dbg += (f" | q_err={(qt - qa).abs().max().item():.2f} pad=({padc[0] * 1000:+.0f},{padc[1] * 1000:+.0f},{padc[2] * 1000:+.0f})mm"
                           f" ang={np.degrees(np.arctan2(padc[1], padc[0])):+.0f} f_trick={ftop[hh, fin].sum().item():.1f}N f_other={ftop.sum().item() - ftop[hh, fin].sum().item():.1f}N")
                Fw = u.cube_contacts.sensors["top"].data.force_matrix_w[0, 0].view(2, len(CC_LINKS), 3)[hh, fin].sum(0)  # (3,) env frame
                padw = u.cube.data.body_pos_w[0, u.core_id] + torch.einsum("ij,j->i", quat_to_mat(u._cube_state()[1][0:1])[0], torch.tensor(padc, device=u.device))
                tau = torch.linalg.cross(padw - u.cube.data.body_pos_w[0, u.core_id], Fw) @ u.roll_axis[0]
                fk_dbg += f" tau_axis={tau.item() * 1000:+.0f}mNm"
                # torque from every other hand link on the layer (application point ~ the link origin): who resists?
                Fall = u.cube_contacts.sensors["top"].data.force_matrix_w[0, 0].view(2, len(CC_LINKS), 3)
                core_w = u.cube.data.body_pos_w[0, u.core_id]
                tau_o = 0.0
                for hh2, sd2 in enumerate(("left", "right")):
                    bid = [u.hands[sd2].body_names.index(f"{sd2}_{ln}") for ln in CC_LINKS]
                    pw = u.hands[sd2].data.body_pos_w[0, bid]  # (27, 3)
                    tl = torch.linalg.cross(pw - core_w, Fall[hh2]) @ u.roll_axis[0]  # (27,)
                    mask_tr = torch.tensor([hh2 == hh and li in fin for li in range(len(CC_LINKS))], device=u.device)
                    tau_o += tl[~mask_tr].sum().item()
                fk_dbg += f" tau_other={tau_o * 1000:+.0f}mNm layer_w={float(u._cube_state()[3][0]):+.2f} press={teacher.f_press[0].item() * 1000:.0f}mm lead={teacher.f_lead[0].item():.1f}"
        else:
            fk_dbg = ""
        print(f"  t={k * u.step_dt:5.2f}s move={ALL_MOVES[int(u.move_id[0])]:3s} phase={PHASES[int(teacher.phase[0])]:10s}{fk_dbg} "
              f"err={np.degrees(float(u.signed_err[0])):+7.1f}deg theta={np.degrees(teacher.theta[0].cpu().numpy()).round(0).tolist()} "
              f"layer={tw:+6.1f} angvel={wv:.2f} done={bool(u.move_done[0])} links L/R={pads} z={cz:.3f} wrist rot actual/target L {rot_act[0]:.0f}/{rot_tgt[0]:.0f} R {rot_act[1]:.0f}/{rot_tgt[1]:.0f}", flush=True)  # fmt: skip
    max_phase = torch.max(max_phase, teacher.phase)
    snap = u.snapshot
    gave_up = (teacher.phase == F_DONE) & ~snap["primitive_done"]  # finger trick out of retries: count as a timeout
    ended = snap["primitive_done"] | snap["dropped"] | (snap["episode_step"] >= u.max_episode_length - 1) | gave_up
    for i in ended.nonzero().flatten().tolist():
        m = ALL_MOVES[int(snap["move_id"][i])]
        out = "success" if snap["primitive_done"][i] else ("dropped" if snap["dropped"][i] else "timeout")
        results[m][out] += 1
        ea = np.degrees(teacher.err_at[i].cpu().numpy())
        errlog[out].append([ea[PHASES.index("release")], ea[PHASES.index("return")], ea[PHASES.index("regrip")], np.degrees(float(snap["twist_error"][i]))])
        if out == "success":
            durations[m].append(int(snap["episode_step"][i]) * u.step_dt)
        else:
            fail_phase[m][PHASES[int(teacher.phase[i])]] += 1
            if out == "timeout":  # which primitive_done condition is missing
                why = []
                if not snap["move_done"][i]:
                    why.append(f"layer err {np.degrees(float(snap['twist_error'][i])):+.0f}deg")
                if float(u.home_err[i]) > 0.1 + np.radians(10):
                    why.append(f"home_err {float(u.home_err[i]):.2f}")
                pads = (u.layer_f[i].sum(-1) > cfg.tactile_contact_threshold).sum(-1).tolist()
                if min(pads) < cfg.home_min_pads:
                    why.append(f"pads {pads}")
                timeout_why[m][" & ".join(w.split(" ")[0] + (" " + w.split(" ")[1] if w.startswith("pads") else "") for w in why) or "?"] += 1
    if ended.any():
        ids = ended.nonzero().flatten()
        teacher.reset(ids)
        max_phase[ids] = 0
    if args.video and k % 2 == 0:
        frames.append(env.render())
for m in sorted(results):
    r = results[m]
    tot = sum(r.values())
    print(f"[teacher] {m:3s}: {tot:4d} episodes | success {r['success'] / tot:6.1%} | dropped {r['dropped'] / tot:6.1%} | timeout {r['timeout'] / tot:6.1%}"
          f" | mean time {np.mean(durations[m]) if durations[m] else float('nan'):.2f}s | failures stuck in {dict(fail_phase[m])} | timeout because {dict(timeout_why[m])}", flush=True)  # fmt: skip
for out, rows in errlog.items():
    r = np.abs(np.array(rows))
    print(f"[teacher] |layer err| deg (median) for {out:8s}: at release {np.nanmedian(r[:, 0]):5.1f}  at return {np.nanmedian(r[:, 1]):5.1f}  "
          f"at regrip {np.nanmedian(r[:, 2]):5.1f}  at end {np.nanmedian(r[:, 3]):5.1f}   (n={len(r)})", flush=True)  # fmt: skip
if args.video:
    import imageio.v2 as imageio

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out, frames, fps=20)
    print(f"[teacher] video -> {out}", flush=True)
env.close()
app.close()
