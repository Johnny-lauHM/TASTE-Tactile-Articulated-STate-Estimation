"""Full-solve executor: scramble -> solver -> physical skill sequence -> chained skill policies, no resets.

Every env solves its own random scramble. The initial complete cube state is the one observation the robot gets (here
the known scramble); afterwards skills are triggered by the completion signal (--trigger truth: simulation ground truth
used as an oracle; --trigger detector: the learned tactile/proprio detector, ground truth only scores). The physical
result of every face turn is read from the layer angle at completion (rounded to quarter turns) and replayed on the
logical cube, so an over- or under-turn shows up as a wrong final state, never as a silent success.

    python scripts/solve/execute.py --headless --policy roll=<ckpt> --policy flick=<ckpt> --num_envs 64
"""
import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--policy", action="append", default=[], help="family=checkpoint (families: flick, roll, regrip; an old bimanual_twist flick checkpoint is detected and adapted)")
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--scramble_len", type=int, default=25)
p.add_argument("--max_moves", type=int, default=0, help="debug: truncate the face-turn solution")
p.add_argument("--formula", default="", help="execute this hand-frame skill/face-turn formula instead of a solve")
p.add_argument("--skill_timeout", type=float, default=3.0)
p.add_argument("--recover", action="store_true", help="sighted recovery: retry stalled turns, replan from the observed state")
p.add_argument("--align_tol", type=float, default=0.14, help="rad: a layer this close to a quarter counts as aligned")
p.add_argument("--max_retries", type=int, default=2)
p.add_argument("--max_replans", type=int, default=6)
p.add_argument("--face_skills", default="U,U',R,R',L,L'", help="face turns the robot can execute directly")
p.add_argument("--trigger", default="truth", choices=["truth", "detector", "vision"])  # vision: env.success_vis (--set cube_vis=True)
p.add_argument("--student", default=None, help="student checkpoint: one recurrent proprio+tactile policy for all skills")
p.add_argument("--estimator", default=None, help="blind: cube-state estimator checkpoint (solve.estimator); every controller acts on its estimate")
p.add_argument("--done_window", type=int, default=0, help=">0: the detector fires on --done_hold positive steps among the last done_window (tolerates single-step dips) instead of consecutive ones")
p.add_argument("--late_s", type=float, default=0.0, help=">0: from this skill time on the detector needs only --late_hold steps (late completions are rarely premature)")
p.add_argument("--late_hold", type=int, default=3)
p.add_argument("--blind_retry", action="store_true", help="blind recovery: a face turn that times out with the ESTIMATED layer between quarters is retried")
p.add_argument("--blind_continue", action="store_true", help="blind: a premature completion does not end the run (the robot cannot "
               "know): the physical outcome is applied and the plan goes on; solved = the true final cube state")
p.add_argument("--timeout_continue", action="store_true", help="a skill timeout does not end the run (the robot's own clock): the physical "
               "outcome is applied and the plan goes on; solved = the true final cube state")
p.add_argument("--log_qd", action="store_true", help="per skill: finger joint speed statistics (vs the real hand's ~3 rad/s "
               "POSITION / ~4.3 rad/s MIT cap), written to <out>/qd_stats.json")
p.add_argument("--timeout_settle", action="store_true", help="with --timeout_continue: after a timed-out skill, run a hold "
               "(both hands back to the ready grip, until it completes or times out) before the next skill")
p.add_argument("--timeout_regrip", action="store_true", help="with --timeout_continue: after a timed-out skill (other than a regrip) "
               "each hand re-grasps in turn (re-centers the cube) before the plan goes on")
p.add_argument("--redo_missed", action="store_true", help="a face turn that ends (completion or time-out) with the layer aligned on a "
               "WRONG quarter (the turn did not happen) is repeated (flicks) instead of going on; blind: judged on the estimated layer")
p.add_argument("--finish_tol", type=float, default=0.0, help=">0 [deg]: a face turn that completes with the layer more than this "
               "off a quarter is repeated (realign before re-gripping); judged on the estimate (blind) unless --finish_oracle")
p.add_argument("--finish_oracle", action="store_true", help="--finish_tol judged on the true layer angle (diagnostic)")
p.add_argument("--finish_mode", default="target", choices=["target", "residual"], help="target: error to the intended quarter "
               "(also catches missed turns); residual: off the nearest quarter, only with the layer within 45 deg of the target")
p.add_argument("--retry_reset", action="store_true", help="a retried skill also restarts the completion detector window and the "
               "policies' state (as an advance); off = the frozen system's behavior")
p.add_argument("--done_threshold", type=float, default=0.5)
p.add_argument("--done_hold", type=int, default=-1, help="consecutive detector steps above threshold (-1: the success window, cfg.stable_s, + 1)")
p.add_argument("--ready_cache", default="cache/solve_ready_udr_v2.npz")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max_time", type=float, default=0.0, help=">0: stop after this many s of sim time (diagnostics; running envs count as time-outs)")
p.add_argument("--out", default="outputs/solve/execute")
p.add_argument("--video", action="store_true")
p.add_argument("--video_env", type=int, default=0, help="env the video follows")
p.add_argument("--record", default="", help="save every env's state every 2nd step (replay any env: scripts/solve/replay.py)")
p.add_argument("--plan_reliability", default="", help="comma list of handoff_matrix jsons / executor reports: plan by minimal "
               "failure probability (cube_logic.ReliabilityCost of their transition counts) instead of nominal time")
p.add_argument("--warmup", type=float, default=0.0, help="s of holding the ready grip (zero action) after the reset, before the first skill")
p.add_argument("--set", action="append", default=[])
p.add_argument("--mod", action="append", default=[], help="constants of rubik_lab.solve.skills, e.g. ROLL_SHIFT=0.016")
p.add_argument("--reseat_tol", type=float, default=0.35, help="rad: re-seat (and let the detents re-align) layers stopped this close to a quarter")
p.add_argument("--reseat", action="store_true", help="sighted recovery: a skill that ends tilted / unfinished but with the layers aligned re-seats the cube (regrip_left, regrip_right) and replans")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.video:
    a.enable_cameras = True
app = AppLauncher(a).app
import sys as _sys, traceback as _tb, os as _os
def _die(t, v, b):
    _tb.print_exception(t, v, b); _sys.stdout.flush(); _sys.stderr.flush(); _os._exit(1)
_sys.excepthook = _die  # the simulation app hangs on close after an exception

import ast  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rubik_lab.solve import skills as S  # noqa: E402
for _kv in a.mod:
    _k, _v = _kv.split("=", 1)
    setattr(S, _k, __import__("ast").literal_eval(_v))
from rubik_lab.solve.cube_logic import Cube, best_solution, bound_layer_moves, random_scramble  # noqa: E402
from rubik_lab.solve.binding import quarter_residual  # noqa: E402
from rubik_lab.solve.env import SolveEnv  # noqa: E402
from rubik_lab.solve.env_cfg import SolveEnvCfg  # noqa: E402
from rubik_lab.solve.policies import load_policies  # noqa: E402

FAMILY = {s: ("flick" if s in S.FLICK else "roll" if s in S.ROLL_DELTA else "regrip" if s in S.REGRIP_HAND else "hold") for s in S.SKILLS}

cfg = SolveEnvCfg()
cfg.scene.num_envs = a.num_envs
cfg.ready_cache = a.ready_cache
cfg.episode_length_s = 1e6
cfg.terminate_on_success = False
cfg.skills = ("hold",)
cfg.seed = a.seed
for kv in a.set:
    k, v = kv.split("=", 1)
    setattr(cfg, k, ast.literal_eval(v))
cfg.cube_est = bool(a.estimator)
if a.video:
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = a.video_env
    cfg.viewer.eye = (-0.30, -0.28, 0.42)
    cfg.viewer.lookat = (0.0, 0.0, 0.3)
    cfg.viewer.resolution = (640, 480)
if a.estimator and torch.load(a.estimator, map_location="cpu", weights_only=False)["kwargs"].get("img_feat", 0):
    cfg.tactile.image = True  # the estimator reads the 60x60 TacMap pad images
env = SolveEnv(cfg, render_mode="rgb_array" if a.video else None)
env.chain_mode = True
dev, n = env.device, env.num_envs
obs_key = "student" if a.student else "policy"
if a.student:
    policies = load_policies(env, {"all": a.student}, obs_key)
else:
    policies = load_policies(env, dict(kv.split("=", 1) for kv in a.policy), obs_key)

# ---------------------------------------------------------------- plans
rng = np.random.default_rng(a.seed)
plans = []
trans_cost = None
if a.plan_reliability:
    from rubik_lab.solve.cube_logic import ReliabilityCost
    trans_cost = ReliabilityCost.from_files(a.plan_reliability.split(","))
for i in range(n):
    if a.formula:
        scramble, sol = [], a.formula.split()
        skills = sol
        start = Cube()
    else:
        scramble = random_scramble(a.scramble_len, rng)
        start = Cube().apply(scramble)
        sol, skills, _, _ = best_solution(start, face_skills=tuple(a.face_skills.split(",")), trans_cost=trans_cost)
        if a.max_moves:
            skills = skills[: a.max_moves]
    plans.append(dict(scramble=scramble, solution=sol, skills=skills, start=start.s))

ptr = torch.zeros(n, dtype=torch.long, device=dev)  # skills completed (all plans, including replans)
queue = [list(pl["skills"]) for pl in plans]  # remaining skills per env (replaced on a replan)
status = ["running"] * n
records = [[] for _ in range(n)]
mis_done = [False] * n  # cfg.cube_jam_deg: a layer was out of line when the run ended (not solved)
performed = [[] for _ in range(n)]
retries = [0] * n
replans = [0] * n
FACE_TURN = set(S.LAYER_TARGET)


def missed_turn(i, k):  # the layer sits on a quarter that is not the target: the turn did not happen
    if k not in S.FLICK:  # finger flicks only: a repeated wrist roll would start from a rolled wrist
        return False
    err = float(env.est_cube[i, 9]) if est is not None else float(env.layer_err[i])
    if a.trigger == "vision":  # the layer angle as the camera reports it
        err = float(env.layer_target[i] - env.perceived()["twist"][i])
    err = abs(float(np.arctan2(np.sin(err), np.cos(err))))
    return err > 0.25 * np.pi and abs(float(quarter_residual(torch.tensor(err)))) < a.align_tol
obs, _ = env.reset()
est = None
if a.estimator:
    from rubik_lab.solve.estimator import Estimator
    est = Estimator(a.estimator, n, dev)
    env.est_cube = est.step(obs)
for _ in range(round(a.warmup / env.step_dt)):  # the freshly written grip settles before the first skill
    obs, *_ = env.step(torch.zeros(n, 56, device=dev))
    if est is not None:
        env.est_cube = est.step(obs)


def set_next(ids):  # next planned skill of each env (obs_next_skill), -1 when none
    env.next_skill[ids] = torch.tensor([S.SKILL_INDEX[queue[i][1]] if len(queue[i]) > 1 else -1 for i in ids], device=dev)


first = torch.tensor([S.SKILL_INDEX[q[0]] if q else 0 for q in queue], device=dev)
env.start_skill(torch.arange(n, device=dev), first)
set_next(list(range(n)))
obs = env._get_observations()
if est is not None:  # the last warm-up step again, with the first skill commanded (as a chain advance in training)
    env.est_cube = est.step(obs, ids=torch.arange(n, device=dev))
    obs = env._get_observations()
frames = []
rec_q, rec_c, rec_n = [], [], []
rec_x = {}  # blind runs: what the system perceived and decided (dashboard of scripts/solve/replay.py)
painter = None
if a.video:
    import isaaclab.utils.math as mu
    from rubik_lab.solve.visual import StickerPainter
    painter = StickerPainter()

    def paint():
        R_ready = mu.matrix_from_quat(env.ready["cube_quat"][env.ready_idx[a.video_env]].unsqueeze(0))[0]
        R_core = mu.matrix_from_quat(env.cube.data.root_quat_w[a.video_env].unsqueeze(0))[0]
        painter.recolor((R_ready.T @ R_core).cpu().numpy(), Cube(plans[a.video_env]["start"]).apply(performed[a.video_env]).s)
    paint()
t0 = time.time()
step = 0
if a.trigger == "detector" and not (a.student or a.estimator):
    raise ValueError("the detector trigger uses the student's / estimator's completion head")
if a.trigger == "vision" and not env.cfg.cube_vis:
    raise ValueError("the vision trigger needs the camera model (--set cube_vis=True)")
above = torch.zeros(n, dtype=torch.long, device=dev)
qd_stats = {}
hist = torch.zeros(n, max(1, a.done_window), dtype=torch.bool, device=dev)  # last detector outputs (done_window)


def physical_outcome(i, k):
    """quarter turns of the bound layer (intended or not) as a hand-frame face turn, of the 3-body cube's middle slab as a
    slice turn (the rebind erases both), plus the x rotation."""
    twist = float(env.cube.data.joint_pos[i, env.twist_id])
    mid = float(env.cube.data.joint_pos[i, env.mid_id]) if env.mid_id is not None else 0.0
    performed[i] += bound_layer_moves(S.BIND_FACE[k], twist, mid)
    if k in S.CUBE_QUARTERS_ABOUT_R:  # did the whole cube end nearer the rotated or the original orientation?
        import isaaclab.utils.math as mu_
        lq = env._logical_quat()[i : i + 1]
        if mu_.quat_error_magnitude(lq, env.cube_goal[i : i + 1]) < mu_.quat_error_magnitude(lq, env.cube_ref[i : i + 1]):
            performed[i].append(k)
    return twist


def replan(i):
    """sighted recovery: observe the true cube state and the hand roll states, solve again from there."""
    cube = Cube(plans[i]["start"]).apply(performed[i])
    roll = env.roll[i].round().long().tolist()
    if cube.solved():
        queue[i] = [f"regrip_{h}" for h, r in zip(("left", "right"), roll) if r != 0]
    else:
        _, sk, _, _ = best_solution(cube, face_skills=tuple(a.face_skills.split(",")), init_roll=tuple(roll))
        queue[i] = sk
    replans[i] += 1


while any(s == "running" for s in status):
    if a.max_time > 0 and step * env.step_dt >= a.max_time:  # diagnostic cut: the running envs end as time-outs
        status = ["timeout" if s_ == "running" else s_ for s_ in status]
        break
    with torch.no_grad():
        act = torch.zeros(n, 56, device=dev)
        for fam, pol in policies.items():
            if fam == "all":
                act = pol(obs[obs_key])
                continue
            m = torch.tensor([FAMILY[S.SKILLS[k]] == fam.split("_")[0] for k in env.skill.tolist()], device=dev)  # flick_legacy -> flick
            if m.any():
                act[m] = pol(obs[obs_key])[m]
        obs, _, term, trunc, _ = env.step(act)
        if est is not None:
            env.est_cube = est.step(obs)
    step += 1
    if step % 400 == 0:  # progress: skills done, replans, status counts
        st = {x: status.count(x) for x in set(status)}
        solved_now = sum(Cube(plans[i]["start"]).apply(performed[i]).solved() for i in range(n))
        print(f"[exec] t={step * env.step_dt:.0f}s status {st} skills done med {int(ptr.float().median())} max {int(ptr.max())} "
              f"replans med {int(np.median(replans))} cubes solved now {solved_now}", flush=True)
    if a.record and step % 2 == 0:
        rec_q.append(torch.stack([env.hands[sd].data.joint_pos for sd in ("left", "right")], 1).half().cpu())
        rec_c.append(torch.cat([env.cube.data.root_pos_w - env.scene.env_origins, env.cube.data.root_quat_w,
                                env.cube.data.joint_pos[:, env.twist_id : env.twist_id + 1]], -1).cpu())
        rec_n.append([len(p_) for p_ in performed])
        if est is not None:
            x = dict(est=env.est_cube.float(), truth=env.cube_state_true().float(), done=est.done.float(), above=above.float(),
                     skill=env.skill.float(), t_skill=env.t_skill.float(), ptr=ptr.float(),
                     force=env.tactile.ft[..., :3].norm(dim=-1).half())  # (N, 2 hands, 5 pads) [N]
            if "tac_img" in obs:
                x["tac_img"] = obs["tac_img"]
            for k_, v_ in x.items():
                rec_x.setdefault(k_, []).append(v_.cpu())
    if a.log_qd:  # (N, 2 hands, J) finger speeds of running envs, binned by the current skill
        run = torch.tensor([s_ == "running" for s_ in status], device=dev)
        v = torch.stack([env.hands[s_].data.joint_vel[:, env.finger_ids[s_]] for s_ in ("left", "right")], 1).abs()
        for k_ in env.skill[run].unique().tolist():
            m_ = run & (env.skill == k_); vk = v[m_]
            st_ = qd_stats.setdefault(S.SKILLS[k_], dict(n=0, gt3=torch.zeros(v.shape[-1], device=dev), gt43=torch.zeros(v.shape[-1], device=dev),
                                                          vmax=torch.zeros(v.shape[-1], device=dev), envmax=[]))
            st_["n"] += vk.shape[0] * vk.shape[1]
            st_["gt3"] += (vk > 3.0).float().sum((0, 1)); st_["gt43"] += (vk > 4.3).float().sum((0, 1))
            st_["vmax"] = torch.maximum(st_["vmax"], vk.amax((0, 1)))
    if a.video and step % 2 == 0 and status[a.video_env] == "running":
        frames.append(env.render())
    if a.trigger == "detector":
        pd = est.done if est is not None else policies["all"].last_done
        if a.done_window > 0:
            hist = torch.cat([hist[:, 1:], (pd > a.done_threshold).unsqueeze(-1)], -1)
            above = hist.sum(-1)
        else:
            above = torch.where(pd > a.done_threshold, above + 1, torch.zeros_like(above))
        hold = a.done_hold if a.done_hold > 0 else round(env.cfg.stable_s / env.step_dt) + 1
        # the completion window is only open once the skill's nominal motion is over (the robot's own timing)
        hold_t = torch.where(env.t_skill >= a.late_s, a.late_hold, hold) if a.late_s > 0 else hold
        done_sig = (above >= hold_t) & (env.t_skill >= env.k_duration[env.skill])
    elif a.trigger == "vision":
        done_sig = env.success_vis.clone()
    else:
        done_sig = env.success.clone()
    advance, retry_ids = [], []
    for i in range(n):
        if status[i] != "running":
            continue
        k = S.SKILLS[env.skill[i].item()]
        if env.dropped[i] or env.broken[i]:
            status[i] = "dropped"
            continue
        if a.redo_missed and (done_sig[i] or env.t_skill[i] > a.skill_timeout) and retries[i] < a.max_retries and missed_turn(i, k):
            records[i].append(dict(skill=k, t=round(float(env.t_skill[i]), 3), redo=True, truth_success=bool(env.success[i]),
                                   layer_err_deg=round(float(np.degrees(float(env.layer_err[i]))), 2)))
            retries[i] += 1
            retry_ids.append(i)
            continue
        if a.finish_tol > 0 and done_sig[i] and k in S.FLICK and retries[i] < a.max_retries:
            # finger flicks only (a repeated wrist roll would start from the rolled wrist); off the intended quarter
            err = float(env.layer_err[i]) if (a.finish_oracle or est is None) else float(env.est_cube[i, 9])
            err = abs(float(np.arctan2(np.sin(err), np.cos(err))))
            if a.finish_mode == "residual":
                err = err if err < 0.25 * np.pi else 0.0  # a missed turn is redo_missed's business
            if err > np.radians(a.finish_tol):
                records[i].append(dict(skill=k, t=round(float(env.t_skill[i]), 3), finish=True, truth_success=bool(env.success[i]),
                                       residual_deg=round(float(np.degrees(abs(float(quarter_residual(env.cube.data.joint_pos[i, env.twist_id]))))), 1)))
                retries[i] += 1
                retry_ids.append(i)
                continue
        if done_sig[i]:
            truth = bool(env.success[i])
            twist = physical_outcome(i, k)
            records[i].append(dict(skill=k, t=round(float(env.t_skill[i]), 3), truth_success=truth,
                                   twist_deg=round(np.degrees(twist), 2), tilt_deg=round(float(np.degrees(float(env.tilt[i]))), 2),
                                   res_deg=round(float(np.degrees(float(quarter_residual(env.cube.data.joint_pos[i, env.twist_id])))), 2),
                                   layer_err_deg=round(float(np.degrees(float(env.layer_err[i]))), 2)))
            if est is not None:
                records[i][-1]["est_err_deg"] = round(float(np.degrees(float(env.est_cube[i, 9]))), 2)
            if bool(env.jam[i]):
                records[i][-1]["jam"] = True
            if env.mid_id is not None:
                records[i][-1]["mid_deg"] = round(float(np.degrees(float(env.cube.data.joint_pos[i, env.mid_id]))), 2)
            if not truth:  # detector fired early: which completion criteria did not hold, how long the window had run
                records[i][-1].update(stable_steps=int(env.stable_steps[i]), failing=[c for c, v in env.crit.items() if c != "handoff" and not bool(v[i])])
            if not truth and not a.blind_continue:
                status[i] = "premature"
                continue
            if not truth:
                records[i][-1]["continued"] = True
            ptr[i] += 1
            queue[i].pop(0)
            retries[i] = 0
        elif env.t_skill[i] > a.skill_timeout:
            resid = abs(float(quarter_residual(env.cube.data.joint_pos[i, env.twist_id])))
            if a.trigger == "vision":  # recovery decisions on what the camera reports
                resid = abs(float(quarter_residual(env.perceived()["twist"][i])))
            records[i].append(dict(skill=k, t=round(float(env.t_skill[i]), 3), timeout=True, residual_deg=round(np.degrees(resid), 1),
                                   truth_success=bool(env.success[i]), failing=[c for c, v in env.crit.items() if c != "handoff" and not bool(v[i])],
                                   layer_err_deg=round(float(np.degrees(float(env.layer_err[i]))), 2)))
            if bool(env.jam[i]):
                records[i][-1]["jam"] = True
            if env.mid_id is not None:
                records[i][-1]["mid_deg"] = round(float(np.degrees(float(env.cube.data.joint_pos[i, env.mid_id]))), 2)
            if a.blind_retry and est is not None and k in FACE_TURN and retries[i] < a.max_retries:
                est_twist = env.layer_target[i] - env.est_cube[i, 9]  # the layer angle as touch reports it
                if abs(float(quarter_residual(est_twist))) > a.align_tol:
                    retries[i] += 1
                    records[i][-1]["blind_retry"] = True
                    retry_ids.append(i)
                    continue
            if a.timeout_continue:
                physical_outcome(i, k)
                records[i][-1]["continued"] = True
                ptr[i] += 1
                queue[i].pop(0)
                retries[i] = 0
                if a.timeout_regrip and k not in S.REGRIP_HAND:
                    queue[i][:0] = ["regrip_left", "regrip_right"]
                if a.timeout_settle and k != "hold" and queue[i]:
                    queue[i][:0] = ["hold"]
            elif not a.recover:
                status[i] = "timeout"
                continue
            elif k in FACE_TURN and resid > a.align_tol and retries[i] < a.max_retries:
                retries[i] += 1  # the layer stopped between quarters: try the same turn again (no re-binding)
                retry_ids.append(i)
                continue
            elif resid <= a.align_tol and replans[i] < a.max_replans and float(env.tilt[i]) < env.cfg.tilt_tol and k not in S.REGRIP_HAND and not (a.reseat and k in S.FLICK):
                physical_outcome(i, k)
                retries[i] = 0
                replan(i)
            elif a.reseat and resid <= a.reseat_tol and replans[i] < a.max_replans:
                # tilted cube or an unfinished regrip: re-grasp with each hand in turn (the regrip's close + restore
                # re-centers the cube), then solve again from the observed state
                physical_outcome(i, k)
                retries[i] = 0
                cube = Cube(plans[i]["start"]).apply(performed[i])
                rest = [] if cube.solved() else best_solution(cube, face_skills=tuple(a.face_skills.split(",")))[1]
                queue[i] = ["regrip_left", "regrip_right"] + rest
                replans[i] += 1
            else:
                status[i] = "failed"
                print(f"[exec] env {i} failed in {k}: layer residual {np.degrees(resid):.1f} deg, tilt {np.degrees(float(env.tilt[i])):.1f} deg, retries {retries[i]}, replans {replans[i]}", flush=True)
                continue
        else:
            continue
        if not queue[i]:
            if a.formula or Cube(plans[i]["start"]).apply(performed[i]).solved() or not a.recover or replans[i] >= a.max_replans:
                status[i] = "done"
                mis_done[i] = bool(env.misaligned[i])
                print(f"[exec] env {i} done at t={step * env.step_dt:.0f}s: solved {Cube(plans[i]['start']).apply(performed[i]).solved()}, "
                      f"skills {int(ptr[i])}, replans {replans[i]}", flush=True)
                continue
            replan(i)  # plan exhausted but the cube is not solved: observe and solve again
            if not queue[i]:
                status[i] = "done"
                mis_done[i] = bool(env.misaligned[i])
                continue
        advance.append(i)
    if retry_ids:
        ids = torch.tensor(retry_ids, device=dev)
        env.start_skill(ids, env.skill[ids], retry=True)
        if a.retry_reset:
            above[ids] = 0
            hist[ids] = False
            for pol in policies.values():
                pol.reset(ids)
    if advance:
        ids = torch.tensor(advance, device=dev)
        nxt = torch.tensor([S.SKILL_INDEX[queue[i][0]] for i in advance], device=dev)
        env.start_skill(ids, nxt)
        set_next(advance)
        if painter is not None and a.video_env in advance:
            paint()
        above[ids] = 0
        hist[ids] = False
        for pol in policies.values():
            pol.reset(ids)
    if advance or retry_ids:
        obs = env._get_observations()
        if est is not None:  # this step again for them: their commanded skill changed (as a chain advance in training)
            env.est_cube = est.step(obs, ids=torch.tensor(advance + retry_ids, device=dev))
    if est is not None:
        obs = env._get_observations()  # the controllers' cube observation from the current estimate

# ---------------------------------------------------------------- report
results = []
for i, pl in enumerate(plans):
    final = Cube(pl["start"]).apply(performed[i])
    results.append(dict(scramble=" ".join(pl["scramble"]), solution=" ".join(pl["solution"]), n_skills=len(pl["skills"]),
                        status=status[i], skills_done=int(ptr[i]), replans=replans[i], solved=final.solved() and status[i] == "done" and not mis_done[i], misaligned=mis_done[i],
                        time_s=round(sum(r["t"] for r in records[i]), 2), records=records[i]))
solved = sum(r["solved"] for r in results)
per_skill = {}
for r in results:
    for q in r["records"]:
        d = per_skill.setdefault(q["skill"], {"n": 0, "ok": 0, "t": 0.0})
        d["n"] += 1
        d["ok"] += int(bool(q.get("truth_success")))
        d["t"] += q["t"] if q.get("truth_success") else 0.0
per_skill = {k: dict(n=v["n"], rate=round(v["ok"] / v["n"], 3), mean_t=round(v["t"] / max(1, v["ok"]), 2)) for k, v in per_skill.items()}
summary = dict(n=n, solved=solved, per_skill=per_skill, status={s: status.count(s) for s in set(status)},
               mean_skills_done=float(np.mean([r["skills_done"] for r in results])), mean_replans=float(np.mean(replans)),
               mean_time_s_solved=float(np.mean([r["time_s"] for r in results if r["solved"]])) if solved else None,
               trigger=a.trigger, policies=a.policy, wall_s=round(time.time() - t0, 1))
print(json.dumps(summary, indent=1), flush=True)
os.makedirs(a.out, exist_ok=True)
json.dump(dict(summary=summary, results=results), open(os.path.join(a.out, "report.json"), "w"), indent=1)
if a.log_qd:
    from rubik_lab.constants import FINGER_JOINTS
    json.dump({k_: dict(samples=st_["n"], joints=FINGER_JOINTS, vmax=st_["vmax"].tolist(), frac_gt3=(st_["gt3"] / max(st_["n"], 1)).tolist(),
                        frac_gt4p3=(st_["gt43"] / max(st_["n"], 1)).tolist()) for k_, st_ in qd_stats.items()},
              open(os.path.join(a.out, "qd_stats.json"), "w"), indent=1)
if a.record and rec_q:
    torch.save(dict(hand_q=torch.stack(rec_q, 1), cube=torch.stack(rec_c, 1), n_performed=torch.tensor(rec_n).T,
                    performed=performed, plans=plans, status=status, ready_idx=env.ready_idx.cpu(), step_dt=2 * env.step_dt,
                    ready_cache=a.ready_cache, arm=env.cfg.arm, arm_cfg={k_: getattr(env.cfg, k_) for k_ in ('fr3_layout', 'fr3_urdf_dir', 'fr3_usd_dir')}, done_hold=a.done_hold, done_threshold=a.done_threshold, skills=list(S.SKILLS),
                    k_duration=env.k_duration.cpu(), blind={k_: torch.stack(v_, 1) for k_, v_ in rec_x.items()}), a.record)
    print(f"[exec] recorded {len(rec_q)} frames of {n} envs -> {a.record}", flush=True)
if a.video and frames:
    import imageio.v2 as imageio
    imageio.mimsave(os.path.join(a.out, f"env{a.video_env}.mp4"), frames, fps=20)
os._exit(0)
