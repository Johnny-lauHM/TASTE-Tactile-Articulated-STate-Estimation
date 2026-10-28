"""Scripted (zero-residual) skill success under nominal / cfg variants, with per-ready-grip success counts.

    python scripts/solve/sweep_regrip.py --headless --skills regrip_right,regrip_left --mod REGRIP_HOLD_PRESS=0.4 --set hold_press=0.3 --tag c1
--mod sets constants of rubik_lab.solve.skills (nominal shape), --set sets SolveEnvCfg fields. Result: outputs/solve/sweep/<tag>.json
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--skills", default="regrip_right,regrip_left")
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--episodes", type=int, default=1)
p.add_argument("--ready_cache", default="cache/solve_ready_udr_v3.npz")
p.add_argument("--mod", action="append", default=[])
p.add_argument("--set", action="append", default=[])
p.add_argument("--checkpoint", default=None)
p.add_argument("--tag", default="sweep")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import sys as _sys, traceback as _tb
def _die(t, v, b):
    _tb.print_exception(t, v, b); _sys.stdout.flush(); os._exit(1)
_sys.excepthook = _die
import ast, json, torch
from rubik_lab.solve import skills as S
for kv in a.mod:
    k, v = kv.split("=", 1)
    setattr(S, k, ast.literal_eval(v))
from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.policies import TeacherPolicy

res = {}
env = None
for sk in a.skills.split(","):
    if env is None:
        cfg = SolveEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.skills = (sk,); cfg.ready_cache = a.ready_cache
        for kv in a.set:
            k, v = kv.split("=", 1)
            *path, k = k.split(".")  # dotted keys reach nested cfgs, e.g. sim.gravity
            obj = cfg
            for name in path:
                obj = getattr(obj, name)
            setattr(obj, k, ast.literal_eval(v))
        env = SolveEnv(cfg)
    else:
        env.skill_pool[:] = S.SKILL_INDEX[sk]
    obs = env.reset()[0]
    pol = TeacherPolicy(a.checkpoint, obs["policy"].shape[-1], env.device) if a.checkpoint else None
    n, dev = env.num_envs, env.device
    done = torch.zeros(n, dtype=torch.long, device=dev)
    reasons = {"success": 0, "violation": 0, "dropped": 0, "timeout": 0}
    per_grip = {}
    cur_idx, peak_rot, peak_pos = env.ready_idx.clone(), torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    pr_l, pp_l, tw_l = [], [], []
    while (done < a.episodes).any():
        act = torch.zeros(n, 56, device=dev) if pol is None else pol(obs["policy"])
        peak_rot, peak_pos = env.peak_rot.clone(), env.peak_pos.clone()
        obs, _, term, trunc, _ = env.step(act)
        fin = (term | trunc) & (done < a.episodes)
        if fin.any():
            s = env._last_success[fin]
            reasons["success"] += int(s.sum()); reasons["timeout"] += int((trunc[fin] & ~s).sum())
            reasons["violation"] += int(env._last_violation[fin].sum()); reasons["dropped"] += int(env._last_dropped[fin].sum())
            for i, ok in zip(cur_idx[fin].tolist(), s.tolist()):
                g = per_grip.setdefault(i, [0, 0]); g[0] += int(ok); g[1] += 1
            pr_l.append(torch.rad2deg(peak_rot[fin]).cpu()); pp_l.append((peak_pos[fin] * 1000).cpu())
            tw_l.append(torch.rad2deg(env._last_twist[fin]).cpu())
            done[fin] += 1
        cur_idx = env.ready_idx.clone()
    q = torch.tensor([0.1, 0.5, 0.9])
    pct = lambda x: [round(float(v), 1) for v in torch.quantile(torch.cat(x).float(), q)]  # noqa: E731
    res[sk] = dict(rate=reasons["success"] / int(done.sum()), **reasons, peak_rot_pct=pct(pr_l), peak_pos_mm_pct=pct(pp_l), twist_end_pct=pct(tw_l),
                   per_grip={k: v for k, v in sorted(per_grip.items())})
    print(f"[sweep] {a.tag} {sk}: " + json.dumps({k: v for k, v in res[sk].items() if k != "per_grip"}), flush=True)
os.makedirs("outputs/solve/sweep", exist_ok=True)
json.dump(dict(args=vars(a), res=res), open(f"outputs/solve/sweep/{a.tag}.json", "w"), indent=1, default=str)
os._exit(0)
