"""Per-skill success of a teacher checkpoint (deterministic) or of the scripted nominal (--zero), fresh ready starts.

    python scripts/solve/eval_skills.py --headless --checkpoint <model.pt> --skills "x,R" --num_envs 512
"""
import argparse, json, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default=None)
p.add_argument("--zero", action="store_true")
p.add_argument("--noise", type=float, default=0.0)
p.add_argument("--skills", default="x")
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--episodes", type=int, default=1, help="episodes per env")
p.add_argument("--ready_cache", default="cache/solve_ready_udr_v3.npz")
p.add_argument("--set", action="append", default=[])
p.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import sys as _sys, traceback as _tb, os as _os
def _die(t, v, b):
    _tb.print_exception(t, v, b); _sys.stdout.flush(); _os._exit(1)
_sys.excepthook = _die
import ast, torch
from rubik_lab.solve import skills as S
from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.policies import TeacherPolicy

res = {}
for sk in a.skills.split(","):
    cfg = SolveEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.skills = (sk,); cfg.ready_cache = a.ready_cache
    for kv in a.set:
        k, v = kv.split("=", 1); setattr(cfg, k, ast.literal_eval(v))
    env = SolveEnv(cfg) if not res else env
    if res:
        env.skill_pool[:] = S.SKILL_INDEX[sk]
    obs = env.reset()[0]
    pol = None if a.zero else TeacherPolicy(a.checkpoint, obs["policy"].shape[-1], env.device)
    n = env.num_envs
    done = torch.zeros(n, dtype=torch.long, device=env.device); ok = torch.zeros_like(done); dur = torch.zeros(n, device=env.device)
    critacc = {}
    twist_end = []
    reasons = {"success": 0, "violation": 0, "dropped": 0, "timeout": 0}
    while (done < a.episodes).any():
        act = torch.zeros(n, 56, device=env.device) if pol is None else pol(obs["policy"])
        act = act + a.noise * torch.randn_like(act)
        succ_before, viol, drop, t = env.success.clone(), None, None, env.t_skill.clone()
        obs, _, term, trunc, _ = env.step(act)
        late = env.t_skill > env.k_duration[env.skill]
        if late.any():
            for c, v in env.crit.items():
                critacc[c] = critacc.get(c, 0.0) + v[late].float().mean().item()
            critacc["_n"] = critacc.get("_n", 0) + 1
        fin = (term | trunc) & (done < a.episodes)
        if fin.any():
            # the reset already happened: use the statistics kept by the env (success flag of the finished episode)
            s = env._last_success[fin]
            ok[fin] += s.long(); done[fin] += 1
            reasons["success"] += int(s.sum()); reasons["timeout"] += int((trunc[fin] & ~s).sum())
            reasons["violation"] += int(env._last_violation[fin].sum()); reasons["dropped"] += int(env._last_dropped[fin].sum())
            twist_end.append(torch.rad2deg(env._last_twist[fin]).cpu())
            dur[fin] += torch.where(s, env._last_t[fin], torch.zeros_like(env._last_t[fin]))
    res[sk] = dict(rate=ok.sum().item() / done.sum().item(), mean_time_success=(dur.sum() / ok.sum().clamp(min=1)).item(), **reasons)
    te = torch.cat(twist_end)
    res[sk]["twist_end_pct"] = [round(float(x), 1) for x in torch.quantile(te, torch.tensor([0.1, 0.5, 0.9]))]
    print(sk, json.dumps(res[sk]), {c: round(v / critacc["_n"], 3) for c, v in critacc.items() if c != "_n"}, flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
os._exit(0)
