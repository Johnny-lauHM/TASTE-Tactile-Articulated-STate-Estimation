"""Episode success of an old bimanual_twist flick checkpoint (LegacyFlickPolicy) as the U / U' / D / D' skills of SolveEnv.

    python scripts/solve/eval_legacy_flick.py --headless --checkpoint logs/rsl_rl/rubik_flick/2026-10-17_18-41-54/model_1900.pt
--residual <solve teacher ckpt>: the legacy policy runs inside the env as the nominal (cfg.flick_legacy_ckpt) and this
residual policy acts on top (--residual zero: nominal only).
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--skills", default="U,U',D,D'")
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--episodes", type=int, default=1)
p.add_argument("--ready_cache", default="cache/solve_ready_udr_v3.npz")
p.add_argument("--adapter", default="{}", help="LegacyFlickPolicy keyword arguments (json)")
p.add_argument("--residual", default="")
p.add_argument("--stochastic", action="store_true", help="sample the residual policy (training noise) instead of its mean")
p.add_argument("--set", action="append", default=[])
p.add_argument("--out", default="outputs/solve/legacy_flick.json")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import sys as _sys, traceback as _tb
def _die(t, v, b):
    _tb.print_exception(t, v, b); _sys.stdout.flush(); os._exit(1)
_sys.excepthook = _die
import ast, json, torch
from rubik_lab.solve import skills as S
from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.policies import LegacyFlickPolicy, TeacherPolicy

cfg = SolveEnvCfg(); cfg.scene.num_envs = a.num_envs; cfg.skills = tuple(a.skills.split(",")); cfg.ready_cache = a.ready_cache
for kv in a.set:
    k, v = kv.split("=", 1); setattr(cfg, k, ast.literal_eval(v))
if a.residual:
    cfg.flick_legacy_ckpt, cfg.flick_legacy_kw = a.checkpoint, json.loads(a.adapter)
env = SolveEnv(cfg)
if not a.residual:
    pol = LegacyFlickPolicy(env, a.checkpoint, **json.loads(a.adapter))
else:
    pol = env.flick_legacy
    res_pol = None if a.residual == "zero" else TeacherPolicy(a.residual, env._get_observations()["policy"].shape[-1], env.device)
n, dev = env.num_envs, env.device
res = {}
for sk in cfg.skills:
    env.skill_pool[:] = S.SKILL_INDEX[sk]
    env.skill_w = torch.ones(len(env.skill_pool), device=dev)
    env.reset()
    pol.reset(torch.arange(n, device=dev))
    done = torch.zeros(n, dtype=torch.long, device=dev)
    out = dict(success=0, violation=0, dropped=0, timeout=0)
    pr, pp, le, tt, crit = [], [], [], [], {}
    per_grip, cur_idx, by_ep = {}, env.ready_idx.clone(), {}
    while (done < a.episodes).any():
        if not a.residual:
            act = pol()
        else:
            if res_pol is None:
                act = torch.zeros(n, 56, device=dev)
            elif a.stochastic:
                from tensordict import TensorDict
                o = env._get_observations()["policy"]
                with torch.no_grad():
                    act = res_pol.net.act(TensorDict({"policy": o}, batch_size=[n]))
            else:
                act = res_pol(env._get_observations()["policy"])
        peak_rot, peak_pos, lerr, t = env.peak_rot.clone(), env.peak_pos.clone(), env.layer_err.clone() if hasattr(env, "layer_err") else torch.zeros(n, device=dev), env.t_skill.clone()
        cr = {c: v.clone() for c, v in env.crit.items()} if hasattr(env, "crit") else {}
        _, _, term, trunc, _ = env.step(act)
        fin = (term | trunc) & (done < a.episodes)
        if fin.any():
            s = env._last_success[fin]
            out["success"] += int(s.sum()); out["timeout"] += int((trunc[fin] & ~s).sum())
            out["violation"] += int(env._last_violation[fin].sum()); out["dropped"] += int(env._last_dropped[fin].sum())
            pr.append(torch.rad2deg(peak_rot[fin])); pp.append(peak_pos[fin] * 1000); le.append(torch.rad2deg(lerr[fin])); tt.append(t[fin][s])
            for c, v in cr.items():  # criteria on the last step of failed episodes
                if (~s).any():
                    crit.setdefault(c, []).append(v[fin][~s].float())
            for ep, ok in zip(done[fin].tolist(), s.tolist()):
                by_ep.setdefault(ep, [0, 0]); by_ep[ep][0] += int(ok); by_ep[ep][1] += 1
            for i, ok, e_ in zip(cur_idx[fin].tolist(), s.tolist(), torch.rad2deg(lerr[fin]).tolist()):
                g = per_grip.setdefault(i, [0, 0, []]); g[0] += int(ok); g[1] += 1; g[2].append(round(e_, 1))
            done[fin] += 1
        cur_idx = env.ready_idx.clone()
    q = torch.tensor([0.1, 0.5, 0.9], device=dev)
    pct = lambda x: [round(float(v), 1) for v in torch.quantile(torch.cat(x).float(), q)] if x and torch.cat(x).numel() else None  # noqa: E731
    tot = int(done.sum())
    res[sk] = dict(rate=out["success"] / tot, n=tot, **{k: v for k, v in out.items() if k != "success"}, t_success_pct=pct(tt), peak_rot_deg_pct=pct(pr),
                   peak_pos_mm_pct=pct(pp), layer_err_end_deg_pct=pct(le),
                   failed_crit={c: round(float(torch.cat(v).mean()), 2) for c, v in crit.items()},
                   by_episode={k: round(v[0] / v[1], 3) for k, v in sorted(by_ep.items())})
    print(f"[legacy_flick] {sk}: " + json.dumps(res[sk]), flush=True)
    res[sk]["per_grip"] = {k: v for k, v in sorted(per_grip.items())}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(dict(args=vars(a), res=res), open(a.out, "w"), indent=1)
os._exit(0)
