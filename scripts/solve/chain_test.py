"""Chained skill sequence over many envs (no resets), scripted nominal or a teacher checkpoint; reports how many envs
get through each step. A regrip token may name the grip mode its hand closes into: regrip_right>E (edge, the
alternative grasp of cfg.grip_alt_cache) or regrip_right>W (back to the wide ready grip).

    python scripts/solve/chain_test.py --headless --seq "regrip_right>E R R' regrip_right>W regrip_left" \
        --ready_cache cache/solve_ready_ud_v1.npz --set "grip_alt_cache={'right': 'cache/solve_alt_right_edge_v0.npz'}"
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--seq", required=True)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--ready_cache", default="cache/solve_ready_ud_v1.npz")
p.add_argument("--checkpoint", default=None, help="teacher policy for every skill (default: scripted nominal)")
p.add_argument("--skill_timeout", type=float, default=3.5)
p.add_argument("--set", action="append", default=[])
p.add_argument("--video", default="", help="mp4 path: record env 0")
p.add_argument("--tag", default="last")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.video:
    a.enable_cameras = True
app = AppLauncher(a).app
import sys as _sys, traceback as _tb
def _die(t, v, b):
    _tb.print_exception(t, v, b); _sys.stdout.flush(); os._exit(1)
_sys.excepthook = _die
import ast, json, torch
from rubik_lab.solve import skills as S
from rubik_lab.solve.env import SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg
from rubik_lab.solve.policies import TeacherPolicy

cfg = SolveEnvCfg()
cfg.scene.num_envs = a.num_envs
cfg.ready_cache = a.ready_cache
cfg.episode_length_s = 1e6
cfg.terminate_on_success = False
cfg.skills = ("hold",)
for kv in a.set:
    k, v = kv.split("=", 1)
    *path, k = k.split(".")
    obj = cfg
    for name in path:
        obj = getattr(obj, name)
    setattr(obj, k, ast.literal_eval(v))
if a.video:
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (-0.30, -0.28, 0.42)
    cfg.viewer.lookat = (0.0, 0.0, 0.3)
    cfg.viewer.resolution = (640, 480)
env = SolveEnv(cfg, render_mode="rgb_array" if a.video else None)
env.chain_mode = True
dev, n = env.device, env.num_envs
obs, _ = env.reset()
pol = TeacherPolicy(a.checkpoint, obs["policy"].shape[-1], dev) if a.checkpoint else None
tokens = a.seq.split()
MODE = {"W": 0, "E": 1}


def begin(ids, tok):
    sk, _, mode = tok.partition(">")
    env.start_skill(ids, torch.full((len(ids),), S.SKILL_INDEX[sk], device=dev))
    if mode:
        env.grip_mode_next[ids, ["left", "right"].index(S.REGRIP_HAND[sk])] = MODE[mode]


alive = torch.ones(n, dtype=torch.bool, device=dev)
step_of = torch.zeros(n, dtype=torch.long, device=dev)  # index of the token each env is executing
begin(torch.arange(n, device=dev), tokens[0])
obs = env._get_observations()
passed = [0] * len(tokens)
fails = [dict(timeout=0, dropped=0, violation=0) for _ in tokens]
frames = []
while alive.any():
    with torch.no_grad():
        act = torch.zeros(n, 56, device=dev) if pol is None else pol(obs["policy"])
        obs, _, _, _, _ = env.step(act)
    if a.video and alive[0]:
        frames.append(env.render())
    dead = alive & (env.dropped | env.violation | (env.t_skill > a.skill_timeout))
    for i in torch.nonzero(dead).flatten().tolist():
        j = int(step_of[i])
        why = "dropped" if env.dropped[i] else "violation" if env.violation[i] else "timeout"
        fails[j][why] += 1
        if why == "timeout":  # which completion criteria were still unmet
            for c, v in env.crit.items():
                fails[j].setdefault("unmet", {}).setdefault(c, 0)
                fails[j]["unmet"][c] += int(not bool(v[i]))
    alive &= ~dead
    done = alive & env.success
    for i in torch.nonzero(done).flatten().tolist():
        passed[int(step_of[i])] += 1
    step_of[done] += 1
    fin = done & (step_of >= len(tokens))
    alive &= ~fin
    nxt = done & ~fin
    for j in range(len(tokens)):
        ids = torch.nonzero(nxt & (step_of == j)).flatten()
        if len(ids):
            begin(ids, tokens[j])
res = [dict(skill=t, passed=passed[j], of=(passed[j - 1] if j else n), **fails[j]) for j, t in enumerate(tokens)]
for r in res:
    print(f"[chain] {r['skill']:16s} {r['passed']:4d}/{r['of']:4d}  fails {dict((k, r[k]) for k in ('timeout', 'violation', 'dropped'))} unmet {r.get('unmet', {})}", flush=True)
print(f"[chain] complete: {passed[-1]}/{n} = {passed[-1] / n:.1%}", flush=True)
os.makedirs("outputs/solve/chain", exist_ok=True)
json.dump(dict(args=vars(a), res=res, steps=step_of.tolist(), ready_idx=env.ready_idx.tolist(), alt_idx=env.alt_idx.tolist()),
          open(f"outputs/solve/chain/{a.tag}.json", "w"), indent=1, default=str)
if a.video and frames:
    import imageio.v2 as imageio
    imageio.mimsave(a.video, [f[..., :3] for f in frames[::2]], fps=20)
    print(f"[chain] video -> {a.video}", flush=True)
os._exit(0)
