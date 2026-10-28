"""Unit cases of the cube-realism bookkeeping (env._rebind with cfg.cube_jam_deg: per-face residual slots; cfg.cube_3body).
Exits 1 on any failure.
    python scripts/solve/test_jam.py --headless [--set cube_3body=True]
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--set", action="append", default=[])
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import ast
import math

import torch

from rubik_lab.solve.env import FACE_NORMAL, SolveEnv
from rubik_lab.solve.env_cfg import SolveEnvCfg

cfg = SolveEnvCfg()
cfg.scene.num_envs = 4
cfg.ready_cache = "cache/solve_ready_ud_all4_flick.npz"
cfg.episode_length_s = 1000.0
cfg.skills = ("hold",)
cfg.terminate_on_success = False
cfg.link_contacts = False
cfg.contact_model = "tacmap"
cfg.arm, cfg.fr3_urdf_dir, cfg.fr3_usd_dir = "fr3", "assets_franka/left_m90", "assets_franka/left_m90"
cfg.fr3_layout = "assets_franka/layout_left_m90_c3.json"
cfg.cube_jam_deg = 15.0
for kv in a.set:
    k, v = kv.split("=", 1)
    setattr(cfg, k, ast.literal_eval(v))
env = SolveEnv(cfg)
env.chain_mode = True
env.reset()
dev = env.device
d = env.cube.data
D = math.radians
THREE = env.mid_id is not None
fails = []


def I(*e):
    return torch.tensor(e, device=dev)


def bind(ids, f):
    env._rebind(ids, torch.tensor(FACE_NORMAL[f], device=dev).expand(len(ids), 3).contiguous())


def deg(x):
    return [round(math.degrees(v), 1) for v in x.tolist()]


def twist():
    return deg(d.joint_pos[:, env.twist_id])


def mid():
    return deg(d.joint_pos[:, env.mid_id]) if THREE else [0.0] * 4


def set_joint(jid, ids, vals):
    env.cube.write_joint_state_to_sim(torch.tensor([[D(v)] for v in vals], device=dev), torch.zeros(len(ids), 1, device=dev),
                                      joint_ids=[jid], env_ids=ids)


def show(tag):
    print(f"{tag:28s} twist {twist()} mid {mid()} jam {env.jam.int().tolist()} misaligned {env.misaligned.int().tolist()} "
          f"slots {[deg(r) for r in env.face_res]}", flush=True)


def check(name, got, want):
    ok = got == want
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {got} (want {want})", flush=True)
    if not ok:
        fails.append(name)


ALL = I(0, 1, 2, 3)
bind(ALL, "U")
set_joint(env.twist_id, ALL, [5, 25, 25, 25])
show("U free, residuals set")
bind(ALL, "U")
show("re-bind U (same face)")
check("same face keeps the residual", [round(x) for x in twist()], [5, 25, 25, 25])
bind(ALL, "R")
show("bind R (orthogonal)")
check("5 deg snapped by the corner cutting", env.face_res[0].abs().max().item() < 1e-6, True)
check("25 deg: U slot kept, R jammed", env.jam.tolist(), [False, True, True, True])
env.cube.write_joint_state_to_sim(d.joint_pos[:, env.twist_id:env.twist_id + 1].clone(), torch.full((4, 1), 40.0, device=dev),
                                  joint_ids=[env.twist_id], env_ids=ALL)  # kick every layer at 40 rad/s
env.step(torch.zeros(4, env.cfg.action_space, device=dev))
show("kicked, 1 step later")
check("free layer moves more than jammed ones (x5)", abs(twist()[0]) > 5 * max(abs(x) for x in twist()[1:]), True)
check("jammed layers held (< 2 deg)", all(abs(x) < 2 for x in twist()[1:]), True)
set_joint(env.twist_id, ALL, [0, 0, 0, 0])

# env 1: two misaligned layers on one axis (U 25, then D -30), then an orthogonal face: both kept
bind(I(1), "D")
check("parallel face passes", env.jam[1].item(), False)
set_joint(env.twist_id, I(1), [-30])
bind(I(1), "R")
show("env1: U 25 and D -30, bind R")
s1 = deg(env.face_res[1])
check("both slots kept (U 25, D -30)", (round(s1[4]), round(s1[5])), (25, -30))
check("jammed", env.jam[1].item(), True)

# env 2: back to U restores, realigning clears
bind(I(2), "U")
show("env2 back to U")
check("returning restores the residual", round(twist()[2]), 25)
check("its slot is cleared", env.face_res[2].abs().max().item() < 1e-6, True)
check("still misaligned (free 25 deg)", env.misaligned[2].item(), True)
set_joint(env.twist_id, I(2), [0])
bind(I(2), "U")
check("realigned layer is fine", env.misaligned[2].item(), False)

# env 3: capture / restore keeps the slots and the jam
st = env.capture_full(I(3))
env.face_res[3] = 0.0
env._write_full(I(3), st)
check("capture/restore keeps the slot", round(deg(env.face_res[3])[4]), 25)
check("capture/restore keeps the jam", env.jam[3].item(), True)

# env 0: opposite face on the same axis (no corner cutting): 2-body stores U, 3-body swaps twist and middle
bind(I(0), "U")
set_joint(env.twist_id, I(0), [10])
if THREE:
    set_joint(env.mid_id, I(0), [20])
bind(I(0), "D")
show("env0: U 10 (mid 20), bind D")
if THREE:
    check("3-body swap: twist <- middle, middle <- twist", (round(twist()[0]), round(mid()[0])), (20, 10))
else:
    check("2-body: U 10 kept in its slot, D free at 0", (round(deg(env.face_res[0])[4]), round(twist()[0])), (10, 0))
bind(I(0), "U")
check("back to U: 10 again", round(twist()[0]), 10)
if THREE:
    check("back to U: middle 20 again", round(mid()[0]), 20)
print("RESULT", "PASS" if not fails else f"FAIL {fails}", flush=True)
os._exit(0 if not fails else 1)
