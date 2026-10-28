"""DAgger distillation of the privileged skill teachers into one recurrent proprio+tactile student.

The student is rolled out (with probability 1 - beta per env-step the teacher's action is executed instead, beta
decays to 0) and trained on the teacher's action at every visited state, plus the completion label (ground-truth
success latch, held for success_linger_s) and the remaining task error. Hidden states reset at every episode start,
which is also every skill start (the executor resets them the same way).

    python scripts/solve/distill.py --headless --teacher roll=<ckpt> --teacher flick=<ckpt> --iterations 3000
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--teacher", action="append", default=[])
p.add_argument("--skills", default="U,U',R,R',L,L',x,x',regrip_left,regrip_right")
p.add_argument("--num_envs", type=int, default=2048)
p.add_argument("--iterations", type=int, default=3000)
p.add_argument("--horizon", type=int, default=32)
p.add_argument("--beta_iters", type=int, default=300, help="iterations over which teacher mixing decays to 0")
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--ready_cache", default="cache/solve_ready_udr_v3.npz")
p.add_argument("--out", default="logs/solve_student/run")
p.add_argument("--resume", default=None)
p.add_argument("--done_weight", type=float, default=1.0)
p.add_argument("--err_weight", type=float, default=0.5)
p.add_argument("--set", action="append", default=[])
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app
import sys as _sys, traceback as _tb, os as _os
def _die(t, v, b):
    _tb.print_exception(t, v, b); _sys.stdout.flush(); _sys.stderr.flush(); _os._exit(1)
_sys.excepthook = _die  # the simulation app hangs on close after an exception

import ast  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rubik_lab.solve import skills as S  # noqa: E402
from rubik_lab.solve.env import SolveEnv  # noqa: E402
from rubik_lab.solve.env_cfg import SolveEnvCfg  # noqa: E402
from rubik_lab.solve.policies import TeacherPolicy  # noqa: E402
from rubik_lab.solve.student import StudentNet  # noqa: E402

FAMILY = {s: ("flick" if s in S.FLICK else "roll" if s in S.ROLL_DELTA else "regrip" if s in S.REGRIP_HAND else "hold") for s in S.SKILLS}

cfg = SolveEnvCfg()
cfg.scene.num_envs = a.num_envs
cfg.skills = tuple(a.skills.split(","))
cfg.ready_cache = a.ready_cache
cfg.buffer_size, cfg.buffer_prob = 20000, 0.6
cfg.terminate_on_success = False
cfg.success_linger_s = 0.3
cfg.episode_length_s = 3.5
for kv in a.set:
    k, v = kv.split("=", 1)
    setattr(cfg, k, ast.literal_eval(v))
env = SolveEnv(cfg)
dev, n = env.device, env.num_envs
obs = env.reset()[0]
teachers = {}
for kv in a.teacher:
    fam, path = kv.split("=", 1)
    teachers[fam] = TeacherPolicy(path, obs["policy"].shape[-1], dev)
fam_of_skill = torch.tensor([list(teachers).index(FAMILY[s]) if FAMILY[s] in teachers else -1 for s in S.SKILLS], device=dev)

kw = dict(obs_dim=obs["student"].shape[-1])
net = StudentNet(**kw).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
start_it = 0
if a.resume:
    blob = torch.load(a.resume, map_location=dev, weights_only=False)
    net.load_state_dict(blob["student_state_dict"])
    opt.load_state_dict(blob["optimizer"])
    start_it = blob["iteration"]
os.makedirs(a.out, exist_ok=True)
json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)


def teacher_action(o):
    act = torch.zeros(n, 56, device=dev)
    fi = fam_of_skill[env.skill]
    for j, pol in enumerate(teachers.values()):
        m = fi == j
        if m.any():
            act[m] = pol(o)[m]
    return act


h = net.init_hidden(n, dev)
reset_next = torch.ones(n, dtype=torch.bool, device=dev)
log_f = open(os.path.join(a.out, "log.jsonl"), "a")
t0 = time.time()
for it in range(start_it, start_it + a.iterations):
    beta = max(0.0, 1.0 - it / max(1, a.beta_iters))
    O, A, D, E, R, M = [], [], [], [], [], []
    h0 = h.detach()
    with torch.no_grad():
        for t in range(a.horizon):
            so = obs["student"]
            net.norm.update(so)
            ta = teacher_action(obs["policy"])
            sa, _, _, h = net(so.unsqueeze(0), h, reset_next.unsqueeze(0))
            use_t = torch.rand(n, device=dev) < beta
            act = torch.where(use_t.unsqueeze(-1), ta, sa[0])
            O.append(so)
            A.append(ta)
            R.append(reset_next.clone())
            # labels describing the state this observation was taken in
            D.append(env.success.float())
            E.append(env.prev_err.clone())
            M.append((fam_of_skill[env.skill] >= 0).float())
            obs, _, term, trunc, _ = env.step(act)
            reset_next = term | trunc
    O, A, D, E, R, M = (torch.stack(x, 0) for x in (O, A, D, E, R, M))
    pa, pd, pe, _ = net(O, h0, R)
    w = M.unsqueeze(-1)
    loss_a = ((pa - A).square() * w).sum() / (w.sum() * 56 + 1)
    pos = D.sum().clamp(min=1)
    pw = ((M.sum() - pos) / pos).clamp(max=20.0)  # done is rare: balance the classes
    loss_d = F.binary_cross_entropy_with_logits(pd, D, weight=M, pos_weight=pw)
    loss_e = (F.smooth_l1_loss(pe, E.clamp(max=3.2), reduction="none") * M).sum() / (M.sum() + 1)
    loss = loss_a + a.done_weight * loss_d + a.err_weight * loss_e
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    if it % 10 == 0:
        with torch.no_grad():
            pred = (pd > 0).float()
            tp = (pred * D * M).sum()
            prec = (tp / (pred * M).sum().clamp(min=1)).item()
            rec = (tp / (D * M).sum().clamp(min=1)).item()
        log = env.extras.get("log", {})
        rec_d = dict(it=it, beta=round(beta, 3), loss_a=round(loss_a.item(), 5), loss_d=round(loss_d.item(), 4),
                     loss_e=round(loss_e.item(), 4), done_prec=round(prec, 3), done_rec=round(rec, 3),
                     t=round(time.time() - t0, 1), **{k: round(float(v), 3) for k, v in log.items() if k.startswith("succ_")})
        print(json.dumps(rec_d), flush=True)
        log_f.write(json.dumps(rec_d) + "\n")
        log_f.flush()
    if (it + 1) % 200 == 0 or it == start_it + a.iterations - 1:
        torch.save(dict(student_state_dict=net.state_dict(), student_kwargs=kw, optimizer=opt.state_dict(), iteration=it + 1,
                        teachers=a.teacher, skills=a.skills), os.path.join(a.out, f"student_{it + 1}.pt"))
os._exit(0)
