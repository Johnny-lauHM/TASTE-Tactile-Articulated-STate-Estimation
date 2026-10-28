"""Tactile estimability probe: how well can a recurrent estimator recover the cube state from what the hands sense?

Trains one GRU estimator per input variant on cache/tactile/probe.pt (scripts/solve/collect_tactile.py) and reports test
errors (envs held out) of: cube offset from its ready-grip pose [mm], its rotation [deg], the layer residual to the
nearest quarter [deg], the completion window (done). The key number is the cube offset at the moment the sighted regrip
servo latches it (end of the unroll, regrip t ~ 1.25 s): a blind regrip needs it from touch.

Variants: proprio (no pad F/T) | ft_normal (+ delivered normal-only F/T, as now) | wrench (proprio + full pad wrench with
shear) | wrench_centroid (+ contact centroid per pad).

    python scripts/solve/tactile_probe.py --data cache/tactile/probe.pt
"""
import argparse
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--data", default="cache/tactile/probe.pt")
p.add_argument("--variants", default="proprio,ft_normal,wrench,wrench_centroid")
p.add_argument("--iters", type=int, default=4000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--window", type=int, default=96)
p.add_argument("--burn", type=int, default=32)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--no_history", action="store_true", help="MLP on the current step only (no memory)")
p.add_argument("--out", default="outputs/solve/tactile_probe.json")
a = p.parse_args()
dev = "cuda"
# several collection files (same number of steps) are concatenated along the env dimension
_parts = [torch.load(f) for f in a.data.split(",")]
d = {k: (torch.cat([q[k] for q in _parts], 1) if torch.is_tensor(_parts[0][k]) else _parts[0][k]) for k in _parts[0]}
del _parts
T, N = d["reset"].shape
lab = d["labels"].float()  # (T, N, 13)
names = d["label_names"]
L = {k: i for i, k in enumerate(names)}
skill = lab[..., L["skill"]].long()
t_sk = lab[..., L["t_skill"]]
REGRIP, X = (11, 12), (9, 10)  # rubik_lab.solve.skills.SKILLS indices
is_x = (skill == X[0]) | (skill == X[1])
is_reg = (skill == REGRIP[0]) | (skill == REGRIP[1])

# targets (scaled) and masks: cube offset / rotation are undefined mid-x (the cube turns through the symmetry)
tgt = torch.cat([lab[..., 0:3] / 5.0, lab[..., 3:6] / 5.0, lab[..., L["layer_resid_deg"]: L["layer_resid_deg"] + 1] / 10.0], -1)
pose_mask = (~is_x).float()
done = lab[..., L["done"]]
st = d["student"].float()
lay = d["student_layout"]
ft_cols = [h * lay["per_hand"] + c for h in range(lay["n_hands"]) for c in range(*lay["ft_slice"])]
keep = [c for c in range(st.shape[-1]) if c not in set(ft_cols)]
# "clean" proprioception: joint angles and velocities only, plus the commanded skill / phase / rolls / next skill. Left
# out: finger targets, wrist tracking error and the previous action - in teacher rollouts they carry privileged
# information (the sighted regrip servo and the legacy flick act on the true cube pose, the teacher's action sees it)
ph = lay["per_hand"]
com0 = lay["n_hands"] * ph
clean = [h * ph + c for h in range(lay["n_hands"]) for c in range(0, 44)] + list(range(com0, com0 + 17)) + list(range(st.shape[-1] - 13, st.shape[-1]))
ft_n = st[..., ft_cols]


def inputs(v):
    parts = {"proprio": [st[..., keep]], "ft_normal": [st], "wrench": [st[..., keep], d["wrench"].float()],
             "wrench_centroid": [st[..., keep], d["wrench"].float(), d["centroid"].float()],
             "q": [st[..., clean]], "q_ft_normal": [st[..., clean], ft_n], "q_wrench": [st[..., clean], d["wrench"].float()],
             "q_wrench_centroid": [st[..., clean], d["wrench"].float(), d["centroid"].float()]}[v]
    return torch.cat(parts, -1)


n_tr = int(0.8 * N)


class Est(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(din, a.hidden), nn.ELU(), nn.Linear(a.hidden, a.hidden), nn.ELU())
        self.gru = None if a.no_history else nn.GRU(a.hidden, a.hidden)
        self.head = nn.Sequential(nn.Linear(a.hidden, 128), nn.ELU(), nn.Linear(128, 8))

    def forward(self, x, h=None, resets=None):  # x (T, B, D)
        z = self.enc(x)
        if self.gru is None:
            return self.head(z), None
        outs = []
        for t in range(z.shape[0]):
            if resets is not None and h is not None:
                h = h * (~resets[t]).float().view(1, -1, 1)
            o, h = self.gru(z[t : t + 1], h)
            outs.append(o[0])
        return self.head(torch.stack(outs, 0)), h


def run(v):
    X = inputs(v)
    mu, sd = X[:, :n_tr].reshape(-1, X.shape[-1]).mean(0), X[:, :n_tr].reshape(-1, X.shape[-1]).std(0) + 1e-3
    Xn = ((X - mu) / sd).clamp(-8, 8)
    net = Est(X.shape[-1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    for it in range(a.iters):
        e = torch.randint(0, n_tr, (a.batch,))
        s = torch.randint(0, T - a.window, (a.batch,))
        idx = s.view(1, -1) + torch.arange(a.window).view(-1, 1)  # (W, B)
        g = lambda z: z[idx, e.view(1, -1)].to(dev)  # noqa: E731
        out, _ = net(g(Xn), None, g(d["reset"]))
        w = torch.zeros(a.window, 1, device=dev)
        w[a.burn :] = 1.0
        pm = g(pose_mask).unsqueeze(-1) * w.unsqueeze(-1)
        yt = g(tgt)
        loss_pose = (F.smooth_l1_loss(out[..., :6], yt[..., :6], reduction="none") * pm).sum() / (pm.sum() * 6 + 1)
        loss_res = (F.smooth_l1_loss(out[..., 6], yt[..., 6], reduction="none") * w).sum() / (w.sum() * a.batch)
        loss_done = (F.binary_cross_entropy_with_logits(out[..., 7], g(done), reduction="none") * w).sum() / (w.sum() * a.batch)
        loss = loss_pose + loss_res + 0.5 * loss_done
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 1000 == 0:
            print(f"[probe] {v} it {it} loss {loss.item():.4f}", flush=True)
    # test: full sequences of the held-out envs, hidden state carried
    net.eval()
    preds = []
    with torch.no_grad():
        h = None
        for c0 in range(0, T, 250):
            o, h = net(Xn[c0 : c0 + 250, n_tr:].to(dev), h, d["reset"][c0 : c0 + 250, n_tr:].to(dev))
            preds.append(o.cpu())
    pr = torch.cat(preds, 0)
    sl = slice(n_tr, N)
    valid = torch.zeros(T, N - n_tr, dtype=torch.bool)
    valid[200:] = True  # skip the first seconds (hidden state warming up)
    pm = (pose_mask[:, sl] > 0) & valid
    err_p = (pr[..., 0:3] - tgt[:, sl, 0:3]).norm(dim=-1) * 5.0  # mm
    err_r = (pr[..., 3:6] - tgt[:, sl, 3:6]).norm(dim=-1) * 5.0  # deg
    triv_p = (tgt[:, sl, 0:3] - tgt[:, :n_tr, 0:3][pose_mask[:, :n_tr] > 0].mean(0)).norm(dim=-1) * 5.0
    triv_r = (tgt[:, sl, 3:6] - tgt[:, :n_tr, 3:6][pose_mask[:, :n_tr] > 0].mean(0)).norm(dim=-1) * 5.0
    # the regrip servo's latch: last steps of the unroll
    latch = is_reg[:, sl] & (t_sk[:, sl] > 1.15) & (t_sk[:, sl] <= 1.25) & valid
    reg0 = is_reg[:, sl] & (t_sk[:, sl] <= 0.1) & valid  # regrip start: before the servo moves the hand
    err_res = (pr[..., 6] - tgt[:, sl, 6]).abs() * 10.0
    pd = torch.sigmoid(pr[..., 7]) > 0.5
    dt = done[:, sl] > 0.5
    tp = (pd & dt & valid).sum().item()
    prec, rec = tp / max(1, (pd & valid).sum().item()), tp / max(1, (dt & valid).sum().item())
    q = lambda z, m, qq: float(torch.quantile(z[m].float(), qq)) if m.any() else math.nan  # noqa: E731
    res = dict(
        offset_mm_p50=q(err_p, pm, 0.5), offset_mm_p90=q(err_p, pm, 0.9), offset_trivial_p50=q(triv_p, pm, 0.5),
        rot_deg_p50=q(err_r, pm, 0.5), rot_deg_p90=q(err_r, pm, 0.9), rot_trivial_p50=q(triv_r, pm, 0.5),
        latch_offset_mm_p50=q(err_p, latch, 0.5), latch_offset_mm_p90=q(err_p, latch, 0.9), latch_trivial_p50=q(triv_p, latch, 0.5),
        latch_rot_deg_p50=q(err_r, latch, 0.5), latch_rot_trivial_p50=q(triv_r, latch, 0.5), n_latch=int(latch.sum()),
        start_offset_mm_p50=q(err_p, reg0, 0.5), start_offset_mm_p90=q(err_p, reg0, 0.9), start_trivial_p50=q(triv_p, reg0, 0.5),
        start_rot_deg_p50=q(err_r, reg0, 0.5), start_rot_trivial_p50=q(triv_r, reg0, 0.5),
        layer_resid_deg_p50=q(err_res, valid, 0.5), layer_resid_deg_p90=q(err_res, valid, 0.9),
        done_precision=prec, done_recall=rec)
    print(f"[probe] {v}: " + ", ".join(f"{k} {val:.2f}" if isinstance(val, float) else f"{k} {val}" for k, val in res.items()), flush=True)
    return res


results = {v: run(v) for v in a.variants.split(",")}
json.dump(dict(args=vars(a), results=results), open(a.out, "w"), indent=1)
print(f"[probe] -> {a.out}", flush=True)
