"""Kinematic synthesis of human-like edge grasps (pure PyTorch, no Isaac Sim).

Per hand: thumb pad on the F face, index + middle pads on the B face, all inside the hand's own outer column
(right: R column, left: L column) and the middle row; every other link keeps >= clearance from the cube; no
penetration, no hand-hand collision, forearms clear of the cube. Cube = 57 mm box at the origin (logical frame:
F=-x, B=+x, R=-y, L=+y, U=+z). Many random initializations are optimized in parallel with Adam.
Output: cache/synth_grasps.npz with wrist poses (cube frame) and finger joints of the best solutions.
    python scripts/synth_grasps.py --batch 2048 --iters 1500
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "source/rubik_lab"))
from rubik_lab.hand_kinematics import HandFK, axis_angle_to_mat, mat_to_quat, quat_to_mat  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=2048)
p.add_argument("--iters", type=int, default=1500)
p.add_argument("--keep", type=int, default=128)
p.add_argument("--clearance", type=float, default=0.004)
p.add_argument("--contacts", default="thumb:F,index:B,middle:B")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max_normal_cost", type=float, default=0.15, help="per contact (1 + n_pad . n_face); 0.3 ~ 45 deg")
p.add_argument("--region_left", default="edge", choices=["edge", "center", "wide"])
p.add_argument("--ref", default=None, help="paired synthesis: start from these grasps (cache npz) and stay close to them")
p.add_argument("--ref_copies", type=int, default=16)
p.add_argument("--prox_pos", type=float, default=30.0, help="weight of |pos - pos_ref|^2 [1/m^2 units]")
p.add_argument("--prox_rot", type=float, default=0.3, help="weight of |rotation from ref|^2 [1/rad^2]")
p.add_argument("--region_right", default="edge", choices=["edge", "center", "wide"])
p.add_argument("--center_tol", type=float, default=0.004, help="center grip: pads within this of the center line [m]")
p.add_argument("--rows", default="mid", choices=["mid", "mid_low"], help="allowed rows for F/B contacts")
p.add_argument("--swept", default="", help="moving-layer faces, e.g. \"U\" or \"F\": holding links stay out of the volume the turning layer sweeps")
p.add_argument("--free", default="index", help="fingers exempt from --swept (the trick finger / fingers the env lifts), e.g. index,ring")
p.add_argument("--swept_margin", type=float, default=0.0025)
p.add_argument("--out", default="cache/synth_grasps.npz")
a = p.parse_args()
torch.manual_seed(a.seed)
dev = "cuda"
from rubik_lab.constants import CUBE_SIZE  # noqa: E402

H = CUBE_SIZE / 2
ROW = CUBE_SIZE / 3
FACE_N = {"F": (-1.0, 0, 0), "B": (1.0, 0, 0), "U": (0, 0, 1.0), "D": (0, 0, -1.0)}
SWEPT_AXES = [torch.tensor(FACE_N[f], device=dev) for f in a.swept.split(",") if f]
FREE_FINGERS = set(a.free.split(","))
R_SWEEP = H * math.sqrt(2) + a.swept_margin  # the turning layer's corners
contacts = [tuple(c.split(":")) for c in a.contacts.split(",")]
SIDE_SIGN = {"right": -1.0, "left": 1.0}  # outer column of the right hand is -y
fks = {s: HandFK(s, dev, pts_per_link=48) for s in ("left", "right")}
B = a.batch


def sdf_box(x):  # (..., 3) -> signed distance to the cube
    q = x.abs() - H
    return torch.linalg.norm(q.clamp(min=0), dim=-1) + q.max(-1).values.clamp(max=0)


def rand_quat(n):
    q = torch.randn(n, 4, device=dev)
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)


var = {}
ref = None
if a.ref:
    # paired synthesis: B = n_ref * copies, every copy starts at its reference grasp (cube frame) with a little noise
    ref = np.load(os.path.join(ROOT, a.ref))
    n_ref = len(ref["cube_pos"])
    B = n_ref * a.ref_copies
    ref_id = torch.arange(n_ref, device=dev).repeat_interleave(a.ref_copies)
    Rc = quat_to_mat(torch.tensor(ref["cube_quat"], device=dev, dtype=torch.float32))[ref_id]
    cpos = torch.tensor(ref["cube_pos"], device=dev, dtype=torch.float32)[ref_id]
for s, sg in SIDE_SIGN.items():
    fk = fks[s]
    if ref is not None:
        p_env = torch.tensor(ref[f"{s}_ee_pos"], device=dev, dtype=torch.float32)[ref_id]
        R_env = quat_to_mat(torch.tensor(ref[f"{s}_ee_quat"], device=dev, dtype=torch.float32))[ref_id]
        pos0 = torch.einsum("bji,bj->bi", Rc, p_env - cpos) + 0.001 * torch.randn(B, 3, device=dev)
        R0 = Rc.transpose(1, 2) @ R_env
        qf = torch.tensor(ref[f"{s}_fingers"], device=dev, dtype=torch.float32)[ref_id]
        frac = ((qf - fk.lower) / (fk.upper - fk.lower)).clamp(0.02, 0.98)
        q_raw0 = torch.log(frac / (1 - frac)) + 0.2 * torch.randn(B, 22, device=dev)
    else:
        pos0 = torch.stack([torch.empty(B, device=dev).uniform_(-0.14, 0.06), sg * torch.empty(B, device=dev).uniform_(0.05, 0.16),
                            torch.empty(B, device=dev).uniform_(-0.14, 0.06)], -1)  # fmt: skip
        R0 = quat_to_mat(rand_quat(B))
        q_raw0 = torch.randn(B, 22, device=dev).mul(0.5)
    var[s] = {
        "pos": pos0.clone().requires_grad_(True),
        "rot": (1e-3 * torch.randn(B, 3, device=dev)).requires_grad_(True),
        "q_raw": q_raw0.clone().requires_grad_(True),
        "R0": R0,
        "pos_ref": pos0.clone().detach(),
    }
opt = torch.optim.Adam(
    [{"params": [v["pos"]], "lr": 3e-3} for v in var.values()]
    + [{"params": [v["rot"]], "lr": 2e-2} for v in var.values()]
    + [{"params": [v["q_raw"]], "lr": 5e-2} for v in var.values()]
)


def evaluate(details=False):
    total = torch.zeros(B, device=dev)
    pts_all, info = {}, {}
    for s, sg in SIDE_SIGN.items():
        fk, v = fks[s], var[s]
        R = axis_angle_to_mat(v["rot"]) @ v["R0"]
        q = fk.lower + torch.sigmoid(v["q_raw"]) * (fk.upper - fk.lower)
        fr = fk.forward(v["pos"], R, q)
        pads = fk.pads(fr)
        L = {}
        L["plane"] = L["inplane"] = L["normal"] = torch.zeros(B, device=dev)
        contact_links = set()
        for spec in contacts:
            finger, face = spec[0], spec[1]
            sub = spec[2] if len(spec) > 2 else None  # optional per-contact region (human home grip)
            c, m = pads[finger]
            n = torch.tensor(FACE_N[face], device=dev)
            L["plane"] = L["plane"] + (c @ n - H) ** 2
            region = a.region_left if sg > 0 else a.region_right
            if len(spec) > 3:  # optional per-contact lateral region, e.g. ring:B:lowx:edge (flicking finger far out)
                region = spec[3]
            if region == "center":  # pinch line through the center of mass: the hand can hold the cube alone
                y_lo, y_hi = -a.center_tol, a.center_tol
            elif region == "wide":  # own column + the near half of the middle column (human holding hand)
                y_lo, y_hi = (-0.002, 0.0255) if sg > 0 else (-0.0255, 0.002)
            else:  # edge: own outer column only, 3-4 mm margin (the twisting hand)
                y_lo, y_hi = (0.0135, 0.0255) if sg > 0 else (-0.0255, -0.0135)
            L["inplane"] = L["inplane"] + (c[:, 1] - y_hi).clamp(min=0) ** 2 + (y_lo - c[:, 1]).clamp(min=0) ** 2
            if face in ("F", "B"):  # rows: mid / midlow / low (or the global --rows default)
                rows = sub or a.rows
                z_lo, z_hi = {"mid": (-0.007, 0.007), "mid_low": (-H + 0.003, H / 3 - 0.0025), "midlow": (-H + 0.003, H / 3 - 0.0025),
                              "low": (-H + 0.003, -H / 3 - 0.002), "top": (H / 3 + 0.002, H - 0.003),
                              # human U-grip / D-grip: the pad centre on the far side of the middle row so the pad
                              # (~8 mm half-extent) stays off the U layer (lowr) or the D layer (high)
                              "lowr": (-H + 0.003, -0.001), "high": (0.001, H - 0.003),
                              "lowx": (-H + 0.003, -0.017),
                              "mid3": (-0.003, 0.003),  # mid3: pad centred on the middle row (one grip for U and D)
                              "midhi": (0.002, 0.007)}[rows]  # midhi: aimed 4-5 mm high, the settled grasp sags ~5 mm under gravity/press  # fmt: skip  lowx: bottom row proper (D flick finger)
                if region == "center":
                    z_lo, z_hi = -a.center_tol, a.center_tol
                L["inplane"] = L["inplane"] + (c[:, 2] - z_hi).clamp(min=0) ** 2 + (z_lo - c[:, 2]).clamp(min=0) ** 2
            else:  # U/D faces: front-back range (back / front half, or anywhere 3 mm from the edges)
                x_lo, x_hi = {"back": (0.010, H - 0.003), "front": (-H + 0.003, -0.010), "midx": (-0.005, 0.005),
                              "backx": (-0.001, H - 0.003), "frontx": (-H + 0.003, 0.001),
                              "midx3": (-0.003, 0.003),  # midx3: one grip for F and B
                              "midxf": (-0.008, -0.003)}.get(sub, (-H + 0.003, H - 0.003))  # fmt: skip  midxf: aimed 5 mm front, the thumb slides back when pressed
                L["inplane"] = L["inplane"] + (c[:, 0] - x_hi).clamp(min=0) ** 2 + (x_lo - c[:, 0]).clamp(min=0) ** 2
            L["normal"] = L["normal"] + (1.0 + m @ n)
            contact_links |= {f"{s}_{finger}_elastomer", f"{s}_{finger}_DP"}
        pts = fk.points(fr)
        pen = clr = torch.zeros(B, device=dev)
        min_clear = torch.full((B,), 1.0, device=dev)
        max_pen = torch.zeros(B, device=dev)
        for name, P in pts.items():
            d = sdf_box(P)
            tol = 0.0015 if name in contact_links else 0.0
            pen = pen + ((-d - tol).clamp(min=0) ** 2).sum(-1)
            max_pen = torch.max(max_pen, (-d).max(-1).values)
            if name not in contact_links:
                clr = clr + ((a.clearance - d).clamp(min=0) ** 2).sum(-1)
                min_clear = torch.min(min_clear, d.min(-1).values)
        L["pen"], L["clear"] = pen, clr
        # holding links (not the trick / lifted fingers) must stay out of the cylinder-band the turning layer sweeps
        sw = torch.zeros(B, device=dev)
        max_sw = torch.zeros(B, device=dev)
        for name, P in pts.items():
            if name.split("_")[1] in FREE_FINGERS:
                continue
            for ax in SWEPT_AXES:
                al = P @ ax
                rad = torch.linalg.norm(P - al.unsqueeze(-1) * ax, dim=-1)
                depth = torch.minimum(torch.minimum(al - (H - ROW - a.swept_margin), (H + a.swept_margin) - al), R_SWEEP - rad).clamp(min=0)
                sw = sw + (depth**2).sum(-1)
                max_sw = torch.max(max_sw, depth.max(-1).values)
        L["swept"] = sw
        z_w = R[:, :, 2]
        t = torch.linspace(0.03, 0.25, 8, device=dev)
        fore = v["pos"].unsqueeze(1) - t.view(1, -1, 1) * z_w.unsqueeze(1)
        L["forearm"] = ((0.05 - sdf_box(fore)).clamp(min=0) ** 2).sum(-1)
        L["prior"] = (-z_w[:, 2]).clamp(min=0) ** 2  # fingers not pointing down
        # paired synthesis: stay close to the reference wrist pose (minimal regrip motion)
        L["prox"] = (a.prox_pos * ((v["pos"] - v["pos_ref"]) ** 2).sum(-1) + a.prox_rot * (v["rot"] ** 2).sum(-1)) if ref is not None else torch.zeros(B, device=dev)
        w = {"plane": 1e4, "inplane": 1e4, "normal": 1.0, "pen": 1e4, "clear": 1e3, "forearm": 1e3, "prior": 0.1, "prox": 1.0, "swept": 1e3}
        total = total + sum(w[k] * L[k] for k in w)
        info.setdefault("terms", {})[s] = {k: w[k] * L[k] for k in w}
        pts_all[s] = torch.cat([P[:, ::4] for P in pts.values()], 1)
        info[s] = {"q": q, "R": R, "pos": v["pos"], "max_pen": max_pen, "min_clear": min_clear, "max_swept": max_sw,
                   "plane": L["plane"], "inplane": L["inplane"], "normal": L["normal"]}  # fmt: skip
    d_lr = torch.cdist(pts_all["left"], pts_all["right"]).flatten(1).min(-1).values
    total = total + 1e3 * (0.01 - d_lr).clamp(min=0) ** 2
    info["d_lr"] = d_lr
    return total, info


t0 = time.time()
for it in range(a.iters):
    loss, _ = evaluate()
    opt.zero_grad()
    loss.sum().backward()
    opt.step()
    if it % 250 == 0 or it == a.iters - 1:
        print(f"iter {it:5d}  loss median {loss.median().item():.4f}  best {loss.min().item():.5f}  ({time.time() - t0:.0f}s)", flush=True)

with torch.no_grad():
    loss, info = evaluate()
b = int(loss.argmin())
for sd in SIDE_SIGN:
    print(f"[synth] best solution {sd} terms: " + ", ".join(f"{k}={v[b].item():.4f}" for k, v in info["terms"][sd].items())
          + f" | max_pen {info[sd]['max_pen'][b] * 1e3:.1f} mm min_clear {info[sd]['min_clear'][b] * 1e3:.1f} mm swept {info[sd]['max_swept'][b] * 1e3:.1f} mm", flush=True)  # fmt: skip
print(f"[synth] best hand-hand distance {info['d_lr'][b] * 1e3:.1f} mm", flush=True)
ok = torch.ones(B, dtype=torch.bool, device=dev)
for s in SIDE_SIGN:
    i = info[s]
    ok &= (i["max_pen"] < 0.002) & (i["min_clear"] > 0.002) & (i["plane"] < 3 * 0.002**2) & (i["inplane"] < 1e-6) & (i["normal"] < len(contacts) * a.max_normal_cost)
    ok &= i["max_swept"] < 0.001
    checks = {"pen": i["max_pen"] < 0.002, "clear": i["min_clear"] > 0.002, "plane": i["plane"] < 3 * 0.002**2, "inplane": i["inplane"] < 1e-6,
              "normal": i["normal"] < len(contacts) * a.max_normal_cost, "swept": i["max_swept"] < 0.001}  # fmt: skip
    print(f"[synth] {s} pass rates: " + ", ".join(f"{k} {v.float().mean():.0%}" for k, v in checks.items()), flush=True)
ok &= info["d_lr"] > 0.005
order = torch.argsort(torch.where(ok, loss, loss + 1e6))
keep = order[: min(a.keep, int(ok.sum()))]
if ref is not None:  # best valid copy per reference grasp
    best = {}
    for i in order.tolist():
        if not ok[i]:
            break
        best.setdefault(int(ref_id[i]), i)
    keep = torch.tensor(sorted(best.values()), device=dev, dtype=torch.long)
print(f"[synth] {int(ok.sum())}/{B} solutions satisfy all constraints; keeping {len(keep)}", flush=True)
out = {"loss": loss[keep]}
if ref is not None:
    out["ref_id"] = ref_id[keep].float()
    for s in SIDE_SIGN:
        d = torch.linalg.norm(info[s]["pos"][keep] - var[s]["pos_ref"][keep], dim=-1)
        r = torch.linalg.norm(var[s]["rot"][keep], dim=-1)
        print(f"[synth] paired {s}: wrist move median {d.median() * 100:.1f} cm (max {d.max() * 100:.1f}), rotation median {torch.rad2deg(r.median()):.0f} deg", flush=True)
for s in SIDE_SIGN:
    i = info[s]
    out[f"{s}_wrist_pos"] = i["pos"][keep]
    out[f"{s}_wrist_quat"] = mat_to_quat(i["R"][keep])
    out[f"{s}_fingers"] = i["q"][keep]
    out[f"{s}_max_pen_mm"] = i["max_pen"][keep] * 1e3
    out[f"{s}_min_clear_mm"] = i["min_clear"][keep] * 1e3
if len(keep):
    for s in SIDE_SIGN:
        print(f"  {s}: max penetration {out[f'{s}_max_pen_mm'].max():.2f} mm, min clearance of other links "
              f"{out[f'{s}_min_clear_mm'].min():.2f} mm", flush=True)  # fmt: skip
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    np.savez(os.path.join(ROOT, a.out), **{k: v.detach().cpu().numpy() for k, v in out.items()})
    print(f"[synth] saved {os.path.join(ROOT, a.out)}", flush=True)
