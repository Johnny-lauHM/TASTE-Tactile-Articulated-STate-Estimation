"""Human finger tricks (flicks) from the human home grip: online IK of the flick path for the scripted teacher.

A flick keeps one fingertip pad on a point of the moving layer while the layer turns by theta (the point rotates with the
layer: no sliding), optionally with a small wrist motion. The path is solved by batched IK on the hand FK (pure torch)
in the logical cube frame, then converted to the env's residual actions. Feasibility study: scripts/flick_feasibility.py.
"""

from __future__ import annotations

import math

import torch


from .constants import CUBE_SIZE, FINGER_JOINTS
from .hand_kinematics import HandFK, axis_angle_to_mat

H = CUBE_SIZE / 2
NORMALS = {"U": (0, 0, 1.0), "D": (0, 0, -1.0), "F": (-1.0, 0, 0), "B": (1.0, 0, 0), "R": (0, -1.0, 0), "L": (0, 1.0, 0)}
# move -> (hand, finger, contact face of the moving layer, wrist assist, lateral side). Every trick pushes the back face
# (B) of the moving layer forward on the flicking hand's own side: a push by the pad's normal force (torque = lateral
# offset x force). Friction drive on the U face failed in sim (pad force ~0.4 N, the pad lifted off while flicking).
# This legacy path keeps wrist commands fixed. Earlier wrist-assist trials disturbed the grasp; they do not prove
# coordinated support or small wrist motions impossible. Retries re-plan the same contact-face family.
FLICKS = {
    "U": ("right", "index", "B", False, -1.0),  # index behind the top layer, pushes its back face sideways
    "U'": ("left", "index", "B", False, +1.0),
    "D": ("left", "ring", "B", False, +1.0),
    "D'": ("right", "ring", "B", False, -1.0),
    # F/B from the U/D-pinch grip (thumb under, middle over the cube): the index on top pushes the front/back row of the
    # top face sideways
    "F": ("right", "index", "U", False, -1.0),
    "F'": ("left", "index", "U", False, +1.0),
    # B/B' from the same F grip (user: wrists stay where they are, no palm-up regrasp): the ring finger already rests on
    # the back row of the U face and pushes it sideways, mirror image of the index doing F
    "B": ("left", "ring", "U", False, +1.0),
    "B'": ("right", "ring", "U", False, -1.0),
}  # fmt: skip
LAT_MIN, LAT_MAX = H - 0.016, H - 0.007  # [m] contact offset from the layer axis (torque arm): near the edge (user), the pad's own, clamped
PRESS = 0.003  # [m] pad target below the face (the finger PD offset is the pushing force)
PRESS_CL = 0.004  # [m] extra depth commanded by the closed-loop tracker (MCP: 4.8 Nm/rad, 1.9 Nm limit)


def rigid_finger_box_penetration(fk, frames, finger, layer_rotation, clearance=0.):
    """Sampled active-link depth into the two cube boxes (diagnostic IK only).

    Pad compression against the top is intentional; pad/core and all rigid
    finger/body intersections are penalized. Zero depth is not a collision proof.
    """
    links=[name for name in fk.link_pts if '_'+finger+'_' in name]
    points=fk.points(frames,links=links)
    depths=[]
    for name,pts in points.items():
        center=pts.new_tensor([0.,0.,-H/3])
        half=pts.new_tensor([H,H,2*H/3])+clearance
        depths.append((half-(pts-center).abs()).amin(-1).amax(-1))
        if not name.endswith('_elastomer'):
            local=torch.einsum('bji,bpj->bpi',layer_rotation,pts)
            center=pts.new_tensor([0.,0.,2*H/3])
            half=pts.new_tensor([H,H,H/3])+clearance
            depths.append((half-(local-center).abs()).amin(-1).amax(-1))
    if not depths:raise ValueError('No active finger collision points')
    return torch.stack(depths,-1).amax(-1).clamp(min=0)


def surface_patch_target(point, rotation, normal, axis, tangent, lateral_sign, lateral_range, press):
    """Project a pad center onto a side-face patch, not a fixed material point.

    Retains the existing edge lever-arm interval and positive compression torque;
    height stays 3mm inside both edges of the upper-layer side face. This is a
    geometric constraint, not a sticking-contact or force-transmission model.
    """
    local=torch.einsum('bji,bj->bi',rotation,point)
    lateral=lateral_sign*(lateral_sign*(local@tangent)).clamp(*lateral_range)
    height=(local@axis).clamp(H/3+.003,H-.003)
    target=(H-press)*normal+lateral[:,None]*tangent+height[:,None]*axis
    return torch.einsum('bij,bj->bi',rotation,target)


class FlickPlanner:
    def __init__(self, device, steps_deg: int = 5, max_deg: int = 70, approach_step: float = 0.003, lateral_range=None, contact_height=None, contact_press=PRESS):
        self.dev = device
        if not 0 <= contact_press < H:raise ValueError("Contact prepress must lie in [0, half cube size)")
        self.contact_press = contact_press
        self.lat_min, self.lat_max = (LAT_MIN, LAT_MAX) if lateral_range is None else lateral_range
        if not 0 < self.lat_min <= self.lat_max < H:
            raise ValueError("Contact lateral range must be inside the cube side face")
        self.contact_height = H / 3 + 0.5 * (H - H / 3) if contact_height is None else contact_height
        if not H / 3 < self.contact_height < H:
            raise ValueError("Contact height must lie inside the moving layer side face")
        self.fk = {s: HandFK(s, device, pts_per_link=8) for s in ("left", "right")}
        self.thetas = torch.deg2rad(torch.arange(0, max_deg + 1, steps_deg, device=device, dtype=torch.float32))
        self.approach_step = approach_step
        self.n_app = 12  # approach waypoints (lift off, go behind the face, press in)
        self.W = self.n_app + len(self.thetas)

    def _approach(self, pad0, p0, ax, cn):
        """(B, n_app, 3) path from the grasp pad position: back off the face first (the grasp presses the pad in; a
        pressed pad cannot slide), go 10 mm outside the contact face at the contact height, then press in to p0
        (the last approach waypoint is p0 itself)."""
        out_dir = ax  # the layer's outer side
        a0 = pad0 + 0.008 * out_dir  # lift off first (a pad dragged along the face turns the layer the wrong way)
        a0b = a0 + 0.007 * cn
        a1 = p0 + 0.007 * cn + 0.003 * out_dir
        a2 = p0 + 0.007 * cn
        keys = [pad0, a0, a0b, a1, a2, p0]
        nk = len(keys) - 1
        seg = torch.stack([torch.linalg.norm(keys[i + 1] - keys[i], dim=-1) for i in range(nk)], 1)  # (B, nk)
        cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(1)], 1)  # (B, nk + 1)
        s = torch.linspace(0, 1, self.n_app + 1, device=self.dev)[1:].unsqueeze(0) * cum[:, -1:]  # (B, n_app)
        idx = torch.searchsorted(cum.contiguous(), s.contiguous()).clamp(1, nk)
        K = torch.stack(keys, 1)  # (B, nk + 1, 3)
        lo, hi = K.gather(1, (idx - 1).unsqueeze(-1).expand(-1, -1, 3)), K.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        c0, c1 = cum.gather(1, idx - 1), cum.gather(1, idx)
        w = ((s - c0) / (c1 - c0).clamp(min=1e-6)).unsqueeze(-1)
        return lo + w * (hi - lo)

    def solve(self, move, wrist_pos_c, wrist_R_c, fingers, iters=60, phi=None, rigid_clearance=False, wrist_budget=None, surface_contact=False):
        """Logical-cube-frame wrist pose (B,3),(B,3,3) and finger joints (B,22) of the flicking hand ->
        finger targets (B, W, 22), wrist deltas dpos (B, W, 3) / drot (B, W, 3) in the cube frame, reachable (B, W).
        Waypoints: n_app approach points, then the layer angles self.thetas. `phi` (B,) [rad]: rotation the layer
        already has (in the move's direction); the contact point is taken on the turned layer."""
        side, finger, cface, assist, lat = FLICKS[move]
        fk, dev = self.fk[side], self.dev
        B = fingers.shape[0]
        face = move[0]
        ax = torch.tensor(NORMALS[face], device=dev)
        sgn = -1.0 if not move.endswith("'") else 1.0  # clockwise = -90 deg about the outward normal
        cn = torch.tensor(NORMALS[cface], device=dev)
        if surface_contact and cface==face:raise ValueError("Surface patch diagnostic requires a side contact face")
        cols = [i for i, j in enumerate(FINGER_JOINTS) if j.startswith(finger)]
        # the holding fingers compensate the wrist assist so their pads stay put (else the wrist drags the cube along)
        hold = [f for f in ("thumb", "index", "middle") if f != finger]
        hold_cols = [i for i, j in enumerate(FINGER_JOINTS) if j.split("_")[0] in hold]
        phi = torch.zeros(B, device=dev) if phi is None else phi
        R_phi = axis_angle_to_mat(ax.expand(B, 3) * sgn * phi.unsqueeze(-1))
        with torch.no_grad():
            pads_init = fk.pads(fk.forward(wrist_pos_c, wrist_R_c, fingers))
            pad0 = pads_init[finger][0]
            pad0_l = torch.einsum("bji,bj->bi", R_phi, pad0)  # in the (un-turned) layer frame
            hold0 = {f: pads_init[f][0] for f in hold}
        # contact point: on the contact face, PRESS below it, inside the moving layer, LATERAL to the hand's side
        tang = torch.linalg.cross(cn, ax) if cface != face else None  # lateral direction on the contact face
        p0 = pad0_l - ((pad0_l @ cn) - (H - self.contact_press)).unsqueeze(-1) * cn
        if cface == face:  # friction drive on the outer face: torque arm = radius from the axis, keep it 18-26 mm
            rad = p0 - (p0 @ ax).unsqueeze(-1) * ax
            rn = rad.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            p0 = p0 + (rn.clamp(0.018, 0.026) / rn - 1.0) * rad
        if cface != face:
            along = p0 @ ax
            p0 = p0 + (self.contact_height - along).unsqueeze(-1) * ax  # centre of the layer's side face
            t = tang * (1.0 if (tang @ torch.tensor(NORMALS["L"], device=dev)) > 0 else -1.0)  # unit +y (L side)
            p0 = p0 + (lat * torch.clamp(lat * (p0 @ t), self.lat_min, self.lat_max) - p0 @ t).unsqueeze(-1) * t
        with torch.no_grad():
            app = self._approach(pad0, torch.einsum("bij,bj->bi", R_phi, p0), ax, torch.einsum("bij,j->bi", R_phi, cn))
        raw = fingers[:, cols].clone().requires_grad_(True)
        dpos = torch.zeros(B, 3, device=dev, requires_grad=assist)
        drot = torch.zeros(B, 3, device=dev, requires_grad=assist)
        raw_h = fingers[:, hold_cols].clone().requires_grad_(assist)
        groups = [{"params": [raw], "lr": 0.03}] + ([{"params": [dpos], "lr": 1e-3}, {"params": [drot], "lr": 1e-2}, {"params": [raw_h], "lr": 0.02}] if assist else [])
        W = self.W
        out_q = torch.zeros(B, W, 22, device=dev)
        out_dp = torch.zeros(B, W, 3, device=dev)
        out_dr = torch.zeros(B, W, 3, device=dev)
        ok = torch.zeros(B, W, dtype=torch.bool, device=dev)
        self.last_pads = torch.zeros(B, W, 3, device=dev)  # planned pad centers (debug)
        self.last_targets = torch.zeros(B, W, 3, device=dev)
        self.last_ok_raw = torch.zeros(B, W, dtype=torch.bool, device=dev)
        self.last_norm = torch.zeros(B, W, 2, device=dev)  # (m.tn, m.ax) of the planned pad
        self.last_tn = torch.zeros(B, W, 3, device=dev)  # face normal at each waypoint (zero during the approach)
        for k in range(W):
            if k < self.n_app:
                target, tn, tol = app[:, k], torch.einsum("bij,j->bi", R_phi, cn), 0.006
                w_n, w_ax = (1e-5 if k < self.n_app - 2 else 1e-4), 0.0
            else:
                Rl = axis_angle_to_mat(ax.expand(B, 3) * sgn * (phi + self.thetas[k - self.n_app]).unsqueeze(-1))
                target = torch.einsum("bij,bj->bi", Rl, p0)
                tn, w_n, w_ax, tol = torch.einsum("bij,j->bi", Rl, cn), 2e-4, (0.0 if cface == face else 1e-4), 0.004
            opt = torch.optim.Adam(groups)
            n_it = iters if k == 0 else (iters // 4 if k < self.n_app else iters // 2)  # warm-started from the previous waypoint
            for _ in range(n_it):
                q = fingers.clone()
                q[:, cols] = torch.max(torch.min(raw, fk.upper[cols]), fk.lower[cols])
                if assist:
                    q[:, hold_cols] = torch.max(torch.min(raw_h, fk.upper[hold_cols]), fk.lower[hold_cols])
                R = axis_angle_to_mat(drot) @ wrist_R_c if assist else wrist_R_c
                frames = fk.forward(wrist_pos_c + (dpos if assist else 0), R, q)
                pads = fk.pads(frames)
                c, m = pads[finger]
                if surface_contact and k>=self.n_app-1:
                    target=surface_patch_target(c,R_phi if k<self.n_app else Rl,cn,ax,t,lat,(self.lat_min,self.lat_max),self.contact_press)
                # pad normal: into the face, and never along the layer axis (that component lifts the cube instead of
                # turning the layer; a curled finger can't present the pad flat, so in-plane tilt is tolerated)
                loss = ((c - target) ** 2).sum(-1) + w_n * (1 + (m * tn).sum(-1)) + w_ax * (m @ ax) ** 2
                if rigid_clearance:
                    depth=rigid_finger_box_penetration(fk,frames,finger,R_phi if k<self.n_app else Rl,clearance=.0005)
                    loss=loss+10*depth.square()
                if assist:
                    loss = loss + 10 * ((dpos.norm(dim=-1) - 0.01).clamp(min=0) ** 2) + 1e-3 * ((drot.norm(dim=-1) - math.radians(10)).clamp(min=0) ** 2)
                    loss = loss + 4 * sum(((pads[f][0] - hold0[f]) ** 2).sum(-1) for f in hold)
                opt.zero_grad()
                loss.sum().backward()
                opt.step()
                if assist and wrist_budget is not None:
                    with torch.no_grad():
                        for value,budget in zip((dpos,drot),wrist_budget):
                            radius=(budget-1e-6).clamp(min=0)[:,None]
                            value.mul_((radius/value.norm(dim=-1,keepdim=True).clamp(min=1e-12)).clamp(max=1))
            with torch.no_grad():
                q = fingers.clone()
                q[:, cols] = torch.max(torch.min(raw, fk.upper[cols]), fk.lower[cols])
                if assist:
                    q[:, hold_cols] = torch.max(torch.min(raw_h, fk.upper[hold_cols]), fk.lower[hold_cols])
                R = axis_angle_to_mat(drot) @ wrist_R_c if assist else wrist_R_c
                frames = fk.forward(wrist_pos_c + (dpos if assist else 0), R, q)
                pads = fk.pads(frames)
                c, m = pads[finger]
                if surface_contact and k>=self.n_app-1:
                    target=surface_patch_target(c,R_phi if k<self.n_app else Rl,cn,ax,t,lat,(self.lat_min,self.lat_max),self.contact_press)
                good = torch.linalg.norm(c - target, dim=-1) < tol
                if rigid_clearance:
                    good &= rigid_finger_box_penetration(fk,frames,finger,R_phi if k<self.n_app else Rl)<=.0001
                if assist:  # the grip must stay (holding pads within 4 mm of where they were)
                    good &= torch.stack([torch.linalg.norm(pads[f][0] - hold0[f], dim=-1) for f in hold], -1).amax(-1) < 0.004
                if k >= self.n_app - 1:
                    good &= (m * tn).sum(-1) < (-0.3 if cface == face else -0.1)  # the pad normal turns with the finger, not the face
                    if cface != face:
                        good &= (m @ ax).abs() < 0.8
                ok[:, k] = good
                self.last_pads[:, k] = c
                self.last_targets[:, k] = target
                self.last_ok_raw[:, k] = good
                self.last_norm[:, k, 0], self.last_norm[:, k, 1] = (m * tn).sum(-1), m @ ax
                if k >= self.n_app:
                    self.last_tn[:, k] = tn
                out_q[:, k], out_dp[:, k], out_dr[:, k] = q, dpos.detach(), drot.detach()
        # reachable prefix only (the flick is executed up to the last continuously reachable waypoint)
        ok = torch.cumprod(ok.float(), dim=1).bool()
        return out_q, out_dp, out_dr, ok

    def track(self, move, wrist_pos_c, wrist_R_c, fingers, q_init, target, tn=None, iters=10, press=None, align_normal=False, pad_normal=None):
        """closed loop: finger joints (B,22) that put the trick pad on `target` (B,3) from the *current* wrist pose
        (the wrist drifts when the grasp forces change, so open-loop joint paths miss by ~1-2 cm)."""
        side, finger = FLICKS[move][:2]
        fk = self.fk[side]
        cols = [i for i, j in enumerate(FINGER_JOINTS) if j.startswith(finger)]
        if tn is not None:  # push through the face (the contact is soft and the PD lags); `press` (B,) ramps up on a stall
            target = target - (PRESS_CL if press is None else press.unsqueeze(-1)) * tn
        raw = q_init[:, cols].clone().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=0.03)
        for _ in range(iters):
            q = fingers.clone()
            q[:, cols] = torch.max(torch.min(raw, fk.upper[cols]), fk.lower[cols])
            c, m = fk.pads(fk.forward(wrist_pos_c, wrist_R_c, q))[finger]
            loss = ((c - target) ** 2).sum(-1)
            if pad_normal is not None:
                loss = loss + 2e-4 * (1 - (m * pad_normal).sum(-1))
            if align_normal and tn is not None:
                axis = torch.tensor(NORMALS[move[0]], device=self.dev)
                in_contact_path = (tn.norm(dim=-1) > 0.5).float()
                loss = loss + in_contact_path * (2e-4 * (1 + (m * tn).sum(-1)) + 1e-4 * (m @ axis) ** 2)
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
        with torch.no_grad():
            q = fingers.clone()
            q[:, cols] = torch.max(torch.min(raw, fk.upper[cols]), fk.lower[cols])
        return q

    def reach_deg(self, ok):
        """continuously reachable layer angle [deg] from the ok mask of solve()."""
        n = ok.sum(-1) - self.n_app
        return torch.where(n > 0, (n - 1).float() * float(torch.rad2deg(self.thetas[1])), torch.zeros_like(n, dtype=torch.float))


def to_cube_frame(env, side, mode: str = "plan"):
    """wrist pose and finger joints of one hand in the logical cube frame (pos (N,3), R (N,3,3), fingers (N,22), Rc).
    mode "plan": the *actual* wrist pose (the grasp squeezes, so the wrist sits mm outside its target and planning from
    the target put the pad ~1 cm off) with the *target* finger joints (the actual ones have the pads pressed in);
    mode "actual": actual wrist and fingers."""
    from .hand_kinematics import quat_to_mat

    _, logical_q, _, _ = env._cube_state()
    if mode == "cache" and env.grasp_cache is not None:  # the cached grasp itself (same input as the offline reach study)
        g = {k: v[env.episode_grasp_idx.clamp(min=0)] for k, v in env.grasp_cache.items()}
        Rc_g = quat_to_mat(g["cube_quat"])
        pos_c = torch.einsum("bji,bj->bi", Rc_g, g[f"{side}_ee_pos"] - g["cube_pos"])
        return pos_c, Rc_g.transpose(1, 2) @ quat_to_mat(g[f"{side}_ee_quat"]), g[f"{side}_fingers"], quat_to_mat(logical_q)
    p, q = env.ctrl[side].ee_pose_w()
    fingers = env.finger_grasp[side] if mode == "plan" else env.hands[side].data.joint_pos[:, env.finger_ids[side]]
    core_p = env.cube.data.body_pos_w[:, env.core_id]
    Rc = quat_to_mat(logical_q)
    pos_c = torch.einsum("bji,bj->bi", Rc, p - core_p)
    R_c = Rc.transpose(1, 2) @ quat_to_mat(q)
    return pos_c, R_c, fingers, Rc
