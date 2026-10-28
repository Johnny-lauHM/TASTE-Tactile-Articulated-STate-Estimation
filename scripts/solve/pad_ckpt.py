"""Extend a solve teacher checkpoint by extra observation columns at the end (e.g. obs_act_hist): the first layers'
new input columns get zero weights, the normalizers mean 0 / std 1 there, and Adam's moments zeros, so the padded
policy acts exactly as the original and resumes with its optimizer state.

    python scripts/solve/pad_ckpt.py <in.pt> <extra columns> <out.pt>
"""
import sys

import torch

src, extra, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = torch.load(src, map_location="cpu", weights_only=False)
sd = d["model_state_dict"]
n_in = None
for k, v in list(sd.items()):
    if k.endswith((".actor.0.weight", ".critic.0.weight")):
        n_in = v.shape[1]
        sd[k] = torch.cat([v, torch.zeros(v.shape[0], extra)], 1)
    elif "obs_normalizer." in k and k.endswith(("_mean", "_var", "_std")):
        sd[k] = torch.cat([v, (torch.zeros if k.endswith("_mean") else torch.ones)(*v.shape[:-1], extra)], -1)
for st in d.get("optimizer_state_dict", {}).get("state", {}).values():  # the first-layer weights: shape (h, n_in)
    for k, v in st.items():
        if torch.is_tensor(v) and v.dim() == 2 and v.shape[1] == n_in:
            st[k] = torch.cat([v, torch.zeros(v.shape[0], extra)], 1)
torch.save(d, out)
print(f"[pad] {src}: {n_in} -> {n_in + extra} inputs -> {out}")
