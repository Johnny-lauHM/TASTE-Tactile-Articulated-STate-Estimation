"""Windows over recorded episodes (scripts/collect_tactile_data.py) for the completion detector.

A sample is the last `window` policy steps of one episode ending at row t; the label is the state at t:
    0 in progress | 1 done (full primitive ready: primitive_done) | 2 failed (drop, violation, broken; timeout is handled by executor)
Inputs use only what the real hand provides: Wave SE F/T (+ optional 60x60 images), finger joints, finger PD error
(target - position, a proxy for joint effort), and the commanded move id. No cube state.
"""

from __future__ import annotations

import h5py
import json
import numpy as np
import torch

LABELS = ("in_progress", "done", "failed")
from rubik_lab.moves import ALL_MOVES
N_MOVES = len(ALL_MOVES)


class TactileWindows(torch.utils.data.Dataset):
    def __init__(self, path: str, window: int = 16, images: bool = False, episodes: np.ndarray | None = None,
                 preload_max_gb: float = 8.0, label_key: str = "primitive_done", timeout_is_failure: bool = False):
        self.f = h5py.File(path, "r")
        if images:
            if "tactile_image" not in self.f:
                self.f.close()
                raise ValueError("Image training requested but dataset has no tactile_image")
            cfg = json.loads(self.f.attrs.get("config_json", "{}"))
            versions = [v for v in (self.f.attrs.get("tactile_geometry_version"),
                         cfg.get("tactile", {}).get("geometry_version")) if v is not None]
            if not versions or any(v != 2 for v in versions):
                self.f.close()
                raise ValueError("Image training requires corrected TacMap geometry_version=2; historical or unversioned images are invalid")
        self.window, self.images = window, bool(images)
        uid = self.f["episode_uid"][:]
        order = np.lexsort((self.f["step"][:], uid))  # rows grouped by episode, in time order
        uid_sorted = uid[order]
        keep = np.ones(len(order), bool) if episodes is None else np.isin(uid_sorted, episodes)
        self.order, self.uid_sorted = order, uid_sorted
        # position of each sorted row within its episode
        start = np.r_[0, np.flatnonzero(np.diff(uid_sorted)) + 1]
        pos = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
        self.ep_start = np.repeat(start, np.diff(np.r_[start, len(order)]))
        self.index = np.flatnonzero(keep)
        self.pos = pos
        g = lambda k: self.f[k][:][order]  # noqa: E731
        self.ft = torch.from_numpy(g("tactile_ft").astype(np.float32)).flatten(1)  # (R, 60)
        fp, ftg = g("finger_pos").astype(np.float32), g("finger_tgt").astype(np.float32)
        self.prop = torch.from_numpy(np.concatenate([fp.reshape(len(fp), -1), (ftg - fp).reshape(len(fp), -1)], 1))
        self.move = torch.from_numpy(g("move_id").astype(np.int64))
        lab = np.zeros(len(order), np.int64)
        if label_key not in self.f:
            raise ValueError(f"Dataset has no {label_key}; collect complete-cycle labels first")
        lab[g(label_key).astype(bool)] = 1
        failed = g("dropped").astype(bool)
        for key in ("cycle_failed", "broken"):
            if key in self.f:
                failed |= g(key).astype(bool)
        if timeout_is_failure and "timeout" in self.f:
            failed |= g("timeout").astype(bool) & (lab != 1)
        lab[failed] = 2
        self.label = torch.from_numpy(lab)
        self.uids = np.unique(uid_sorted)
        self.img_mem = None
        if self.images and self.f["tactile_image"].size / 1e9 <= preload_max_gb:
            self.img_mem = torch.from_numpy(self.f["tactile_image"][:][order])  # uint8, sorted rows

    def __len__(self):
        return len(self.index)

    def class_counts(self):
        return torch.bincount(self.label[self.index], minlength=len(LABELS))

    def __getitem__(self, i):
        r = self.index[i]
        rows = np.maximum(r - np.arange(self.window)[::-1], self.ep_start[r])  # pad by repeating the first step
        x = {
            "ft": self.ft[rows] * torch.tensor(0.2),
            "prop": self.prop[rows],
            "move": torch.nn.functional.one_hot(self.move[r], N_MOVES).float(),
        }
        if self.images and self.img_mem is not None:
            x["image"] = self.img_mem[rows].float().div(255.0)  # (T, 2, 5, 60, 60)
        elif self.images:
            src = self.order[rows]
            u, inv = np.unique(src, return_inverse=True)
            img = self.f["tactile_image"][u][inv]  # h5py needs sorted unique indices
            x["image"] = torch.from_numpy(img).float().div(255.0)  # (T, 2, 5, 60, 60)
        return x, self.label[r]


def split_episodes(path: str, val_frac: float = 0.2, seed: int = 0, group_key: str = "episode_uid"):
    """Group holdout, keeping fragments from one raw trial on the same side.

    episode_uid automatically resolves merged fragments to (source file, raw UID).
    source_file holds out complete recording runs; grasp_idx holds out grasps.
    """
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between zero and one")
    with h5py.File(path, "r") as f:
        if group_key not in f:
            raise ValueError(f"Missing split field {group_key}")
        uid, groups = f["episode_uid"][:], f[group_key][:]
        if group_key == "episode_uid" and "source_file" in f and "source_episode_uid" in f:
            _, groups = np.unique(np.stack([f["source_file"][:], f["source_episode_uid"][:]], -1),
                                  axis=0, return_inverse=True)
    unique = np.unique(groups)
    if group_key == "grasp_idx" and (unique < 0).any():
        raise ValueError("Unknown grasp IDs cannot be used for grasp holdout")
    if len(unique) < 2:
        raise ValueError("At least two independent groups are needed for train/val")
    # Every episode must belong to exactly one group, or windows could leak.
    pairs = np.unique(np.stack([uid, groups], -1), axis=0)
    if len(pairs) != len(np.unique(uid)):
        raise ValueError("An episode contains more than one split group")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    k = min(len(unique)-1, max(1, int(len(unique) * val_frac)))
    is_val = np.isin(pairs[:, 1], unique[:k])
    return pairs[~is_val, 0], pairs[is_val, 0]
