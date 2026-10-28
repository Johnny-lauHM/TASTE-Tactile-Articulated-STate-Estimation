"""Train the move-completion detector on a dataset from scripts/collect_tactile_data.py (plain PyTorch, no Isaac).

    python scripts/train_detector.py data/tactile_dataset.h5 [--images] [--epochs 20]
"""

import argparse
import json
from pathlib import Path
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "source/rubik_lab"))
from rubik_lab.detector import LABELS, CompletionDetector, TactileWindows  # noqa: E402
from rubik_lab.detector.data import split_episodes  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("data")
p.add_argument("--seed", type=int, default=123)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--images", action="store_true")
p.add_argument("--window", type=int, default=16)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--out", default="logs/detector/detector.pt")
p.add_argument("--split_by", choices=["grasp_idx", "episode_uid"], default="grasp_idx")
p.add_argument("--label_key", choices=["primitive_done", "move_done"], default="primitive_done")
p.add_argument("--device", default=None)
p.add_argument("--include_timeout_failures", action="store_true", help="legacy label ablation; normally executor handles timeout")
a = p.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

tr_ep, va_ep = split_episodes(a.data, group_key=a.split_by, seed=a.seed)
tr = TactileWindows(a.data, a.window, a.images, tr_ep, label_key=a.label_key, timeout_is_failure=a.include_timeout_failures)
va = TactileWindows(a.data, a.window, a.images, va_ep, label_key=a.label_key)
counts = tr.class_counts()
print(f"train {len(tr)} windows / {len(tr_ep)} episodes, val {len(va)} / {len(va_ep)}; train class counts {dict(zip(LABELS, counts.tolist()))}")
if (counts == 0).any():
    raise ValueError(f"Dataset lacks required classes {dict(zip(LABELS, counts.tolist()))}; collect real successes and failures before training")
weights = counts.sum() / counts.float()  # missing classes already rejected; equal class contribution
weights = weights / weights.sum() * len(LABELS)
model = CompletionDetector(images=tr.images).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(dev))
dl = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=sh, num_workers=0)  # noqa: E731
to = lambda x: {k: v.to(dev) for k, v in x.items()}  # noqa: E731

for ep in range(a.epochs):
    model.train()
    tot = 0.0
    for x, y in dl(tr, True):
        loss = loss_fn(model(to(x)), y.to(dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item() * len(y)
    model.eval()
    conf = torch.zeros(3, 3, dtype=torch.long)
    probabilities, validation_labels = [], []
    with torch.no_grad():
        for x, y in dl(va, False):
            prob = model(to(x)).softmax(-1).cpu()
            pr = prob.argmax(-1)
            probabilities.append(prob[:, 1].numpy())
            validation_labels.append(y.numpy())
            conf += torch.bincount(y * 3 + pr, minlength=9).view(3, 3)
    rec = (conf.diag() / conf.sum(1).clamp(min=1)).tolist()
    prec = (conf.diag() / conf.sum(0).clamp(min=1)).tolist()
    print(f"epoch {ep + 1:3d} loss {tot / len(tr):.4f} | val recall {dict(zip(LABELS, np.round(rec, 3)))} "
          f"precision {dict(zip(LABELS, np.round(prec, 3)))}", flush=True)  # fmt: skip
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
torch.save({"model": model.state_dict(), "images": tr.images, "window": a.window, "confusion": conf, "schema_version": 3, "timeout_is_failure": a.include_timeout_failures, "label_key": a.label_key, "split_by": a.split_by, "train_episodes": tr_ep, "val_episodes": va_ep}, a.out)
print(f"saved {a.out}\nconfusion (rows=true, cols=pred) {LABELS}:\n{conf}")

from rubik_lab.detector.metrics import trigger_metrics
validation_uid = va.uid_sorted[va.index]
metrics = {
    'scope': 'held-out grasp validation, not closed-loop blind-solving acceptance',
    'seed': a.seed, 'timeout_is_failure': a.include_timeout_failures, 'label_key': a.label_key, 'split_by': a.split_by,
    'train_class_counts': counts.tolist(), 'val_class_counts': va.class_counts().tolist(),
    'confusion': conf.tolist(),
    'trigger_metrics': [trigger_metrics(validation_uid, np.concatenate(validation_labels),
                        np.concatenate(probabilities), threshold)
                        for threshold in (0.5, 0.8, 0.9, 0.95, 0.99)],
}
Path(a.out + '.json').write_text(json.dumps(metrics, indent=2)+'\n')
print(json.dumps(metrics), flush=True)
