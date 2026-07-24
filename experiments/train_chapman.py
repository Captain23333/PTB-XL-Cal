"""Train SimpleResNet1D on Chapman-Shaoxing (5-family labels), single seed=42.

Saves:
  checkpoints/hard_bce_chapman_seed42.pt
  results/cal_probs/probs_hard_bce_chapman_seed42.npz
    val_logits / val_targets / test_logits / test_targets / test_patient_ids

Usage:
  conda run -n ecg python experiments/train_chapman.py
  or inside tmux: python experiments/train_chapman.py
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.data_chapman import get_chapman_loader, FAMILIES
from src.models import build_chapman_model

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
PROBS_DIR = ROOT / "results" / "cal_probs"
PROBS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 40
BATCH = 64
LR = 1e-3
GPU = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Data ──────────────────────────────────────────────────────────────────────
train_loader = get_chapman_loader("train", batch_size=BATCH, num_workers=4)
val_loader   = get_chapman_loader("val",   batch_size=256,   num_workers=4)
test_loader  = get_chapman_loader("test",  batch_size=256,   num_workers=4)
print(f"Train {len(train_loader.dataset)} | Val {len(val_loader.dataset)} | Test {len(test_loader.dataset)}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = build_chapman_model(n_classes=5, backbone_name="simplresnet1d").to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params/1e6:.2f}M")

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = OneCycleLR(optimizer, max_lr=LR,
                       steps_per_epoch=len(train_loader), epochs=EPOCHS)
criterion = nn.BCEWithLogitsLoss()


def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    all_logits, all_targets, all_pids = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y_hard"].to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out["logits"], y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * len(x)
            n += len(x)
            all_logits.append(out["logits"].detach().cpu().float())
            all_targets.append(y.cpu().float())
            all_pids.append(batch["patient_id"].cpu())
    logits  = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    pids    = torch.cat(all_pids).numpy()
    return total_loss / n, logits, targets, pids


def macro_auc_safe(targets, probs):
    from sklearn.metrics import roc_auc_score
    cols = [k for k in range(targets.shape[1]) if targets[:, k].sum() > 0]
    return roc_auc_score(targets[:, cols], probs[:, cols], average="macro") if cols else float("nan")


# ── Training loop ─────────────────────────────────────────────────────────────
best_val_auc = 0.0
ckpt_path = CKPT_DIR / "hard_bce_chapman_seed42.pt"

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, _, _, _ = run_epoch(train_loader, train=True)
    val_loss, val_logits, val_targets, _ = run_epoch(val_loader)
    val_probs = torch.sigmoid(torch.from_numpy(val_logits)).numpy()
    val_auc = macro_auc_safe(val_targets, val_probs)
    print(f"Epoch {epoch:3d}  tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
          f"val_auc={val_auc:.4f}  ({time.time()-t0:.0f}s)")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), ckpt_path)
        print(f"  -> Saved checkpoint (val_auc={val_auc:.4f})")

print(f"\nTraining done. Best val AUC: {best_val_auc:.4f}")

# ── Extract logits for calibration ───────────────────────────────────────────
model.load_state_dict(torch.load(ckpt_path, map_location=device))
_, val_logits,  val_targets,  _         = run_epoch(val_loader)
_, test_logits, test_targets, test_pids = run_epoch(test_loader)

npz_path = PROBS_DIR / "probs_hard_bce_chapman_seed42.npz"
np.savez(npz_path,
         val_logits=val_logits, val_targets=val_targets,
         test_logits=test_logits, test_targets=test_targets,
         test_patient_ids=test_pids,
         family_names=np.array(FAMILIES))
print(f"Saved: {npz_path}")

# Quick sanity print
test_probs = torch.sigmoid(torch.from_numpy(test_logits)).numpy()
test_auc = macro_auc_safe(test_targets, test_probs)
print(f"Test AUC (best ckpt): {test_auc:.4f}")
for i, fam in enumerate(FAMILIES):
    prev = test_targets[:, i].mean()
    print(f"  {fam}: prevalence={prev:.3f}")
