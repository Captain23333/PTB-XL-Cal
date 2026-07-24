"""Calibration analysis for Chapman-Shaoxing.

Applies C0–C5 to the Chapman SimpleResNet1D model (single seed 42),
runs the lex-selector, and reports family-level ECE.

Output: results/chapman_calib.csv  +  console table for the appendix.

Usage:
    conda run -n ecg python experiments/calib_chapman.py
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.calibrators import (
    fit_raw, fit_temperature_scalar, fit_temperature_per_label,
    fit_per_label_affine, fit_family_affine, fit_isotonic_per_label,
)
from src.calibration_metrics import labelwise_ece

ROOT = Path(__file__).resolve().parent.parent
PROBS_DIR = ROOT / "results" / "cal_probs"

FAMILIES = ["NORM", "MI", "STTC", "CD", "HYP"]
N_FAM = len(FAMILIES)

# For C4 family_affine, we need a family index array (one entry per label).
# With 5 labels = 5 families, each label belongs to exactly one family.
FAM_IDX = np.arange(N_FAM)   # label i → family i (trivially)

CALS = ["C0_raw", "C1_single_T", "C2_per_label_T",
        "C3_per_label_affine", "C4_family_affine", "C5_isotonic"]
CAL_PARAMS = {"C0_raw": 0, "C1_single_T": 1, "C2_per_label_T": 5,
              "C3_per_label_affine": 10, "C4_family_affine": 5, "C5_isotonic": -1}


def fit_and_apply(cal, val_logits, val_targets, test_logits):
    val_probs = 1 / (1 + np.exp(-val_logits))
    test_probs_raw = 1 / (1 + np.exp(-test_logits))
    if cal == "C0_raw":
        fn = fit_raw()
        return fn(test_logits), fn(val_logits)
    elif cal == "C1_single_T":
        fn = fit_temperature_scalar(val_logits, val_targets)
        return fn(test_logits), fn(val_logits)
    elif cal == "C2_per_label_T":
        fn = fit_temperature_per_label(val_logits, val_targets)
        return fn(test_logits), fn(val_logits)
    elif cal == "C3_per_label_affine":
        fn = fit_per_label_affine(val_logits, val_targets)
        return fn(test_logits), fn(val_logits)
    elif cal == "C4_family_affine":
        fn = fit_family_affine(val_logits, val_targets, FAM_IDX)
        return fn(test_logits), fn(val_logits)
    elif cal == "C5_isotonic":
        fn = fit_isotonic_per_label(val_probs, val_targets)
        return fn(test_probs_raw), fn(val_probs)
    raise ValueError(cal)


def macro_auc_safe(targets, probs):
    cols = [k for k in range(targets.shape[1]) if targets[:, k].sum() > 0]
    return roc_auc_score(targets[:, cols], probs[:, cols], average="macro") if cols else float("nan")


def lex_select(val_rows):
    best_ece = min(r["val_ece"] for r in val_rows)
    se = np.std([r["val_ece"] for r in val_rows]) / max(np.sqrt(len(val_rows)), 1)
    cands = [r for r in val_rows if r["val_ece"] <= best_ece + se]
    cands.sort(key=lambda r: (r["val_gap"], -r["val_auc"], r["n_params"]))
    return cands[0]["calibrator"]


# ── Load Chapman data ─────────────────────────────────────────────────────────
npz_path = PROBS_DIR / "probs_hard_bce_chapman_seed42.npz"
if not npz_path.exists():
    print(f"ERROR: {npz_path} not found. Run train_chapman.py first.")
    sys.exit(1)

d = np.load(npz_path)
val_logits  = d["val_logits"];  val_targets  = d["val_targets"].astype(float)
test_logits = d["test_logits"]; test_targets = d["test_targets"].astype(float)
test_pids   = d["test_patient_ids"]

print(f"Val  : {val_logits.shape}  Test: {test_logits.shape}")
print(f"Families: {FAMILIES}")
print("Test prevalence:")
for i, f in enumerate(FAMILIES):
    print(f"  {f}: {test_targets[:,i].mean():.3f}")

# ── Calibrate ─────────────────────────────────────────────────────────────────
print("\n=== Calibrating C0–C5 ===")
rows = []
val_rows_cal = []
for cal in CALS:
    test_probs, val_probs = fit_and_apply(cal, val_logits, val_targets, test_logits)
    val_ece,  _, _ = labelwise_ece(val_probs, val_targets)
    test_ece, _, _ = labelwise_ece(test_probs, test_targets)
    val_auc  = macro_auc_safe(val_targets, val_probs)
    test_auc = macro_auc_safe(test_targets, test_probs)

    # Per-family ECE on test
    fam_ece = {}
    for i, fam in enumerate(FAMILIES):
        if test_targets[:, i].sum() > 0:
            ece, _, _ = labelwise_ece(test_probs[:, [i]], test_targets[:, [i]])
        else:
            ece = float("nan")
        fam_ece[fam] = ece

    fam_str = "  ".join(f"{f}={fam_ece[f]*1e3:.2f}" for f in FAMILIES)
    print(f"  {cal:20s}  test_ece={test_ece*1e3:.2f}  auc={test_auc:.4f}  {fam_str}")

    rows.append({"calibrator": cal, "test_ece": test_ece, "test_auc": test_auc, **fam_ece})
    val_rows_cal.append({"calibrator": cal, "val_ece": val_ece, "val_auc": val_auc,
                         "val_gap": 0.0, "n_params": CAL_PARAMS[cal]})

# ── Lex-selector ──────────────────────────────────────────────────────────────
winner = lex_select(val_rows_cal)
print(f"\nLex-selector winner: {winner}")

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "chapman_calib.csv", index=False)
print(f"Saved: results/chapman_calib.csv")

# ── NORM failure check ────────────────────────────────────────────────────────
print("\n=== Appendix table (Chapman calibration) ===")
hdr = f"{'Calibrator':22s}  {'ECE×1e3':>8}  {'AUC':>6}"
for f in FAMILIES:
    hdr += f"  {f+' ECE×1e3':>13}"
hdr += f"  {'MaxRatio':>8}"
print(hdr)
for r in rows:
    max_fam_ece = max(r[f] for f in FAMILIES if not np.isnan(r[f]))
    max_ratio = max_fam_ece / r["test_ece"] if r["test_ece"] > 0 else float("nan")
    worst_fam = max((f for f in FAMILIES if not np.isnan(r[f])), key=lambda f: r[f])
    sel_mark = " ✓" if r["calibrator"] == winner else ""
    line = f"  {r['calibrator']:20s}  {r['test_ece']*1e3:8.2f}  {r['test_auc']:6.4f}"
    for f in FAMILIES:
        v = r[f]*1e3 if not np.isnan(r[f]) else float("nan")
        line += f"  {v:13.2f}"
    line += f"  {max_ratio:8.2f} ({worst_fam}){sel_mark}"
    print(line)
