"""Calibration evaluation for InceptionTime backbone (Task A).

Applies C0-C5 calibrators to 3 seeds, runs lex-selector on val fold9,
prints results comparable to SimpleResNet1D table.

Usage:
    python experiments/calib_inceptiontime.py
"""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from pathlib import Path
import numpy as np
import pandas as pd

from src.calibrators import (
    fit_raw, fit_temperature_scalar, fit_temperature_per_label,
    fit_per_label_affine, fit_family_affine, fit_isotonic_per_label,
)
from src.calibration_metrics import (
    labelwise_ece, macro_brier, threshold_transfer_gap,
)
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PROBS_DIR = ROOT / "results/cal_probs"

# Family index for C4 (matches locked protocol)
# clinical_families.json: {family_name: [stmt, ...]}
FAMILY_JSON = ROOT / "results/clinical_families.json"
fam_data = json.load(open(FAMILY_JSON))

# Build stmt_to_family_idx mapping (load stmt order from scp_statements.csv)
import pandas as _pd
_scp = _pd.read_csv(ROOT / "data/scp_statements.csv", index_col=0)
STMT_LIST = _scp.index.tolist()
_fam_names = sorted(fam_data.keys())  # CD, HYP, MI, NORM, STTC
_stmt2fam = {}
for fi, fn in enumerate(_fam_names):
    for s in fam_data[fn]:
        _stmt2fam[s] = fi
FAM_IDX = np.array([_stmt2fam.get(s, 0) for s in STMT_LIST])

SEEDS = [42, 123, 2024]
CALIBRATORS = ["C0_raw", "C1_single_T", "C2_per_label_T",
               "C3_per_label_affine", "C4_family_affine", "C5_isotonic"]
CAL_PARAMS = {"C0_raw": 0, "C1_single_T": 1, "C2_per_label_T": 71,
              "C3_per_label_affine": 142, "C4_family_affine": 10, "C5_isotonic": -1}


def macro_auc_safe(targets, probs):
    cols = [k for k in range(targets.shape[1]) if targets[:, k].sum() > 0]
    if not cols:
        return float("nan")
    return roc_auc_score(targets[:, cols], probs[:, cols], average="macro")


def fit_and_apply(cal_name, val_logits, val_targets, test_logits):
    val_probs = 1 / (1 + np.exp(-val_logits))
    if cal_name == "C0_raw":
        fn = fit_raw()
        return fn(test_logits), fn(val_logits)
    elif cal_name == "C1_single_T":
        fn = fit_temperature_scalar(val_logits, val_targets)
        return fn(test_logits), fn(val_logits)
    elif cal_name == "C2_per_label_T":
        fn = fit_temperature_per_label(val_logits, val_targets)
        return fn(test_logits), fn(val_logits)
    elif cal_name == "C3_per_label_affine":
        fn = fit_per_label_affine(val_logits, val_targets)
        return fn(test_logits), fn(val_logits)
    elif cal_name == "C4_family_affine":
        fn = fit_family_affine(val_logits, val_targets, FAM_IDX)
        return fn(test_logits), fn(val_logits)
    elif cal_name == "C5_isotonic":
        fn = fit_isotonic_per_label(val_probs, val_targets)
        test_probs = 1 / (1 + np.exp(-test_logits))
        return fn(test_probs), fn(val_probs)
    raise ValueError(cal_name)


def lex_select(val_rows):
    """Lexicographic selector applied to val metrics only."""
    best_ece = min(r["val_ece"] for r in val_rows)
    se = np.std([r["val_ece"] for r in val_rows]) / np.sqrt(len(val_rows))
    candidates = [r for r in val_rows if r["val_ece"] <= best_ece + se]
    best_gap = min(r["val_gap"] for r in candidates)
    candidates = [r for r in candidates if r["val_gap"] <= best_gap + 1e-6]
    best_auc = max(r["val_auc"] for r in candidates)
    candidates = [r for r in candidates if r["val_auc"] >= best_auc - 1e-6]
    candidates.sort(key=lambda r: r["n_params"])
    return candidates[0]["calibrator"]


rows = []
for seed in SEEDS:
    path = PROBS_DIR / f"probs_hard_bce_inceptiontime_seed{seed}.npz"
    d = np.load(path)
    val_logits  = d["val_logits"]
    val_targets = d["val_targets"].astype(float)
    test_logits  = d["test_logits"]
    test_targets = d["test_targets"].astype(float)

    val_metrics_per_cal = []
    for cal in CALIBRATORS:
        test_probs, val_probs = fit_and_apply(cal, val_logits, val_targets, test_logits)
        val_ece, _, _ = labelwise_ece(val_probs, val_targets)
        val_auc = macro_auc_safe(val_targets, val_probs)
        val_gap, _, _, _ = threshold_transfer_gap(val_probs, val_targets, val_probs, val_targets)
        test_ece, _, _ = labelwise_ece(test_probs, test_targets)
        test_brier = macro_brier(test_probs, test_targets)
        test_auc = macro_auc_safe(test_targets, test_probs)
        val_metrics_per_cal.append({
            "calibrator": cal, "val_ece": val_ece, "val_auc": val_auc,
            "val_gap": val_gap, "n_params": CAL_PARAMS[cal],
        })
        rows.append({
            "backbone": "inceptiontime", "seed": seed, "calibrator": cal,
            "val_ece": val_ece, "test_ece": test_ece,
            "test_brier": test_brier, "test_auc": test_auc,
            "val_auc": val_auc,
        })
        print(f"  seed{seed} {cal:25s} val_ece={val_ece*1e3:.2f} test_ece={test_ece*1e3:.2f} test_auc={test_auc:.4f}")

    winner = lex_select(val_metrics_per_cal)
    print(f"  seed{seed} => lex-selector: {winner}\n")

df = pd.DataFrame(rows)
out_path = ROOT / "results/block_inceptiontime_calib.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# Summary table
print("\n=== InceptionTime Calibration Summary (mean ± std across 3 seeds) ===")
print(f"{'Calibrator':<25} {'val ECE×1e3':>12} {'test ECE×1e3':>13} {'test AUC':>10}")
print("-" * 65)
for cal in CALIBRATORS:
    sub = df[df.calibrator == cal]
    ve = sub.val_ece.mean() * 1e3
    te = sub.test_ece.mean() * 1e3
    te_std = sub.test_ece.std() * 1e3
    au = sub.test_auc.mean()
    print(f"{cal:<25} {ve:>12.2f} {te:>10.2f}±{te_std:.2f} {au:>10.4f}")

# Lex-selector per seed
print("\n=== Lex-selector results ===")
for seed in SEEDS:
    sub_rows = [r for r in rows if r["seed"] == seed]
    val_rows = []
    for cal in CALIBRATORS:
        r = next(x for x in sub_rows if x["calibrator"] == cal)
        val_rows.append({"calibrator": cal, "val_ece": r["val_ece"],
                          "val_auc": r["val_auc"], "val_gap": 0.0,
                          "n_params": CAL_PARAMS[cal]})
    winner = lex_select(val_rows)
    w_row = next(x for x in sub_rows if x["calibrator"] == winner)
    print(f"  seed{seed}: {winner}  (test_ece={w_row['test_ece']*1e3:.2f}×1e-3, test_auc={w_row['test_auc']:.4f})")
