"""Block 1 -- Q1: Which calibrator family helps?

hard_bce only (B0) x C0..C5 x 3 seeds + 1 ECG-FM row (if available).
Computes labelwise_ece, macro_brier, slope/intercept, threshold gap @ 0.9 / 0.8 / 0.95.
Patient-paired bootstrap CI (B=1000) of ECE-reduction vs C0.

Outputs results/block1_q1.csv (long format).
"""
from __future__ import annotations
import sys, time, json
import os
sys.path.insert(0, os.environ.get("PTBXL_CAL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
import numpy as np
import pandas as pd

from src.calib_pipeline import (PROBS_DIR, family_index_per_stmt, fit_all_calibrators,
                                 apply_calibrator, macro_auc_safe, load_probs, CAL_PARAM_COUNT)
from src.calibration_metrics import (labelwise_ece, macro_brier, calibration_slope_intercept,
                                       threshold_transfer_gap, patient_paired_bootstrap_ci)

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
OUT_CSV = ROOT / "results/block1_q1.csv"
OUT_BOOT = ROOT / "results/block1_q1_bootstrap.csv"


def run_one_cell(method, seed, path, fam_idx):
    print(f"  cell {method} seed{seed}  loading {path.name}")
    d = load_probs(path)
    val_logits = d["val_logits"]; val_targets = d["val_targets"]
    test_logits = d["test_logits"]; test_targets = d["test_targets"]
    test_pid = d["test_patient_ids"]
    cals = fit_all_calibrators(val_logits, val_targets, fam_idx)
    rows = []
    test_probs_per_cal = {}
    for cal_name, (fn, mode) in cals.items():
        p_test = apply_calibrator(fn, mode, test_logits)
        test_probs_per_cal[cal_name] = p_test
        ece, per_stmt, ex_ece = labelwise_ece(p_test, test_targets)
        br = macro_brier(p_test, test_targets)
        slopes, intercepts, _ = calibration_slope_intercept(p_test, test_targets)
        # threshold transfer at 0.9/0.8/0.95
        v_p = apply_calibrator(fn, mode, val_logits)
        gap_rows = {}
        for sens in (0.9, 0.8, 0.95):
            ms, msp, _, ex = threshold_transfer_gap(
                v_p, val_targets, p_test, test_targets,
                target_sens=sens, sens_tol=0.005)
            gap_rows[f"sens{sens:.2f}_gap_sens"] = ms
            gap_rows[f"sens{sens:.2f}_gap_spec"] = msp
            gap_rows[f"sens{sens:.2f}_excluded"] = len(ex)
        auc = macro_auc_safe(test_targets, p_test)
        rows.append({
            "backbone": method, "seed": seed, "calibrator": cal_name,
            "macro_ece": ece, "macro_brier": br,
            "calib_slope_mean": float(np.nanmean(slopes)),
            "calib_intercept_mean": float(np.nanmean(intercepts)),
            "macro_auc": auc,
            "n_excluded_ece": len(ex_ece),
            "n_params": CAL_PARAM_COUNT.get(cal_name, -1),
            **gap_rows,
        })
    return rows, test_probs_per_cal, test_targets, test_pid


def bootstrap_ece_vs_c0(test_probs_per_cal, test_targets, test_pid, B=1000):
    """Paired bootstrap of (ECE_C0 - ECE_Cx) on patients. Returns rows."""
    if "C0_raw" not in test_probs_per_cal:
        return []
    p_a = test_probs_per_cal["C0_raw"]
    rows = []
    def ece_metric(t, p):
        e, _, _ = labelwise_ece(p, t)
        return e
    for cal_name, p_b in test_probs_per_cal.items():
        if cal_name == "C0_raw":
            continue
        # patient_paired_bootstrap_ci returns mean(metric_b - metric_a); we want ECE reduction
        # so report -diff = ECE_C0 - ECE_Cx (positive = improvement)
        diff_mean, lo, hi, p2 = patient_paired_bootstrap_ci(
            ece_metric, p_a, p_b, test_targets, test_pid, n_boot=B, seed=42)
        rows.append({
            "calibrator": cal_name,
            "ece_reduction_mean": -diff_mean,
            "ci_lo": -hi, "ci_hi": -lo,
            "p_two_sided": p2,
        })
    return rows


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Block 1 Q1 starting")
    fam_idx, fam_order = family_index_per_stmt()

    rows_all = []
    boot_rows_all = []

    # SimpleResNet1D × hard_bce × 3 seeds
    for s in (42, 123, 2024):
        path = PROBS_DIR / f"probs_hard_bce_seed{s}.npz"
        if not path.exists():
            print(f"  SKIP missing {path.name}")
            continue
        rows, probs_pc, tgt, pid = run_one_cell("simpleresnet1d_hard_bce", s, path, fam_idx)
        rows_all.extend(rows)
        for br in bootstrap_ece_vs_c0(probs_pc, tgt, pid, B=1000):
            br["backbone"] = "simpleresnet1d_hard_bce"
            br["seed"] = s
            boot_rows_all.append(br)

    # ECG-FM row (single seed=42) if available
    ecgfm_path = PROBS_DIR / "probs_ecgfm_linprobe_seed42.npz"
    if ecgfm_path.exists():
        rows, probs_pc, tgt, pid = run_one_cell("ecgfm_linprobe", 42, ecgfm_path, fam_idx)
        rows_all.extend(rows)
        for br in bootstrap_ece_vs_c0(probs_pc, tgt, pid, B=1000):
            br["backbone"] = "ecgfm_linprobe"
            br["seed"] = 42
            boot_rows_all.append(br)
    else:
        print("  ECG-FM probs absent -- Q1 will report SimpleResNet1D only")

    pd.DataFrame(rows_all).to_csv(OUT_CSV, index=False)
    pd.DataFrame(boot_rows_all).to_csv(OUT_BOOT, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] saved {OUT_CSV.name} ({len(rows_all)} rows) + "
          f"{OUT_BOOT.name} ({len(boot_rows_all)} rows)")


if __name__ == "__main__":
    main()
