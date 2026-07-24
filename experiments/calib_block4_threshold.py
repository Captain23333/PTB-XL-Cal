"""Block 4 -- Operating-point stability probe.

Full 6 calibrators x 8 methods x 3 seeds + ECG-FM row.
For each cell:
  - per-statement tau_k fit on val for sens=0.9 (locked); supplementary 0.8 / 0.95
  - macro |sens_test - target| and macro |spec_test - spec_val|
Spearman correlation between cell labelwise_ECE and cell sens_gap (across cells).

Output:
  results/block4_threshold.csv
  results/block4_spearman.json
"""
from __future__ import annotations
import sys, time, json
import os
sys.path.insert(0, os.environ.get("PTBXL_CAL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
import numpy as np
import pandas as pd

from src.calib_pipeline import (PROBS_DIR, BASE_METHODS, SEEDS, family_index_per_stmt,
                                 fit_all_calibrators, apply_calibrator, load_probs)
from src.calibration_metrics import labelwise_ece, threshold_transfer_gap

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
OUT_CSV = ROOT / "results/block4_threshold.csv"
OUT_SPEAR = ROOT / "results/block4_spearman.json"


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Block 4 starting")
    fam_idx, _ = family_index_per_stmt()
    rows = []
    for m in BASE_METHODS + ["ecgfm_linprobe"]:
        seeds_avail = SEEDS if m != "ecgfm_linprobe" else [42]
        for s in seeds_avail:
            path = PROBS_DIR / f"probs_{m}_seed{s}.npz"
            if not path.exists():
                continue
            d = load_probs(path)
            cals = fit_all_calibrators(d["val_logits"], d["val_targets"], fam_idx)
            for cal_name, (fn, mode) in cals.items():
                p_val = apply_calibrator(fn, mode, d["val_logits"])
                p_test = apply_calibrator(fn, mode, d["test_logits"])
                ece, _, _ = labelwise_ece(p_test, d["test_targets"])
                row = {"method": m, "seed": s, "calibrator": cal_name, "labelwise_ece": ece}
                for sens in (0.9, 0.8, 0.95):
                    ms, msp, _, ex = threshold_transfer_gap(
                        p_val, d["val_targets"], p_test, d["test_targets"],
                        target_sens=sens, sens_tol=0.005)
                    row[f"sens_gap_{sens:.2f}"] = ms
                    row[f"spec_gap_{sens:.2f}"] = msp
                    row[f"excluded_{sens:.2f}"] = len(ex)
                rows.append(row)
            print(f"  {m} seed{s} done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"  saved {OUT_CSV.name}: {len(df)} rows")

    # Spearman ECE vs sens_gap_0.9
    from scipy.stats import spearmanr
    valid = df.dropna(subset=["labelwise_ece", "sens_gap_0.90"])
    if len(valid) >= 3:
        rho, p = spearmanr(valid["labelwise_ece"], valid["sens_gap_0.90"])
    else:
        rho, p = float("nan"), float("nan")
    out = {"spearman_rho": float(rho), "p_value": float(p),
           "n_cells": int(len(valid))}
    with open(OUT_SPEAR, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Spearman rho={rho:.3f} p={p:.4f}")


if __name__ == "__main__":
    main()
