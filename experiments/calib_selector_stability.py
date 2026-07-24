"""Major-5 rebuttal experiment: lex-selector val->test stability.

Reviewer concern: on Chapman the selector picked C5 isotonic (val-best) but C3
was better on test (9.4 vs 6.9), a 36% gap. Does the selector routinely pick a
test-suboptimal calibrator, or is Chapman an out-of-distribution outlier?

For every (method, seed, dataset) cell we fit all six calibrators on val, pick
the val-best by labelwise ECE (the dominant lex criterion), then record its RANK
by test labelwise ECE among the six and its ECE gap to the test-best calibrator.
If the val-selected calibrator is test-top-1/top-2 in in-distribution cells and
only Chapman drops it, the selector is stable and Chapman quantifies the cost of
distribution shift.

Output: results/selector_stability.csv + console summary.

Usage:
    conda run -n ecg python experiments/calib_selector_stability.py
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from src.calibrators import (fit_raw, fit_temperature_scalar,
    fit_temperature_per_label, fit_per_label_affine, fit_family_affine,
    fit_isotonic_per_label)
from src.calibration_metrics import labelwise_ece
from src.data import PTBXLDataset

PROBS_DIR = ROOT / "results" / "cal_probs"
SEEDS = [42, 123, 2024]
PTBXL_METHODS = ["hard_bce", "asl", "focal", "label_smooth", "chmc_coh",
                 "scp_soft_full", "soft", "soft_parent_ord"]
CAL_NAMES = ["C0", "C1", "C2", "C3", "C4", "C5"]


def ptbxl_fam_idx():
    fams = json.load(open(ROOT / "results" / "clinical_families.json"))
    order = ["NORM", "MI", "STTC", "CD", "HYP"]
    stmt2fam = {m: f for f, ms in fams.items() for m in ms}
    stmts = PTBXLDataset([10]).all_stmts
    idx = np.zeros(len(stmts), dtype=np.int64)
    for i, s in enumerate(stmts):
        if s in stmt2fam:
            idx[i] = order.index(stmt2fam[s])
    return idx


def all_cal_eces(vL, vT, tL, tT, fam_idx):
    """Return (val_eces[6], test_eces[6]) for C0..C5."""
    vLt = torch.as_tensor(vL, dtype=torch.float32)
    vTt = torch.as_tensor(vT, dtype=torch.float32)
    vp = expit(vL).astype(np.float64)
    fns = [
        (fit_raw(), "logit"),
        (fit_temperature_scalar(vLt, vTt), "logit"),
        (fit_temperature_per_label(vLt, vTt), "logit"),
        (fit_per_label_affine(vLt, vTt), "logit"),
        (fit_family_affine(vLt, vTt, fam_idx), "logit"),
        (fit_isotonic_per_label(vp, vT), "prob"),
    ]
    val_e, test_e = [], []
    for fn, mode in fns:
        if mode == "logit":
            vpc, tpc = fn(vL), fn(tL)
        else:
            vpc, tpc = fn(expit(vL).astype(np.float64)), fn(expit(tL).astype(np.float64))
        val_e.append(labelwise_ece(vpc, vT)[0])
        test_e.append(labelwise_ece(tpc, tT)[0])
    return np.array(val_e), np.array(test_e)


def record_cell(name, val_e, test_e, rows):
    val_pick = int(np.argmin(val_e))
    test_order = np.argsort(test_e)
    test_rank = int(np.where(test_order == val_pick)[0][0]) + 1  # 1-based
    gap = (test_e[val_pick] - test_e.min()) * 1e3
    rows.append({"cell": name, "val_pick": CAL_NAMES[val_pick],
                 "test_rank_of_pick": test_rank,
                 "test_ece_pick_x1e3": test_e[val_pick]*1e3,
                 "test_ece_best_x1e3": test_e.min()*1e3,
                 "gap_x1e3": gap})
    return test_rank, gap


def main():
    rows = []
    fam_idx = ptbxl_fam_idx()

    # PTB-XL cells
    for m in PTBXL_METHODS:
        for s in SEEDS:
            f = PROBS_DIR / f"probs_{m}_seed{s}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            ve, te = all_cal_eces(d["val_logits"], d["val_targets"].astype(float),
                                  d["test_logits"], d["test_targets"].astype(float),
                                  fam_idx)
            record_cell(f"ptbxl/{m}/s{s}", ve, te, rows)

    # ECG-FM probe (1 seed)
    f = PROBS_DIR / "probs_ecgfm_linprobe_seed42.npz"
    if f.exists():
        d = np.load(f)
        ve, te = all_cal_eces(d["val_logits"], d["val_targets"].astype(float),
                              d["test_logits"], d["test_targets"].astype(float), fam_idx)
        record_cell("ptbxl/ecgfm/s42", ve, te, rows)

    # Chapman cell (5-label, fam = identity)
    f = PROBS_DIR / "probs_hard_bce_chapman_seed42.npz"
    if f.exists():
        d = np.load(f)
        ve, te = all_cal_eces(d["val_logits"], d["val_targets"].astype(float),
                              d["test_logits"], d["test_targets"].astype(float),
                              np.arange(5))
        record_cell("chapman/hard_bce/s42", ve, te, rows)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "selector_stability.csv", index=False)

    ind = df[df["cell"].str.startswith("ptbxl")]
    print(f"=== In-distribution (PTB-XL) cells: n={len(ind)} ===")
    print(f"  val-pick is C5 isotonic in {(ind.val_pick=='C5').sum()}/{len(ind)} cells")
    print(f"  test-rank of val-pick: top-1 in {(ind.test_rank_of_pick==1).sum()}, "
          f"top-2 in {(ind.test_rank_of_pick<=2).sum()}/{len(ind)}")
    print(f"  mean test ECE gap of pick = {ind.gap_x1e3.mean():.2f}e-3 "
          f"(max {ind.gap_x1e3.max():.2f}e-3)")
    print("\n=== Out-of-distribution (Chapman) ===")
    ch = df[df["cell"].str.startswith("chapman")]
    for _, r in ch.iterrows():
        print(f"  {r.cell}: val-pick={r.val_pick}, test-rank={r.test_rank_of_pick}, "
              f"gap={r.gap_x1e3:.2f}e-3 (pick {r.test_ece_pick_x1e3:.2f} vs best {r.test_ece_best_x1e3:.2f})")
    print("\nSaved: results/selector_stability.csv")


if __name__ == "__main__":
    main()
