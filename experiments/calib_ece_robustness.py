"""Major-3 rebuttal experiment: ECE measurement robustness for the NORM finding.

Reviewer concern: the 15-bin adaptive ECE is a biased estimator, and because
NORM is high-prevalence its bin occupancy differs from low-prevalence labels, so
the high NORM/macro ratio might be an estimation artefact rather than real
miscalibration. We rule this out three ways on hard-BCE + C5 (3 seeds, fold 10):

  (1) Binning sensitivity: recompute the NORM/macro ratio for
      n_bins in {10,12,15,18,20}. If NORM stays >2x at every bin count, the
      finding is not a 15-bin artefact.
  (2) Binning-free triangulation: a Kolmogorov-Smirnov calibration statistic
      (no bins). Report NORM vs macro under KS.
  (3) Estimation-variance check: bootstrap the NORM ECE and show its reliability
      deviation exceeds its own estimation standard error (deviation >> SE).

Output: results/ece_robustness.csv + console table.

Usage:
    conda run -n ecg python experiments/calib_ece_robustness.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.special import expit

from src.calibrators import fit_isotonic_per_label
from src.calibration_metrics import labelwise_ece

PROBS_DIR = ROOT / "results" / "cal_probs"
SEEDS = [42, 123, 2024]
NORM_COL = 4
RNG = np.random.default_rng(20240605)


def ks_calibration(prob, targ):
    """Binning-free KS calibration statistic for one label.
    max |F_pred(t) - F_emp(t)| over the sorted predictions."""
    order = np.argsort(prob)
    p = prob[order]
    y = targ[order]
    n = len(p)
    # cumulative mean of predicted vs observed, normalised
    cum_pred = np.cumsum(p) / n
    cum_obs = np.cumsum(y) / n
    return float(np.max(np.abs(cum_pred - cum_obs)))


def macro_ks(prob, targ, exclude_n_pos_lt=5):
    K = prob.shape[1]
    vals = []
    for k in range(K):
        if targ[:, k].sum() < exclude_n_pos_lt:
            continue
        vals.append(ks_calibration(prob[:, k], targ[:, k]))
    return float(np.mean(vals)), vals


def load_calibrated(seed):
    d = np.load(PROBS_DIR / f"probs_hard_bce_seed{seed}.npz")
    vL, vT = d["val_logits"], d["val_targets"].astype(float)
    tL, tT = d["test_logits"], d["test_targets"].astype(float)
    cal = fit_isotonic_per_label(expit(vL).astype(np.float64), vT)
    tc = cal(expit(tL).astype(np.float64))
    return tc, tT


def main():
    # Precompute calibrated test probs per seed
    cal_data = {s: load_calibrated(s) for s in SEEDS}

    rows = []

    # (1) Binning sensitivity
    print("=== (1) Binning sensitivity: NORM/macro ECE ratio ===")
    print(f"{'n_bins':>7} {'NORM ECE':>9} {'macro ECE':>10} {'ratio':>7}")
    for nb in [10, 12, 15, 18, 20]:
        norm_e, macro_e, ratio = [], [], []
        for s in SEEDS:
            tc, tT = cal_data[s]
            me, _, _ = labelwise_ece(tc, tT, n_bins=nb)
            ne, _, _ = labelwise_ece(tc[:, [NORM_COL]], tT[:, [NORM_COL]], n_bins=nb)
            norm_e.append(ne); macro_e.append(me); ratio.append(ne / me)
        print(f"{nb:>7} {np.mean(norm_e)*1e3:>9.2f} {np.mean(macro_e)*1e3:>10.2f} "
              f"{np.mean(ratio):>7.2f}")
        rows.append({"check": "binning", "param": nb,
                     "norm_x1e3": np.mean(norm_e)*1e3,
                     "macro_x1e3": np.mean(macro_e)*1e3,
                     "ratio": np.mean(ratio)})

    # (2) Binning-free KS
    print("\n=== (2) Binning-free KS calibration statistic ===")
    norm_ks, macro_ks_v = [], []
    for s in SEEDS:
        tc, tT = cal_data[s]
        mks, _ = macro_ks(tc, tT)
        nks = ks_calibration(tc[:, NORM_COL], tT[:, NORM_COL])
        norm_ks.append(nks); macro_ks_v.append(mks)
    nks_m, mks_m = np.mean(norm_ks), np.mean(macro_ks_v)
    print(f"  NORM KS = {nks_m*1e3:.2f}e-3 | macro KS = {mks_m*1e3:.2f}e-3 | "
          f"ratio = {nks_m/mks_m:.2f}")
    rows.append({"check": "ks", "param": "binfree",
                 "norm_x1e3": nks_m*1e3, "macro_x1e3": mks_m*1e3,
                 "ratio": nks_m/mks_m})

    # (3) Estimation-variance check: bootstrap NORM ECE
    print("\n=== (3) NORM ECE vs its estimation SE (bootstrap) ===")
    B = 1000
    for s in SEEDS:
        tc, tT = cal_data[s]
        N = tc.shape[0]
        boot = []
        for _ in range(B):
            idx = RNG.integers(0, N, N)
            ne, _, _ = labelwise_ece(tc[idx][:, [NORM_COL]], tT[idx][:, [NORM_COL]])
            boot.append(ne)
        boot = np.array(boot)
        point, se = boot.mean(), boot.std()
        print(f"  seed {s}: NORM ECE = {point*1e3:.2f}e-3, SE = {se*1e3:.2f}e-3, "
              f"deviation/SE = {point/se:.1f}")
        rows.append({"check": "boot_se", "param": s,
                     "norm_x1e3": point*1e3, "macro_x1e3": se*1e3,
                     "ratio": point/se})

    pd.DataFrame(rows).to_csv(ROOT / "results" / "ece_robustness.csv", index=False)
    print("\nSaved: results/ece_robustness.csv")


if __name__ == "__main__":
    main()
