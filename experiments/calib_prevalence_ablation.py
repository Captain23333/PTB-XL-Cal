"""Major-2 rebuttal experiment: PTB-XL internal prevalence-controlled ablation.

The reviewer notes that the Chapman-Shaoxing contrast confounds three variables
at once (NORM prevalence 43%->18%, label count 71->5, task structure). This
script isolates *prevalence alone* inside PTB-XL: we keep all non-NORM records,
subsample NORM-positive records to hit a target prevalence, and measure whether
the NORM-family ECE ratio (NORM ECE / macro ECE) falls toward the Chapman value
(0.81x) as prevalence drops to 18%.

Nothing is retrained. We reuse the cached hard-BCE test/val predictions and only
resample rows + refit the post-hoc calibrator.

Two estimands:
  (A) matched   : subsample BOTH val and test to target prevalence, refit C5 on
                  the resampled val, evaluate on the resampled test. Mirrors the
                  Chapman pipeline (calibrator fit and evaluated at same prevalence).
  (B) fixed-cal : fit C5 once on the FULL (43%) val, evaluate on resampled test
                  only. Isolates the test-distribution effect with the deployed
                  calibrator held fixed.

Output: results/prevalence_ablation.csv  +  console table for the paper.

Usage:
    conda run -n ecg python experiments/calib_prevalence_ablation.py
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
NORM_COL = 4                     # verified: stmts.index('NORM') == 4
PREV_GRID = [0.18, 0.25, 0.30, 0.35, None]   # None = full (~0.438, no subsample)
B = 200                          # bootstrap resamples per (seed, prevalence)
RNG = np.random.default_rng(20240605)


def subsample_to_prevalence(targets, p, rng):
    """Return row indices keeping all NORM-negative records and a random subset
    of NORM-positive records so that NORM prevalence == p. p=None keeps all."""
    norm = targets[:, NORM_COL].astype(int)
    pos = np.where(norm == 1)[0]
    neg = np.where(norm == 0)[0]
    if p is None:
        return np.arange(targets.shape[0])
    n_neg = len(neg)
    k = int(round(p * n_neg / (1.0 - p)))          # k/(k+n_neg) == p
    k = min(k, len(pos))
    keep_pos = rng.choice(pos, size=k, replace=False)
    idx = np.concatenate([neg, keep_pos])
    idx.sort()
    return idx


def eval_ratio(val_logits, val_targets, test_logits, test_targets, mode):
    """Fit C5 and return (norm_ece, macro_ece, ratio, norm_npos)."""
    val_probs = expit(val_logits).astype(np.float64)
    test_probs = expit(test_logits).astype(np.float64)
    cal = fit_isotonic_per_label(val_probs, val_targets)
    test_cal = cal(test_probs)
    macro_ece, _, _ = labelwise_ece(test_cal, test_targets)
    norm_ece, _, _ = labelwise_ece(test_cal[:, [NORM_COL]], test_targets[:, [NORM_COL]])
    ratio = norm_ece / macro_ece if macro_ece > 0 else np.nan
    return norm_ece, macro_ece, ratio, int(test_targets[:, NORM_COL].sum())


def main():
    rows = []
    for mode in ["matched", "fixed_cal"]:
        for p in PREV_GRID:
            agg = {"norm_ece": [], "macro_ece": [], "ratio": [], "norm_npos": []}
            for seed in SEEDS:
                d = np.load(PROBS_DIR / f"probs_hard_bce_seed{seed}.npz")
                vL, vT = d["val_logits"], d["val_targets"].astype(float)
                tL, tT = d["test_logits"], d["test_targets"].astype(float)

                # fixed_cal: fit once on full val
                full_val_cal = None
                if mode == "fixed_cal":
                    vp = expit(vL).astype(np.float64)
                    full_val_cal = fit_isotonic_per_label(vp, vT)

                reps = 1 if p is None else B
                for _ in range(reps):
                    if mode == "matched":
                        vidx = subsample_to_prevalence(vT, p, RNG)
                        tidx = subsample_to_prevalence(tT, p, RNG)
                        ne, me, r, npos = eval_ratio(vL[vidx], vT[vidx],
                                                     tL[tidx], tT[tidx], mode)
                    else:  # fixed_cal
                        tidx = subsample_to_prevalence(tT, p, RNG)
                        tp = expit(tL[tidx]).astype(np.float64)
                        tc = full_val_cal(tp)
                        me, _, _ = labelwise_ece(tc, tT[tidx])
                        ne, _, _ = labelwise_ece(tc[:, [NORM_COL]], tT[tidx][:, [NORM_COL]])
                        r = ne / me if me > 0 else np.nan
                        npos = int(tT[tidx][:, NORM_COL].sum())
                    agg["norm_ece"].append(ne); agg["macro_ece"].append(me)
                    agg["ratio"].append(r); agg["norm_npos"].append(npos)

            prev_label = "full(0.438)" if p is None else f"{p:.2f}"
            ratios = np.array(agg["ratio"])
            rows.append({
                "mode": mode, "prevalence": prev_label,
                "norm_ece_x1e3": np.nanmean(agg["norm_ece"]) * 1e3,
                "macro_ece_x1e3": np.nanmean(agg["macro_ece"]) * 1e3,
                "ratio_mean": np.nanmean(ratios),
                "ratio_lo": np.nanpercentile(ratios, 2.5),
                "ratio_hi": np.nanpercentile(ratios, 97.5),
                "norm_npos_mean": np.mean(agg["norm_npos"]),
            })

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "prevalence_ablation.csv"
    df.to_csv(out, index=False)

    print(f"\nB={B} bootstrap resamples per (seed,prevalence), 3 seeds pooled.")
    print(f"Chapman reference: NORM ratio = 0.81x at 18% prevalence (5-label).\n")
    for mode in ["matched", "fixed_cal"]:
        print(f"=== mode = {mode} ===")
        print(f"{'prev':>12} {'NORM npos':>10} {'NORM ECE':>9} {'macro ECE':>10} "
              f"{'ratio':>7} {'95% CI':>16}")
        for _, r in df[df["mode"] == mode].iterrows():
            ci = f"[{r['ratio_lo']:.2f}, {r['ratio_hi']:.2f}]"
            print(f"{r['prevalence']:>12} {r['norm_npos_mean']:>10.0f} "
                  f"{r['norm_ece_x1e3']:>9.2f} {r['macro_ece_x1e3']:>10.2f} "
                  f"{r['ratio_mean']:>7.2f} {ci:>16}")
        print()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
