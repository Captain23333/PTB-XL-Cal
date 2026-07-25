"""Block 3 -- Q3: Which clinical-family subgroups break global ECE?

Reads results/block2_q2_selected_calibrators.json to know per-method best+second-best
calibrator. For each (method, calibrator in {best, second}, family in 5):
  - family-restricted labelwise ECE
  - family-restricted macro-Brier
  - family-restricted slope/intercept (per-stmt averaged)
Failure cell = family_ECE > 2 * method's macro_ECE (with the chosen calibrator).
Save:
  results/block3_q3_family.csv
  results/block3_q3_family_strict.csv  (prespecified 80-cell primary subset:
      8 base methods x {C5_isotonic, C3_per_label_affine} x 5 families; this
      is the subset paper_figures.py / paper text use for the "16/16 NORM"
      headline claim -- reconstructed here as a deterministic filter of
      block3_q3_family.csv after the original generation step was lost)
  results/block3_q3_failure_cells.csv
  results/block3_q3_heatmap.npz  (8x5 matrix at C3 calibrator slot for the figure)
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
                                 family_membership_mask, fit_all_calibrators, apply_calibrator,
                                 load_probs)
from src.calibration_metrics import (labelwise_ece, macro_brier, calibration_slope_intercept)

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
SEL_PATH = ROOT / "results/block2_q2_selected_calibrators.json"
OUT_FAM = ROOT / "results/block3_q3_family.csv"
OUT_FAM_STRICT = ROOT / "results/block3_q3_family_strict.csv"
OUT_FAIL = ROOT / "results/block3_q3_failure_cells.csv"
OUT_HEAT = ROOT / "results/block3_q3_heatmap.npz"

# Pre-registered primary subset (locked before test evaluation, see
# docs/refine/FINAL_PROPOSAL.md): 8 base methods (excludes the single-seed
# ecgfm_linprobe sanity row) x top-2 calibrators (C5 isotonic, C3 per-label
# affine) x 5 clinical families = 80 cells.
STRICT_CALIBRATORS = ("C5_isotonic", "C3_per_label_affine")


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Block 3 Q3 starting")
    fam_idx, fam_order = family_index_per_stmt()
    fam_masks = family_membership_mask(fam_order)
    if not SEL_PATH.exists():
        print(f"  ERROR: missing {SEL_PATH}; run block 2 first")
        return
    selected = json.load(open(SEL_PATH))

    fam_rows = []
    fail_rows = []
    # 8 methods x 5 families heatmap at C3 (per-label affine) -- for figure
    methods_for_heatmap = BASE_METHODS
    heatmap_C3 = np.full((len(methods_for_heatmap), len(fam_order)), np.nan)
    heatmap_C4 = np.full((len(methods_for_heatmap), len(fam_order)), np.nan)

    for m in BASE_METHODS + ["ecgfm_linprobe"]:
        sel = selected.get(m)
        if sel is None:
            continue
        cals_to_eval = [sel["best"]]
        if sel.get("second"):
            cals_to_eval.append(sel["second"])
        # also force-include C3 + C4 for the heatmap
        for must in ("C3_per_label_affine", "C4_family_affine"):
            if must not in cals_to_eval:
                cals_to_eval.append(must)

        seeds_avail = SEDS if False else (SEEDS if m != "ecgfm_linprobe" else [42])
        # average across available seeds
        per_cal_per_seed = {c: {} for c in cals_to_eval}
        for s in seeds_avail:
            path = PROBS_DIR / f"probs_{m}_seed{s}.npz"
            if not path.exists():
                continue
            d = load_probs(path)
            cals = fit_all_calibrators(d["val_logits"], d["val_targets"], fam_idx)
            test_logits = d["test_logits"]; test_targets = d["test_targets"]
            for c in cals_to_eval:
                if c not in cals:
                    continue
                fn, mode = cals[c]
                p_test = apply_calibrator(fn, mode, test_logits)
                # macro test ECE
                macro_ece, _, _ = labelwise_ece(p_test, test_targets)
                per_cal_per_seed[c][s] = {"probs": p_test, "targets": test_targets, "macro_ece": macro_ece}

        # aggregate family metrics
        for c in cals_to_eval:
            cells = per_cal_per_seed[c]
            if not cells:
                continue
            macro_ece_avg = float(np.nanmean([cells[s]["macro_ece"] for s in cells]))
            for fi, fname in enumerate(fam_order):
                mask = fam_masks[fname]
                if mask.sum() == 0:
                    continue
                f_eces, f_briers, f_slopes, f_intercepts = [], [], [], []
                for s in cells:
                    p = cells[s]["probs"][:, mask]
                    t = cells[s]["targets"][:, mask]
                    e, _, _ = labelwise_ece(p, t)
                    b = macro_brier(p, t)
                    sl, it, _ = calibration_slope_intercept(p, t)
                    f_eces.append(e); f_briers.append(b)
                    f_slopes.append(float(np.nanmean(sl)))
                    f_intercepts.append(float(np.nanmean(it)))
                fam_e = float(np.nanmean(f_eces))
                fam_rows.append({
                    "method": m, "calibrator": c, "family": fname,
                    "n_stmts": int(mask.sum()),
                    "family_ece": fam_e,
                    "family_brier": float(np.nanmean(f_briers)),
                    "family_slope_mean": float(np.nanmean(f_slopes)),
                    "family_intercept_mean": float(np.nanmean(f_intercepts)),
                    "method_macro_ece_at_this_cal": macro_ece_avg,
                    "ratio": fam_e / macro_ece_avg if macro_ece_avg > 0 else float("nan"),
                })
                if fam_e > 2 * macro_ece_avg:
                    fail_rows.append({
                        "method": m, "calibrator": c, "family": fname,
                        "family_ece": fam_e, "method_macro_ece": macro_ece_avg,
                        "ratio": fam_e / macro_ece_avg,
                    })
                if m in methods_for_heatmap:
                    mi = methods_for_heatmap.index(m)
                    if c == "C3_per_label_affine":
                        heatmap_C3[mi, fi] = fam_e
                    if c == "C4_family_affine":
                        heatmap_C4[mi, fi] = fam_e
            print(f"  {m} cal={c} done")

    fam_df = pd.DataFrame(fam_rows)
    fam_df.to_csv(OUT_FAM, index=False)
    pd.DataFrame(fail_rows).to_csv(OUT_FAIL, index=False)
    np.savez_compressed(OUT_HEAT,
                         heatmap_C3=heatmap_C3, heatmap_C4=heatmap_C4,
                         methods=np.array(methods_for_heatmap),
                         families=np.array(fam_order))

    # Pre-registered 80-cell primary subset used by paper_figures.py / the
    # paper's Q3 "16/16 NORM" headline claim: 8 base methods x {C5, C3} x 5
    # families. Derived deterministically from fam_df, not re-fit, so it is
    # guaranteed consistent with block3_q3_family.csv by construction.
    strict = fam_df[
        fam_df["method"].isin(BASE_METHODS)
        & fam_df["calibrator"].isin(STRICT_CALIBRATORS)
    ].copy()
    strict["failure_cell"] = strict["ratio"] >= 2
    strict.to_csv(OUT_FAM_STRICT, index=False)

    print(f"  saved {OUT_FAM.name} ({len(fam_rows)} rows), "
          f"{OUT_FAIL.name} ({len(fail_rows)} rows), {OUT_HEAT.name}, "
          f"{OUT_FAM_STRICT.name} ({len(strict)} rows, "
          f"{int(strict['failure_cell'].sum())} failure cells)")


# tiny typo guard from above
SEDS = SEEDS

if __name__ == "__main__":
    main()
