"""Block 5 -- Sensitivity & Fairness (no-GPU appendix; addresses R1 review feedback).

Three additions for ML4H findings review round 2:

1) Family-ECE sensitivity: per-method, compute family_ECE_max AND family_ECE_mean
   AND worst_non_NORM_family_ECE for the best-of-cal selection. Shows whether
   the Pareto Q2 result is driven primarily by the NORM singleton.

2) Sex / age subgroup calibration on hard_bce + C5_isotonic (the chosen winner).
   Fairness check: does ECE differ across (male/female) × (age tertile)?

3) Q4 demotion validation: confirm C0-C4 produce identical sens-gap within
   each (method, seed) — supports the decision to move Q4 to appendix.
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
from src.calibration_metrics import labelwise_ece

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
META = ROOT / "data/ptbxl_database.csv"
OUT_FAM_SENS = ROOT / "results/block5_family_sensitivity.csv"
OUT_FAIR = ROOT / "results/block5_fairness.csv"
OUT_Q4_TIE = ROOT / "results/block5_q4_monotonic_tie.csv"


def fam_ece_stats(p_test, t_test, fam_masks, exclude_family="NORM"):
    """Return dict {family: ece}, max, mean, worst_non_excluded."""
    eces = {}
    for f, mask in fam_masks.items():
        if mask.sum() == 0:
            eces[f] = float("nan"); continue
        e, _, _ = labelwise_ece(p_test[:, mask], t_test[:, mask])
        eces[f] = e
    arr = np.array([v for v in eces.values() if not np.isnan(v)])
    nonex = np.array([v for f, v in eces.items() if f != exclude_family and not np.isnan(v)])
    return {
        "fam_ece_max": float(np.nanmax(arr)),
        "fam_ece_mean": float(np.nanmean(arr)),
        "fam_ece_worst_non_norm": float(np.nanmax(nonex)) if len(nonex) else float("nan"),
        **{f"fam_ece_{f}": v for f, v in eces.items()},
    }


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Block 5 sensitivity & fairness starting")
    fam_idx, fam_order = family_index_per_stmt()
    fam_masks = family_membership_mask(fam_order)
    meta = pd.read_csv(META)[["ecg_id", "patient_id", "age", "sex", "strat_fold"]]
    print(f"  meta: {len(meta)} rows; sex 0=male/1=female; age range "
          f"{meta['age'].min():.0f}-{meta['age'].max():.0f}")

    # ---- (1) Family sensitivity for best-of-cal per method ----
    with open(ROOT / "results/block2_q2_selected_calibrators.json") as f:
        selected = json.load(f)
    rows = []
    for m, sel in selected.items():
        cal = sel["best"]
        seed_list = SEEDS if m != "ecgfm_linprobe" else [42]
        for s in seed_list:
            path = PROBS_DIR / f"probs_{m}_seed{s}.npz"
            if not path.exists(): continue
            d = load_probs(path)
            cals = fit_all_calibrators(d["val_logits"], d["val_targets"], fam_idx)
            fn, mode = cals[cal]
            p_test = apply_calibrator(fn, mode, d["test_logits"])
            ece, _, _ = labelwise_ece(p_test, d["test_targets"])
            fam_stats = fam_ece_stats(p_test, d["test_targets"], fam_masks)
            rows.append({"method": m, "seed": s, "calibrator": cal,
                         "macro_ece": float(ece), **fam_stats})
            print(f"  fam_sens {m} seed{s}: max={fam_stats['fam_ece_max']:.4f} "
                  f"mean={fam_stats['fam_ece_mean']:.4f} "
                  f"non_NORM_max={fam_stats['fam_ece_worst_non_norm']:.4f}")
    pd.DataFrame(rows).to_csv(OUT_FAM_SENS, index=False)
    print(f"  saved {OUT_FAM_SENS.name}: {len(rows)} rows")

    # ---- (2) Fairness on hard_bce + C5_isotonic (3 seeds) ----
    fair_rows = []
    for s in SEEDS:
        path = PROBS_DIR / f"probs_hard_bce_seed{s}.npz"
        d = load_probs(path)
        cals = fit_all_calibrators(d["val_logits"], d["val_targets"], fam_idx)
        fn, mode = cals["C5_isotonic"]
        p_test = apply_calibrator(fn, mode, d["test_logits"])
        # Test ecg_ids in same order as test rows (cal_probs stores test_patient_ids,
        # but PTB-XL ecg_id, patient_id, age, sex are 1:1 with records; we use patient_id).
        test_pids = d["test_patient_ids"]
        # Join via patient_id (one patient may have multiple records; use record-level join via ecg_id?
        # The recompute script saves test_patient_ids per record; here pid_array len == n_records.
        # Use the FIRST occurrence of each patient_id from meta for age/sex (constant per patient).
        meta_pid = meta.drop_duplicates("patient_id")[["patient_id", "age", "sex"]]
        pid_lookup = meta_pid.set_index("patient_id")
        # For each test record, get age, sex
        ages = pid_lookup.reindex(test_pids)["age"].values
        sexes = pid_lookup.reindex(test_pids)["sex"].values
        # Age tertiles based on test distribution
        a1, a2 = np.nanpercentile(ages, [33.33, 66.67])
        age_bin = np.where(ages <= a1, "young", np.where(ages <= a2, "mid", "old"))
        for sx_label, sx in [("male", 0), ("female", 1)]:
            mask = (sexes == sx)
            if mask.sum() < 20: continue
            ece, _, _ = labelwise_ece(p_test[mask], d["test_targets"][mask])
            fair_rows.append({"seed": s, "subgroup_type": "sex", "subgroup": sx_label,
                              "n": int(mask.sum()), "macro_ece": float(ece)})
        for ab in ["young", "mid", "old"]:
            mask = (age_bin == ab)
            if mask.sum() < 20: continue
            ece, _, _ = labelwise_ece(p_test[mask], d["test_targets"][mask])
            fair_rows.append({"seed": s, "subgroup_type": "age", "subgroup": ab,
                              "n": int(mask.sum()), "macro_ece": float(ece)})
        # Overall for reference
        ece_all, _, _ = labelwise_ece(p_test, d["test_targets"])
        fair_rows.append({"seed": s, "subgroup_type": "all", "subgroup": "all",
                          "n": int(len(p_test)), "macro_ece": float(ece_all)})
    pd.DataFrame(fair_rows).to_csv(OUT_FAIR, index=False)
    print(f"  saved {OUT_FAIR.name}: {len(fair_rows)} rows")
    fair_df = pd.DataFrame(fair_rows)
    print("\n  Fairness ECE (mean over 3 seeds):")
    print(fair_df.groupby(["subgroup_type", "subgroup"])["macro_ece"].agg(["mean", "std", "count"]).round(4))

    # ---- (3) Q4 monotonic-tie validation ----
    df_q4 = pd.read_csv("results/block4_threshold.csv")
    mono = ['C0_raw','C1_temp_scalar','C2_temp_per_label','C3_per_label_affine','C4_family_affine']
    tie_rows = []
    for (m, s), g in df_q4.groupby(['method','seed']):
        sg = g[g['calibrator'].isin(mono)]
        n_unique = sg['sens_gap_0.90'].nunique()
        # also include C5 vs mean of C0-C4
        c5_gap = g[g['calibrator']=='C5_isotonic']['sens_gap_0.90'].iloc[0]
        mono_gap = sg['sens_gap_0.90'].mean()
        tie_rows.append({"method": m, "seed": s, "n_unique_among_C0_C4": int(n_unique),
                         "mono_sens_gap": float(mono_gap),
                         "C5_isotonic_sens_gap": float(c5_gap),
                         "C5_changes_gap": bool(abs(c5_gap - mono_gap) > 1e-6)})
    pd.DataFrame(tie_rows).to_csv(OUT_Q4_TIE, index=False)
    print(f"  saved {OUT_Q4_TIE.name}: {len(tie_rows)} rows")
    tie_df = pd.DataFrame(tie_rows)
    print(f"\n  Q4 mono-tie: {(tie_df['n_unique_among_C0_C4']==1).sum()}/{len(tie_df)} "
          f"cells have C0-C4 produce identical sens_gap. "
          f"C5 differs in {tie_df['C5_changes_gap'].sum()}/{len(tie_df)} cells.")

    print(f"\n[{time.strftime('%H:%M:%S')}] Block 5 DONE")


if __name__ == "__main__":
    main()
