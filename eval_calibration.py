"""PTB-XL-Cal release-quality scaffold entry point.

Usage:
    python eval_calibration.py \
        --predictions 'results/cal_probs/probs_*.npz' \
        --val-split splits/val_fold9.csv \
        --test-split splits/test_fold10.csv \
        --leaderboard leaderboard_worked_example.json

Self-test:
    python eval_calibration.py --ci-test

Pipeline (locked):
    1. Load each predictions_<method>_seed<s>.npz file matching the glob.
       Required keys: val_logits, val_targets, test_logits, test_targets,
                       val_patient_ids, test_patient_ids.
    2. Fit 6 calibrators (C0..C5) on val.
    3. Apply each to test; compute the locked metric suite:
        - macro labelwise ECE (15 adaptive equal-freq bins, exclude N_pos<5)
        - macro Brier
        - macro AUC (sklearn, min_pos=10)
        - calibration slope + intercept
        - threshold-transfer gap @ sens=0.9
        - per-family ECE on the 5-family taxonomy
    4. Apply lexicographic selector per method (val-side selection, test
       evaluation only after the choice is fixed).
    5. Emit leaderboard JSON conforming to leaderboard_schema.json.
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.calib_pipeline import (family_index_per_stmt, family_membership_mask,
                                 fit_all_calibrators, apply_calibrator, macro_auc_safe,
                                 select_best_calibrator, CALIBRATORS, CAL_PARAM_COUNT)
from src.calibration_metrics import (labelwise_ece, macro_brier,
                                       calibration_slope_intercept, threshold_transfer_gap)


PROTOCOL_VERSION = "1.0"
ECE_PROTOCOL = "labelwise, 15 adaptive equal-frequency bins, N_pos>=5 (test)"
SELECTION_RULE = "lex_ECE_threshold_AUC_param"


def parse_method_seed(path):
    name = Path(path).stem
    if not name.startswith("probs_") and not name.startswith("predictions_"):
        return None, None
    body = name.split("_", 1)[1]  # skip "probs_" / "predictions_"
    if "_seed" not in body:
        return None, None
    method, seed_s = body.rsplit("_seed", 1)
    try:
        seed = int(seed_s)
    except ValueError:
        return None, None
    return method, seed


def evaluate_one_file(path, fam_idx, fam_masks, fam_order):
    method, seed = parse_method_seed(path)
    if method is None:
        return None
    d = np.load(path)
    if not all(k in d.files for k in ("val_logits", "val_targets", "test_logits",
                                        "test_targets", "val_patient_ids", "test_patient_ids")):
        return None
    cals = fit_all_calibrators(d["val_logits"], d["val_targets"], fam_idx)
    per_cal_full = {}
    for cal_name, (fn, mode) in cals.items():
        p_val = apply_calibrator(fn, mode, d["val_logits"])
        p_test = apply_calibrator(fn, mode, d["test_logits"])
        v_ece, v_per, _ = labelwise_ece(p_val, d["val_targets"])
        v_gap, _, _, _ = threshold_transfer_gap(
            p_val, d["val_targets"], p_val, d["val_targets"],
            target_sens=0.9, sens_tol=0.005)
        v_auc = macro_auc_safe(d["val_targets"], p_val)
        t_ece, t_per, t_excluded = labelwise_ece(p_test, d["test_targets"])
        t_brier = macro_brier(p_test, d["test_targets"])
        t_auc = macro_auc_safe(d["test_targets"], p_test)
        slopes, intercepts, _ = calibration_slope_intercept(p_test, d["test_targets"])
        t_gap, t_gap_sp, _, _ = threshold_transfer_gap(
            p_val, d["val_targets"], p_test, d["test_targets"],
            target_sens=0.9, sens_tol=0.005)
        fam_ece = {}
        for f, mask in fam_masks.items():
            if mask.sum() == 0:
                fam_ece[f] = float("nan"); continue
            fe, _, _ = labelwise_ece(p_test[:, mask], d["test_targets"][:, mask])
            fam_ece[f] = fe
        per_cal_full[cal_name] = {
            "macro_auc": float(t_auc),
            "labelwise_ece": float(t_ece),
            "macro_brier": float(t_brier),
            "calib_slope": float(np.nanmean(slopes)),
            "calib_intercept": float(np.nanmean(intercepts)),
            "threshold_transfer_gap_at_sens_0.9": float(t_gap),
            "family_ece": {f: float(v) for f, v in fam_ece.items()},
            "excluded_statements": [int(i) for i in t_excluded],
            "n_params": CAL_PARAM_COUNT.get(cal_name, -1),
            "_val": {"macro_ece": float(v_ece), "per_stmt_ece": v_per,
                      "threshold_gap_sens09": float(v_gap), "macro_auc": float(v_auc),
                      "n_params": CAL_PARAM_COUNT.get(cal_name, -1)},
        }
    # Selector on val
    sel_input = {c: per_cal_full[c]["_val"] for c in per_cal_full}
    chosen, trace = select_best_calibrator(sel_input)
    return method, seed, per_cal_full, chosen, trace


def write_leaderboard(items, out_path):
    out = {
        "protocol_version": PROTOCOL_VERSION,
        "ece_protocol": ECE_PROTOCOL,
        "selection_rule": SELECTION_RULE,
        "results": items,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="results/cal_probs/probs_*.npz")
    ap.add_argument("--val-split", default="splits/val_fold9.csv")
    ap.add_argument("--test-split", default="splits/test_fold10.csv")
    ap.add_argument("--leaderboard", default="leaderboard_worked_example.json")
    ap.add_argument("--ci-test", action="store_true",
                    help="Run worked-example self-test on results/cal_probs/probs_hard_bce_seed42.npz")
    args = ap.parse_args()

    fam_idx, fam_order = family_index_per_stmt()
    fam_masks = family_membership_mask(fam_order)

    if args.ci_test:
        ROOT = Path(__file__).resolve().parent
        path = ROOT / "results/cal_probs/probs_hard_bce_seed42.npz"
        if not path.exists():
            print("CI test cannot run -- probs_hard_bce_seed42.npz absent")
            sys.exit(2)
        res = evaluate_one_file(path, fam_idx, fam_masks, fam_order)
        method, seed, full, chosen, trace = res
        print(f"CI: method={method} seed={seed} chosen={chosen}")
        # Basic sanity bounds
        ece_c0 = full["C0_raw"]["labelwise_ece"]
        ece_c3 = full["C3_per_label_affine"]["labelwise_ece"]
        print(f"  ECE_C0={ece_c0:.4f}  ECE_C3={ece_c3:.4f}")
        assert 0 < ece_c0 < 0.2, f"C0 ECE out of expected range: {ece_c0}"
        assert 0 < ece_c3 < 0.1, f"C3 ECE out of expected range: {ece_c3}"
        print("CI: PASS")
        sys.exit(0)

    files = sorted(glob.glob(args.predictions))
    print(f"[{time.strftime('%H:%M:%S')}] {len(files)} prediction files")
    items = []
    for path in files:
        try:
            res = evaluate_one_file(path, fam_idx, fam_masks, fam_order)
        except Exception as e:
            print(f"  FAIL {path}: {e}")
            continue
        if res is None:
            print(f"  SKIP {path} (unparseable)")
            continue
        method, seed, full, chosen, trace = res
        for cal_name, m in full.items():
            items.append({
                "method": method, "calibrator": cal_name, "seed": seed,
                "backbone": ("ecgfm_linprobe" if "ecgfm" in method else "simpleresnet1d_4m"),
                "is_selected": (cal_name == chosen),
                "macro_auc": m["macro_auc"],
                "labelwise_ece": m["labelwise_ece"],
                "macro_brier": m["macro_brier"],
                "calib_slope": m["calib_slope"],
                "calib_intercept": m["calib_intercept"],
                "threshold_transfer_gap_at_sens_0.9": m["threshold_transfer_gap_at_sens_0.9"],
                "family_ece": m["family_ece"],
                "excluded_statements": m["excluded_statements"],
                "n_params": m["n_params"],
                "selection_trace": (trace if cal_name == chosen else None),
            })
        print(f"  done {method} seed{seed} chosen={chosen}")
    write_leaderboard(items, args.leaderboard)
    print(f"[{time.strftime('%H:%M:%S')}] wrote {args.leaderboard} ({len(items)} items)")


if __name__ == "__main__":
    main()
