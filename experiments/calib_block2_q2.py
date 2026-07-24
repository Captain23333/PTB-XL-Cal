"""Block 2 -- Q2: Which base methods stay on the Pareto front after best-of-cal?

Full 6 calibrators x 8 base methods x 3 seeds grid + 1 ECG-FM row.
1) compute metric suite per cell (val + test)
2) per-method (averaged across seeds via aggregated val ECE), apply locked
   lexicographic selector on VAL fold to pick best calibrator
3) report Pareto front in (macro_AUC, labelwise_ECE, family_ECE_max) on TEST
4) patient-paired bootstrap (B=500) dominance probability vs hard_bce+best
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
                                 macro_auc_safe, load_probs, CAL_PARAM_COUNT,
                                 select_best_calibrator)
from src.calibration_metrics import (labelwise_ece, macro_brier, threshold_transfer_gap,
                                       patient_paired_bootstrap_ci)

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
OUT_GRID = ROOT / "results/block2_q2_full_grid.csv"
OUT_PARETO = ROOT / "results/block2_q2_pareto.csv"
OUT_DOM = ROOT / "results/block2_q2_dominance.csv"
OUT_SEL = ROOT / "results/block2_q2_selected_calibrators.json"


def family_ece_max(p_test, t_test, fam_masks):
    """Max family-level macro ECE across 5 families."""
    out = {}
    for f, mask in fam_masks.items():
        if mask.sum() == 0:
            out[f] = float("nan")
            continue
        e, _, _ = labelwise_ece(p_test[:, mask], t_test[:, mask])
        out[f] = e
    return out


def evaluate_cell(method, seed, path, fam_idx, fam_masks):
    d = load_probs(path)
    val_logits = d["val_logits"]; val_targets = d["val_targets"]
    test_logits = d["test_logits"]; test_targets = d["test_targets"]
    val_pid = d["val_patient_ids"]; test_pid = d["test_patient_ids"]
    cals = fit_all_calibrators(val_logits, val_targets, fam_idx)
    out = {}
    for cal_name, (fn, mode) in cals.items():
        p_val = apply_calibrator(fn, mode, val_logits)
        p_test = apply_calibrator(fn, mode, test_logits)
        # VAL metrics for selector
        v_ece, v_per, _ = labelwise_ece(p_val, val_targets)
        v_gap_s, _, _, _ = threshold_transfer_gap(
            p_val, val_targets, p_val, val_targets, target_sens=0.9, sens_tol=0.005)
        v_auc = macro_auc_safe(val_targets, p_val)
        # TEST metrics
        t_ece, t_per, t_ex = labelwise_ece(p_test, test_targets)
        t_brier = macro_brier(p_test, test_targets)
        t_auc = macro_auc_safe(test_targets, p_test)
        t_gap_s, t_gap_sp, _, _ = threshold_transfer_gap(
            p_val, val_targets, p_test, test_targets, target_sens=0.9, sens_tol=0.005)
        fam_ece = family_ece_max(p_test, test_targets, fam_masks)
        out[cal_name] = {
            "val_macro_ece": v_ece, "val_per_stmt_ece": v_per, "val_threshold_gap_sens09": v_gap_s,
            "val_macro_auc": v_auc,
            "test_macro_ece": t_ece, "test_macro_brier": t_brier, "test_macro_auc": t_auc,
            "test_threshold_gap_sens09": t_gap_s, "test_threshold_gap_spec09": t_gap_sp,
            "test_per_stmt_ece": t_per,
            "test_family_ece": fam_ece,
            "test_probs": p_test,
            "n_params": CAL_PARAM_COUNT.get(cal_name, -1),
        }
    return out, test_targets, test_pid


def aggregate_for_selector(per_seed_cells):
    """Average val metrics across seeds per calibrator, return per_calibrator_metrics
    dict with keys: macro_ece, per_stmt_ece, threshold_gap_sens09, macro_auc, n_params."""
    cals = list(per_seed_cells[list(per_seed_cells.keys())[0]].keys())
    out = {}
    for c in cals:
        eces = [per_seed_cells[s][c]["val_macro_ece"] for s in per_seed_cells]
        per_st = np.nanmean(np.stack([per_seed_cells[s][c]["val_per_stmt_ece"] for s in per_seed_cells]), axis=0)
        gaps = [per_seed_cells[s][c]["val_threshold_gap_sens09"] for s in per_seed_cells]
        aucs = [per_seed_cells[s][c]["val_macro_auc"] for s in per_seed_cells]
        np_ = per_seed_cells[list(per_seed_cells.keys())[0]][c]["n_params"]
        out[c] = {
            "macro_ece": float(np.nanmean(eces)),
            "per_stmt_ece": per_st,
            "threshold_gap_sens09": float(np.nanmean(gaps)),
            "macro_auc": float(np.nanmean(aucs)),
            "n_params": np_,
        }
    return out


def pareto_front(points):
    """points: list of dicts with 'macro_auc' (max), 'labelwise_ece' (min), 'family_ece_max' (min).
    Returns list of indices on Pareto front (3D)."""
    n = len(points)
    on_front = [True] * n
    for i in range(n):
        if not on_front[i]:
            continue
        for j in range(n):
            if i == j: continue
            pi = points[i]; pj = points[j]
            # j dominates i iff j is >= on AUC AND <= on ECE AND <= on famECE, with strict in at least one
            ge = pj["macro_auc"] >= pi["macro_auc"]
            le1 = pj["labelwise_ece"] <= pi["labelwise_ece"]
            le2 = pj["family_ece_max"] <= pi["family_ece_max"]
            strict = (pj["macro_auc"] > pi["macro_auc"] or
                      pj["labelwise_ece"] < pi["labelwise_ece"] or
                      pj["family_ece_max"] < pi["family_ece_max"])
            if ge and le1 and le2 and strict:
                on_front[i] = False
                break
    return [i for i, v in enumerate(on_front) if v]


def dominance_bootstrap(target_probs, target_targets, target_pid,
                         ref_probs, ref_targets, ref_pid,
                         fam_masks, B=500, seed=42):
    """Probability that target Pareto-dominates ref on (AUC>, ECE<, famECE_max<)
    across patient-paired resamples."""
    assert np.array_equal(target_pid, ref_pid)
    rng = np.random.default_rng(seed)
    pid = np.asarray(target_pid)
    unique, inv = np.unique(pid, return_inverse=True)
    by_p = [np.where(inv == i)[0] for i in range(len(unique))]
    Np = len(unique)
    wins = 0
    counted = 0
    for _ in range(B):
        chosen = rng.integers(0, Np, size=Np)
        rows = np.concatenate([by_p[i] for i in chosen])
        try:
            t_auc = macro_auc_safe(target_targets[rows], target_probs[rows])
            r_auc = macro_auc_safe(ref_targets[rows], ref_probs[rows])
            t_ece, _, _ = labelwise_ece(target_probs[rows], target_targets[rows])
            r_ece, _, _ = labelwise_ece(ref_probs[rows], ref_targets[rows])
            t_fmax = max(family_ece_max(target_probs[rows], target_targets[rows], fam_masks).values())
            r_fmax = max(family_ece_max(ref_probs[rows], ref_targets[rows], fam_masks).values())
            if t_auc >= r_auc and t_ece <= r_ece and t_fmax <= r_fmax and (
                t_auc > r_auc or t_ece < r_ece or t_fmax < r_fmax):
                wins += 1
            counted += 1
        except Exception:
            continue
    return wins / max(1, counted)


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Block 2 Q2 starting")
    fam_idx, fam_order = family_index_per_stmt()
    fam_masks = family_membership_mask(fam_order)

    grid_rows = []
    per_method_seeds = {}  # method -> {seed: per_cal_dict}
    per_method_targets = {}  # method -> {seed: (test_targets, test_pid)}

    # 8 base methods x 3 seeds
    for m in BASE_METHODS:
        for s in SEEDS:
            path = PROBS_DIR / f"probs_{m}_seed{s}.npz"
            if not path.exists():
                print(f"  MISSING {path.name}")
                continue
            try:
                cells, tgt, pid = evaluate_cell(m, s, path, fam_idx, fam_masks)
            except Exception as e:
                print(f"  cell fail {m} seed{s}: {e}")
                continue
            per_method_seeds.setdefault(m, {})[s] = cells
            per_method_targets.setdefault(m, {})[s] = (tgt, pid)
            for cal_name, vals in cells.items():
                grid_rows.append({
                    "method": m, "seed": s, "calibrator": cal_name,
                    "val_macro_ece": vals["val_macro_ece"],
                    "val_threshold_gap_sens09": vals["val_threshold_gap_sens09"],
                    "val_macro_auc": vals["val_macro_auc"],
                    "test_macro_ece": vals["test_macro_ece"],
                    "test_macro_brier": vals["test_macro_brier"],
                    "test_macro_auc": vals["test_macro_auc"],
                    "test_threshold_gap_sens09": vals["test_threshold_gap_sens09"],
                    "test_threshold_gap_spec09": vals["test_threshold_gap_spec09"],
                    **{f"test_family_ece_{f}": vals["test_family_ece"][f] for f in fam_order},
                    "n_params": vals["n_params"],
                })
            print(f"  cell done {m} seed{s}")

    # ECG-FM row
    ecgfm_path = PROBS_DIR / "probs_ecgfm_linprobe_seed42.npz"
    if ecgfm_path.exists():
        try:
            cells, tgt, pid = evaluate_cell("ecgfm_linprobe", 42, ecgfm_path, fam_idx, fam_masks)
            per_method_seeds["ecgfm_linprobe"] = {42: cells}
            per_method_targets["ecgfm_linprobe"] = {42: (tgt, pid)}
            for cal_name, vals in cells.items():
                grid_rows.append({
                    "method": "ecgfm_linprobe", "seed": 42, "calibrator": cal_name,
                    "val_macro_ece": vals["val_macro_ece"],
                    "val_threshold_gap_sens09": vals["val_threshold_gap_sens09"],
                    "val_macro_auc": vals["val_macro_auc"],
                    "test_macro_ece": vals["test_macro_ece"],
                    "test_macro_brier": vals["test_macro_brier"],
                    "test_macro_auc": vals["test_macro_auc"],
                    "test_threshold_gap_sens09": vals["test_threshold_gap_sens09"],
                    "test_threshold_gap_spec09": vals["test_threshold_gap_spec09"],
                    **{f"test_family_ece_{f}": vals["test_family_ece"][f] for f in fam_order},
                    "n_params": vals["n_params"],
                })
        except Exception as e:
            print(f"  ecgfm cell fail: {e}")

    pd.DataFrame(grid_rows).to_csv(OUT_GRID, index=False)
    print(f"  saved {OUT_GRID.name}: {len(grid_rows)} rows")

    # ---- Per-method best calibrator (lex selector on VAL averaged across seeds) ----
    selected = {}
    for m, seed_cells in per_method_seeds.items():
        per_cal = aggregate_for_selector(seed_cells)
        # rename keys to match selector convention
        # selector wants threshold_gap_sens09 (already)
        chosen, trace = select_best_calibrator(per_cal)
        # also pick 2nd-best (drop chosen, repeat)
        per_cal2 = {k: v for k, v in per_cal.items() if k != chosen}
        if per_cal2:
            second, _ = select_best_calibrator(per_cal2)
        else:
            second = None
        selected[m] = {"best": chosen, "second": second, "trace": trace}
    with open(OUT_SEL, "w") as f:
        json.dump(selected, f, indent=2)

    # ---- Pareto front on TEST using per-seed averaged metrics with chosen calibrator ----
    pareto_rows = []
    points = []
    method_keys = []
    for m, sel in selected.items():
        c = sel["best"]
        # average TEST metrics across seeds for this cal
        vals_per_seed = [per_method_seeds[m][s][c] for s in per_method_seeds[m]]
        auc = float(np.nanmean([v["test_macro_auc"] for v in vals_per_seed]))
        ece = float(np.nanmean([v["test_macro_ece"] for v in vals_per_seed]))
        fmax = float(np.nanmean([max(v["test_family_ece"].values()) for v in vals_per_seed]))
        points.append({"macro_auc": auc, "labelwise_ece": ece, "family_ece_max": fmax})
        method_keys.append((m, c))
    front_idx = pareto_front(points) if points else []
    for i, (m, c) in enumerate(method_keys):
        pareto_rows.append({
            "method": m, "calibrator": c,
            **points[i],
            "on_pareto_front": i in front_idx,
        })
    pd.DataFrame(pareto_rows).to_csv(OUT_PARETO, index=False)
    print(f"  saved {OUT_PARETO.name}: {len(pareto_rows)} rows; pareto front size = {len(front_idx)}")

    # ---- Dominance bootstrap vs hard_bce + best-cal ----
    dom_rows = []
    if "hard_bce" in selected:
        ref_method = "hard_bce"; ref_cal = selected["hard_bce"]["best"]
        # use seed=42 representative (need shared targets/pid)
        ref_seed = 42
        if ref_seed in per_method_seeds[ref_method]:
            ref_probs = per_method_seeds[ref_method][ref_seed][ref_cal]["test_probs"]
            ref_tgt, ref_pid = per_method_targets[ref_method][ref_seed]
            for m, sel in selected.items():
                if m == ref_method:
                    continue
                # use the matching seed if present, else first available
                use_seed = ref_seed if ref_seed in per_method_seeds[m] else next(iter(per_method_seeds[m]))
                tgt_m, pid_m = per_method_targets[m][use_seed]
                if not np.array_equal(pid_m, ref_pid):
                    # ECG-FM seed=42 vs hard_bce seed=42 -> same loader so should match
                    print(f"  WARN pid mismatch for {m}; skipping dominance")
                    continue
                tcal = sel["best"]
                tprobs = per_method_seeds[m][use_seed][tcal]["test_probs"]
                p = dominance_bootstrap(tprobs, tgt_m, pid_m,
                                          ref_probs, ref_tgt, ref_pid,
                                          fam_masks, B=500, seed=42)
                dom_rows.append({
                    "method": m, "calibrator": tcal, "seed_used": use_seed,
                    "ref_method": ref_method, "ref_calibrator": ref_cal,
                    "dominance_prob_3d": p,
                })
    pd.DataFrame(dom_rows).to_csv(OUT_DOM, index=False)
    print(f"  saved {OUT_DOM.name}: {len(dom_rows)} rows")

    print(f"[{time.strftime('%H:%M:%S')}] Block 2 Q2 DONE")


if __name__ == "__main__":
    main()
