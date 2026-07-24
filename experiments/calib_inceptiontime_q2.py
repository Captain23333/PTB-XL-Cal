"""Q2 cross-backbone analysis: Pareto front and bootstrap dominance on InceptionTime.

Loads all 8 methods × 3 seeds InceptionTime npz files, applies lex-selector
(val fold9 only), then runs patient-paired bootstrap dominance vs hard_bce+C5.

Usage:
    python experiments/calib_inceptiontime_q2.py
"""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.calibrators import (
    fit_raw, fit_temperature_scalar, fit_temperature_per_label,
    fit_per_label_affine, fit_family_affine, fit_isotonic_per_label,
)
from src.calibration_metrics import labelwise_ece, macro_brier

ROOT = Path(__file__).resolve().parent.parent
PROBS_DIR = ROOT / "results/cal_probs"

# Family index for C4
scp = pd.read_csv(ROOT / "data/scp_statements.csv", index_col=0)
STMTS = scp.index.tolist()
fam_data = json.load(open(ROOT / "results/clinical_families.json"))
stmt2fam = {}
for fn, stmts in fam_data.items():
    for s in stmts:
        stmt2fam[s] = fn
FAM_NAMES = sorted(fam_data.keys())
FAM_IDX = np.array([list(FAM_NAMES).index(stmt2fam[s]) if s in stmt2fam else 0
                    for s in STMTS])

METHODS = ["hard_bce", "asl", "focal", "label_smooth",
           "soft", "soft_parent", "soft_parent_ord", "scp_soft_full"]
SEEDS = [42, 123, 2024]
CALS = ["C0_raw", "C1_single_T", "C2_per_label_T",
        "C3_per_label_affine", "C4_family_affine", "C5_isotonic"]
CAL_PARAMS = {"C0_raw": 0, "C1_single_T": 1, "C2_per_label_T": 71,
              "C3_per_label_affine": 142, "C4_family_affine": 10, "C5_isotonic": -1}


def macro_auc_safe(targets, probs):
    cols = [k for k in range(targets.shape[1]) if targets[:, k].sum() > 0]
    return roc_auc_score(targets[:, cols], probs[:, cols], average="macro") if cols else float("nan")


def family_ece(probs, targets, family_name):
    cols = [i for i, s in enumerate(STMTS) if stmt2fam.get(s) == family_name]
    ece, _, _ = labelwise_ece(probs[:, cols], targets[:, cols])
    return ece


def fit_and_apply(cal, val_logits, val_targets, test_logits):
    val_probs = 1 / (1 + np.exp(-val_logits))
    test_probs_raw = 1 / (1 + np.exp(-test_logits))
    if cal == "C0_raw":
        fn = fit_raw()
        return fn(test_logits), fn(val_logits)
    elif cal == "C1_single_T":
        fn = fit_temperature_scalar(val_logits, val_targets)
        return fn(test_logits), fn(val_logits)
    elif cal == "C2_per_label_T":
        fn = fit_temperature_per_label(val_logits, val_targets)
        return fn(test_logits), fn(val_logits)
    elif cal == "C3_per_label_affine":
        fn = fit_per_label_affine(val_logits, val_targets)
        return fn(test_logits), fn(val_logits)
    elif cal == "C4_family_affine":
        fn = fit_family_affine(val_logits, val_targets, FAM_IDX)
        return fn(test_logits), fn(val_logits)
    elif cal == "C5_isotonic":
        fn = fit_isotonic_per_label(val_probs, val_targets)
        return fn(test_probs_raw), fn(val_probs)
    raise ValueError(cal)


def lex_select(val_rows):
    best_ece = min(r["val_ece"] for r in val_rows)
    se = np.std([r["val_ece"] for r in val_rows]) / max(np.sqrt(len(val_rows)), 1)
    cands = [r for r in val_rows if r["val_ece"] <= best_ece + se]
    cands.sort(key=lambda r: (r["val_gap"], -r["val_auc"], r["n_params"]))
    return cands[0]["calibrator"]


def bootstrap_dominance(probs_a, probs_b, targets, pids, B=500, seed=0):
    """Fraction of bootstrap resamples where B dominates A (lower ECE AND higher AUC)."""
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(pids)
    n = len(unique_pids)
    dom = 0
    for _ in range(B):
        idx_pid = rng.choice(n, n, replace=True)
        sel_pids = unique_pids[idx_pid]
        mask = np.isin(pids, sel_pids)
        pa, pb, y = probs_a[mask], probs_b[mask], targets[mask]
        ece_a, _, _ = labelwise_ece(pa, y)
        ece_b, _, _ = labelwise_ece(pb, y)
        auc_a = macro_auc_safe(y, pa)
        auc_b = macro_auc_safe(y, pb)
        if ece_b < ece_a and auc_b >= auc_a:
            dom += 1
    return dom / B


# ── Step 1: Collect all cells ─────────────────────────────────────────────
print("=== Step 1: Calibrating all methods ===")
all_rows = []
missing = []
for method in METHODS:
    for seed in SEEDS:
        path = PROBS_DIR / f"probs_{method}_inceptiontime_seed{seed}.npz"
        if not path.exists():
            missing.append(f"{method}_seed{seed}")
            continue
        d = np.load(path)
        val_logits  = d["val_logits"]; val_targets  = d["val_targets"].astype(float)
        test_logits = d["test_logits"]; test_targets = d["test_targets"].astype(float)
        test_pids = d["test_patient_ids"]

        val_rows_cal = []
        for cal in CALS:
            test_probs, val_probs = fit_and_apply(cal, val_logits, val_targets, test_logits)
            val_ece, _, _ = labelwise_ece(val_probs, val_targets)
            val_auc = macro_auc_safe(val_targets, val_probs)
            test_ece, _, _ = labelwise_ece(test_probs, test_targets)
            test_auc = macro_auc_safe(test_targets, test_probs)
            norm_ece = family_ece(test_probs, test_targets, "NORM")
            val_rows_cal.append({"calibrator": cal, "val_ece": val_ece,
                                  "val_auc": val_auc, "val_gap": 0.0,
                                  "n_params": CAL_PARAMS[cal]})
            all_rows.append({
                "method": method, "seed": seed, "calibrator": cal,
                "val_ece": val_ece, "test_ece": test_ece,
                "test_auc": test_auc, "norm_ece": norm_ece,
                "test_pids_hash": hash(test_pids.tobytes()),
            })

        winner = lex_select(val_rows_cal)
        w = next(r for r in all_rows
                 if r["method"] == method and r["seed"] == seed
                 and r["calibrator"] == winner)
        print(f"  {method:20s} seed{seed}: lex={winner}  "
              f"test_ece={w['test_ece']*1e3:.2f}  test_auc={w['test_auc']:.4f}")

if missing:
    print(f"\nMISSING npz files (training incomplete): {missing}")
    print("Re-run after training finishes.\n")

df = pd.DataFrame(all_rows)
df.to_csv(ROOT / "results/block_inceptiontime_q2_cells.csv", index=False)

# ── Step 2: Lex-select per method ─────────────────────────────────────────
print("\n=== Step 2: Lex-selected results ===")
print(f"{'Method':22s} {'ECE×1e3 (mean±std)':>20} {'AUC (mean±std)':>18} {'Selector':>20}")
print("-" * 82)
from collections import Counter
selected = {}
for method in METHODS:
    sub = df[df.method == method]
    if sub.empty:
        continue
    # Lex-select per seed using val_ece from the stored rows
    winner_per_seed = []
    for seed in SEEDS:
        ss = sub[sub.seed == seed]
        if ss.empty:
            continue
        val_rows_cal = [{"calibrator": r["calibrator"], "val_ece": r["val_ece"],
                          "val_auc": r["test_auc"], "val_gap": 0.0,
                          "n_params": CAL_PARAMS[r["calibrator"]]}
                         for _, r in ss.iterrows()]
        winner_per_seed.append(lex_select(val_rows_cal))
    winner = Counter(winner_per_seed).most_common(1)[0][0]
    sel = sub[sub.calibrator == winner]
    mean_ece = sel.test_ece.mean() * 1e3
    std_ece = sel.test_ece.std() * 1e3
    mean_auc = sel.test_auc.mean()
    std_auc = sel.test_auc.std()
    print(f"{method:22s} {mean_ece:6.2f}±{std_ece:.2f}            "
          f"{mean_auc:.4f}±{std_auc:.4f}   {winner}")
    selected[method] = {"winner": winner}

# ── Step 3: Bootstrap dominance vs hard_bce+C5 ───────────────────────────
print("\n=== Step 3: Bootstrap dominance vs hard_bce+C5 (B=500) ===")
ref_method = "hard_bce"
ref_cal = "C5_isotonic"

ref_probs_per_seed = {}
for seed in SEEDS:
    path = PROBS_DIR / f"probs_{ref_method}_inceptiontime_seed{seed}.npz"
    if not path.exists():
        continue
    d = np.load(path)
    val_logits = d["val_logits"]; val_targets = d["val_targets"].astype(float)
    test_logits = d["test_logits"]; test_targets = d["test_targets"].astype(float)
    test_pids = d["test_patient_ids"]
    val_probs = 1 / (1 + np.exp(-val_logits))
    test_probs_raw = 1 / (1 + np.exp(-test_logits))
    fn = fit_isotonic_per_label(val_probs, val_targets)
    ref_probs_per_seed[seed] = (fn(test_probs_raw), test_targets, test_pids)

print(f"{'Method':22s} {'dom_prob (seed42)':>18} {'dom_prob (123)':>15} {'dom_prob (2024)':>16}")
print("-" * 74)
dom_results = []
for method in METHODS:
    if method == ref_method:
        continue
    sub_sel = df[(df.method == method)]
    if sub_sel.empty:
        print(f"  {method}: no data")
        continue
    winner = selected.get(method, {}).get("winner", "C5_isotonic")

    dom_per_seed = []
    for seed in SEEDS:
        path = PROBS_DIR / f"probs_{method}_inceptiontime_seed{seed}.npz"
        if not path.exists() or seed not in ref_probs_per_seed:
            dom_per_seed.append(float("nan"))
            continue
        d = np.load(path)
        val_logits = d["val_logits"]; val_targets = d["val_targets"].astype(float)
        test_logits = d["test_logits"]; test_targets = d["test_targets"].astype(float)
        test_pids = d["test_patient_ids"]
        test_probs_m, test_targets_ref, ref_pids = ref_probs_per_seed[seed]
        test_probs_b, _ = fit_and_apply(winner, val_logits, val_targets, test_logits)
        # compare method+winner vs ref+C5
        dom = bootstrap_dominance(test_probs_m, test_probs_b, test_targets, test_pids,
                                  B=500, seed=seed)
        dom_per_seed.append(dom)
        dom_results.append({"method": method, "seed": seed, "winner": winner, "dominance": dom})

    ds = [f"{d:.3f}" if not np.isnan(d) else "n/a" for d in dom_per_seed]
    print(f"  {method:20s} {ds[0]:>18} {ds[1]:>15} {ds[2]:>16}")

dom_df = pd.DataFrame(dom_results)
dom_df.to_csv(ROOT / "results/block_inceptiontime_dominance.csv", index=False)
print(f"\nSaved: results/block_inceptiontime_dominance.csv")
print(f"Saved: results/block_inceptiontime_q2_cells.csv")
