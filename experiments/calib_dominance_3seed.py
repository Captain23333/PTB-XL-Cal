"""3-seed dominance recompute (CPU-only, near-mandatory before submission).

Reuses the cached per-(method, seed) probs to compute Q2 paired-bootstrap
dominance probability of each method vs hard_bce, MATCHED seed-by-seed.

Outputs results/block2_q2_dominance_3seed.csv with per-seed values + mean.
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
from experiments.calib_block2_q2 import evaluate_cell, dominance_bootstrap

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
SEL_PATH = ROOT / "results/block2_q2_selected_calibrators.json"
OUT_CSV = ROOT / "results/block2_q2_dominance_3seed.csv"


def main():
    print(f"[{time.strftime('%H:%M:%S')}] 3-seed dominance recompute starting")
    fam_idx, fam_order = family_index_per_stmt()
    fam_masks = family_membership_mask(fam_order)
    with open(SEL_PATH) as f:
        selected = json.load(f)

    ref_method = "hard_bce"
    ref_cal = selected[ref_method]["best"]
    print(f"  reference = {ref_method} + {ref_cal}")

    # Cache reference probs for each seed
    ref_cache = {}
    for s in SEEDS:
        path = PROBS_DIR / f"probs_{ref_method}_seed{s}.npz"
        if not path.exists():
            print(f"  MISSING ref {path.name}")
            continue
        cells, tgt, pid = evaluate_cell(ref_method, s, path, fam_idx, fam_masks)
        ref_cache[s] = (cells[ref_cal]["test_probs"], tgt, pid)
        print(f"  ref seed{s} cached ({len(pid)} samples)")

    rows = []
    for m in BASE_METHODS + ["ecgfm_linprobe"]:
        if m == ref_method:
            continue
        tcal = selected.get(m, {}).get("best")
        if tcal is None:
            print(f"  no selected cal for {m}; skip")
            continue
        per_seed = []
        for s in SEEDS if m != "ecgfm_linprobe" else [42]:
            path = PROBS_DIR / f"probs_{m}_seed{s}.npz"
            if not path.exists():
                continue
            if s not in ref_cache:
                continue
            try:
                cells, tgt, pid = evaluate_cell(m, s, path, fam_idx, fam_masks)
                tprobs = cells[tcal]["test_probs"]
                ref_probs, ref_tgt, ref_pid = ref_cache[s]
                if not np.array_equal(pid, ref_pid):
                    print(f"  WARN pid mismatch {m} seed{s}")
                    continue
                p = dominance_bootstrap(tprobs, tgt, pid,
                                          ref_probs, ref_tgt, ref_pid,
                                          fam_masks, B=500, seed=42)
                per_seed.append((s, p))
                print(f"  {m:20s} seed{s}: dominance={p:.4f}")
            except Exception as e:
                print(f"  fail {m} seed{s}: {e}")
        if per_seed:
            probs = [p for _, p in per_seed]
            rows.append({
                "method": m, "calibrator": tcal,
                "ref_method": ref_method, "ref_calibrator": ref_cal,
                "n_seeds": len(per_seed),
                "dominance_seed42": probs[0] if len(probs) >= 1 else float("nan"),
                "dominance_seed123": probs[1] if len(probs) >= 2 else float("nan"),
                "dominance_seed2024": probs[2] if len(probs) >= 3 else float("nan"),
                "dominance_mean": float(np.mean(probs)),
                "dominance_std": float(np.std(probs)) if len(probs) > 1 else 0.0,
                "dominance_min": float(np.min(probs)),
                "dominance_max": float(np.max(probs)),
            })

    df = pd.DataFrame(rows).sort_values("dominance_mean", ascending=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[{time.strftime('%H:%M:%S')}] saved {OUT_CSV.name}: {len(rows)} rows")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
