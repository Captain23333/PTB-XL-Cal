"""
Supplement S1-S7 minimal runners.
S1: likelihood mapping ablation (linear / sqrt / sigmoid / learned)
S2: hierarchy loss form ablation (max-margin² / noisy-OR / hinge)
S3: λ_ord, λ_hier sensitivity (3×3 grid, 1 seed)
S4: hierarchy scope (diag-only / all-71)
S5: backbone interchange (xresnet / lightweight)
S7: HVR ε sensitivity (analysis only on saved probs)
S8: macro-AUPRC main results (analysis only)
S6 (ECG-FM portability) is left as TODO (requires ECG-FM weights).
"""
import sys
import os
sys.path.insert(0, os.environ.get("PTBXL_CAL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.data import get_loaders
from src.evaluate import macro_auc, hvr, macro_auprc

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))


def s7_hvr_eps_sensitivity():
    """HVR with ε ∈ {0, 0.05, 0.1} on saved probs."""
    print("=== S7: HVR ε sensitivity ===")
    probs = np.load(ROOT / "results/block_a_probs.npz")
    hier_data = json.load(open(ROOT / "results/hierarchy_edges.json"))
    _, _, _, train_ds = get_loaders(batch_size=1, num_workers=0)
    edges = hier_data["edges"]
    rows = []
    methods = sorted({k.split("_p_stmt")[0] for k in probs.files if k.endswith("_p_stmt")})
    for m in methods:
        for eps in [0.0, 0.05, 0.1]:
            v = hvr(probs[f"{m}_p_super"], probs[f"{m}_p_sub"], probs[f"{m}_p_stmt"],
                    edges, train_ds.super2idx, train_ds.sub2idx, train_ds.stmt2idx, eps=eps)
            rows.append({"method": m, "epsilon": eps, "hvr": round(v, 4)})
            print(f"  {m:18s} eps={eps:.2f} -> HVR={v:.4f}")
    pd.DataFrame(rows).to_csv(ROOT / "results/supp_s7_hvr_eps.csv", index=False)


def s8_auprc():
    """macro-AUPRC for all methods on test set."""
    print("=== S8: macro-AUPRC ===")
    probs = np.load(ROOT / "results/block_a_probs.npz")
    rows = []
    methods = sorted({k.split("_p_stmt")[0] for k in probs.files if k.endswith("_p_stmt")})
    for m in methods:
        y = probs[f"{m}_y_stmt"]
        p = probs[f"{m}_p_stmt"]
        ap = macro_auprc(y, p)
        rows.append({"method": m, "macro_auprc": round(ap, 4)})
        print(f"  {m:18s} -> macro-AUPRC={ap:.4f}")
    pd.DataFrame(rows).to_csv(ROOT / "results/supp_s8_auprc.csv", index=False)


def s1_likelihood_mapping():
    """Train SCP-Soft with 4 likelihood mappings (1 seed each) - schedules training, not analysis only."""
    print("=== S1: likelihood mapping ablation ===")
    print("This requires training 4 variants. Use experiments/run_supplements.sh to launch.")
    print("Output: results/supp_s1_mapping.csv (after training completes)")


def s3_lambda_grid():
    """3x3 grid for (λ_ord, λ_hier). Stub — requires training."""
    print("=== S3: λ sensitivity grid (3x3) ===")
    print("This requires 9 training runs. Use experiments/run_supplements.sh to launch.")


def main():
    """Run analysis-only supplements (S7, S8). Training-required supplements
    are launched separately via run_supplements.sh."""
    if not (ROOT / "results/block_a_probs.npz").exists():
        print("ERROR: results/block_a_probs.npz missing. Run experiments/02_block_a_eval.py first.")
        return
    s7_hvr_eps_sensitivity()
    print()
    s8_auprc()
    print()
    print("Analysis-only supplements (S7, S8) COMPLETE.")
    print("For training-required supplements (S1, S2, S3, S4, S5):")
    print("  bash experiments/run_supplements.sh")


if __name__ == "__main__":
    main()
