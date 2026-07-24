"""
Block D — Pareto + dominance analysis.

Output:
  figures/fig_block_d_pareto.png   - 3-panel 2D scatter (AUC×ECE, AUC×HVR, ECE×HVR)
  results/block_d_dominance.csv    - bootstrap Pareto-dominance frequency vs each baseline
"""
import sys
import os
sys.path.insert(0, os.environ.get("PTBXL_CAL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.evaluate import macro_auc, macro_brier, macro_ece, hvr

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))


def main():
    print("=" * 60)
    print("Block D - Pareto + Dominance")
    print("=" * 60)

    df = pd.read_csv(ROOT / "results/block_a_raw.csv")
    print(f"Loaded {len(df)} rows from block_a_raw.csv")

    # ------- 3-panel 2D scatter -------
    pairs = [("macro_auc", "macro_ece", "AUC × ECE"),
             ("macro_auc", "hvr", "AUC × HVR"),
             ("macro_ece", "hvr", "ECE × HVR")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    methods = df["method"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for ax, (m1, m2, title) in zip(axes, pairs):
        for color, method in zip(colors, methods):
            sub = df[df["method"] == method]
            marker = "*" if "scp_soft" in method else ("D" if "chmc" in method else "o")
            size = 220 if ("scp_soft" in method or "chmc" in method) else 100
            ax.scatter(sub[m1], sub[m2], label=method, color=color, s=size,
                       marker=marker, alpha=0.85, edgecolors="black", linewidths=0.6)
        ax.set_xlabel(m1); ax.set_ylabel(m2); ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(ROOT / "figures/fig_block_d_pareto.png", dpi=200)
    plt.close()
    print(f"Saved: figures/fig_block_d_pareto.png")

    # ------- Pareto dominance frequency via bootstrap -------
    # Load probs (seed=42 representative)
    probs = np.load(ROOT / "results/block_a_probs.npz")
    method_probs = {}
    for key in probs.files:
        # key like "hard_bce_p_stmt", "scp_soft_full_p_stmt", etc.
        if key.endswith("_p_stmt"):
            m = key[:-len("_p_stmt")]
            method_probs[m] = {
                "y_stmt": probs[f"{m}_y_stmt"] if f"{m}_y_stmt" in probs.files else probs[f"hard_bce_y_stmt"],
                "p_stmt": probs[f"{m}_p_stmt"],
                "p_super": probs[f"{m}_p_super"],
                "p_sub": probs[f"{m}_p_sub"],
            }
    print(f"Loaded probs for {len(method_probs)} methods")

    # Bootstrap dominance freq for SCP-Soft vs each baseline
    if "scp_soft_full" not in method_probs:
        print("scp_soft_full probs missing -- skipping dominance analysis")
        return
    hier_data = json.load(open(ROOT / "results/hierarchy_edges.json"))
    edges = hier_data["edges"]
    # Need indices; load lazily
    from src.data import get_loaders
    _, _, _, train_ds = get_loaders(batch_size=1, num_workers=0)

    target = method_probs["scp_soft_full"]
    rng = np.random.default_rng(0)
    B = 500   # bootstrap samples (lower than 1000 for speed since we compute 3 metrics)
    N = target["y_stmt"].shape[0]
    dom_rows = []
    for m in method_probs:
        if m == "scp_soft_full": continue
        baseline = method_probs[m]
        wins = 0; ties = 0; losses = 0
        for _ in range(B):
            idx = rng.integers(0, N, N)
            try:
                t_auc, _ = macro_auc(target["y_stmt"][idx], target["p_stmt"][idx])
                b_auc, _ = macro_auc(baseline["y_stmt"][idx], baseline["p_stmt"][idx])
                t_ece = macro_ece(target["y_stmt"][idx], target["p_stmt"][idx])
                b_ece = macro_ece(baseline["y_stmt"][idx], baseline["p_stmt"][idx])
                t_hvr = hvr(target["p_super"][idx], target["p_sub"][idx], target["p_stmt"][idx],
                            edges, train_ds.super2idx, train_ds.sub2idx, train_ds.stmt2idx)
                b_hvr = hvr(baseline["p_super"][idx], baseline["p_sub"][idx], baseline["p_stmt"][idx],
                            edges, train_ds.super2idx, train_ds.sub2idx, train_ds.stmt2idx)
                # SCP-Soft dominates if: AUC ≥, ECE ≤, HVR ≤, with ≥1 strict
                ge_auc = t_auc >= b_auc - 1e-6
                le_ece = t_ece <= b_ece + 1e-6
                le_hvr = t_hvr <= b_hvr + 1e-6
                strict = (t_auc > b_auc + 1e-6) or (t_ece < b_ece - 1e-6) or (t_hvr < b_hvr - 1e-6)
                if ge_auc and le_ece and le_hvr and strict:
                    wins += 1
                else:
                    losses += 1
            except Exception:
                continue
        dom_freq = wins / max(1, wins + losses)
        ci_lo = max(0, dom_freq - 1.96 * np.sqrt(dom_freq * (1-dom_freq) / max(1, wins+losses)))
        ci_hi = min(1, dom_freq + 1.96 * np.sqrt(dom_freq * (1-dom_freq) / max(1, wins+losses)))
        dom_rows.append({
            "baseline": m, "wins": wins, "losses": losses,
            "dominance_freq": round(dom_freq, 4),
            "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
            "n_bootstrap": wins + losses,
        })
        print(f"vs {m:18s}: SCP-Soft dominates {dom_freq*100:.1f}% (95% CI [{ci_lo*100:.1f}, {ci_hi*100:.1f}]%)")
    pd.DataFrame(dom_rows).to_csv(ROOT / "results/block_d_dominance.csv", index=False)
    print(f"\nSaved: results/block_d_dominance.csv")
    print("Block D COMPLETE.")


if __name__ == "__main__":
    main()
