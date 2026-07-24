"""
Block C — Stratified analyses (1 figure 3 panels + 1 clinical-family table).

Panel 1: macro-AUC by likelihood bin (15/35/50/80/100)
Panel 2: macro-AUC by frequency tertile (top/mid/tail)
Panel 3: reliability stratified by likelihood bin
Table:   per-clinical-family (MI/STTC/HYP/CD) macro-AUC + macro-ECE
"""
import sys
import os
sys.path.insert(0, os.environ.get("PTBXL_CAL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.data import get_loaders, LIK_TO_BIN
from src.evaluate import macro_ece, adaptive_ece

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))


def load_probs():
    """Load saved probs from Block A (seed=42 representative)."""
    npz = np.load(ROOT / "results/block_a_probs.npz")
    return {key: npz[key] for key in npz.files}


def likelihood_bin_aucs(p_pred, lik_matrix, min_pos=5):
    """For each bin b in {1..5}, compute statement-level AUC then macro-average.
    Positives = (i,j) where lik_matrix[i,j] == b; negatives = (i,j) where lik_matrix[i,j] == 0.
    """
    bin_to_value = {1: 15, 2: 35, 3: 50, 4: 80, 5: 100}
    out = {}
    for b, val in bin_to_value.items():
        aucs = []
        for j in range(p_pred.shape[1]):
            pos_mask = (lik_matrix[:, j] == b)
            neg_mask = (lik_matrix[:, j] == 0)
            if pos_mask.sum() < min_pos or neg_mask.sum() < min_pos:
                continue
            y = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
            pp = np.concatenate([p_pred[pos_mask, j], p_pred[neg_mask, j]])
            try:
                aucs.append(roc_auc_score(y, pp))
            except Exception:
                pass
        out[val] = float(np.mean(aucs)) if aucs else float("nan")
    return out


def frequency_tertile_aucs(y, p, min_pos=10):
    """Split statements into top/mid/tail tertile by positive count; macro-AUC each."""
    pos_counts = (y > 0.5).sum(axis=0)
    order = np.argsort(pos_counts)[::-1]
    J = y.shape[1]
    tertiles = {"top": order[:J//3], "mid": order[J//3:2*J//3], "tail": order[2*J//3:]}
    out = {}
    for name, idx in tertiles.items():
        aucs = []
        for j in idx:
            if y[:, j].sum() < min_pos: continue
            try: aucs.append(roc_auc_score(y[:, j], p[:, j]))
            except: pass
        out[name] = float(np.mean(aucs)) if aucs else float("nan")
    return out


def reliability_by_bin(y, p, lik_matrix, n_bins=8):
    """For each likelihood bin, compute reliability curve points (mean_pred, mean_actual)."""
    bin_to_value = {1: 15, 2: 35, 3: 50, 4: 80, 5: 100}
    curves = {}
    for b, val in bin_to_value.items():
        # Pool all (i,j) with lik=b as positives + lik=0 as negatives across all statements
        all_p, all_y = [], []
        for j in range(p.shape[1]):
            pos = lik_matrix[:, j] == b
            neg = lik_matrix[:, j] == 0
            all_p.append(p[pos, j])
            all_y.append(np.ones(pos.sum()))
            all_p.append(p[neg, j])
            all_y.append(np.zeros(neg.sum()))
        all_p = np.concatenate(all_p)
        all_y = np.concatenate(all_y)
        if len(all_p) == 0:
            curves[val] = (np.array([]), np.array([]))
            continue
        # Equal-width prob bins
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.digitize(all_p, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        mean_p = []; mean_y = []
        for k in range(n_bins):
            mask = idx == k
            if mask.sum() < 10: continue
            mean_p.append(all_p[mask].mean())
            mean_y.append(all_y[mask].mean())
        curves[val] = (np.array(mean_p), np.array(mean_y))
    return curves


def main():
    print("=" * 60)
    print("Block C - Stratified Analyses")
    print("=" * 60)

    probs = load_probs()
    families = json.load(open(ROOT / "results/clinical_families.json"))
    _, _, test_loader, train_ds = get_loaders(batch_size=1, num_workers=0)

    # Build likelihood matrix for test set
    db = pd.read_csv(ROOT / "data/ptbxl_database.csv")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    test_db = db[db["strat_fold"] == 10].reset_index(drop=True)
    N = len(test_db); J = 71
    lik_matrix = np.zeros((N, J), dtype=int)
    for i, row in test_db.iterrows():
        for stmt, lik in row["scp_codes"].items():
            if stmt in train_ds.stmt2idx:
                lik_matrix[i, train_ds.stmt2idx[stmt]] = LIK_TO_BIN.get(int(round(float(lik))), 0)
    print(f"Test likelihood matrix built: shape={lik_matrix.shape}")

    p_scp = probs["scp_soft_full_p_stmt"]
    p_bce = probs["hard_bce_p_stmt"]
    y = probs["scp_soft_full_y_stmt"]   # same y for all methods

    # Panel 1: AUC by likelihood bin
    bin_scp = likelihood_bin_aucs(p_scp, lik_matrix)
    bin_bce = likelihood_bin_aucs(p_bce, lik_matrix)
    print("Panel 1 - AUC by likelihood bin:")
    for v in [15, 35, 50, 80, 100]:
        print(f"  bin {v}: SCP-Soft={bin_scp[v]:.4f}  hard-BCE={bin_bce[v]:.4f}")

    # Panel 2: freq tertile
    freq_scp = frequency_tertile_aucs(y, p_scp)
    freq_bce = frequency_tertile_aucs(y, p_bce)
    print("Panel 2 - AUC by frequency tertile:")
    for tier in ["top", "mid", "tail"]:
        print(f"  {tier}: SCP-Soft={freq_scp[tier]:.4f}  hard-BCE={freq_bce[tier]:.4f}")

    # Panel 3: reliability by likelihood bin
    rel_scp = reliability_by_bin(y, p_scp, lik_matrix)
    rel_bce = reliability_by_bin(y, p_bce, lik_matrix)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    bins_x = [15, 35, 50, 80, 100]
    axes[0].plot(bins_x, [bin_bce[b] for b in bins_x], "-o", label="hard-BCE", lw=2)
    axes[0].plot(bins_x, [bin_scp[b] for b in bins_x], "-o", label="SCP-Soft", lw=2)
    axes[0].set_xlabel("Physician likelihood bin"); axes[0].set_ylabel("macro-AUC")
    axes[0].set_title("(a) AUC by likelihood bin"); axes[0].legend(); axes[0].grid(alpha=0.3)

    x = np.arange(3); width = 0.35
    axes[1].bar(x - width/2, [freq_bce[t] for t in ["top","mid","tail"]], width, label="hard-BCE")
    axes[1].bar(x + width/2, [freq_scp[t] for t in ["top","mid","tail"]], width, label="SCP-Soft")
    axes[1].set_xticks(x); axes[1].set_xticklabels(["top 1/3", "mid 1/3", "tail 1/3"])
    axes[1].set_ylabel("macro-AUC"); axes[1].set_title("(b) AUC by frequency tertile")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    axes[2].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    for i, b in enumerate(bins_x):
        mp, my = rel_scp[b]
        if len(mp): axes[2].plot(mp, my, "-o", color=colors[i], label=f"SCP-Soft lik={b}", lw=1.5, ms=4)
    axes[2].set_xlabel("Mean predicted probability"); axes[2].set_ylabel("Empirical positive rate")
    axes[2].set_title("(c) Reliability by likelihood bin"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ROOT / "figures/fig_block_c_stratified.png", dpi=200)
    plt.close()
    print(f"Saved: figures/fig_block_c_stratified.png")

    # Clinical family table
    rows = []
    for fam, stmts in families.items():
        js = [train_ds.stmt2idx[s] for s in stmts if s in train_ds.stmt2idx]
        if not js: continue
        # macro AUC over family stmts only
        aucs_scp, aucs_bce = [], []
        eces_scp, eces_bce = [], []
        for j in js:
            if y[:, j].sum() < 10: continue
            try:
                aucs_scp.append(roc_auc_score(y[:, j], p_scp[:, j]))
                aucs_bce.append(roc_auc_score(y[:, j], p_bce[:, j]))
            except: pass
            eces_scp.append(adaptive_ece(y[:, j], p_scp[:, j]))
            eces_bce.append(adaptive_ece(y[:, j], p_bce[:, j]))
        rows.append({
            "family": fam, "n_stmts": len(js),
            "auc_scp": np.mean(aucs_scp) if aucs_scp else np.nan,
            "auc_bce": np.mean(aucs_bce) if aucs_bce else np.nan,
            "ece_scp": np.mean(eces_scp) if eces_scp else np.nan,
            "ece_bce": np.mean(eces_bce) if eces_bce else np.nan,
        })
    df_fam = pd.DataFrame(rows).round(4)
    df_fam.to_csv(ROOT / "results/block_c_family.csv", index=False)
    print("\nClinical family table:")
    print(df_fam)

    # Save raw stratified data
    pd.DataFrame({"likelihood_bin": [15, 35, 50, 80, 100],
                  "auc_scp_soft": [bin_scp[b] for b in [15,35,50,80,100]],
                  "auc_hard_bce": [bin_bce[b] for b in [15,35,50,80,100]]}).to_csv(
        ROOT / "results/block_c_likelihood.csv", index=False)
    pd.DataFrame({"tertile": ["top", "mid", "tail"],
                  "auc_scp_soft": [freq_scp[t] for t in ["top","mid","tail"]],
                  "auc_hard_bce": [freq_bce[t] for t in ["top","mid","tail"]]}).to_csv(
        ROOT / "results/block_c_frequency.csv", index=False)

    print("\nBlock C COMPLETE.")


if __name__ == "__main__":
    main()
