"""Generate the 4 main figures + scaffold flow diagram.

Reads results/block{1,2,3,4}_*.csv/json/npz and writes figures/{png,pdf}.
"""
from __future__ import annotations
import sys, time, json
import os
sys.path.insert(0, os.environ.get("PTBXL_CAL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def save_both(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.{{png,pdf}}")


def fig1_q1():
    p = ROOT / "results/block1_q1.csv"
    if not p.exists():
        print("  fig1: missing block1_q1.csv -- skipped"); return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    backbones = sorted(df["backbone"].unique())
    cal_order = ["C0_raw", "C1_temp_scalar", "C2_temp_per_label",
                 "C3_per_label_affine", "C4_family_affine", "C5_isotonic"]
    width = 0.8 / max(1, len(backbones))
    for bi, bb in enumerate(backbones):
        sub = df[df["backbone"] == bb].groupby("calibrator")["macro_ece"].agg(["mean", "std"]).reindex(cal_order)
        x = np.arange(len(cal_order)) + bi * width - width * (len(backbones) - 1) / 2
        ax.bar(x, sub["mean"].values, width=width, yerr=sub["std"].fillna(0).values,
               label=bb, capsize=3)
        for xi, mv in zip(x, sub["mean"].values):
            if not np.isnan(mv):
                ax.text(xi, mv, f"{mv:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(np.arange(len(cal_order)))
    ax.set_xticklabels([c.split("_", 1)[0] + "\n" + c.split("_", 1)[1] for c in cal_order], fontsize=8)
    ax.set_ylabel("Macro labelwise ECE")
    ax.set_title("Q1: Calibrator family on hard_bce (mean +/- std over seeds)")
    ax.legend(fontsize=8)
    save_both(fig, "fig1_q1_calibrator_family")


def fig2_q2_pareto():
    p = ROOT / "results/block2_q2_pareto.csv"
    g = ROOT / "results/block2_q2_full_grid.csv"
    if not p.exists() or not g.exists():
        print("  fig2: missing pareto/grid csv -- skipped"); return
    pareto = pd.read_csv(p)
    grid = pd.read_csv(g)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel a: AUC vs ECE
    axes[0].scatter(grid["test_macro_ece"], grid["test_macro_auc"], c="lightgray",
                    s=18, alpha=0.5, label="all cells")
    on_pf = pareto[pareto["on_pareto_front"]]
    off_pf = pareto[~pareto["on_pareto_front"]]
    axes[0].scatter(off_pf["labelwise_ece"], off_pf["macro_auc"], c="C0", s=60,
                    label="best-cal off-front", edgecolor="k")
    axes[0].scatter(on_pf["labelwise_ece"], on_pf["macro_auc"], c="C3", s=80,
                    label="Pareto front", edgecolor="k", marker="^")
    for _, r in pareto.iterrows():
        axes[0].annotate(r["method"], (r["labelwise_ece"], r["macro_auc"]),
                         fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel("labelwise ECE"); axes[0].set_ylabel("macro AUC")
    axes[0].set_title("(a) AUC x ECE"); axes[0].legend(fontsize=7)

    # Panel b: AUC vs family ECE max
    axes[1].scatter(pareto["family_ece_max"], pareto["macro_auc"], c="C0", s=60, edgecolor="k")
    for _, r in pareto.iterrows():
        axes[1].annotate(r["method"], (r["family_ece_max"], r["macro_auc"]),
                         fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[1].set_xlabel("family ECE (max across 5)"); axes[1].set_ylabel("macro AUC")
    axes[1].set_title("(b) AUC x family-ECE_max")

    # Panel c: ECE vs threshold gap
    axes[2].scatter(grid["test_macro_ece"], grid["test_threshold_gap_sens09"],
                    c="lightgray", s=18, alpha=0.5)
    sub = grid.merge(pareto[["method", "calibrator"]], on=["method", "calibrator"])
    axes[2].scatter(sub["test_macro_ece"], sub["test_threshold_gap_sens09"],
                    c="C0", s=50, edgecolor="k", label="best per method")
    axes[2].set_xlabel("labelwise ECE"); axes[2].set_ylabel("threshold gap @ sens=0.9")
    axes[2].set_title("(c) ECE x threshold gap"); axes[2].legend(fontsize=7)

    plt.suptitle("Q2: Pareto front and dominance space")
    save_both(fig, "fig2_q2_pareto_dominance")


def fig3_q3_heatmap():
    p = ROOT / "results/block3_q3_heatmap.npz"
    if not p.exists():
        print("  fig3: missing heatmap npz -- skipped"); return
    d = np.load(p, allow_pickle=True)
    H = d["heatmap_C3"]
    methods = list(d["methods"])
    families = list(d["families"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(H, annot=True, fmt=".3f", xticklabels=families, yticklabels=methods,
                cmap="YlOrRd", ax=ax)
    ax.set_title("Q3: Family-level ECE under C3 (per-label affine)")
    save_both(fig, "fig3_q3_family_reliability_heatmap")


def fig4_threshold():
    p = ROOT / "results/block4_threshold.csv"
    sp = ROOT / "results/block4_spearman.json"
    if not p.exists():
        print("  fig4: missing block4 csv -- skipped"); return
    df = pd.read_csv(p)
    rho_info = {"spearman_rho": float("nan")}
    if sp.exists():
        rho_info = json.load(open(sp))
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = df.dropna(subset=["labelwise_ece", "sens_gap_0.90"])
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(sub["method"].unique()):
        s = sub[sub["method"] == m]
        ax.scatter(s["labelwise_ece"], s["sens_gap_0.90"], label=m,
                   color=cmap(i % 10), alpha=0.7, s=40)
    ax.set_xlabel("labelwise ECE")
    ax.set_ylabel("|sens_test - 0.9|")
    ax.set_title(f"Block 4: ECE vs threshold-transfer gap (Spearman rho = {rho_info['spearman_rho']:.3f})")
    ax.legend(fontsize=7, ncol=2)
    save_both(fig, "fig4_block4_threshold_transfer")


def figS_scaffold_flow():
    """Block diagram of eval_calibration.py pipeline."""
    fig, ax = plt.subplots(figsize=(10, 4))
    boxes = [
        (0.05, 0.55, 0.18, 0.30, "predictions_*.npz\n(val + test logits)"),
        (0.30, 0.75, 0.18, 0.18, "fit 6 calibrators\non val (C0..C5)"),
        (0.30, 0.30, 0.18, 0.18, "apply to test"),
        (0.55, 0.55, 0.18, 0.30, "locked metric suite\n(ECE/Brier/slope/\nthreshold-gap)"),
        (0.78, 0.55, 0.18, 0.30, "lex selector\n+ leaderboard.json"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, color="#cdebff", ec="black"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
    arrows = [(0.23, 0.70, 0.30, 0.85), (0.23, 0.70, 0.30, 0.40),
              (0.48, 0.85, 0.55, 0.70), (0.48, 0.40, 0.55, 0.65),
              (0.73, 0.70, 0.78, 0.70)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("eval_calibration.py pipeline (scaffold)")
    save_both(fig, "figS_scaffold_flow")


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Figures starting")
    fig1_q1()
    fig2_q2_pareto()
    fig3_q3_heatmap()
    fig4_threshold()
    figS_scaffold_flow()
    print(f"[{time.strftime('%H:%M:%S')}] Figures DONE")


if __name__ == "__main__":
    main()
