"""Paper figures for PTB-XL-Cal (PMLR ML4H 2026 findings).

Generates:
  - figures/fig1_pareto_dominance_3seed.{png,pdf}   (HERO: 3-panel)
  - figures/fig2_norm_failure_mode.{png,pdf}        (NORM-family reliability + family-ECE bar)
  - figures/fig3_q1_calibrator_family.{png,pdf}     (Q1 ECE bar with bootstrap CI)
  - figures/figS3_q4_monotonic_tie.{png,pdf}        (appendix: Q4 mis-spec scatter)
  - figures/table1_q1.tex                            (LaTeX Q1 table)
  - figures/table2_q2_pareto.tex                     (LaTeX Q2 Pareto + 3-seed dominance + family)
  - figures/latex_includes.tex                       (LaTeX \includegraphics snippets)

All outputs at 300 DPI, serif, no titles inside figures (titles in LaTeX caption).
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
FIG = ROOT / "figures"
RES = ROOT / "results"

# --- Style ---
matplotlib.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "stix",
})

# Color palette (tab10-ish + colorblind-safe)
C_HARD_BCE = "#1f77b4"   # blue (reference)
C_ASL      = "#2ca02c"   # green
C_SOFT     = "#9467bd"   # purple
C_ECGFM    = "#ff7f0e"   # orange
C_DOMINATED = "#7f7f7f"  # gray (dominated)
C_D6       = "#d62728"   # red (D6 cautionary)

METHOD_LABEL = {
    "hard_bce":         "hard-BCE",
    "asl":              "ASL",
    "focal":            "focal",
    "label_smooth":     "LS($\\epsilon$=0.1)",
    "chmc_coh":         "CHMC-proj",
    "scp_soft_full":    "SCP-Soft (D6)",
    "soft":             "soft-likelihood",
    "soft_parent_ord":  "soft+parent+ord",
    "ecgfm_linprobe":   "ECG-FM (probe)",
}
PARETO_METHODS = {"hard_bce", "asl", "soft", "ecgfm_linprobe"}


def _save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}.{{png,pdf}}")


# ============================================================
# FIG 1 — HERO: Pareto + 3-seed dominance heatmap
# ============================================================

def fig1_hero():
    pareto = pd.read_csv(RES / "block2_q2_pareto.csv")
    dom = pd.read_csv(RES / "block2_q2_dominance_3seed.csv")

    fig = plt.figure(figsize=(7.0, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # ---- Panel A: macro-AUC vs labelwise-ECE ----
    for _, row in pareto.iterrows():
        m = row["method"]
        on_pareto = bool(row["on_pareto_front"])
        is_ref = (m == "hard_bce")
        color = {
            "hard_bce": C_HARD_BCE, "asl": C_ASL, "soft": C_SOFT,
            "ecgfm_linprobe": C_ECGFM, "scp_soft_full": C_D6,
        }.get(m, C_DOMINATED)
        marker_size = 110 if is_ref else 70
        edge = "black"
        if is_ref:
            ax_a.scatter(row["macro_auc"], row["labelwise_ece"], marker="*",
                         s=180, c=color, edgecolor=edge, linewidth=0.8, zorder=5,
                         label=METHOD_LABEL[m] + " (ref)")
        elif on_pareto:
            ax_a.scatter(row["macro_auc"], row["labelwise_ece"], marker="o",
                         s=marker_size, c=color, edgecolor=edge, linewidth=0.6, zorder=4,
                         label=METHOD_LABEL[m])
        else:
            ax_a.scatter(row["macro_auc"], row["labelwise_ece"], marker="o",
                         s=marker_size, facecolor="none", edgecolor=color, linewidth=0.9,
                         zorder=3, alpha=0.9,
                         label=METHOD_LABEL[m])
    ax_a.set_xlabel("macro-AUC (test)")
    ax_a.set_ylabel("labelwise ECE (test)")
    ax_a.set_yscale("log")
    ax_a.text(-0.18, 1.04, "A", transform=ax_a.transAxes, fontsize=11, fontweight="bold")
    ax_a.legend(loc="upper left", bbox_to_anchor=(1.0, 1.02), frameon=False,
                fontsize=7, handlelength=1.1, labelspacing=0.3, ncol=1,
                borderaxespad=0.).set_visible(False)  # legend goes on Panel B instead

    # ---- Panel B: macro-AUC vs family-ECE-max ----
    for _, row in pareto.iterrows():
        m = row["method"]
        on_pareto = bool(row["on_pareto_front"])
        is_ref = (m == "hard_bce")
        color = {
            "hard_bce": C_HARD_BCE, "asl": C_ASL, "soft": C_SOFT,
            "ecgfm_linprobe": C_ECGFM, "scp_soft_full": C_D6,
        }.get(m, C_DOMINATED)
        if is_ref:
            ax_b.scatter(row["macro_auc"], row["family_ece_max"], marker="*",
                         s=180, c=color, edgecolor="black", linewidth=0.8, zorder=5,
                         label="hard-BCE+iso (ref)")
        elif on_pareto:
            ax_b.scatter(row["macro_auc"], row["family_ece_max"], marker="o",
                         s=70, c=color, edgecolor="black", linewidth=0.6, zorder=4,
                         label=METHOD_LABEL[m])
        else:
            ax_b.scatter(row["macro_auc"], row["family_ece_max"], marker="o",
                         s=70, facecolor="none", edgecolor=color, linewidth=0.9,
                         zorder=3, alpha=0.9,
                         label=METHOD_LABEL[m])
    ax_b.set_xlabel("macro-AUC (test)")
    ax_b.set_ylabel("max family ECE (test)")
    ax_b.text(-0.18, 1.04, "B", transform=ax_b.transAxes, fontsize=11, fontweight="bold")
    ax_b.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                fontsize=7, handlelength=1.2, labelspacing=0.5)

    # Annotate "NORM-driven" on the family-ECE panel
    ax_b.annotate("(NORM-driven)", xy=(0.62, 0.05), xycoords="axes fraction",
                  fontsize=7, color="dimgray", ha="left")

    # ---- Panel C: 3-seed dominance heatmap ----
    # rows = methods (ordered: dominated first, then dominators), cols = seed42, seed123, seed2024, mean
    # Sort by dominance_mean ascending so dominated (0) on top, "soft" at bottom (highest dominance)
    dom_sorted = dom.sort_values("dominance_mean", ascending=True).reset_index(drop=True)
    method_order = dom_sorted["method"].tolist()
    heat = np.full((len(method_order), 4), np.nan)
    for i, row in dom_sorted.iterrows():
        heat[i, 0] = row["dominance_seed42"]
        heat[i, 1] = row["dominance_seed123"]
        heat[i, 2] = row["dominance_seed2024"]
        heat[i, 3] = row["dominance_mean"]

    cmap = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#fee0d2", "#fc9272", "#de2d26"])
    im = ax_c.imshow(heat, cmap=cmap, vmin=0, vmax=0.16, aspect="auto")
    ax_c.set_xticks(range(4))
    ax_c.set_xticklabels(["seed 42", "seed 123", "seed 2024", "mean"])
    ax_c.set_yticks(range(len(method_order)))
    ax_c.set_yticklabels([METHOD_LABEL[m] for m in method_order], fontsize=8)
    ax_c.set_xlabel("Patient-paired bootstrap dominance probability vs hard-BCE + isotonic (B=500)")
    ax_c.text(-0.10, 1.04, "C", transform=ax_c.transAxes, fontsize=11, fontweight="bold")

    # Annotate each cell
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            if np.isnan(v):
                ax_c.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
            else:
                text = "0.000" if v < 0.0005 else f"{v:.3f}"
                color = "black" if v < 0.08 else "white"
                weight = "bold" if v < 0.0005 else "normal"
                ax_c.text(j, i, text, ha="center", va="center",
                          fontsize=7.5, color=color, fontweight=weight)

    # Colorbar
    cax = fig.add_axes([0.93, 0.10, 0.012, 0.30])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("dominance prob.", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # Tighten layout (manual since gridspec + extra axes)
    fig.subplots_adjust(left=0.10, right=0.86, top=0.96, bottom=0.08)
    _save(fig, "fig1_pareto_dominance_3seed")


# ============================================================
# FIG 2 — NORM failure mode (heatmap left + family-ECE bar right)
# ============================================================

def fig2_norm():
    strict = pd.read_csv(RES / "block3_q3_family_strict.csv")
    sens = pd.read_csv(RES / "block5_family_sensitivity.csv")

    # Average over seeds for the bar chart
    sens_agg = sens.groupby("method").agg(
        fam_ece_max=("fam_ece_max", "mean"),
        fam_ece_mean=("fam_ece_mean", "mean"),
        fam_ece_worst_non_norm=("fam_ece_worst_non_norm", "mean"),
        fam_ece_NORM=("fam_ece_NORM", "mean"),
    ).reset_index()

    method_order = ["hard_bce", "asl", "focal", "label_smooth", "chmc_coh",
                    "scp_soft_full", "soft", "soft_parent_ord", "ecgfm_linprobe"]
    sens_agg["order"] = sens_agg["method"].map({m: i for i, m in enumerate(method_order)})
    sens_agg = sens_agg.sort_values("order").reset_index(drop=True)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.0, 3.4),
                                       gridspec_kw={"width_ratios": [1.0, 1.4], "wspace": 0.38})

    # --- LEFT: 8 methods × top-2 calibrators heatmap for NORM family ECE ratio ---
    # Pivot strict subset to (method × calibrator) for NORM family
    norm = strict[strict["family"] == "NORM"].copy()
    # Plot ratio (family_ece / macro_ece) on a 8 × 2 heatmap
    cals = ["C5_isotonic", "C3_per_label_affine"]
    methods_8 = ["hard_bce", "asl", "focal", "label_smooth", "chmc_coh",
                 "scp_soft_full", "soft", "soft_parent_ord"]
    H = np.zeros((len(methods_8), len(cals)))
    for i, m in enumerate(methods_8):
        for j, c in enumerate(cals):
            sub = norm[(norm["method"] == m) & (norm["calibrator"] == c)]
            H[i, j] = float(sub["ratio"].iloc[0]) if len(sub) else np.nan
    cmap_r = LinearSegmentedColormap.from_list(
        "ratio", ["#ffffff", "#fee08b", "#f46d43", "#a50026"])
    im_l = ax_l.imshow(H, cmap=cmap_r, vmin=2.0, vmax=12.0, aspect="auto")
    ax_l.set_xticks(range(len(cals)))
    ax_l.set_xticklabels(["C5 iso", "C3 affine"], fontsize=8)
    ax_l.set_yticks(range(len(methods_8)))
    ax_l.set_yticklabels([METHOD_LABEL[m] for m in methods_8], fontsize=8)
    ax_l.set_xlabel("calibrator (top-2 per lex selector)")
    ax_l.set_title("NORM family ECE / method's macro ECE", fontsize=9, pad=4)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v = H[i, j]
            color = "black" if v < 6 else "white"
            ax_l.text(j, i, f"{v:.1f}×", ha="center", va="center", fontsize=8, color=color)
    # Use axes_grid1 to attach colorbar directly to the left panel (auto-tracks position).
    div_l = make_axes_locatable(ax_l)
    cax_l = div_l.append_axes("right", size="5%", pad=0.08)
    cb_l = fig.colorbar(im_l, cax=cax_l)
    cb_l.set_label("ratio", fontsize=7)
    cb_l.ax.tick_params(labelsize=6)

    # --- RIGHT: per-method family ECE max / mean / worst-non-NORM bars ---
    n = len(sens_agg)
    x = np.arange(n)
    w = 0.28
    bars_max = ax_r.bar(x - w, sens_agg["fam_ece_max"], width=w,
                          color="#a50026", label="max family ECE", edgecolor="black", linewidth=0.4)
    bars_mean = ax_r.bar(x, sens_agg["fam_ece_mean"], width=w,
                          color="#fdae61", label="mean family ECE", edgecolor="black", linewidth=0.4)
    bars_nn = ax_r.bar(x + w, sens_agg["fam_ece_worst_non_norm"], width=w,
                          color="#74add1", label="worst non-NORM family ECE", edgecolor="black", linewidth=0.4)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([METHOD_LABEL[m] for m in sens_agg["method"]],
                          rotation=35, ha="right", fontsize=7.5)
    ax_r.set_ylabel("family-level ECE (mean over seeds)")
    ax_r.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False,
                 fontsize=7, ncol=3)
    ax_r.set_ylim(0, 0.055)
    ax_r.set_title("max is NORM-driven; non-NORM $\\leq$ macro ECE", fontsize=9, pad=4)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.28)
    _save(fig, "fig2_norm_failure_mode")


# ============================================================
# FIG 3 — Q1 calibrator family ECE (bar chart with bootstrap CI)
# ============================================================

def fig3_q1():
    q1 = pd.read_csv(RES / "block1_q1.csv")
    boot = pd.read_csv(RES / "block1_q1_bootstrap.csv")

    # Per-backbone: mean over seeds for each calibrator + std
    fig, ax = plt.subplots(figsize=(6.5, 3.4))

    backbones = ["simpleresnet1d_hard_bce", "ecgfm_linprobe"]
    bb_labels = ["SimpleResNet1D 4M + hard-BCE\n(3 seeds, mean ± std)",
                 "ECG-FM linear probe\n(1 seed; sanity row)"]
    cal_order = ["C0_raw", "C1_temp_scalar", "C2_temp_per_label",
                 "C3_per_label_affine", "C4_family_affine", "C5_isotonic"]
    cal_short = ["C0 raw", "C1 single-T", "C2 per-label T",
                 "C3 per-label affine", "C4 family affine", "C5 isotonic"]

    n = len(cal_order)
    x = np.arange(n)
    w = 0.36
    cmap_bb = ["#1f77b4", "#ff7f0e"]
    for i, bb in enumerate(backbones):
        sub = q1[q1["backbone"] == bb]
        means = []; stds = []
        for c in cal_order:
            vals = sub[sub["calibrator"] == c]["macro_ece"].values
            if len(vals) > 0:
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0.0)
            else:
                means.append(np.nan); stds.append(0.0)
        bars = ax.bar(x + (i - 0.5) * w, means, width=w, yerr=stds,
                      color=cmap_bb[i], edgecolor="black", linewidth=0.4,
                      capsize=2.5, label=bb_labels[i], alpha=0.9)
        # Annotate values
        for j, (m_v, s_v) in enumerate(zip(means, stds)):
            if not np.isnan(m_v):
                ax.text(x[j] + (i - 0.5) * w, m_v + max(stds) * 0.6 + 0.0006,
                         f"{m_v*100:.2f}", ha="center", va="bottom",
                         fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(cal_short, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("labelwise ECE (test fold 10)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False,
               fontsize=7.5, ncol=2)
    ax.set_ylim(0, 0.026)

    # Annotate "winner" with a star ABOVE the vertical "0.58"/"0.61" number labels.
    # The rotated text spans roughly 0.0035 data units above the bar; place star at v + 0.0055.
    for i, bb in enumerate(backbones):
        sub = q1[(q1["backbone"] == bb) & (q1["calibrator"] == "C5_isotonic")]
        v = sub["macro_ece"].mean()
        ax.scatter(5 + (i - 0.5) * w, v + 0.0055, marker="*",
                    s=130, color="gold", edgecolor="black", linewidth=0.5, zorder=10,
                    clip_on=False)
    ax.text(0.01, 0.86, r"$\bigstar$ = locked-selector choice",
             transform=ax.transAxes, fontsize=7.5, color="dimgray",
             ha="left", va="top")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.20)
    _save(fig, "fig3_q1_calibrator_family")


# ============================================================
# FIG S3 — Appendix: Q4 monotonic-tie validation
# ============================================================

def figS3_q4_tie():
    tie = pd.read_csv(RES / "block5_q4_monotonic_tie.csv")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    methods = sorted(tie["method"].unique())
    cmap = plt.cm.tab10
    for i, m in enumerate(methods):
        sub = tie[tie["method"] == m]
        # x = mono_sens_gap, y = C5_isotonic_sens_gap
        ax.scatter(sub["mono_sens_gap"], sub["C5_isotonic_sens_gap"],
                    s=60, color=cmap(i / max(1, len(methods) - 1)),
                    label=METHOD_LABEL[m], edgecolor="black", linewidth=0.4, alpha=0.85)

    # y = x reference line
    lo = min(tie["mono_sens_gap"].min(), tie["C5_isotonic_sens_gap"].min()) * 0.9
    hi = max(tie["mono_sens_gap"].max(), tie["C5_isotonic_sens_gap"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, label="y = x (no effect)")
    ax.set_xlabel("monotonic calibrators C0–C4 sens-gap @ 0.9\n(identical within each (method, seed))")
    ax.set_ylabel("isotonic C5 sens-gap @ 0.9")
    ax.legend(loc="upper left", frameon=False, fontsize=7, ncol=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    # Annotate
    ax.text(0.99, 0.02,
             f"25/25 cells: C0–C4 tied; C5 differs in 25/25.\nQ4 probe is mis-specified for monotonic calibrators.",
             transform=ax.transAxes, fontsize=7.5, color="dimgray",
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                       edgecolor="gray", linewidth=0.4))

    fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.16)
    _save(fig, "figS3_q4_monotonic_tie")


# ============================================================
# TABLES (LaTeX)
# ============================================================

def table1_q1():
    q1 = pd.read_csv(RES / "block1_q1.csv")
    boot = pd.read_csv(RES / "block1_q1_bootstrap.csv")
    cal_order = ["C0_raw", "C1_temp_scalar", "C2_temp_per_label",
                 "C3_per_label_affine", "C4_family_affine", "C5_isotonic"]
    cal_short = ["C0 raw", "C1 single-T", "C2 per-label T",
                 "C3 per-label affine", "C4 family affine", "C5 isotonic"]

    def fmt_mean_std(bb, c):
        sub = q1[(q1["backbone"] == bb) & (q1["calibrator"] == c)]
        vals = sub["macro_ece"].values
        if len(vals) == 0:
            return "—"
        if len(vals) == 1:
            return f"{vals[0]*1000:.1f}"
        m, s = np.mean(vals), np.std(vals)
        return f"{m*1000:.1f} $\\pm$ {s*1000:.1f}"

    def fmt_boot_ci(bb, c):
        # bootstrap rows have ece_reduction_mean, ci_lo, ci_hi vs C0 baseline
        if c == "C0_raw":
            return "(ref.)"
        sub = boot[(boot["backbone"] == bb) & (boot["calibrator"] == c)]
        if len(sub) == 0:
            return "—"
        rm = sub["ece_reduction_mean"].mean() * 1000
        lo = sub["ci_lo"].mean() * 1000
        hi = sub["ci_hi"].mean() * 1000
        return f"{rm:+.1f} [{lo:+.1f}, {hi:+.1f}]"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Q1 --- Post-hoc calibrators meaningfully improve probability quality on hard-BCE.} ",
        r"Labelwise ECE on test fold 10 (units: $\times 10^{-3}$). SimpleResNet1D 4M values are mean\,$\pm$\,std over 3 seeds; ECG-FM is one seed (inference-only sanity row). $\Delta$ columns: patient-paired bootstrap mean ECE reduction vs the raw-probability baseline C0 with 95\% CI (\emph{positive} = ECE went down = improvement). The lex selector chooses C5 isotonic for both backbones, and for every other base method (Table~\ref{tab:q2}).}",
        r"\label{tab:q1}",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"              & \multicolumn{2}{c}{SimpleResNet1D + hard-BCE} & \multicolumn{2}{c}{ECG-FM linear probe} \\",
        r"              & \multicolumn{2}{c}{(3 seeds)}                 & \multicolumn{2}{c}{(1 seed)}            \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Calibrator    & ECE          & $\Delta$ vs C0 [95\% CI]      & ECE   & $\Delta$ vs C0 [95\% CI] \\",
        r"\midrule",
    ]
    for c, cs in zip(cal_order, cal_short):
        r1 = fmt_mean_std("simpleresnet1d_hard_bce", c)
        r2 = fmt_boot_ci("simpleresnet1d_hard_bce", c)
        r3 = fmt_mean_std("ecgfm_linprobe", c)
        r4 = fmt_boot_ci("ecgfm_linprobe", c)
        mark = r"\,$\star$" if c == "C5_isotonic" else ""
        lines.append(f"{cs}{mark} & {r1} & {r2} & {r3} & {r4} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.3em}\\",
        r"{\footnotesize $\star$ = lex-selector winner. Patient-paired bootstrap $B{=}1000$; statements with $N_{\text{pos}}<5$ excluded uniformly.}",
        r"\end{table}",
    ]
    out = "\n".join(lines)
    (FIG / "table1_q1.tex").write_text(out)
    print(f"  saved figures/table1_q1.tex")


def table2_q2():
    pareto = pd.read_csv(RES / "block2_q2_pareto.csv")
    dom = pd.read_csv(RES / "block2_q2_dominance_3seed.csv")
    sens = pd.read_csv(RES / "block5_family_sensitivity.csv").groupby("method").agg(
        fam_ece_mean=("fam_ece_mean", "mean"),
        fam_ece_worst_non_norm=("fam_ece_worst_non_norm", "mean"),
    ).reset_index()

    # Join
    df = pareto.merge(dom[["method", "dominance_mean", "dominance_min", "dominance_max", "n_seeds"]],
                       on="method", how="left")
    df = df.merge(sens, on="method", how="left")
    # Order: hard_bce first (ref), then by AUC desc
    order_priority = {"hard_bce": -1e9}
    df["order_key"] = df.apply(lambda r: order_priority.get(r["method"], -r["macro_auc"]), axis=1)
    df = df.sort_values("order_key").reset_index(drop=True)

    def fmt_dom(row):
        if row["method"] == "hard_bce":
            return "(ref.)"
        n = int(row["n_seeds"])
        mean = row["dominance_mean"]
        if pd.isna(mean):
            return "—"
        if n == 1:
            return f"{mean:.3f} (1 seed)"
        return f"{mean:.3f} [{row['dominance_min']:.3f}, {row['dominance_max']:.3f}]"

    def fmt_dom_compact(row):
        if row["method"] == "hard_bce":
            return "(ref.)"
        n = int(row["n_seeds"])
        mean = row["dominance_mean"]
        if pd.isna(mean):
            return "---"
        if n == 1:
            return f"{mean:.3f} (1 seed)"
        return f"{mean:.3f}\\,[{row['dominance_min']:.2f},{row['dominance_max']:.2f}]"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Q2 --- Pareto front under the locked lex best-of-calibrator selector; D6 cautionary.}",
        r"Per-method best calibrator on val fold 9 (lex rule: ECE $\to$ 1\,SE band $\to$ threshold-transfer gap @ sens=0.9 $\to$ 1\,SE band $\to$ macro-AUC $\to$ param count). All 9 methods choose \textbf{C5 isotonic}. Metrics on test fold 10, 3-seed mean (ECG-FM is 1 seed). Dominance prob.\ = fraction of $B{=}500$ patient-paired bootstrap resamples in which the row jointly improves (AUC$\uparrow$, ECE$\downarrow$, max fam.\ ECE$\downarrow$) over (hard-BCE\,$+$\,C5), reported as 3-seed mean ($[$min, max$]$). \textbf{SCP-Soft (D6), soft+parent+ord, and CHMC-proj are dominated at 0.000 in every seed.}}",
        r"\label{tab:q2}",
        r"\begin{tabular}{l r r r r r c l}",
        r"\toprule",
        r"                       & macro  & label. & fam. ECE & fam. ECE & non-NORM &        & dominance \\",
        r"Method (+ C5 iso)      &   AUC  &    ECE &      max &     mean & fam. ECE & Pareto & vs ref.\ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        m = r["method"]
        on_p = bool(r["on_pareto_front"])
        is_ref = (m == "hard_bce")
        is_d6 = (m == "scp_soft_full")
        name = METHOD_LABEL[m]
        if is_ref:
            name = r"\textbf{" + name + r"} (ref.)"
        elif is_d6:
            name = r"\textit{" + name + r"}"
        fam_mean = r["fam_ece_mean"] if pd.notna(r["fam_ece_mean"]) else 0
        fam_nn = r["fam_ece_worst_non_norm"] if pd.notna(r["fam_ece_worst_non_norm"]) else 0
        pareto_str = r"$\checkmark$" if on_p else r"$\cdot$"
        lines.append(
            f"{name} & {r['macro_auc']:.4f} & {r['labelwise_ece']*1000:.2f} & "
            f"{r['family_ece_max']*1000:.2f} & {fam_mean*1000:.2f} & "
            f"{fam_nn*1000:.2f} & {pareto_str} & {fmt_dom_compact(r)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.3em}\\",
        r"{\footnotesize ECE values in $\times 10^{-3}$. The Pareto front in (AUC, labelwise ECE, max fam.\ ECE) contains hard-BCE, ASL, soft-likelihood, and ECG-FM (all with C5 isotonic). Worst non-NORM family ECE $\leq$ method's macro ECE in every row, confirming the NORM-only failure mode (see Fig.~\ref{fig:norm}).}",
        r"\end{table*}",
    ]
    out = "\n".join(lines)
    (FIG / "table2_q2_pareto.tex").write_text(out)
    print(f"  saved figures/table2_q2_pareto.tex")


# ============================================================
# LATEX INCLUDES
# ============================================================

def latex_includes():
    snippets = [
        (r"% === Hero Fig 1: Pareto + 3-seed dominance heatmap ===",
         r"\begin{figure*}[t]",
         r"  \centering",
         r"  \includegraphics[width=0.92\textwidth]{figures/fig1_pareto_dominance_3seed.pdf}",
         r"  \caption{\textbf{Q2 — Pareto front under the locked lex best-of-calibrator selector, with 3-seed paired-bootstrap dominance vs the reference (hard-BCE + C5 isotonic).} ",
         r"  \textbf{(A)} macro-AUC vs labelwise ECE; closed circles are non-dominated, open circles dominated, $\star$ = reference. ",
         r"  \textbf{(B)} macro-AUC vs maximum family-level ECE; the NORM family drives the max. ",
         r"  \textbf{(C)} Patient-paired bootstrap dominance probability vs hard-BCE + isotonic, per seed and mean across seeds. ",
         r"  \emph{SCP-Soft (D6), soft+parent+ord, CHMC-proj, and label-smoothing are dominated at 0.000 in every seed}, providing strong negative evidence that training-time weak-label objectives cannot recover via post-hoc calibration on PTB-XL.}",
         r"  \label{fig:hero}",
         r"\end{figure*}"),
        (r"% === Fig 2: NORM-family failure mode ===",
         r"\begin{figure}[t]",
         r"  \centering",
         r"  \includegraphics[width=\linewidth]{figures/fig2_norm_failure_mode.pdf}",
         r"  \caption{\textbf{Q3 — calibration failure is concentrated in the NORM family.} ",
         r"  Left: NORM family ECE divided by the method's macro ECE, for the strict prespecified subset of 8 base methods $\times$ top-2 calibrators by lex selector. Ratios $\geq 2$ are prespecified ``failure cells'' — 16/16 cells fall in NORM. ",
         r"  Right: per-method family-ECE max, mean, and worst non-NORM (3-seed mean). The max is consistently NORM-driven; worst non-NORM is $\leq$ each method's macro ECE, ruling out broad multi-family heterogeneity.}",
         r"  \label{fig:norm}",
         r"\end{figure}"),
        (r"% === Fig 3: Q1 calibrator family ECE ===",
         r"\begin{figure}[t]",
         r"  \centering",
         r"  \includegraphics[width=\linewidth]{figures/fig3_q1_calibrator_family.pdf}",
         r"  \caption{\textbf{Q1 — post-hoc calibrators meaningfully improve probability quality on hard-BCE.} ",
         r"  Labelwise ECE on test fold 10 (units: $\times 10^{-3}$) for the 6 locked calibrators applied to (i) SimpleResNet1D 4M + hard-BCE (mean $\pm$ std over 3 seeds) and (ii) ECG-FM linear probe (1 seed, inference-only sanity row). ",
         r"  $\star$ = lex selector winner (C5 isotonic in both cases). 3-seed patient-paired bootstrap CIs reported in Table~\ref{tab:q1}.}",
         r"  \label{fig:q1}",
         r"\end{figure}"),
        (r"% === Appendix Fig S1: scaffold flow (existing) ===",
         r"\begin{figure}[t]",
         r"  \centering",
         r"  \includegraphics[width=0.9\linewidth]{figures/figS_scaffold_flow.pdf}",
         r"  \caption{\textbf{Scaffold pipeline.} ",
         r"  Per-(method, seed) prediction NPZ on val/test fold $\rightarrow$ calibrator fit on val $\rightarrow$ apply to test $\rightarrow$ locked metric suite $\rightarrow$ lex selector $\rightarrow$ leaderboard JSON with uniform exclusion sets emitted. ",
         r"  CI self-test reproduces a worked example within bootstrap CI in $<1$ minute.}",
         r"  \label{fig:scaffold}",
         r"\end{figure}"),
        (r"% === Appendix Fig S3: Q4 monotonic-tie validation ===",
         r"\begin{figure}[t]",
         r"  \centering",
         r"  \includegraphics[width=0.85\linewidth]{figures/figS3_q4_monotonic_tie.pdf}",
         r"  \caption{\textbf{Appendix: Q4 operating-point stability probe is mis-specified for monotonic calibrators.} ",
         r"  For every (method, seed) cell, the 5 monotonic calibrators C0--C4 produce \emph{identical} sens-gap @ 0.9 (verified 25/25 cells). Only the non-monotonic isotonic C5 moves the gap. The internally prespecified Spearman $\rho$ statistic therefore measures an isotonic-vs-monotonic-family binary distinction, not a continuous calibration $\to$ stability relationship. We demote Q4 to this appendix and reframe as ``calibration-family invariance of operating-point stability under monotonic transforms.''}",
         r"  \label{fig:q4tie}",
         r"\end{figure}"),
    ]
    out = "\n\n".join("\n".join(s) for s in snippets)
    (FIG / "latex_includes.tex").write_text(out)
    print(f"  saved figures/latex_includes.tex")


# ============================================================
# MAIN
# ============================================================

def main():
    print("[paper_figures] generating Fig 1 hero ...")
    fig1_hero()
    print("[paper_figures] generating Fig 2 NORM failure mode ...")
    fig2_norm()
    print("[paper_figures] generating Fig 3 Q1 calibrator family ...")
    fig3_q1()
    print("[paper_figures] generating Fig S3 Q4 mono-tie (appendix) ...")
    figS3_q4_tie()
    print("[paper_figures] generating Table 1 Q1 (LaTeX) ...")
    table1_q1()
    print("[paper_figures] generating Table 2 Q2 (LaTeX) ...")
    table2_q2()
    print("[paper_figures] writing latex_includes.tex ...")
    latex_includes()
    print("[paper_figures] DONE.")


if __name__ == "__main__":
    main()
