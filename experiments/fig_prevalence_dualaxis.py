"""Main-text figure: prevalence-controlled ablation, dual-axis.

Left axis  : NORM/macro ECE ratio (4.30 -> 2.10) with 95% bootstrap band.
Right axis : NORM family ECE absolute value (24.8 -> 19.2).
Both fall together as NORM prevalence drops, so the ratio decline is not a
macro-denominator artefact. Chapman 5-label reference (0.81x at 18%) marked.

Usage:
    conda run -n ecg python experiments/fig_prevalence_dualaxis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
df = pd.read_csv(ROOT / "results" / "prevalence_ablation.csv")
m = df[df["mode"] == "matched"].copy()
m["prev"] = m["prevalence"].map(lambda s: 0.438 if s.startswith("full") else float(s))
m = m.sort_values("prev")

x = m["prev"].values * 100
ratio = m["ratio_mean"].values
lo, hi = m["ratio_lo"].values, m["ratio_hi"].values
norm_ece = m["norm_ece_x1e3"].values

C_RATIO = "#4C72B0"   # blue (left)
C_ECE = "#55A868"     # green (right)
C_CHAP = "#C44E52"    # red

fig, axL = plt.subplots(figsize=(3.5, 2.8))

# Left axis: ratio + CI band
axL.fill_between(x, lo, hi, color=C_RATIO, alpha=0.15)
axL.plot(x, ratio, "-o", color=C_RATIO, lw=2, ms=5, label="NORM/macro ratio (left)")
axL.axhline(2.0, ls="--", color="0.45", lw=1)
axL.text(30, 2.06, r"$2\times$ failure threshold", fontsize=6, color="0.35")
axL.scatter([18], [0.81], marker="*", s=150, color=C_CHAP, zorder=6, clip_on=False)
axL.annotate("Chapman\n(5-label) 0.81", (18, 0.81), fontsize=6, color=C_CHAP,
             xytext=(20.5, 0.55), va="center")
axL.set_xlabel("NORM prevalence (%)", fontsize=8)
axL.set_ylabel(r"NORM / macro ECE ratio", fontsize=8, color=C_RATIO)
axL.tick_params(axis="y", labelcolor=C_RATIO, labelsize=7)
axL.tick_params(axis="x", labelsize=7)
axL.set_ylim(0, 5.5)
axL.set_xlim(16, 46)

# Right axis: NORM family ECE absolute
axR = axL.twinx()
axR.plot(x, norm_ece, "-s", color=C_ECE, lw=2, ms=4.5, label="NORM family ECE (right)")
axR.set_ylabel(r"NORM family ECE ($\times10^{-3}$)", fontsize=8, color=C_ECE)
axR.tick_params(axis="y", labelcolor=C_ECE, labelsize=7)
axR.set_ylim(15, 28)

# combined legend
lines = axL.get_lines()[:1] + axR.get_lines()[:1]
axL.legend(lines, [l.get_label() for l in lines], fontsize=6.0,
           loc="upper left", framealpha=0.9)

fig.tight_layout()
fig.savefig(FIG / "fig5_prevalence_ablation.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig5_prevalence_ablation.png", dpi=160, bbox_inches="tight")
print("Saved fig5_prevalence_ablation.{pdf,png}")
