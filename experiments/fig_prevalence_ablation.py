"""Figure: PTB-XL internal prevalence-controlled ablation (Major-2 rebuttal).

Twin-panel single-column figure:
  (a) NORM family-ECE / macro-ECE ratio vs NORM prevalence (matched mode),
      with 95% bootstrap band, the 2x failure threshold, and the Chapman 0.81x
      reference at 18%.
  (b) NORM family ECE alone (the denominator-independent signal) vs prevalence.

Usage:
    conda run -n ecg python experiments/fig_prevalence_ablation.py
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

# x = prevalence as float
def pv(s):
    return 0.438 if s.startswith("full") else float(s)
m["prev"] = m["prevalence"].map(pv)
m = m.sort_values("prev")

x = m["prev"].values * 100
ratio = m["ratio_mean"].values
lo = m["ratio_lo"].values
hi = m["ratio_hi"].values
norm_ece = m["norm_ece_x1e3"].values

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))

# Panel (a): ratio
ax = axes[0]
ax.fill_between(x, lo, hi, color="#4C72B0", alpha=0.18, label="95% bootstrap CI")
ax.plot(x, ratio, "-o", color="#4C72B0", lw=2, ms=5, label="PTB-XL (matched, 71-label)")
ax.axhline(2.0, ls="--", color="0.4", lw=1)
ax.text(19, 2.08, r"$2\times$ failure threshold", fontsize=6.5, color="0.3")
ax.scatter([18], [0.81], marker="*", s=130, color="#C44E52", zorder=5,
           label="Chapman 18% (5-label)")
ax.text(18.4, 0.95, "0.81", fontsize=7, color="#C44E52")
ax.set_xlabel("NORM prevalence (%)", fontsize=8)
ax.set_ylabel(r"NORM ECE / macro ECE", fontsize=8)
ax.set_title("(a) Failure ratio vs prevalence", fontsize=8.5)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6.0, loc="upper left", framealpha=0.9)
ax.set_ylim(0, 5.4)

# Panel (b): NORM ECE alone
ax = axes[1]
ax.plot(x, norm_ece, "-s", color="#55A868", lw=2, ms=5)
for xi, yi in zip(x, norm_ece):
    ax.annotate(f"{yi:.1f}", (xi, yi), fontsize=6.5, xytext=(0, 5),
                textcoords="offset points", ha="center", color="#2c6e49")
ax.set_xlabel("NORM prevalence (%)", fontsize=8)
ax.set_ylabel(r"NORM family ECE ($\times10^{-3}$)", fontsize=8)
ax.set_title("(b) NORM ECE (denominator-free)", fontsize=8.5)
ax.tick_params(labelsize=7)
ax.set_ylim(16, 27)

fig.tight_layout()
fig.savefig(FIG / "figS4_prevalence_ablation.pdf", bbox_inches="tight")
fig.savefig(FIG / "figS4_prevalence_ablation.png", dpi=150, bbox_inches="tight")
print("Saved figS4_prevalence_ablation.{pdf,png}")
