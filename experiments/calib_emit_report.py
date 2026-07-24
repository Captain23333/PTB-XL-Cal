"""Emit docs/EXPERIMENT_REPORT.md summarising what was executed by the
PTB-XL-Cal pipeline. Reads results/, figures/, logs/ to populate the
alignment checklist (per /run-analysis Phase 7 spec).
"""
from __future__ import annotations
import sys, time, json
import os
sys.path.insert(0, os.environ.get("PTBXL_CAL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
OUT = DOCS / "EXPERIMENT_REPORT.md"


def maybe_read_csv(p):
    p = ROOT / p
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def maybe_json(p):
    p = ROOT / p
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def file_status(p):
    return "OK" if (ROOT / p).exists() else "MISSING"


def section_artifact_inventory():
    items = [
        "results/cal_probs/",
        "results/block1_q1.csv",
        "results/block1_q1_bootstrap.csv",
        "results/block2_q2_full_grid.csv",
        "results/block2_q2_pareto.csv",
        "results/block2_q2_dominance.csv",
        "results/block2_q2_selected_calibrators.json",
        "results/block3_q3_family.csv",
        "results/block3_q3_failure_cells.csv",
        "results/block3_q3_heatmap.npz",
        "results/block4_threshold.csv",
        "results/block4_spearman.json",
        "figures/fig1_q1_calibrator_family.png",
        "figures/fig2_q2_pareto_dominance.png",
        "figures/fig3_q3_family_reliability_heatmap.png",
        "figures/fig4_block4_threshold_transfer.png",
        "figures/figS_scaffold_flow.png",
        "leaderboard_worked_example.json",
        "leaderboard_schema.json",
        "splits/val_fold9.csv",
        "splits/test_fold10.csv",
    ]
    lines = ["## Artifact inventory", "", "| Path | Status |", "|---|---|"]
    for it in items:
        lines.append(f"| `{it}` | {file_status(it)} |")
    return "\n".join(lines)


def section_q1():
    df = maybe_read_csv("results/block1_q1.csv")
    if df is None:
        return "## Q1 (Block 1) -- Calibrator family\n\nresults/block1_q1.csv MISSING"
    g = df.groupby(["backbone", "calibrator"])["macro_ece"].agg(["mean", "std"]).round(4)
    return "## Q1 (Block 1) -- Calibrator family\n\nMacro labelwise ECE (mean +/- std over seeds):\n\n" + g.to_markdown()


def section_q2():
    pareto = maybe_read_csv("results/block2_q2_pareto.csv")
    dom = maybe_read_csv("results/block2_q2_dominance.csv")
    sel = maybe_json("results/block2_q2_selected_calibrators.json")
    s = ["## Q2 (Block 2) -- Pareto + dominance"]
    if sel:
        s.append("\n### Selected calibrators (lex selector)\n")
        for m, info in sel.items():
            s.append(f"- **{m}**: best={info['best']}  second={info.get('second')}")
    if pareto is not None:
        s.append("\n### Pareto front (best calibrator per method)\n")
        s.append(pareto.round(4).to_markdown(index=False))
    if dom is not None:
        s.append("\n### Dominance vs hard_bce + best-cal\n")
        s.append(dom.round(3).to_markdown(index=False))
    return "\n".join(s)


def section_q3():
    fam = maybe_read_csv("results/block3_q3_family.csv")
    fail = maybe_read_csv("results/block3_q3_failure_cells.csv")
    s = ["## Q3 (Block 3) -- Family decomposition"]
    if fail is not None:
        s.append(f"\nFailure cells (family_ECE > 2x macro_ECE): **{len(fail)}**")
        if len(fail) > 0:
            s.append("\n" + fail.round(4).to_markdown(index=False))
    if fam is not None:
        s.append(f"\nTotal family rows: {len(fam)} -- see `results/block3_q3_family.csv`.")
    return "\n".join(s)


def section_block4():
    sp = maybe_json("results/block4_spearman.json")
    s = ["## Block 4 -- Operating-point stability"]
    if sp:
        s.append(f"\nSpearman rho (labelwise_ECE vs sens_gap@0.9) = **{sp['spearman_rho']:.3f}** "
                 f"(p={sp['p_value']:.4f}, n_cells={sp['n_cells']}).")
    return "\n".join(s)


def section_alignment():
    """Pre-registered question alignment checklist."""
    pareto = maybe_read_csv("results/block2_q2_pareto.csv")
    fail = maybe_read_csv("results/block3_q3_failure_cells.csv")
    sp = maybe_json("results/block4_spearman.json")
    s = ["## Alignment with internally prespecified questions",
         "",
         "| Claim | Evidence path | Status |",
         "|---|---|---|"]
    s.append(f"| Q1: Calibrator family helps within protocol | results/block1_q1.csv + fig1 | "
             f"{file_status('results/block1_q1.csv')} |")
    s.append(f"| Q2: Best-of-cal Pareto front + dominance probabilities | results/block2_q2_*.csv + fig2 | "
             f"{file_status('results/block2_q2_pareto.csv')} |")
    s.append(f"| Q3: Family-level failure cells | results/block3_q3_*.csv + fig3 | "
             f"{file_status('results/block3_q3_family.csv')} |")
    s.append(f"| Op-point stability: ECE -> threshold-gap (Spearman rho) | results/block4_spearman.json + fig4 | "
             f"{file_status('results/block4_spearman.json')} |")
    return "\n".join(s)


def main():
    parts = []
    parts.append(f"# PTB-XL-Cal R1 EXPERIMENT REPORT")
    parts.append(f"\n_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    parts.append(f"\n_Source: docs/refine/FINAL_PROPOSAL.md (round 3, 9.1/10 READY)_\n")
    parts.append(section_artifact_inventory())
    parts.append("")
    parts.append(section_alignment())
    parts.append("")
    parts.append(section_q1())
    parts.append("")
    parts.append(section_q2())
    parts.append("")
    parts.append(section_q3())
    parts.append("")
    parts.append(section_block4())
    parts.append("")
    with open(OUT, "w") as f:
        f.write("\n\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
