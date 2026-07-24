"""
Block E — Clinical case study.
Pick 2 ambiguous test ECGs (1 MI-borderline, 1 HYP-borderline);
plot ECG + hard-BCE vs SCP-Soft predictions and confidence + hierarchy violations.
"""
import sys
import os
sys.path.insert(0, os.environ.get("PTBXL_CAL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.data import get_loaders
from src.models import build_model

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))


def load_model(method, seed=42):
    ckpt_path = ROOT / f"checkpoints/{method}_seed{seed}.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model = build_model().cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    sd = ckpt["state_dict"]
    if not any(k.startswith("module.") for k in sd.keys()) and isinstance(model, nn.DataParallel):
        sd = {f"module.{k}": v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def main():
    print("=" * 60)
    print("Block E - Clinical Case Study")
    print("=" * 60)

    families = json.load(open(ROOT / "results/clinical_families.json"))
    _, _, _, train_ds = get_loaders(batch_size=1, num_workers=0)

    # Load test set
    db = pd.read_csv(ROOT / "data/ptbxl_database.csv")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    test_db = db[db["strat_fold"] == 10].reset_index(drop=True)
    print(f"Test set: {len(test_db)} records")

    # Find borderline cases (likelihood ∈ {30,34,35,50}) for MI and HYP
    cases = {"MI": None, "HYP": None}
    target_liks = {30, 34, 35, 50}
    for fam in ["MI", "HYP"]:
        fam_stmts = set(families.get(fam, []))
        for i, row in test_db.iterrows():
            for stmt, lik in row["scp_codes"].items():
                if stmt in fam_stmts and int(round(float(lik))) in target_liks:
                    cases[fam] = (i, stmt, int(round(float(lik))))
                    break
            if cases[fam] is not None:
                break
        print(f"{fam}-borderline case found: {cases[fam]}")

    # Load both models
    model_bce = load_model("hard_bce")
    model_scp = load_model("scp_soft_full")
    if model_bce is None or model_scp is None:
        print("Required checkpoints missing - cannot run case study")
        return

    # Plot 2 cases × 2 panels
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for row_idx, (fam, info) in enumerate(cases.items()):
        if info is None:
            continue
        ecg_idx, target_stmt, target_lik = info
        meta = test_db.iloc[ecg_idx]
        sig, _ = wfdb.rdsamp(str(ROOT / "data" / meta["filename_lr"]))  # (1000, 12)
        sig = sig.astype(np.float32)

        # Left panel: ECG signal (4 leads compressed)
        ax = axes[row_idx, 0]
        time = np.arange(sig.shape[0]) / 100.0
        leads_to_show = [0, 1, 6, 7]   # I, II, V1, V2
        lead_names = ["I", "II", "V1", "V2"]
        for k, lead in enumerate(leads_to_show):
            ax.plot(time, sig[:, lead] / max(0.1, sig[:, lead].std()) - k*4,
                    lw=0.8, color="black")
            ax.text(-0.4, -k*4, lead_names[k], fontsize=9, va="center")
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        ax.set_title(f"{fam}-borderline: ecg_id={int(meta['ecg_id'])}, "
                     f"{target_stmt} (likelihood={target_lik})", fontsize=11)
        ax.set_xlim(-0.5, 10)

        # Right panel: predictions on family statements
        fam_stmts = [s for s in families[fam] if s in train_ds.stmt2idx]
        x_in = (sig.T - sig.T.mean(axis=1, keepdims=True)) / (sig.T.std(axis=1, keepdims=True) + 1e-6)
        x_in = torch.from_numpy(x_in.astype(np.float32)).unsqueeze(0).cuda()
        with torch.no_grad():
            out_bce = model_bce(x_in)
            out_scp = model_scp(x_in)
            p_bce = torch.sigmoid(out_bce["stmt_logits"]).cpu().numpy()[0]
            p_scp = torch.sigmoid(out_scp["stmt_logits"]).cpu().numpy()[0]
            p_super_bce = torch.sigmoid(out_bce["super_logits"]).cpu().numpy()[0]
            p_super_scp = torch.sigmoid(out_scp["super_logits"]).cpu().numpy()[0]

        ax2 = axes[row_idx, 1]
        n_stmts = len(fam_stmts)
        x_pos = np.arange(n_stmts)
        width = 0.38
        bce_vals = [p_bce[train_ds.stmt2idx[s]] for s in fam_stmts]
        scp_vals = [p_scp[train_ds.stmt2idx[s]] for s in fam_stmts]
        ax2.bar(x_pos - width/2, bce_vals, width, label="hard-BCE", color="tab:red", alpha=0.8)
        ax2.bar(x_pos + width/2, scp_vals, width, label="SCP-Soft", color="tab:blue", alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(fam_stmts, rotation=60, ha="right", fontsize=8)
        ax2.axhline(0.5, color="gray", ls="--", lw=0.6)
        ax2.set_ylabel("Predicted probability"); ax2.set_ylim(0, 1.05)
        # Annotate parent (super) prob
        sup_idx = train_ds.super2idx.get(fam, -1)
        if sup_idx >= 0:
            title = (f"{fam} family predictions | "
                     f"P({fam}_super) BCE={p_super_bce[sup_idx]:.2f} vs SCP={p_super_scp[sup_idx]:.2f}")
        else:
            title = f"{fam} family predictions"
        ax2.set_title(title, fontsize=11)
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(ROOT / "figures/fig_block_e_cases.png", dpi=200)
    plt.close()
    print(f"Saved: figures/fig_block_e_cases.png")

    # Save case info as JSON for paper writing
    case_info = {
        fam: {"ecg_idx": int(info[0]), "target_stmt": info[1], "target_lik": info[2]}
        for fam, info in cases.items() if info is not None
    }
    with open(ROOT / "results/block_e_cases.json", "w") as f:
        json.dump(case_info, f, indent=2)
    print("\nBlock E COMPLETE.")


if __name__ == "__main__":
    main()
