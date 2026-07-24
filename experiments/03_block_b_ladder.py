"""
Block B — 5-step component-isolation ladder.
Train rows 2/3/4 (rows 1 & 5 reused from Block A), then evaluate all 5.
"""
import sys
import os
sys.path.insert(0, os.environ.get("PTBXL_CAL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

from src.data import get_loaders
from src.models import build_model
from src.train import train_one, evaluate_loader

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))

# Ladder rows; row 1 = hard_bce (reused from A); row 5 = scp_soft_full (reused from A)
LADDER = [
    ("hard_bce", "row1: hard-BCE"),
    ("soft", "row2: + stmt-soft"),
    ("soft_parent", "row3: + parent-soft heads"),
    ("soft_parent_ord", "row4: + ordinal CORN"),
    ("scp_soft_full", "row5: + hierarchy (full)"),
]
SEEDS = [42, 123, 2024]


def main():
    print("=" * 60)
    print("Block B - 5-step Component-Isolation Ladder")
    print("=" * 60)

    # Train missing rows (2,3,4 if not already done)
    for method, label in LADDER:
        for s in SEEDS:
            ckpt = ROOT / f"checkpoints/{method}_seed{s}.pt"
            if ckpt.exists():
                print(f"SKIP {method} seed{s}: {ckpt.name} exists")
                continue
            print(f"\n[Block B] Training {method} ({label}) seed={s}")
            train_one(method, s, epochs=50, batch_size=128)

    # Evaluate all 5 rows on test set
    hier_data = json.load(open(ROOT / "results/hierarchy_edges.json"))
    edges = hier_data["edges"]
    _, _, test_loader, train_ds = get_loaders(batch_size=128, num_workers=4)

    rows = []
    for method, label in LADDER:
        for s in SEEDS:
            ckpt_path = ROOT / f"checkpoints/{method}_seed{s}.pt"
            if not ckpt_path.exists():
                print(f"MISSING {ckpt_path}")
                continue
            ckpt = torch.load(ckpt_path, map_location="cuda")
            model = build_model().cuda()
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            sd = ckpt["state_dict"]
            if not any(k.startswith("module.") for k in sd.keys()) and isinstance(model, nn.DataParallel):
                sd = {f"module.{k}": v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            metrics = evaluate_loader(model, test_loader, edges,
                                       train_ds.super2idx, train_ds.sub2idx, train_ds.stmt2idx)
            rows.append({"row_label": label, "method": method, "seed": s, **metrics})
            print(f"{label:36s} seed{s}: AUC={metrics['macro_auc']:.4f} "
                  f"ECE={metrics['macro_ece']:.4f} HVR={metrics['hvr']:.4f}")
            del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/block_b_raw.csv", index=False)
    agg = df.groupby(["row_label", "method"]).agg(
        macro_auc=("macro_auc", "mean"), macro_auc_std=("macro_auc", "std"),
        macro_brier=("macro_brier", "mean"), macro_brier_std=("macro_brier", "std"),
        macro_ece=("macro_ece", "mean"), macro_ece_std=("macro_ece", "std"),
        hvr=("hvr", "mean"), hvr_std=("hvr", "std")).round(4)
    agg.to_csv(ROOT / "results/block_b_agg.csv")
    print("\n=== Block B 5-Step Ladder ===")
    print(agg)
    print("\nBlock B COMPLETE.")


if __name__ == "__main__":
    main()
