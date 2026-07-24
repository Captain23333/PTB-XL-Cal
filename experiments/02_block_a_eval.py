"""
Block A — Evaluate trained checkpoints on test set + add post-hoc baselines.
Run AFTER experiments/run_block_a.sh trains all checkpoints.

Outputs:
  results/block_a_raw.csv      - per (method, seed) full metrics
  results/block_a_agg.csv      - mean ± std per method
  results/block_a_bootstrap.csv - paired bootstrap p-values vs hard_bce
  results/block_a_probs.npz    - probability matrices for downstream blocks
"""
import sys
import os
sys.path.insert(0, os.environ.get("PTBXL_CAL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

from src.data import get_loaders
from src.models import build_model
from src.train import evaluate_loader
from src.evaluate import (macro_auc, macro_brier, macro_ece, hvr,
                          bootstrap_paired_diff)
from src.baselines import fit_temperature, fit_dirichlet, chmc_project

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
TRAINED_METHODS = ["hard_bce", "asl", "focal", "label_smooth", "scp_soft_full"]
SEEDS = [42, 123, 2024]


def load_model(method, seed):
    ckpt_path = ROOT / f"checkpoints/{method}_seed{seed}.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model = build_model().cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    sd = ckpt["state_dict"]
    # Strip 'module.' prefix if present
    if not any(k.startswith("module.") for k in sd.keys()) and isinstance(model, nn.DataParallel):
        sd = {f"module.{k}": v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model


def main():
    print("=" * 60)
    print("Block A Evaluation")
    print("=" * 60)

    hier_data = json.load(open(ROOT / "results/hierarchy_edges.json"))
    edges = hier_data["edges"]
    scp_df = pd.read_csv(ROOT / "data/scp_statements.csv", index_col=0)
    _, val_loader, test_loader, train_ds = get_loaders(batch_size=128, num_workers=4)

    rows = []
    all_probs = {}  # (method, seed) -> probs dict

    # 1) Trained methods (5 × 3 seeds)
    for m in TRAINED_METHODS:
        for s in SEEDS:
            model = load_model(m, s)
            if model is None:
                print(f"SKIP {m} seed{s} - checkpoint missing")
                continue
            metrics, probs = evaluate_loader(
                model, test_loader, edges,
                train_ds.super2idx, train_ds.sub2idx, train_ds.stmt2idx,
                return_probs=True)
            all_probs[(m, s)] = probs
            rows.append({"method": m, "seed": s, **metrics})
            print(f"{m:18s} seed{s}: AUC={metrics['macro_auc']:.4f} "
                  f"Brier={metrics['macro_brier']:.4f} ECE={metrics['macro_ece']:.4f} "
                  f"HVR={metrics['hvr']:.4f}")
            del model
            torch.cuda.empty_cache()

    # 2) Post-hoc baselines on hard_bce: temp scaling + per-label affine logit (legacy 'Dirichlet' name)
    for s in SEEDS:
        if ("hard_bce", s) not in all_probs:
            continue
        # Need val_loader for fitting calibrators
        model = load_model("hard_bce", s)
        if model is None: continue
        _, val_p = evaluate_loader(model, val_loader, edges,
                                   train_ds.super2idx, train_ds.sub2idx, train_ds.stmt2idx,
                                   return_probs=True)
        del model; torch.cuda.empty_cache()

        test_p = all_probs[("hard_bce", s)]
        # logits from probs (inverse sigmoid)
        eps = 1e-7
        val_logits = torch.from_numpy(np.log(val_p["p_stmt"]/(1-val_p["p_stmt"]+eps)+eps)).cuda()
        val_targets = torch.from_numpy(val_p["y_stmt"]).cuda()
        test_logits = torch.from_numpy(np.log(test_p["p_stmt"]/(1-test_p["p_stmt"]+eps)+eps)).cuda()

        # Temperature scaling
        T = fit_temperature(val_logits, val_targets)
        p_temp = torch.sigmoid(test_logits / T).cpu().numpy()
        auc_t, _ = macro_auc(test_p["y_stmt"], p_temp)
        rows.append({"method": "hard_bce+temp", "seed": s,
                     "macro_auc": auc_t,
                     "macro_brier": macro_brier(test_p["y_stmt"], p_temp),
                     "macro_ece": macro_ece(test_p["y_stmt"], p_temp),
                     "hvr": hvr(test_p["p_super"], test_p["p_sub"], p_temp,
                                edges, train_ds.super2idx, train_ds.sub2idx, train_ds.stmt2idx)})
        all_probs[("hard_bce+temp", s)] = {**test_p, "p_stmt": p_temp}
        print(f"hard_bce+temp     seed{s}: AUC={auc_t:.4f}")

        # Per-label affine logit (legacy 'Dirichlet' name)
        A, b = fit_dirichlet(val_logits, val_targets)
        p_dir = torch.sigmoid(A.unsqueeze(0) * test_logits + b.unsqueeze(0)).cpu().numpy()
        auc_d, _ = macro_auc(test_p["y_stmt"], p_dir)
        rows.append({"method": "hard_bce+dir", "seed": s,
                     "macro_auc": auc_d,
                     "macro_brier": macro_brier(test_p["y_stmt"], p_dir),
                     "macro_ece": macro_ece(test_p["y_stmt"], p_dir),
                     "hvr": hvr(test_p["p_super"], test_p["p_sub"], p_dir,
                                edges, train_ds.super2idx, train_ds.sub2idx, train_ds.stmt2idx)})
        all_probs[("hard_bce+dir", s)] = {**test_p, "p_stmt": p_dir}
        print(f"hard_bce+dir      seed{s}: AUC={auc_d:.4f}  # per-label affine logit (legacy 'Dirichlet' name)")

    # 3) CHMC-coh: top-down min-projection on hard_bce 4-head outputs
    for s in SEEDS:
        if ("hard_bce", s) not in all_probs:
            continue
        test_p = all_probs[("hard_bce", s)]
        p_super_coh, p_sub_coh, p_stmt_coh = chmc_project(
            test_p["p_super"], test_p["p_sub"], test_p["p_stmt"],
            train_ds.super_list, train_ds.sub_list, train_ds.all_stmts, scp_df)
        auc_c, _ = macro_auc(test_p["y_stmt"], p_stmt_coh)
        rows.append({"method": "chmc_coh", "seed": s,
                     "macro_auc": auc_c,
                     "macro_brier": macro_brier(test_p["y_stmt"], p_stmt_coh),
                     "macro_ece": macro_ece(test_p["y_stmt"], p_stmt_coh),
                     "hvr": hvr(p_super_coh, p_sub_coh, p_stmt_coh,
                                edges, train_ds.super2idx, train_ds.sub2idx, train_ds.stmt2idx)})
        all_probs[("chmc_coh", s)] = {"y_stmt": test_p["y_stmt"], "p_stmt": p_stmt_coh,
                                       "y_super": test_p["y_super"], "p_super": p_super_coh,
                                       "y_sub": test_p["y_sub"], "p_sub": p_sub_coh}
        print(f"chmc_coh          seed{s}: AUC={auc_c:.4f} HVR={rows[-1]['hvr']:.4f}")

    # 4) Save raw + aggregated tables
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/block_a_raw.csv", index=False)
    agg = df.groupby("method").agg(
        macro_auc=("macro_auc", "mean"), macro_auc_std=("macro_auc", "std"),
        macro_brier=("macro_brier", "mean"), macro_brier_std=("macro_brier", "std"),
        macro_ece=("macro_ece", "mean"), macro_ece_std=("macro_ece", "std"),
        hvr=("hvr", "mean"), hvr_std=("hvr", "std")).round(4)
    agg.to_csv(ROOT / "results/block_a_agg.csv")
    print("\n=== Block A Aggregate ===")
    print(agg)

    # 5) Save probability matrices (use seed=42 representative for downstream blocks)
    save_probs = {}
    for (m, s), p in all_probs.items():
        if s == 42:
            save_probs[f"{m}_y_stmt"] = p["y_stmt"]
            save_probs[f"{m}_p_stmt"] = p["p_stmt"]
            save_probs[f"{m}_p_super"] = p["p_super"]
            save_probs[f"{m}_p_sub"] = p["p_sub"]
    np.savez_compressed(ROOT / "results/block_a_probs.npz", **save_probs)
    print(f"Saved seed=42 probs for {len(save_probs)//4} methods")

    # 6) Bootstrap paired diff vs hard_bce baseline (using seed=42)
    if ("hard_bce", 42) in all_probs:
        ref = all_probs[("hard_bce", 42)]
        boot_rows = []
        for (m, s), p in all_probs.items():
            if m == "hard_bce" or s != 42:
                continue
            for metric_name, fn in [
                ("macro_auc", lambda y, p: macro_auc(y, p)),
                ("macro_brier", lambda y, p: macro_brier(y, p)),
                ("macro_ece", lambda y, p: macro_ece(y, p))]:
                if metric_name == "macro_auc":
                    bs = bootstrap_paired_diff(fn, ref["y_stmt"], ref["p_stmt"], p["p_stmt"],
                                                B=1000, alternative="greater")
                else:
                    # For Brier/ECE, lower is better → test p < ref
                    bs = bootstrap_paired_diff(fn, ref["y_stmt"], ref["p_stmt"], p["p_stmt"],
                                                B=1000, alternative="less")
                boot_rows.append({"method": m, "metric": metric_name, **bs})
        pd.DataFrame(boot_rows).to_csv(ROOT / "results/block_a_bootstrap.csv", index=False)
        print(f"\nSaved bootstrap results")

    print("\nBlock A evaluation COMPLETE.")


if __name__ == "__main__":
    main()
