"""Recompute val + test probs (logits) for all 24 PTB-XL ckpts.

For each (method in {hard_bce, asl, focal, label_smooth, scp_soft_full,
                     soft, soft_parent, soft_parent_ord}, seed in {42,123,2024}):
    forward val_loader and test_loader
    save results/cal_probs/probs_<method>_seed<seed>.npz with:
        val_logits      [N_val, 71]
        val_targets     [N_val, 71]   (y_hard)
        test_logits     [N_test, 71]
        test_targets    [N_test, 71]
        val_patient_ids [N_val]
        test_patient_ids[N_test]
        super_logits_val/test, sub_logits_val/test  (for chmc projection)

Also produces probs_chmc_coh_seed<s>.npz from probs_hard_bce_seed<s>.npz by
applying chmc_project on (p_super, p_sub, p_stmt). For chmc we save the
PROJECTED stmt probs as test_probs (and we lose access to logits since
projection is in prob-space; we mark calibrators C1..C5 as N/A for chmc by
storing test_probs and val_probs in *_probs_pre_calib slots; downstream
scripts treat chmc_coh as "calibrated probs already" and only run C0).

Memory-light: one ckpt at a time.
"""
from __future__ import annotations
import sys, json, time
import os
sys.path.insert(0, os.environ.get("PTBXL_CAL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from src.data import get_loaders
from src.models import build_model
from src.baselines import chmc_project

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "results/cal_probs"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["hard_bce", "asl", "focal", "label_smooth",
           "scp_soft_full", "soft", "soft_parent", "soft_parent_ord"]
SEEDS = [42, 123, 2024]


def _strip_module(sd):
    out = {}
    for k, v in sd.items():
        out[k[len("module."):] if k.startswith("module.") else k] = v
    return out


def load_model(method, seed):
    p = ROOT / f"checkpoints/{method}_seed{seed}.pt"
    if not p.exists():
        return None
    ckpt = torch.load(p, map_location="cuda")
    model = build_model().cuda()
    sd = _strip_module(ckpt["state_dict"])
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def forward_loader(model, loader):
    """Returns dict of numpy arrays for stmt/super/sub LOGITS + y_hard + patient_id."""
    stmt_logits, super_logits, sub_logits, y_hard, pids = [], [], [], [], []
    for batch in loader:
        x = batch["x"].cuda(non_blocking=True)
        out = model(x)
        stmt_logits.append(out["stmt_logits"].detach().cpu().float().numpy())
        super_logits.append(out["super_logits"].detach().cpu().float().numpy())
        sub_logits.append(out["sub_logits"].detach().cpu().float().numpy())
        y_hard.append(batch["y_hard"].numpy())
        pids.append(batch["patient_id"].numpy())
    return dict(
        stmt_logits=np.concatenate(stmt_logits, 0),
        super_logits=np.concatenate(super_logits, 0),
        sub_logits=np.concatenate(sub_logits, 0),
        y_hard=np.concatenate(y_hard, 0),
        patient_ids=np.concatenate(pids, 0),
    )


def main():
    print(f"[{time.strftime('%H:%M:%S')}] === calib_recompute_probs starting ===")
    _, val_loader, test_loader, train_ds = get_loaders(batch_size=128, num_workers=4)
    scp_df = pd.read_csv(ROOT / "data/scp_statements.csv", index_col=0)

    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"val={n_val}  test={n_test}  K={71}")

    for m in METHODS:
        for s in SEEDS:
            out_path = OUT / f"probs_{m}_seed{s}.npz"
            if out_path.exists():
                print(f"[skip exist] {out_path.name}")
                continue
            t0 = time.time()
            model = load_model(m, s)
            if model is None:
                print(f"[MISSING ckpt] {m} seed{s}")
                continue
            try:
                v = forward_loader(model, val_loader)
                t = forward_loader(model, test_loader)
            finally:
                del model
                torch.cuda.empty_cache()
            np.savez_compressed(
                out_path,
                val_logits=v["stmt_logits"].astype(np.float32),
                val_targets=v["y_hard"].astype(np.float32),
                test_logits=t["stmt_logits"].astype(np.float32),
                test_targets=t["y_hard"].astype(np.float32),
                val_patient_ids=v["patient_ids"].astype(np.int64),
                test_patient_ids=t["patient_ids"].astype(np.int64),
                super_logits_val=v["super_logits"].astype(np.float32),
                sub_logits_val=v["sub_logits"].astype(np.float32),
                super_logits_test=t["super_logits"].astype(np.float32),
                sub_logits_test=t["sub_logits"].astype(np.float32),
            )
            print(f"[{time.strftime('%H:%M:%S')}] saved {out_path.name}  "
                  f"val={v['stmt_logits'].shape} test={t['stmt_logits'].shape} "
                  f"in {time.time()-t0:.1f}s")

    # ---- B4 chmc_coh: hard_bce ckpt + projection on probs (val + test) ----
    for s in SEEDS:
        src_path = OUT / f"probs_hard_bce_seed{s}.npz"
        out_path = OUT / f"probs_chmc_coh_seed{s}.npz"
        if not src_path.exists():
            print(f"[chmc skip; need hard_bce seed{s}]")
            continue
        if out_path.exists():
            print(f"[chmc skip exist] {out_path.name}")
            continue
        d = np.load(src_path)
        # Convert val+test to probs, project, save in same format with the
        # PROJECTED stmt probs in *_logits slot transformed back via inverse
        # sigmoid (so downstream calibrators still receive a logit input). For
        # post-hoc cals C1..C5 we use the projected probs as if they were the
        # "raw" output -- this is acceptable per spec (chmc treats projection
        # as the inference-time op).
        eps = 1e-7
        def to_probs(z):
            return 1.0 / (1.0 + np.exp(-z))
        p_super_v = to_probs(d["super_logits_val"])
        p_sub_v = to_probs(d["sub_logits_val"])
        p_stmt_v = to_probs(d["val_logits"])
        p_super_t = to_probs(d["super_logits_test"])
        p_sub_t = to_probs(d["sub_logits_test"])
        p_stmt_t = to_probs(d["test_logits"])

        ps_v_c, pb_v_c, pst_v_c = chmc_project(
            p_super_v, p_sub_v, p_stmt_v,
            train_ds.super_list, train_ds.sub_list, train_ds.all_stmts, scp_df)
        ps_t_c, pb_t_c, pst_t_c = chmc_project(
            p_super_t, p_sub_t, p_stmt_t,
            train_ds.super_list, train_ds.sub_list, train_ds.all_stmts, scp_df)

        # Convert projected stmt probs back to logits for downstream calibrator API
        def to_logits(p):
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p)).astype(np.float32)

        np.savez_compressed(
            out_path,
            val_logits=to_logits(pst_v_c),
            val_targets=d["val_targets"],
            test_logits=to_logits(pst_t_c),
            test_targets=d["test_targets"],
            val_patient_ids=d["val_patient_ids"],
            test_patient_ids=d["test_patient_ids"],
            super_logits_val=to_logits(ps_v_c),
            sub_logits_val=to_logits(pb_v_c),
            super_logits_test=to_logits(ps_t_c),
            sub_logits_test=to_logits(pb_t_c),
        )
        print(f"[chmc_coh seed{s}] saved {out_path.name}")

    # Save splits CSV for downstream eval_calibration.py scaffold
    splits_dir = ROOT / "splits"
    splits_dir.mkdir(exist_ok=True)
    val_ids = val_loader.dataset.db["ecg_id"].astype(int).tolist()
    test_ids = test_loader.dataset.db["ecg_id"].astype(int).tolist()
    pd.DataFrame({"ecg_id": val_ids}).to_csv(splits_dir / "val_fold9.csv", index=False)
    pd.DataFrame({"ecg_id": test_ids}).to_csv(splits_dir / "test_fold10.csv", index=False)
    print(f"saved splits/val_fold9.csv ({len(val_ids)}) splits/test_fold10.csv ({len(test_ids)})")

    print("=== recompute DONE ===")


if __name__ == "__main__":
    main()
