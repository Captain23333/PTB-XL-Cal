"""Generic trainer for any baseline / SCP-Soft variant."""
import json
import torch
import torch.nn as nn
from pathlib import Path

from src.data import get_loaders
from src.models import build_model
from src.losses import SCPSoftLoss
from src.baselines import (hard_bce, AsymmetricLoss, FocalLoss, make_label_smooth)
from src.evaluate import macro_auc, macro_brier, macro_ece, hvr

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def get_loss_fn(method, train_ds, edges):
    s2i = train_ds.super2idx
    b2i = train_ds.sub2idx
    t2i = train_ds.stmt2idx
    if method == "hard_bce":
        return hard_bce
    if method == "asl":
        return AsymmetricLoss()
    if method == "focal":
        return FocalLoss()
    if method == "label_smooth":
        return make_label_smooth(0.1)
    if method == "soft":
        return SCPSoftLoss(edges, s2i, b2i, t2i,
                           use_parent=False, use_ord=False, use_hier=False)
    if method == "soft_parent":
        return SCPSoftLoss(edges, s2i, b2i, t2i,
                           use_ord=False, use_hier=False)
    if method == "soft_parent_ord":
        return SCPSoftLoss(edges, s2i, b2i, t2i, use_hier=False)
    if method == "scp_soft_full":
        return SCPSoftLoss(edges, s2i, b2i, t2i)
    raise ValueError(f"Unknown method: {method}")


def evaluate_loader(model, loader, edges, super2idx, sub2idx, stmt2idx, return_probs=False):
    model.eval()
    all_y_stmt, all_p_stmt = [], []
    all_y_super, all_p_super = [], []
    all_y_sub, all_p_sub = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(batch["x"])
            all_p_stmt.append(torch.sigmoid(out["stmt_logits"]).cpu().float())
            all_p_super.append(torch.sigmoid(out["super_logits"]).cpu().float())
            all_p_sub.append(torch.sigmoid(out["sub_logits"]).cpu().float())
            all_y_stmt.append(batch["y_hard"].cpu())
            all_y_super.append((batch["y_super"] > 0).float().cpu())
            all_y_sub.append((batch["y_sub"] > 0).float().cpu())
    y_stmt = torch.cat(all_y_stmt).numpy()
    p_stmt = torch.cat(all_p_stmt).numpy()
    y_super = torch.cat(all_y_super).numpy()
    p_super = torch.cat(all_p_super).numpy()
    y_sub = torch.cat(all_y_sub).numpy()
    p_sub = torch.cat(all_p_sub).numpy()
    auc, _ = macro_auc(y_stmt, p_stmt)
    metrics = {
        "macro_auc": auc,
        "macro_brier": macro_brier(y_stmt, p_stmt),
        "macro_ece": macro_ece(y_stmt, p_stmt),
        "hvr": hvr(p_super, p_sub, p_stmt, edges, super2idx, sub2idx, stmt2idx),
    }
    if return_probs:
        return metrics, {
            "y_stmt": y_stmt, "p_stmt": p_stmt,
            "y_super": y_super, "p_super": p_super,
            "y_sub": y_sub, "p_sub": p_sub,
        }
    return metrics


def train_one(method, seed, epochs=50, lr=1e-3, batch_size=128,
              ckpt_dir=CKPT_DIR, ckpt_path=None, backbone_name="simplresnet1d"):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_loader, val_loader, test_loader, train_ds = get_loaders(batch_size=batch_size)
    hier_data = json.load(open(ROOT / "results/hierarchy_edges.json"))
    edges = hier_data["edges"]

    model = build_model(backbone_name=backbone_name).cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    loss_fn = get_loss_fn(method, train_ds, edges)

    best_val_auc = -1.0
    patience_left = 5
    if ckpt_path is None:
        suffix = "" if backbone_name == "simplresnet1d" else f"_{backbone_name}"
        ckpt_path = Path(ckpt_dir) / f"{method}{suffix}_seed{seed}.pt"

    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            opt.zero_grad()
            out = model(batch["x"])
            try:
                loss = loss_fn(out, batch, epoch=ep)
            except TypeError:
                loss = loss_fn(out, batch)
            loss.backward()
            opt.step()
            sch.step()

        val_metrics = evaluate_loader(model, val_loader, edges,
                                      train_ds.super2idx, train_ds.sub2idx, train_ds.stmt2idx)
        val_auc = val_metrics["macro_auc"]
        print(f"[{method} seed{seed}] ep{ep:02d} val_auc={val_auc:.4f} val_ece={val_metrics['macro_ece']:.4f} val_hvr={val_metrics['hvr']:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_left = 5
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": ep,
                "val_metrics": val_metrics,
                "method": method,
                "seed": seed,
            }, ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{method} seed{seed}] Early stop at ep{ep} (best val_auc={best_val_auc:.4f})")
                break

    return ckpt_path
