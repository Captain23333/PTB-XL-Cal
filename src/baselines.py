"""Baseline losses + post-hoc calibration + CHMC-coh projection."""
import torch
import torch.nn.functional as F
import numpy as np


def hard_bce(out, batch, epoch=0):
    """Standard binary cross-entropy on stmt head only."""
    return F.binary_cross_entropy_with_logits(out["stmt_logits"], batch["y_hard"])


class AsymmetricLoss:
    """Asymmetric Loss (Ridnik et al., ICCV 2021)."""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        self.g_neg = gamma_neg
        self.g_pos = gamma_pos
        self.clip = clip

    def __call__(self, out, batch, epoch=0):
        x = out["stmt_logits"]
        y = batch["y_hard"]
        x_sig = torch.sigmoid(x)
        if self.clip > 0:
            x_sig = (x_sig + self.clip).clamp(max=1)
        los_pos = y * torch.log(x_sig.clamp(min=1e-8)) * ((1 - x_sig) ** self.g_pos)
        los_neg = (1 - y) * torch.log((1 - x_sig).clamp(min=1e-8)) * (x_sig ** self.g_neg)
        return -(los_pos + los_neg).mean()


class FocalLoss:
    def __init__(self, gamma=2.0):
        self.gamma = gamma

    def __call__(self, out, batch, epoch=0):
        x = out["stmt_logits"]
        y = batch["y_hard"]
        bce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


def make_label_smooth(eps=0.1):
    def loss_fn(out, batch, epoch=0):
        y = batch["y_hard"]
        y_smooth = y * (1 - eps) + (1 - y) * eps
        return F.binary_cross_entropy_with_logits(out["stmt_logits"], y_smooth)
    return loss_fn


# ---------------- Post-hoc calibration ----------------
def fit_temperature(val_logits, val_targets, lr=0.01, n_iter=200):
    T = torch.nn.Parameter(torch.ones(1, device=val_logits.device))
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=n_iter)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(val_logits / T, val_targets)
        loss.backward()
        return loss
    opt.step(closure)
    return T.detach()


def fit_dirichlet(val_logits, val_targets, n_epochs=300, lr=1e-2):
    """DEPRECATED legacy name. This is per-label affine logit calibration
    (Platt-style per-label vector scaling), NOT Kull et al. 2019 Dirichlet
    calibration (which is single-label multi-class softmax).

    Kept for backward compat with experiments/02_block_a_eval.py.
    New code should use src.calibrators.fit_per_label_affine, which returns a
    callable wrapping this same fit but with the locked C3 API.
    """
    N, K = val_logits.shape
    A = torch.nn.Parameter(torch.ones(K, device=val_logits.device))
    b = torch.nn.Parameter(torch.zeros(K, device=val_logits.device))
    opt = torch.optim.Adam([A, b], lr=lr)
    for _ in range(n_epochs):
        opt.zero_grad()
        z = A.unsqueeze(0) * val_logits + b.unsqueeze(0)
        loss = F.binary_cross_entropy_with_logits(z, val_targets)
        loss.backward()
        opt.step()
    return A.detach(), b.detach()


# ---------------- CHMC-coh top-down min-projection ----------------
def chmc_project(p_super_raw, p_sub_raw, p_stmt_raw,
                 super_list, sub_list, stmt_list, scp_df):
    """Top-down min-projection: HVR=0 by construction at all evaluated edges.

    p_*_raw: numpy arrays [N, K_*]
    Returns coherent (p_super, p_sub, p_stmt) numpy arrays.
    """
    p_super_coh = p_super_raw.copy()
    p_sub_coh = np.zeros_like(p_sub_raw)
    p_stmt_coh = p_stmt_raw.copy()

    # sub -> super lookup (each sub belongs to one super)
    sub_to_super = {}
    for sub_c in sub_list:
        rows = scp_df[scp_df["diagnostic_subclass"] == sub_c]
        if len(rows) == 0:
            continue
        sub_to_super[sub_c] = rows["diagnostic_class"].iloc[0]

    # Project sub
    for j, sub_c in enumerate(sub_list):
        super_c = sub_to_super.get(sub_c)
        if super_c is None or super_c not in super_list:
            p_sub_coh[:, j] = p_sub_raw[:, j]
            continue
        i = super_list.index(super_c)
        p_sub_coh[:, j] = np.minimum(p_sub_raw[:, j], p_super_coh[:, i])

    # Project stmt (only diagnostic stmts; non-diag stay raw)
    for k, stmt in enumerate(stmt_list):
        if stmt not in scp_df.index:
            continue
        if scp_df.loc[stmt, "diagnostic"] != 1.0:
            continue
        sub_c = scp_df.loc[stmt, "diagnostic_subclass"]
        if not isinstance(sub_c, str) or sub_c not in sub_list:
            continue
        j = sub_list.index(sub_c)
        p_stmt_coh[:, k] = np.minimum(p_stmt_raw[:, k], p_sub_coh[:, j])

    return p_super_coh, p_sub_coh, p_stmt_coh
