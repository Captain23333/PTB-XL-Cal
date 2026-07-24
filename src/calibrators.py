"""PTB-XL-Cal calibrator suite (C0..C5).

All `fit_*` functions return a CALLABLE with signature:
    calibrator(test_logits_or_probs) -> test_probs_in_[0,1]   shape [N_test, K]

Inputs to fit_*:
    val_logits  : torch.Tensor or np.ndarray [N_val, K]   raw pre-sigmoid logits
    val_targets : torch.Tensor or np.ndarray [N_val, K]   in {0, 1}

C5 (isotonic) takes val_probs (post-sigmoid) instead of val_logits.

Calibrators reference (locked in `eval_calibration.select_best_calibrator`):
    C0 raw                          0    params
    C1 single-scalar temperature    1
    C2 per-label temperature       71
    C3 per-label affine logit      142
    C4 family-pooled affine logit   10
    C5 per-label isotonic          ~per-label nonparametric (complexity ~200 knots)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---- helpers ---------------------------------------------------------------

def _to_t(x, device=None, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(dtype)
    if device is not None:
        x = x.to(device)
    return x


def _device(x):
    if isinstance(x, torch.Tensor):
        return x.device
    return torch.device("cpu")


# ---- C0 raw ---------------------------------------------------------------

def fit_raw(*_args, **_kwargs):
    """C0 -- identity calibrator (no fit). Returns sigmoid(z)."""
    def _apply(test_logits):
        z = _to_t(test_logits, device=torch.device("cpu"))
        return torch.sigmoid(z).numpy().astype(np.float32)
    _apply.calibrator_name = "raw"
    _apply.n_params = 0
    return _apply


# ---- C1 single scalar temperature ----------------------------------------

def fit_temperature_scalar(val_logits, val_targets, lr=0.01, n_iter=200):
    """C1 -- single scalar T over all (sample, statement). LBFGS BCE-with-logits."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = _to_t(val_logits, device=dev)
    y = _to_t(val_targets, device=dev)
    logT = torch.nn.Parameter(torch.zeros(1, device=dev))
    opt = torch.optim.LBFGS([logT], lr=lr, max_iter=n_iter)

    def closure():
        opt.zero_grad()
        T = torch.exp(logT)
        loss = F.binary_cross_entropy_with_logits(z / T, y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        T_final = float(torch.exp(logT).detach().cpu().item())
    except Exception:
        T_final = 1.0
    if not np.isfinite(T_final) or T_final <= 0:
        T_final = 1.0

    def _apply(test_logits):
        zt = _to_t(test_logits, device=torch.device("cpu"))
        return torch.sigmoid(zt / T_final).numpy().astype(np.float32)
    _apply.calibrator_name = "temp_scalar"
    _apply.n_params = 1
    _apply.T = T_final
    return _apply


# ---- C2 per-label temperature --------------------------------------------

def fit_temperature_per_label(val_logits, val_targets, lr=0.05, n_iter=300):
    """C2 -- K per-label temperatures T_k. Adam minimization on BCE-with-logits."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = _to_t(val_logits, device=dev)
    y = _to_t(val_targets, device=dev)
    K = z.shape[1]
    logT = torch.nn.Parameter(torch.zeros(K, device=dev))
    opt = torch.optim.Adam([logT], lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        T = torch.exp(logT).unsqueeze(0)  # [1,K]
        loss = F.binary_cross_entropy_with_logits(z / T, y)
        loss.backward()
        opt.step()
    T_final = torch.exp(logT).detach().cpu().numpy().astype(np.float32)
    T_final = np.where(np.isfinite(T_final) & (T_final > 0), T_final, 1.0)

    def _apply(test_logits):
        from scipy.special import expit
        zt = _to_t(test_logits, device=torch.device("cpu")).numpy()
        return expit(zt / T_final[None, :]).astype(np.float32)
    _apply.calibrator_name = "temp_per_label"
    _apply.n_params = int(K)
    _apply.T_k = T_final
    return _apply


# ---- C3 per-label affine logit -------------------------------------------

def fit_per_label_affine(val_logits, val_targets, n_epochs=300, lr=1e-2):
    """C3 -- per-label affine logit: A_k * z_k + b_k.

    Per-label affine logit calibration (Platt-style per-label vector scaling).
    NOT Kull et al. 2019 Dirichlet calibration (which is single-label multi-class).
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = _to_t(val_logits, device=dev)
    y = _to_t(val_targets, device=dev)
    K = z.shape[1]
    A = torch.nn.Parameter(torch.ones(K, device=dev))
    b = torch.nn.Parameter(torch.zeros(K, device=dev))
    opt = torch.optim.Adam([A, b], lr=lr)
    for _ in range(n_epochs):
        opt.zero_grad()
        logit = A.unsqueeze(0) * z + b.unsqueeze(0)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        loss.backward()
        opt.step()
    A_np = A.detach().cpu().numpy().astype(np.float32)
    b_np = b.detach().cpu().numpy().astype(np.float32)

    def _apply(test_logits):
        from scipy.special import expit
        zt = _to_t(test_logits, device=torch.device("cpu")).numpy()
        logit = A_np[None, :] * zt + b_np[None, :]
        return expit(logit).astype(np.float32)
    _apply.calibrator_name = "per_label_affine"
    _apply.n_params = 2 * int(K)
    _apply.A = A_np
    _apply.b = b_np
    return _apply


# ---- C4 family-pooled affine ---------------------------------------------

def fit_family_affine(val_logits, val_targets, family_index_per_stmt,
                      n_epochs=300, lr=1e-2):
    """C4 -- per-family affine logit: A_f * z_k + b_f for k in family f.

    family_index_per_stmt : np.array[K] of int family ids in {0..F-1}.
    Trains F (A, b) pairs by Adam on BCE-with-logits.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = _to_t(val_logits, device=dev)
    y = _to_t(val_targets, device=dev)
    fam = np.asarray(family_index_per_stmt, dtype=np.int64)
    K = z.shape[1]
    F_count = int(fam.max()) + 1
    A = torch.nn.Parameter(torch.ones(F_count, device=dev))
    b = torch.nn.Parameter(torch.zeros(F_count, device=dev))
    fam_t = torch.from_numpy(fam).to(dev).long()
    opt = torch.optim.Adam([A, b], lr=lr)
    for _ in range(n_epochs):
        opt.zero_grad()
        A_per_k = A[fam_t]  # [K]
        b_per_k = b[fam_t]
        logit = A_per_k.unsqueeze(0) * z + b_per_k.unsqueeze(0)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        loss.backward()
        opt.step()
    A_np = A.detach().cpu().numpy().astype(np.float32)
    b_np = b.detach().cpu().numpy().astype(np.float32)
    A_per_k_np = A_np[fam]
    b_per_k_np = b_np[fam]

    def _apply(test_logits):
        from scipy.special import expit
        zt = _to_t(test_logits, device=torch.device("cpu")).numpy()
        logit = A_per_k_np[None, :] * zt + b_per_k_np[None, :]
        return expit(logit).astype(np.float32)
    _apply.calibrator_name = "family_affine"
    _apply.n_params = 2 * F_count
    _apply.A_per_family = A_np
    _apply.b_per_family = b_np
    return _apply


# ---- C5 per-label isotonic regression ------------------------------------

def fit_isotonic_per_label(val_probs_sigmoid, val_targets):
    """C5 -- per-label isotonic regression. Operates on probs in [0,1] (NOT logits).

    Returns callable that ALSO expects test PROBS (not logits).
    """
    from sklearn.isotonic import IsotonicRegression
    p = np.asarray(val_probs_sigmoid, dtype=np.float64)
    y = np.asarray(val_targets, dtype=np.float64)
    K = p.shape[1]
    irs = []
    for k in range(K):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        # If the column has zero positives or zero negatives, fall back to identity
        if y[:, k].sum() < 1 or (1 - y[:, k]).sum() < 1:
            irs.append(None)
            continue
        try:
            ir.fit(p[:, k], y[:, k])
            irs.append(ir)
        except Exception:
            irs.append(None)

    def _apply(test_probs_sigmoid):
        pt = np.asarray(test_probs_sigmoid, dtype=np.float64)
        out = np.empty_like(pt, dtype=np.float32)
        for k in range(K):
            if irs[k] is None:
                out[:, k] = pt[:, k]
            else:
                out[:, k] = irs[k].predict(pt[:, k])
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    _apply.calibrator_name = "isotonic_per_label"
    _apply.n_params = int(K * 50)  # rough complexity estimate; used for tie-break
    return _apply
