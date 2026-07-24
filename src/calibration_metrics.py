"""PTB-XL-Cal locked metric suite.

All metrics are LOCKED per docs/refine/FINAL_PROPOSAL.md and must NOT be silently
changed (bin counts, exclusion thresholds).
"""
from __future__ import annotations

import warnings
import numpy as np


# -------------------------------------------------------------------------
# labelwise ECE: 15 adaptive equal-frequency bins per statement, macro-averaged.
# Statements with N_pos < 5 in `targets` excluded.
# -------------------------------------------------------------------------

def _adaptive_ece_single(prob_1d, targ_1d, n_bins=15):
    """Equal-frequency adaptive ECE for one statement."""
    sort_idx = np.argsort(prob_1d)
    p = prob_1d[sort_idx]
    y = targ_1d[sort_idx]
    N = len(p)
    if N == 0:
        return 0.0
    edges = np.linspace(0, N, n_bins + 1, dtype=int)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        gap = abs(p[lo:hi].mean() - y[lo:hi].mean())
        ece += (hi - lo) / N * gap
    return float(ece)


def labelwise_ece(probs, targets, n_bins=15, exclude_n_pos_lt=5):
    """Per-statement adaptive ECE (15 equal-freq bins), macro-averaged.

    Returns
    -------
    macro_ece : float
    per_stmt_ece : np.ndarray[K]   NaN where excluded
    excluded_idx : list[int]
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    K = probs.shape[1]
    per = np.full(K, np.nan, dtype=np.float64)
    excluded = []
    for k in range(K):
        if targets[:, k].sum() < exclude_n_pos_lt:
            excluded.append(k)
            continue
        per[k] = _adaptive_ece_single(probs[:, k], targets[:, k], n_bins=n_bins)
    valid = ~np.isnan(per)
    macro = float(np.mean(per[valid])) if valid.any() else float("nan")
    return macro, per, excluded


# -------------------------------------------------------------------------
# macro Brier
# -------------------------------------------------------------------------

def macro_brier(probs, targets):
    """mean((probs - targets)**2) across all (sample, statement)."""
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.mean((probs - targets) ** 2))


# -------------------------------------------------------------------------
# Calibration slope + intercept (per-statement logistic GLM on adaptive bins).
# -------------------------------------------------------------------------

def calibration_slope_intercept(probs, targets, n_bins=15, max_iter=200,
                                 tol=1e-4, exclude_n_pos_lt=5):
    """Per-statement adaptive-bin logistic fit of empirical positive rate on
    predicted probability mean.

    Uses statsmodels GLM(family=Binomial) via logit link. Statement excluded if
    N_pos<exclude_n_pos_lt, fewer than 3 non-degenerate bins, or fit fails to
    converge.

    Returns
    -------
    slopes : np.ndarray[K]   NaN where excluded
    intercepts : np.ndarray[K]   NaN where excluded
    included_idx : list[int]
    """
    import statsmodels.api as sm

    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    K = probs.shape[1]
    slopes = np.full(K, np.nan)
    intercepts = np.full(K, np.nan)
    included = []
    for k in range(K):
        if targets[:, k].sum() < exclude_n_pos_lt:
            continue
        sort_idx = np.argsort(probs[:, k])
        p = probs[sort_idx, k]
        y = targets[sort_idx, k]
        N = len(p)
        edges = np.linspace(0, N, n_bins + 1, dtype=int)
        bin_p, bin_y, bin_n = [], [], []
        for b in range(n_bins):
            lo, hi = edges[b], edges[b + 1]
            if hi <= lo:
                continue
            bin_p.append(p[lo:hi].mean())
            bin_y.append(y[lo:hi].mean())
            bin_n.append(hi - lo)
        if len(bin_p) < 3:
            continue
        bin_p = np.asarray(bin_p)
        bin_y = np.asarray(bin_y)
        bin_n = np.asarray(bin_n, dtype=float)
        # Map empirical_rate to logit-friendly numbers; skip if all 0 or all 1.
        if np.allclose(bin_y, bin_y[0]):
            continue
        # Use logit of predicted prob as covariate (slope=1, intercept=0 means
        # perfectly calibrated).
        x_logit = np.log(np.clip(bin_p, 1e-6, 1 - 1e-6) /
                         (1 - np.clip(bin_p, 1e-6, 1 - 1e-6)))
        # Frequency-weighted GLM with binary expanded successes/failures
        successes = (bin_y * bin_n).round().astype(int)
        failures = (bin_n - successes).astype(int)
        endog = np.column_stack([successes, failures])
        exog = sm.add_constant(x_logit)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.GLM(endog, exog, family=sm.families.Binomial()).fit(
                    maxiter=max_iter, tol=tol, disp=False)
            if not res.converged:
                continue
            intercepts[k] = float(res.params[0])
            slopes[k] = float(res.params[1])
            included.append(k)
        except Exception:
            continue
    return slopes, intercepts, included


# -------------------------------------------------------------------------
# Threshold-transfer gap (locked sens=0.9 anchor; supplementary 0.8 / 0.95).
# -------------------------------------------------------------------------

def _sens_spec_at(probs, targets, tau):
    """Return (sens, spec) at threshold tau on a 1-d binary stmt."""
    pred = probs >= tau
    pos = targets > 0.5
    neg = ~pos
    tp = (pred & pos).sum()
    fn = ((~pred) & pos).sum()
    tn = ((~pred) & neg).sum()
    fp = (pred & neg).sum()
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    return float(sens), float(spec)


def _find_tau_for_target_sens(probs, targets, target_sens=0.9, sens_tol=0.005):
    """Find largest threshold tau s.t. sens(tau) in [target-tol, target+tol].

    Iterate over candidate thresholds = unique probs (descending). The largest
    tau that still meets the sens band is preferred (highest specificity given
    the sens constraint).
    """
    pos = targets > 0.5
    n_pos = int(pos.sum())
    if n_pos < 5:
        return None, None
    # Candidate taus: positive-class probs sorted descending give increasing-sens
    # as we lower tau. Use unique probs to avoid redundant evaluation.
    cand = np.unique(probs)[::-1]  # descending
    # Track best tau within band
    best_tau = None
    best_spec = -1.0
    lo = target_sens - sens_tol
    hi = target_sens + sens_tol
    for tau in cand:
        s, sp = _sens_spec_at(probs, targets, tau)
        if lo <= s <= hi:
            if sp > best_spec:
                best_spec = sp
                best_tau = float(tau)
        elif s > hi:
            # We're past the band as sens overshoots; stop.
            break
    if best_tau is None:
        return None, None
    return best_tau, best_spec


def threshold_transfer_gap(val_probs, val_targets, test_probs, test_targets,
                            target_sens=0.9, sens_tol=0.005, exclude_n_pos_lt=5):
    """Per-statement: find tau on val with sens_val(tau) in [t-tol, t+tol];
    compute |sens_test - target| and |spec_test - spec_val|.

    Returns
    -------
    macro_sens_gap : float
    macro_spec_gap : float
    per_stmt : dict { 'sens_gap': np.array[K], 'spec_gap': np.array[K], 'tau': np.array[K] }
    excluded_idx : list[int]
    """
    val_probs = np.asarray(val_probs, dtype=np.float64)
    val_targets = np.asarray(val_targets, dtype=np.float64)
    test_probs = np.asarray(test_probs, dtype=np.float64)
    test_targets = np.asarray(test_targets, dtype=np.float64)
    K = val_probs.shape[1]
    sens_gap = np.full(K, np.nan)
    spec_gap = np.full(K, np.nan)
    taus = np.full(K, np.nan)
    excluded = []
    for k in range(K):
        if val_targets[:, k].sum() < exclude_n_pos_lt or \
           test_targets[:, k].sum() < exclude_n_pos_lt:
            excluded.append(k)
            continue
        tau, spec_val = _find_tau_for_target_sens(
            val_probs[:, k], val_targets[:, k],
            target_sens=target_sens, sens_tol=sens_tol)
        if tau is None:
            excluded.append(k)
            continue
        s_test, sp_test = _sens_spec_at(test_probs[:, k], test_targets[:, k], tau)
        sens_gap[k] = abs(s_test - target_sens)
        spec_gap[k] = abs(sp_test - spec_val)
        taus[k] = tau
    valid = ~np.isnan(sens_gap)
    macro_s = float(np.mean(sens_gap[valid])) if valid.any() else float("nan")
    macro_sp = float(np.mean(spec_gap[valid])) if valid.any() else float("nan")
    return macro_s, macro_sp, {"sens_gap": sens_gap, "spec_gap": spec_gap, "tau": taus}, excluded


# -------------------------------------------------------------------------
# Patient-paired bootstrap CI (resamples PATIENTS, not records).
# -------------------------------------------------------------------------

def patient_paired_bootstrap_ci(metric_fn, probs_a, probs_b, targets, patient_ids,
                                 n_boot=1000, seed=42, alpha=0.05):
    """Patient-level paired bootstrap of metric_fn(targets, probs_b) - metric_fn(targets, probs_a).

    Parameters
    ----------
    metric_fn : callable(targets, probs) -> float (lower-is-better metrics ok; sign handled by caller)
    probs_a, probs_b : np.ndarray [N, K]
    targets : np.ndarray [N, K]
    patient_ids : np.ndarray [N] of int patient identifiers (resample at this level)

    Returns
    -------
    diff_mean, ci_lo, ci_hi, p_two_sided : floats
    """
    rng = np.random.default_rng(seed)
    pid = np.asarray(patient_ids)
    unique_pids, inv = np.unique(pid, return_inverse=True)
    # Build per-patient row index lists once
    by_p = [np.where(inv == i)[0] for i in range(len(unique_pids))]
    diffs = []
    Np = len(unique_pids)
    for _ in range(n_boot):
        chosen = rng.integers(0, Np, size=Np)
        rows = np.concatenate([by_p[i] for i in chosen])
        try:
            va = metric_fn(targets[rows], probs_a[rows])
            vb = metric_fn(targets[rows], probs_b[rows])
            va = va[0] if isinstance(va, tuple) else va
            vb = vb[0] if isinstance(vb, tuple) else vb
            diffs.append(vb - va)
        except Exception:
            continue
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diffs.mean()), lo, hi, float(p_two)
