"""Metrics: macro-AUC / macro-Brier / macro-ECE / HVR + bootstrap CIs."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def macro_auc(y_true, y_prob, min_pos=10):
    """Macro-averaged ROC-AUC, skipping classes with too few positives."""
    aucs = []
    for k in range(y_true.shape[1]):
        yt = y_true[:, k]
        yp = y_prob[:, k]
        if yt.sum() < min_pos or (1 - yt).sum() < min_pos:
            continue
        try:
            aucs.append(roc_auc_score(yt, yp))
        except Exception:
            pass
    if not aucs:
        return float("nan"), aucs
    return float(np.mean(aucs)), aucs


def macro_auprc(y_true, y_prob, min_pos=10):
    aps = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() < min_pos:
            continue
        aps.append(average_precision_score(y_true[:, k], y_prob[:, k]))
    return float(np.mean(aps)) if aps else float("nan")


def macro_brier(y_true, y_prob):
    return float(np.mean((y_prob - y_true) ** 2))


def adaptive_ece(y_true_1d, y_prob_1d, n_bins=10):
    """Adaptive-bin ECE for a single binary label."""
    sort_idx = np.argsort(y_prob_1d)
    y_true_sorted = y_true_1d[sort_idx]
    y_prob_sorted = y_prob_1d[sort_idx]
    N = len(y_prob_sorted)
    if N == 0:
        return 0.0
    bin_size = N // n_bins
    ece = 0.0
    for b in range(n_bins):
        lo = b * bin_size
        hi = (b + 1) * bin_size if b < n_bins - 1 else N
        if hi <= lo:
            continue
        bin_prob = y_prob_sorted[lo:hi]
        bin_true = y_true_sorted[lo:hi]
        gap = abs(bin_prob.mean() - bin_true.mean())
        ece += (hi - lo) / N * gap
    return ece


def macro_ece(y_true, y_prob, n_bins=10, min_pos=10):
    eces = []
    for k in range(y_true.shape[1]):
        if y_true[:, k].sum() < min_pos:
            continue
        eces.append(adaptive_ece(y_true[:, k], y_prob[:, k], n_bins))
    return float(np.mean(eces)) if eces else float("nan")


def hvr(p_super, p_sub, p_stmt, edges, super2idx, sub2idx, stmt2idx, eps=0.0):
    """Hierarchy violation rate: fraction of (sample, edge) pairs with p_c > p_p + eps."""
    violations = 0
    total = 0
    for e in edges:
        if e["level"] == "super_to_sub":
            p_p = p_super[:, super2idx[e["parent"]]]
            p_c = p_sub[:, sub2idx[e["child"]]]
        else:
            p_p = p_sub[:, sub2idx[e["parent"]]]
            p_c = p_stmt[:, stmt2idx[e["child"]]]
        violations += int((p_c > p_p + eps).sum())
        total += int(p_c.shape[0])
    return violations / total if total else 0.0


def bootstrap_paired_diff(metric_fn, y_true, p_a, p_b, B=1000, seed=0,
                          alpha=0.05, alternative="greater"):
    """Patient-level paired bootstrap on metric_fn(y, p_b) - metric_fn(y, p_a).

    Returns dict with diff_mean, ci_lo, ci_hi, p (one-sided "greater" by default).
    """
    rng = np.random.default_rng(seed)
    N = y_true.shape[0]
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        try:
            v_a = metric_fn(y_true[idx], p_a[idx])
            v_b = metric_fn(y_true[idx], p_b[idx])
            v_a = v_a[0] if isinstance(v_a, tuple) else v_a
            v_b = v_b[0] if isinstance(v_b, tuple) else v_b
            diffs.append(v_b - v_a)
        except Exception:
            continue
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return {"diff_mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    if alternative == "greater":
        p = (diffs <= 0).mean()
    elif alternative == "less":
        p = (diffs >= 0).mean()
    else:
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff_mean": float(diffs.mean()), "ci_lo": float(lo),
            "ci_hi": float(hi), "p": float(p), "n_bootstrap": int(len(diffs))}
