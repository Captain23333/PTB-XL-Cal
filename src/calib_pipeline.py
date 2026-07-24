"""Shared helpers used by the calib_block*.py + eval_calibration.py scripts.

Single source of truth for:
  - the locked calibrator suite (C0..C5) factory
  - calibrator application (logits-vs-probs distinction for C5 isotonic)
  - lexicographic per-method best-calibrator selector
  - family index map
  - prob loading
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import os
ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
PROBS_DIR = ROOT / "results/cal_probs"
FAMILIES_PATH = ROOT / "results/clinical_families.json"

CALIBRATORS = ["C0_raw", "C1_temp_scalar", "C2_temp_per_label",
               "C3_per_label_affine", "C4_family_affine", "C5_isotonic"]
CAL_PARAM_COUNT = {
    "C0_raw": 0,
    "C1_temp_scalar": 1,
    "C2_temp_per_label": 71,
    "C3_per_label_affine": 142,
    "C4_family_affine": 10,
    "C5_isotonic": 200,
}

BASE_METHODS = ["hard_bce", "asl", "focal", "label_smooth",
                "chmc_coh", "scp_soft_full", "soft", "soft_parent_ord"]
SEEDS = [42, 123, 2024]
ECGFM_SEED = 42


# ---- statement order (canonical PTB-XL 71) ---------------------------------

def get_stmt_order():
    """Return the canonical 71-statement order without loading raw PTB-XL.

    The release includes this exact order in ``hierarchy_edges.json`` so that
    calibration evaluation and the CI fixture do not depend on ``wfdb`` or a
    local dataset checkout.
    """
    hierarchy_path = ROOT / "results/hierarchy_edges.json"
    with open(hierarchy_path) as f:
        hierarchy = json.load(f)
    stmts = hierarchy.get("all_stmts", [])
    if len(stmts) != 71:
        raise ValueError(
            f"Expected 71 canonical PTB-XL statements in {hierarchy_path}, "
            f"found {len(stmts)}"
        )
    return stmts


def family_index_per_stmt():
    """Return np.array[K=71] of family indices in {0..4} order [NORM,MI,STTC,CD,HYP],
    plus the family name list."""
    fams_json = json.load(open(FAMILIES_PATH))
    fam_order = ["NORM", "MI", "STTC", "CD", "HYP"]
    stmt2fam = {}
    for f, members in fams_json.items():
        for m in members:
            stmt2fam[m] = f
    stmts = get_stmt_order()
    K = len(stmts)
    fam_idx = np.full(K, -1, dtype=np.int64)
    for i, s in enumerate(stmts):
        if s in stmt2fam:
            fam_idx[i] = fam_order.index(stmt2fam[s])
    # Statements without a family (non-diagnostic) -- assign to NORM bucket as a
    # "default" so C4 has full coverage; mark mask separately.
    fam_idx[fam_idx < 0] = 0
    return fam_idx, fam_order


def family_membership_mask(fam_order):
    """Return dict {fam_name: bool mask[K]} of statements actually in each family
    (NOT the default-bucket assignment)."""
    fams_json = json.load(open(FAMILIES_PATH))
    stmts = get_stmt_order()
    K = len(stmts)
    out = {}
    for f in fam_order:
        mask = np.zeros(K, dtype=bool)
        for s in fams_json.get(f, []):
            if s in stmts:
                mask[stmts.index(s)] = True
        out[f] = mask
    return out


# ---- calibrator factory ----------------------------------------------------

def fit_all_calibrators(val_logits, val_targets, fam_idx):
    """Fit C0..C5; return dict cal_name -> callable that takes (test_logits, test_probs)
    and returns probs."""
    import torch
    from src.calibrators import (fit_raw, fit_temperature_scalar,
        fit_temperature_per_label, fit_per_label_affine, fit_family_affine,
        fit_isotonic_per_label)
    from scipy.special import expit
    val_logits_t = torch.as_tensor(val_logits, dtype=torch.float32)
    val_targets_t = torch.as_tensor(val_targets, dtype=torch.float32)
    val_probs = expit(val_logits).astype(np.float32)
    cals = {}
    cals["C0_raw"] = (fit_raw(), "logits")
    cals["C1_temp_scalar"] = (fit_temperature_scalar(val_logits_t, val_targets_t), "logits")
    cals["C2_temp_per_label"] = (fit_temperature_per_label(val_logits_t, val_targets_t), "logits")
    cals["C3_per_label_affine"] = (fit_per_label_affine(val_logits_t, val_targets_t), "logits")
    cals["C4_family_affine"] = (fit_family_affine(val_logits_t, val_targets_t, fam_idx), "logits")
    cals["C5_isotonic"] = (fit_isotonic_per_label(val_probs, val_targets), "probs")
    return cals


def apply_calibrator(cal_callable, mode, logits):
    """Apply a fit calibrator. mode in {logits, probs}."""
    if mode == "logits":
        return cal_callable(logits)
    elif mode == "probs":
        from scipy.special import expit
        probs = expit(logits).astype(np.float32)
        return cal_callable(probs)
    else:
        raise ValueError(mode)


# ---- locked lexicographic selector -----------------------------------------

def select_best_calibrator(per_calibrator_metrics):
    """Locked Q2 lexicographic rule (per docs/refine/FINAL_PROPOSAL.md sec Q2).

    per_calibrator_metrics : dict cal_name -> dict with keys:
        macro_ece (float), per_stmt_ece (np.array[K] with NaNs for excluded),
        threshold_gap_sens09 (float), macro_auc (float), n_params (int)

    Returns selected cal_name + tie-break trace dict.
    """
    cals = list(per_calibrator_metrics.keys())
    if not cals:
        return None, {}
    # 1. lowest ECE
    eces = np.array([per_calibrator_metrics[c]["macro_ece"] for c in cals])
    eces_finite = np.where(np.isfinite(eces), eces, np.inf)
    best = int(np.argmin(eces_finite))
    best_ece = eces_finite[best]
    # 1 SE on labelwise stmt ECE (sd of per_stmt / sqrt(K_valid))
    pse = per_calibrator_metrics[cals[best]]["per_stmt_ece"]
    valid = np.isfinite(pse)
    if valid.sum() > 1:
        se = float(np.nanstd(pse[valid]) / np.sqrt(valid.sum()))
    else:
        se = 0.0
    # within 1 SE band
    band = [c for c, e in zip(cals, eces_finite) if e <= best_ece + se]
    if len(band) <= 1:
        return cals[best], {"trace": "ece_unique", "band": band, "se": se}

    # 2. lowest threshold-gap @ 0.9
    gaps = np.array([per_calibrator_metrics[c]["threshold_gap_sens09"] for c in band])
    gaps_finite = np.where(np.isfinite(gaps), gaps, np.inf)
    best_g = int(np.argmin(gaps_finite))
    best_gap = gaps_finite[best_g]
    # 1 SE on gap is not separately stored; use 5% of best as a soft band
    se_gap = abs(best_gap) * 0.05
    band2 = [c for c, g in zip(band, gaps_finite) if g <= best_gap + se_gap]
    if len(band2) <= 1:
        return band[best_g], {"trace": "ece_band->gap_unique",
                              "ece_band": band, "gap_band": band2}

    # 3. highest macro_auc
    aucs = np.array([per_calibrator_metrics[c]["macro_auc"] for c in band2])
    aucs_finite = np.where(np.isfinite(aucs), aucs, -np.inf)
    best_a = int(np.argmax(aucs_finite))
    best_auc = aucs_finite[best_a]
    band3 = [c for c, a in zip(band2, aucs_finite) if a >= best_auc - 1e-4]
    if len(band3) <= 1:
        return band2[best_a], {"trace": "ece_band->gap_band->auc_unique",
                               "ece_band": band, "gap_band": band2, "auc_band": band3}

    # 4. smaller param count
    params = np.array([CAL_PARAM_COUNT[c] for c in band3])
    best_p = int(np.argmin(params))
    return band3[best_p], {"trace": "all_tied->param_count",
                           "ece_band": band, "gap_band": band2, "auc_band": band3,
                           "param_band": band3}


# ---- prob loading ----------------------------------------------------------

def list_probs_files(method_filter=None):
    """Return list of (method, seed, path) for all probs_*.npz files."""
    out = []
    for p in sorted(PROBS_DIR.glob("probs_*.npz")):
        name = p.stem  # probs_<method>_seed<seed>
        body = name[len("probs_"):]
        if "_seed" not in body:
            continue
        method, seed_s = body.rsplit("_seed", 1)
        try:
            seed = int(seed_s)
        except ValueError:
            continue
        if method_filter is not None and method not in method_filter:
            continue
        out.append((method, seed, p))
    return out


def load_probs(path):
    """Return dict of arrays from a probs_*.npz file."""
    d = np.load(path)
    return {k: d[k] for k in d.files}


def macro_auc_safe(y_true, y_prob, min_pos=10):
    """Macro AUC compatible with src.evaluate.macro_auc, returns float."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for k in range(y_true.shape[1]):
        yt = y_true[:, k]; yp = y_prob[:, k]
        if yt.sum() < min_pos or (1 - yt).sum() < min_pos:
            continue
        try:
            aucs.append(roc_auc_score(yt, yp))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")
