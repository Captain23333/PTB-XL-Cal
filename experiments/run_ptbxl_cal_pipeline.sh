#!/usr/bin/env bash
set +e  # don't crash on individual failures; keep going
cd "${PTBXL_CAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs results/cal_probs figures splits docs

PY="${PTBXL_CAL_PYTHON:-python3}"

echo "[$(date)] === PTB-XL-Cal R1 pipeline START ==="

echo "[$(date)] === Phase 0: hygiene already done by Agent ==="

echo "[$(date)] === Phase 1: recompute val+test probs for 24 ckpts ==="
$PY experiments/calib_recompute_probs.py 2>&1 | tee logs/calib_recompute.log

echo "[$(date)] === Phase 2: ECG-FM gate (90 min time-box) ==="
timeout 5400 $PY experiments/calib_ecgfm_probe.py 2>&1 | tee logs/calib_ecgfm.log

echo "[$(date)] === Phase 3: Block 1 Q1 ==="
$PY experiments/calib_block1_q1.py 2>&1 | tee logs/calib_block1.log

echo "[$(date)] === Phase 4: Block 2 Q2 (longest, ~1-2h with bootstrap) ==="
$PY experiments/calib_block2_q2.py 2>&1 | tee logs/calib_block2.log

echo "[$(date)] === Phase 5: Block 3 Q3 ==="
$PY experiments/calib_block3_q3.py 2>&1 | tee logs/calib_block3.log

echo "[$(date)] === Phase 6: Block 4 threshold transfer ==="
$PY experiments/calib_block4_threshold.py 2>&1 | tee logs/calib_block4.log

echo "[$(date)] === Phase 7: figures ==="
$PY experiments/calib_make_figures.py 2>&1 | tee logs/calib_figures.log

echo "[$(date)] === Phase 8: scaffold worked-example ==="
$PY eval_calibration.py --predictions 'results/cal_probs/probs_*.npz' \
    --val-split splits/val_fold9.csv --test-split splits/test_fold10.csv \
    --leaderboard leaderboard_worked_example.json 2>&1 | tee logs/calib_scaffold.log

echo "[$(date)] === Phase 9: emit EXPERIMENT_REPORT.md ==="
$PY experiments/calib_emit_report.py 2>&1 | tee logs/calib_report.log

echo "[$(date)] === ALL PTB-XL-CAL R1 BLOCKS COMPLETE ==="
