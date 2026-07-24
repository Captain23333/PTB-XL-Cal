#!/usr/bin/env bash
# Fix CUDA-mismatched torch + missing IPython, then run full L.3 pipeline.
# Usage: nohup bash experiments/fix_and_run_all.sh > logs/fix_and_run_all.log 2>&1 &
set +e   # don't abort on individual step failures; we want to log everything
cd "${PTBXL_CAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs

echo "[$(date)] === FIX STAGE 1: replace torch with CUDA-12-compatible version ==="
pip3 uninstall -y torch torchvision 2>&1 | tail -3
pip3 install --user --timeout 180 \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5

echo ""
echo "[$(date)] === FIX STAGE 2: install missing IPython for fastai ==="
pip3 install --user --timeout 120 ipython 2>&1 | tail -3

echo ""
echo "[$(date)] === FIX STAGE 3: verify everything ==="
python3 -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}:', torch.cuda.get_device_name(i))
import wfdb, sklearn, netcal, IPython
print('wfdb', wfdb.__version__, 'sklearn', sklearn.__version__, 'netcal OK', 'IPython OK')
from fastai.vision.models.xresnet import xresnet1d101
print('fastai xresnet1d101 import OK')
" 2>&1

echo ""
echo "[$(date)] === SMOKE TEST: forward pass + 1 batch ==="
cd "${PTBXL_CAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python3 -c "
import sys; sys.path.insert(0, '.')
import torch
from src.models import build_model
m = build_model()
if torch.cuda.is_available():
    m = m.cuda()
    x = torch.randn(2, 12, 1000).cuda()
else:
    x = torch.randn(2, 12, 1000)
out = m(x)
print('Forward pass OK. Output shapes:')
for k, v in out.items():
    print(f'  {k}: {tuple(v.shape)}')
print('Total params:', sum(p.numel() for p in m.parameters()))
" 2>&1

echo ""
python3 -c "
import sys; sys.path.insert(0, '.')
from src.data import get_loaders
tl, vl, tel, ds = get_loaders(batch_size=4, num_workers=0)
print(f'Train: {len(tl.dataset)} | Val: {len(vl.dataset)} | Test: {len(tel.dataset)}')
b = next(iter(tl))
for k, v in b.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={tuple(v.shape)} dtype={v.dtype}')
" 2>&1

echo ""
echo "[$(date)] === BLOCK A: train 5 methods × 3 seeds (~12h on 2× 2080 Ti) ==="
bash experiments/run_block_a.sh 2>&1 | tee logs/block_a_train_v2.log

echo ""
echo "[$(date)] === BLOCK A EVAL ==="
python3 experiments/02_block_a_eval.py 2>&1 | tee logs/block_a_eval_v2.log

echo ""
echo "[$(date)] === BLOCK B: 5-step ladder ==="
python3 experiments/03_block_b_ladder.py 2>&1 | tee logs/block_b_v2.log

echo ""
echo "[$(date)] === BLOCK C: stratified ==="
python3 experiments/04_block_c_stratified.py 2>&1 | tee logs/block_c_v2.log

echo ""
echo "[$(date)] === BLOCK D: Pareto ==="
python3 experiments/05_block_d_pareto.py 2>&1 | tee logs/block_d_v2.log

echo ""
echo "[$(date)] === BLOCK E: case study ==="
python3 experiments/06_block_e_case_study.py 2>&1 | tee logs/block_e_v2.log

echo ""
echo "[$(date)] === SUPPLEMENTS S7+S8 ==="
python3 experiments/07_supplements.py 2>&1 | tee logs/supplements_v2.log

echo ""
echo "[$(date)] === ALL L.3 BLOCKS COMPLETE ==="
echo "Results: results/block_{a,b,c,d}_*.csv ; figures: figures/fig_block_*.png"
echo ""
echo "Next: /result-to-claim → /journal-review-loop → /paper-plan ..."
