#!/usr/bin/env bash
# v3: use clean conda env 'ecg' (torch 2.4.0+cu121), skip all pip install.
# Usage: nohup bash experiments/fix_and_run_v3.sh > logs/fix_and_run_v3.log 2>&1 & disown
set +e
cd "${PTBXL_CAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs checkpoints results figures

PY="${PTBXL_CAL_PYTHON:-python3}"

echo "[$(date)] === ENV CHECK ==="
$PY -c "
import torch, wfdb, sklearn, pandas
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devs', torch.cuda.device_count())
print('wfdb', wfdb.__version__, 'sklearn', sklearn.__version__, 'pandas', pandas.__version__)
"

echo ""
echo "[$(date)] === SMOKE: forward + 1 batch ==="
$PY -c "
import sys; sys.path.insert(0, '.')
import torch
from src.models import build_model
from src.data import get_loaders
m = build_model().cuda()
print('Total params:', f'{sum(p.numel() for p in m.parameters()):,}')
tl, vl, tel, ds = get_loaders(batch_size=8, num_workers=0)
print(f'Train: {len(tl.dataset)} | Val: {len(vl.dataset)} | Test: {len(tel.dataset)}')
b = next(iter(tl))
b = {k:(v.cuda() if torch.is_tensor(v) else v) for k,v in b.items()}
out = m(b['x'])
print('Forward OK:', {k: tuple(v.shape) for k,v in out.items()})
"

echo ""
echo "[$(date)] === BLOCK A: 5 methods x 3 seeds (~5h on 2x 2080 Ti) ==="
METHODS="hard_bce asl focal label_smooth scp_soft_full"
SEEDS="42 123 2024"
for m in $METHODS; do
    for s in $SEEDS; do
        ckpt="checkpoints/${m}_seed${s}.pt"
        if [ -f "$ckpt" ]; then
            echo "[$(date)] Skip $m seed=$s: $ckpt exists"
            continue
        fi
        echo "[$(date)] === Training $m seed=$s ==="
        $PY -c "
import sys; sys.path.insert(0, '.')
from src.train import train_one
train_one('$m', $s, epochs=50, batch_size=128)
" 2>&1
    done
done

echo ""
echo "[$(date)] === BLOCK A EVAL ==="
$PY experiments/02_block_a_eval.py 2>&1 | tee logs/block_a_eval_v3.log

echo ""
echo "[$(date)] === BLOCK B: 5-step ladder ==="
$PY experiments/03_block_b_ladder.py 2>&1 | tee logs/block_b_v3.log

echo ""
echo "[$(date)] === BLOCK C: stratified ==="
$PY experiments/04_block_c_stratified.py 2>&1 | tee logs/block_c_v3.log

echo ""
echo "[$(date)] === BLOCK D: Pareto ==="
$PY experiments/05_block_d_pareto.py 2>&1 | tee logs/block_d_v3.log

echo ""
echo "[$(date)] === BLOCK E: case study ==="
$PY experiments/06_block_e_case_study.py 2>&1 | tee logs/block_e_v3.log

echo ""
echo "[$(date)] === SUPPLEMENTS S7+S8 ==="
$PY experiments/07_supplements.py 2>&1 | tee logs/supplements_v3.log

echo ""
echo "[$(date)] === ALL L.3 BLOCKS COMPLETE ==="
echo "Results: results/block_{a,b,c,d}_*.csv ; figures: figures/fig_block_*.png"
