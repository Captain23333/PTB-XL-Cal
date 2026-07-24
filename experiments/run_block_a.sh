#!/usr/bin/env bash
# Block A: train 5 baselines + SCP-Soft (3 seeds each) on PTB-XL
# Estimated wall-clock: ~12 hours on 2× 2080 Ti
set -e
cd "${PTBXL_CAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

METHODS="hard_bce asl focal label_smooth scp_soft_full"
SEEDS="42 123 2024"

for m in $METHODS; do
    for s in $SEEDS; do
        ckpt="checkpoints/${m}_seed${s}.pt"
        if [ -f "$ckpt" ]; then
            echo "Skip $m seed=$s: $ckpt exists"
            continue
        fi
        echo "=== Training $m seed=$s ==="
        python3 -c "
import sys; sys.path.insert(0, '.')
from src.train import train_one
train_one('$m', $s, epochs=50, batch_size=128)
"
    done
done

echo ""
echo "=== Block A training complete. Run python3 experiments/02_block_a_eval.py for evaluation ==="
