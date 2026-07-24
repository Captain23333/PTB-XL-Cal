#!/usr/bin/env bash
# Block 0: environment setup + smoke test (run after PTB-XL signals fully downloaded)
set -e
cd "${PTBXL_CAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "=== 1) Install required packages (skip if already present) ==="
pip3 install --user -q torch torchvision wfdb fastai scikit-learn netcal tqdm pyyaml 2>&1 | tail -2

echo ""
echo "=== 2) GPU detection ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

echo ""
echo "=== 3) Smoke test: load 1 record + forward pass ==="
python3 -c "
import sys; sys.path.insert(0, '.')
import wfdb
sig, fields = wfdb.rdsamp('data/records100/00000/00001_lr')
print('Signal shape:', sig.shape, 'fs:', fields['fs'])

import torch
from src.models import build_model
model = build_model().cuda() if torch.cuda.is_available() else build_model()
device = next(model.parameters()).device
x = torch.randn(2, 12, 1000, device=device)
out = model(x)
print('Forward pass OK. Output shapes:')
for k, v in out.items():
    print(f'  {k}: {v.shape}')
print('Total params:', sum(p.numel() for p in model.parameters()))
"

echo ""
echo "=== 4) Smoke test: PTB-XL data loader (1 batch) ==="
python3 -c "
import sys; sys.path.insert(0, '.')
from src.data import get_loaders
train_loader, val_loader, test_loader, train_ds = get_loaders(batch_size=4, num_workers=0)
print(f'Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}')
batch = next(iter(train_loader))
print('Batch keys:', list(batch.keys()))
for k, v in batch.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={tuple(v.shape)} dtype={v.dtype}')
print('Likelihood values in this batch:',
      sorted(set((batch[\"y_stmt\"].numpy() * 100).round().astype(int).flatten().tolist())))
"

echo ""
echo "=== Setup complete. Ready to launch training. ==="
echo "Next: bash experiments/run_block_a.sh (or python3 experiments/02_block_a_main.py)"
