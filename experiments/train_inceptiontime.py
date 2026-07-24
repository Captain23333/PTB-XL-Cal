"""Train InceptionTime backbone with hard_bce on 3 seeds for Task A.

Usage:
    python experiments/train_inceptiontime.py --seed 42 --gpu 0
    python experiments/train_inceptiontime.py --seed 123 --gpu 1
    python experiments/train_inceptiontime.py --seed 2024 --gpu 0
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.train import train_one, evaluate_loader, ROOT
from src.data import get_loaders
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--method", type=str, default="hard_bce")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"[InceptionTime] seed={args.seed} gpu={args.gpu} method={args.method}")

    ckpt_path = train_one(
        method=args.method,
        seed=args.seed,
        epochs=args.epochs,
        backbone_name="inceptiontime",
    )
    print(f"[InceptionTime] Checkpoint saved: {ckpt_path}")

    # Save val+test logits as probs_*.npz for calibration pipeline
    print("[InceptionTime] Extracting val/test logits...")
    torch.manual_seed(args.seed)
    from src.models import build_model
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model = build_model(backbone_name="inceptiontime").cuda()
    state = ckpt["state_dict"]
    # strip DataParallel prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    train_loader, val_loader, test_loader, train_ds = get_loaders(batch_size=128)
    edges = json.load(open(ROOT / "results/hierarchy_edges.json"))["edges"]

    def get_logits_targets_pids(loader):
        logits, targets, pids = [], [], []
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].cuda()
                out = model(x)
                logits.append(out["stmt_logits"].cpu().numpy())
                targets.append(batch["y_hard"].cpu().numpy())
                pids.append(batch["patient_id"].cpu().numpy())
        return (np.concatenate(logits), np.concatenate(targets),
                np.concatenate(pids))

    val_logits, val_targets, val_pids = get_logits_targets_pids(val_loader)
    test_logits, test_targets, test_pids = get_logits_targets_pids(test_loader)

    out_path = ROOT / f"results/cal_probs/probs_hard_bce_inceptiontime_seed{args.seed}.npz"
    np.savez(
        out_path,
        val_logits=val_logits, val_targets=val_targets, val_patient_ids=val_pids,
        test_logits=test_logits, test_targets=test_targets, test_patient_ids=test_pids,
        method="hard_bce_inceptiontime", seed=args.seed,
    )
    print(f"[InceptionTime] Saved: {out_path}")
    print(f"  val_logits={val_logits.shape}, test_logits={test_logits.shape}")

if __name__ == "__main__":
    main()
