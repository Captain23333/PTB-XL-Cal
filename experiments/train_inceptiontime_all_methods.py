"""Train all 7 remaining methods on InceptionTime backbone (Task A Q2).

Usage:
    python experiments/train_inceptiontime_all_methods.py --gpu 0 --worker 0
    python experiments/train_inceptiontime_all_methods.py --gpu 1 --worker 1
"""
import argparse, os, sys, json
import numpy as np
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.train import train_one, evaluate_loader, ROOT
from src.data import get_loaders
from src.models import build_model

METHODS = ["asl", "focal", "label_smooth", "soft", "soft_parent",
           "soft_parent_ord", "scp_soft_full"]
SEEDS = [42, 123, 2024]

def extract_probs(method, seed, gpu):
    out_path = ROOT / f"results/cal_probs/probs_{method}_inceptiontime_seed{seed}.npz"
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return

    ckpt_path = ROOT / f"checkpoints/{method}_inceptiontime_seed{seed}.pt"
    if not ckpt_path.exists():
        print(f"  [skip] checkpoint missing: {ckpt_path.name}")
        return

    ckpt = torch.load(ckpt_path, map_location="cuda")
    model = build_model(backbone_name="inceptiontime").cuda()
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state)
    model.eval()

    _, val_loader, test_loader, _ = get_loaders(batch_size=128)

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
    np.savez(out_path,
             val_logits=val_logits, val_targets=val_targets, val_patient_ids=val_pids,
             test_logits=test_logits, test_targets=test_targets, test_patient_ids=test_pids,
             method=f"{method}_inceptiontime", seed=seed)
    print(f"  Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker", type=int, required=True,
                        help="0 or 1 — splits the work across 2 GPUs")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Interleave: worker 0 takes even indices, worker 1 takes odd indices
    all_jobs = [(m, s) for m in METHODS for s in SEEDS]
    jobs = [j for i, j in enumerate(all_jobs) if i % 2 == args.worker]

    for method, seed in jobs:
        ckpt_path = ROOT / f"checkpoints/{method}_inceptiontime_seed{seed}.pt"
        if ckpt_path.exists():
            print(f"[skip] {method} seed{seed} checkpoint exists")
        else:
            print(f"[train] {method} seed{seed} on GPU{args.gpu}")
            train_one(method=method, seed=seed, epochs=args.epochs,
                      backbone_name="inceptiontime")
        extract_probs(method, seed, args.gpu)

    print(f"[worker {args.worker}] ALL DONE")


if __name__ == "__main__":
    main()
