"""ECG-FM gate (kill-test, time-boxed 90 min).

Tries to download an open-weight 12-lead ECG foundation model from HF;
if any candidate works, extracts features for ALL PTB-XL train+val+test
samples and trains a per-statement linear probe (logistic regression) on
train+val, then writes results/cal_probs/probs_ecgfm_linprobe_seed42.npz
in the same format as the SimpleResNet1D probs.

If ALL candidates fail (download / load / forward / OOM), writes a one-line
decision to logs/ecgfm_gate.log AND IDEA_JOURNEY.md, then triggers fallback:
train an 8M wider SimpleResNet1D (base=128) for 30 epochs, single seed,
and save probs in same format. Fallback is itself best-effort: if compute
budget exhausted, just record the limitation.
"""
from __future__ import annotations
import sys, os, time, traceback, json
import os
sys.path.insert(0, os.environ.get("PTBXL_CAL_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:17899")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:17899")

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "results/cal_probs"
LOGS = ROOT / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
GATE_LOG = LOGS / "ecgfm_gate.log"

GLOBAL_DEADLINE = time.time() + 90 * 60  # 90-minute hard time-box


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(GATE_LOG, "a") as f:
        f.write(line + "\n")


def _journal(line):
    try:
        with open(ROOT / "IDEA_JOURNEY.md", "a") as f:
            f.write("\n" + line + "\n")
    except Exception:
        pass


def _within_budget():
    return time.time() < GLOBAL_DEADLINE


# -------------------- HF candidates --------------------

CANDIDATES = [
    {"repo": "wanglab/ecg-fm", "kind": "transformers"},
    {"repo": "Edoardo-BS/hubert-ecg-base", "kind": "transformers"},
    {"repo": "Edoardo-BS/hubert-ecg-large", "kind": "transformers"},
    {"repo": "Edoardo-BS/hubert-ecg-small", "kind": "transformers"},
    {"repo": "Bsingstad/ECG-FM", "kind": "transformers"},
]


def try_download_one(repo_id, kind):
    """Return (model, fwd_fn) tuple on success else None."""
    from huggingface_hub import snapshot_download
    _log(f"trying download repo={repo_id} kind={kind}")
    try:
        local = snapshot_download(repo_id=repo_id,
                                   local_dir=ROOT / f"checkpoints/_hf_{repo_id.replace('/', '_')}",
                                   local_dir_use_symlinks=False,
                                   max_workers=2,
                                   etag_timeout=30)
        _log(f"  download ok: {local}")
    except Exception as e:
        _log(f"  download fail: {type(e).__name__}: {e}")
        return None
    if not _within_budget():
        return None
    # Try transformers
    try:
        from transformers import AutoModel, AutoConfig
        config = AutoConfig.from_pretrained(local, trust_remote_code=True)
        model = AutoModel.from_pretrained(local, trust_remote_code=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        # Probe forward shape: 12-lead 1000 samples
        x = torch.randn(2, 12, 1000)
        if torch.cuda.is_available():
            x = x.cuda()
        with torch.no_grad():
            try:
                out = model(x)
            except Exception:
                # Try alternate input shape
                out = model(input_values=x)
        # Extract feature
        feat = None
        for attr in ("last_hidden_state", "pooler_output", "feature", "logits"):
            if hasattr(out, attr):
                feat = getattr(out, attr)
                break
        if feat is None and isinstance(out, dict):
            for k in ("last_hidden_state", "pooler_output", "feature", "logits"):
                if k in out:
                    feat = out[k]
                    break
        if feat is None and isinstance(out, torch.Tensor):
            feat = out
        if feat is None:
            _log(f"  no recognizable feature output; type(out)={type(out)}")
            return None
        if feat.dim() == 3:
            feat = feat.mean(dim=1)
        _log(f"  forward ok feat={tuple(feat.shape)}")

        def fwd(batch_x):
            batch_x = batch_x.cuda() if torch.cuda.is_available() else batch_x
            with torch.no_grad():
                try:
                    out = model(batch_x)
                except Exception:
                    out = model(input_values=batch_x)
            f = None
            for a in ("last_hidden_state", "pooler_output", "feature", "logits"):
                if hasattr(out, a):
                    f = getattr(out, a); break
            if f is None and isinstance(out, dict):
                for k in ("last_hidden_state", "pooler_output", "feature", "logits"):
                    if k in out:
                        f = out[k]; break
            if f is None and isinstance(out, torch.Tensor):
                f = out
            if f.dim() == 3:
                f = f.mean(dim=1)
            return f.detach().cpu().float().numpy()

        return model, fwd
    except Exception as e:
        _log(f"  transformers load/forward fail: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# -------------------- Linear probe pipeline ----------------

def extract_features_all(fwd, train_ds, val_ds, test_ds, batch_size=64):
    """Stream features for train/val/test."""
    from torch.utils.data import DataLoader
    feats, ys, pids = {}, {}, {}
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if not _within_budget():
            return None
        loader = DataLoader(ds, batch_size=batch_size, num_workers=4,
                            shuffle=False, pin_memory=True)
        F_acc, Y_acc, P_acc = [], [], []
        for i, batch in enumerate(loader):
            x = batch["x"]
            f = fwd(x)
            F_acc.append(f); Y_acc.append(batch["y_hard"].numpy())
            P_acc.append(batch["patient_id"].numpy())
            if i % 20 == 0:
                _log(f"  feat {name} batch {i}/{len(loader)}")
        feats[name] = np.concatenate(F_acc, 0)
        ys[name] = np.concatenate(Y_acc, 0)
        pids[name] = np.concatenate(P_acc, 0)
    return feats, ys, pids


def linear_probe(feats, ys, pids):
    """Per-statement logistic regression (sklearn). Train on train+val (fold 1-9)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X_train = np.concatenate([feats["train"], feats["val"]], 0)
    Y_train = np.concatenate([ys["train"], ys["val"]], 0)
    X_test = feats["test"]
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    K = Y_train.shape[1]
    val_logits = np.zeros((feats["val"].shape[0], K), dtype=np.float32)
    test_logits = np.zeros((X_test.shape[0], K), dtype=np.float32)
    X_val_only = scaler.transform(feats["val"])
    n_train_only = feats["train"].shape[0]
    Y_train_only = ys["train"]
    X_train_only = scaler.transform(feats["train"])
    for k in range(K):
        if Y_train_only[:, k].sum() < 5 or (1 - Y_train_only[:, k]).sum() < 5:
            # Skip rare label - keep default zero logits
            continue
        try:
            clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=1, solver="lbfgs")
            clf.fit(X_train_only, Y_train_only[:, k])
            # Decision function returns logits
            val_logits[:, k] = clf.decision_function(X_val_only)
            test_logits[:, k] = clf.decision_function(X_test)
        except Exception as e:
            _log(f"  logreg fail at stmt {k}: {e}")
    return val_logits, test_logits


def save_ecgfm_probs(val_logits, test_logits, ys, pids):
    out_path = OUT / "probs_ecgfm_linprobe_seed42.npz"
    np.savez_compressed(
        out_path,
        val_logits=val_logits.astype(np.float32),
        val_targets=ys["val"].astype(np.float32),
        test_logits=test_logits.astype(np.float32),
        test_targets=ys["test"].astype(np.float32),
        val_patient_ids=pids["val"].astype(np.int64),
        test_patient_ids=pids["test"].astype(np.int64),
        # super/sub not available from feature backbone — fill zeros
        super_logits_val=np.zeros((val_logits.shape[0], 5), dtype=np.float32),
        sub_logits_val=np.zeros((val_logits.shape[0], 23), dtype=np.float32),
        super_logits_test=np.zeros((test_logits.shape[0], 5), dtype=np.float32),
        sub_logits_test=np.zeros((test_logits.shape[0], 23), dtype=np.float32),
    )
    _log(f"saved {out_path.name}")


# -------------------- Fallback: 8M wider SimpleResNet1D ----------

def fallback_wider_resnet():
    """Train 2x wider SimpleResNet1D (base=128), 30 epochs, single seed=42,
    then forward val+test and save probs in the same format.
    Wrapped in a try/except so the gate cannot crash the pipeline."""
    _log("FALLBACK: training wider SimpleResNet1D base=128, single seed=42, 30 epochs")
    try:
        from src.data import get_loaders
        from src.models import _SimpleResNet1D
        import torch.nn as nn
        import torch.nn.functional as F

        train_loader, val_loader, test_loader, train_ds = get_loaders(batch_size=128, num_workers=4)

        class WiderHead(nn.Module):
            def __init__(self, base=128):
                super().__init__()
                self.bb = _SimpleResNet1D(c_in=12, base=base)
                self.head_stmt = nn.Linear(256, 71)

            def forward(self, x):
                f = self.bb(x)
                return self.head_stmt(f), f

        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        model = WiderHead(base=128).cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=30,
                                                   steps_per_epoch=len(train_loader))
        deadline = time.time() + 60 * 60  # 60 min cap for fallback
        for ep in range(30):
            if time.time() > deadline:
                _log(f"  fallback time cap; stopping at epoch {ep}")
                break
            model.train()
            for batch in train_loader:
                x = batch["x"].cuda(non_blocking=True)
                y = batch["y_hard"].cuda(non_blocking=True)
                opt.zero_grad()
                logits, _ = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                loss.backward()
                opt.step(); sch.step()
            _log(f"  fallback ep{ep} done")
        model.eval()

        @torch.no_grad()
        def fwd(loader):
            zs, ys, pids = [], [], []
            for batch in loader:
                x = batch["x"].cuda(non_blocking=True)
                z, _ = model(x)
                zs.append(z.cpu().numpy())
                ys.append(batch["y_hard"].numpy())
                pids.append(batch["patient_id"].numpy())
            return np.concatenate(zs), np.concatenate(ys), np.concatenate(pids)

        v_z, v_y, v_p = fwd(val_loader)
        t_z, t_y, t_p = fwd(test_loader)
        np.savez_compressed(
            OUT / "probs_ecgfm_linprobe_seed42.npz",
            val_logits=v_z.astype(np.float32),
            val_targets=v_y.astype(np.float32),
            test_logits=t_z.astype(np.float32),
            test_targets=t_y.astype(np.float32),
            val_patient_ids=v_p.astype(np.int64),
            test_patient_ids=t_p.astype(np.int64),
            super_logits_val=np.zeros((v_z.shape[0], 5), dtype=np.float32),
            sub_logits_val=np.zeros((v_z.shape[0], 23), dtype=np.float32),
            super_logits_test=np.zeros((t_z.shape[0], 5), dtype=np.float32),
            sub_logits_test=np.zeros((t_z.shape[0], 23), dtype=np.float32),
        )
        _log("FALLBACK 8M ckpt saved as 'ecgfm_linprobe_seed42' (mislabel ok per spec)")
        return True
    except Exception as e:
        _log(f"FALLBACK FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# -------------------- Main --------------------

def main():
    _log("=== ECG-FM gate START ===")
    _log(f"deadline = {time.strftime('%H:%M:%S', time.localtime(GLOBAL_DEADLINE))}")
    success = False
    for cand in CANDIDATES:
        if not _within_budget():
            _log("90-min time-box exhausted before all candidates tried")
            break
        try:
            res = try_download_one(cand["repo"], cand["kind"])
        except Exception as e:
            _log(f"unhandled exception on {cand['repo']}: {e}")
            res = None
        if res is None:
            continue
        model, fwd = res
        # Free disk: clean other downloaded candidates' dirs lazily
        try:
            from src.data import PTBXLDataset
            train_ds = PTBXLDataset(list(range(1, 9)))
            val_ds = PTBXLDataset([9])
            test_ds = PTBXLDataset([10])
            extracted = extract_features_all(fwd, train_ds, val_ds, test_ds, batch_size=64)
        except Exception as e:
            _log(f"feature extraction failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        if extracted is None:
            _log("feature extraction time-boxed out")
            break
        feats, ys, pids = extracted
        try:
            v_z, t_z = linear_probe(feats, ys, pids)
            save_ecgfm_probs(v_z, t_z, ys, pids)
            success = True
            _journal(f"### Stage L.3 ECG-FM gate PASS via {cand['repo']} (linear probe) at {time.strftime('%Y-%m-%d %H:%M')}")
            break
        except Exception as e:
            _log(f"linear probe failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

    if not success:
        _log("ECG-FM gate FAILED on all candidates -> fallback to wider SimpleResNet1D 8M")
        _journal(f"### Stage L.3 ECG-FM gate FAILED at {time.strftime('%Y-%m-%d %H:%M')}; falling back to internal wider SimpleResNet1D base=128 (acknowledged re-score risk)")
        fallback_wider_resnet()

    _log("=== ECG-FM gate END ===")


if __name__ == "__main__":
    main()
