"""Chapman-Shaoxing dataset loader.

Outputs 5-class multi-label (NORM, MI, STTC, CD, HYP) binary targets.
Resamples 500 Hz → 100 Hz (factor-5 decimation) so the same
SimpleResNet1D/InceptionTime backbone can be reused unchanged.
"""
import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.signal import decimate

ROOT = Path(__file__).resolve().parent.parent
CHAPMAN_DIR = ROOT / "data" / "chapman"
FAMILIES = ["NORM", "MI", "STTC", "CD", "HYP"]


class ChapmanDataset(Dataset):
    """Returns dict with:
        x          : [12, 1000]  100 Hz (decimated from 500 Hz), per-lead z-norm
        y_hard     : [5]         binary multi-label (NORM/MI/STTC/CD/HYP)
        patient_id : int         (record index used as surrogate patient id)
    """
    def __init__(self, split, meta_path=CHAPMAN_DIR/"chapman_metadata.csv", normalize=True):
        df = pd.read_csv(meta_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.normalize = normalize
        self.families = FAMILIES
        self.n_labels = len(FAMILIES)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = str(CHAPMAN_DIR / row["record_path"])
        sig, meta = wfdb.rdsamp(path)       # nominally (5000, 12) at 500 Hz
        # Ensure exactly 12 leads and pad/truncate to 5000 samples
        sig = sig.astype(np.float32)        # (n_samp, n_leads)
        n_samp, n_leads = sig.shape
        if n_leads < 12:
            sig = np.pad(sig, ((0,0),(0,12-n_leads)))
        else:
            sig = sig[:, :12]
        if n_samp < 5000:
            sig = np.pad(sig, ((0, 5000-n_samp),(0,0)))
        else:
            sig = sig[:5000, :]
        sig = sig.T                         # (12, 5000)
        # Decimate 500 → 100 Hz (factor 5)
        sig = decimate(sig, q=5, axis=-1, zero_phase=True).astype(np.float32)
        # sig is now (12, 1000)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize:
            mu = sig.mean(-1, keepdims=True)
            sd = sig.std(-1, keepdims=True) + 1e-6
            sig = (sig - mu) / sd

        y = np.array([row[f] for f in FAMILIES], dtype=np.float32)

        return {
            "x": torch.from_numpy(sig),
            "y_hard": torch.from_numpy(y),
            "patient_id": i,    # no separate patient IDs in Chapman
        }


def get_chapman_loader(split, batch_size=64, num_workers=4, **kw):
    ds = ChapmanDataset(split, **kw)
    shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0))
