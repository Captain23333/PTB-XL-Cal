"""PTB-XL dataset with likelihood + hierarchy targets.

Critical empirical finding (Day 1): PTB-XL v1.0.3 contains 10 distinct
likelihood values {0, 10, 14, 15, 30, 34, 35, 50, 80, 100}, not the
6 commonly cited in the literature {0, 15, 35, 50, 80, 100}. We map
all values directly via y_soft = ell/100.

For the 6-bin ordinal head we coarsen: {10,14,15} -> bin 1; {30,34,35} -> bin 2;
{50} -> bin 3; {80} -> bin 4; {100} -> bin 5; {0} -> bin 0.
"""
import wfdb
import numpy as np
import pandas as pd
import torch
import ast
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Coarsen 10 values to 6 bins for the ordinal head (preserves ordering)
LIK_TO_BIN = {0: 0, 10: 1, 14: 1, 15: 1, 30: 2, 34: 2, 35: 2,
              50: 3, 80: 4, 100: 5}


class PTBXLDataset(Dataset):
    """Returns dict with:
        x        : [12, 1000]  raw 100Hz signal, per-lead z-normalized
        y_stmt   : [71]        soft target = ell/100 in [0,1]
        y_super  : [5]         max-aggregated descendant likelihood/100
        y_sub    : [23]        same for sub-classes
        y_ord    : [71]        ordinal bin index in {0..5}
        y_hard   : [71]        binary 0/1 (for hard-BCE baselines)
        patient_id : int
    """

    def __init__(self,
                 fold_ids,
                 db_path=ROOT/"data/ptbxl_database.csv",
                 scp_path=ROOT/"data/scp_statements.csv",
                 hier_path=ROOT/"results/hierarchy_edges.json",
                 sig_root=ROOT/"data",
                 normalize=True):
        db = pd.read_csv(db_path)
        db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
        self.db = db[db["strat_fold"].isin(fold_ids)].reset_index(drop=True)

        scp = pd.read_csv(scp_path, index_col=0)
        self.scp = scp
        self.all_stmts = scp.index.tolist()
        self.stmt2idx = {s: i for i, s in enumerate(self.all_stmts)}

        hier = json.load(open(hier_path))
        self.super_list = hier["super_classes"]
        self.sub_list = hier["sub_classes"]
        self.super2idx = {s: i for i, s in enumerate(self.super_list)}
        self.sub2idx = {s: i for i, s in enumerate(self.sub_list)}

        self.sig_root = Path(sig_root)
        self.normalize = normalize

    def __len__(self):
        return len(self.db)

    def _signal(self, row):
        path = self.sig_root / row["filename_lr"]   # 100Hz path (no extension)
        sig, _ = wfdb.rdsamp(str(path))             # (1000, 12)
        sig = sig.T.astype(np.float32)              # (12, 1000)
        if self.normalize:
            sig = (sig - sig.mean(-1, keepdims=True)) / (sig.std(-1, keepdims=True) + 1e-6)
        return sig

    def _build_targets(self, scp_codes):
        y_stmt = np.zeros(71, dtype=np.float32)
        y_ord = np.zeros(71, dtype=np.int64)
        y_hard = np.zeros(71, dtype=np.float32)
        y_super = np.zeros(5, dtype=np.float32)
        y_sub = np.zeros(23, dtype=np.float32)
        for stmt, lik in scp_codes.items():
            if stmt not in self.stmt2idx:
                continue
            j = self.stmt2idx[stmt]
            l = float(lik)
            y_stmt[j] = l / 100.0
            y_ord[j] = LIK_TO_BIN.get(int(round(l)), 0)
            y_hard[j] = 1.0 if l > 0 else 0.0
            row = self.scp.loc[stmt]
            if row["diagnostic"] == 1.0:
                sup, sub = row["diagnostic_class"], row["diagnostic_subclass"]
                if isinstance(sup, str) and sup in self.super2idx:
                    k = self.super2idx[sup]
                    y_super[k] = max(y_super[k], l / 100.0)
                if isinstance(sub, str) and sub in self.sub2idx:
                    k = self.sub2idx[sub]
                    y_sub[k] = max(y_sub[k], l / 100.0)
        return y_stmt, y_super, y_sub, y_ord, y_hard

    def __getitem__(self, i):
        row = self.db.iloc[i]
        x = self._signal(row)
        y_stmt, y_super, y_sub, y_ord, y_hard = self._build_targets(row["scp_codes"])
        return {
            "x": torch.from_numpy(x),
            "y_stmt": torch.from_numpy(y_stmt),
            "y_super": torch.from_numpy(y_super),
            "y_sub": torch.from_numpy(y_sub),
            "y_ord": torch.from_numpy(y_ord),
            "y_hard": torch.from_numpy(y_hard),
            "patient_id": int(row["patient_id"]) if pd.notna(row["patient_id"]) else -1,
        }


def get_loaders(batch_size=128, num_workers=8, train_folds=tuple(range(1, 9)),
                val_fold=9, test_fold=10):
    train_ds = PTBXLDataset(list(train_folds))
    val_ds = PTBXLDataset([val_fold])
    test_ds = PTBXLDataset([test_fold])
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_ds
