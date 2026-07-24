"""
Day 1 — Data exploration for SCP-Soft (PTB-XL).
Outputs:
  results/likelihood_dist.csv       - per-statement × likelihood-bin counts
  results/hierarchy_edges.json      - 5 super + 24 sub + 71 stmt + 68-edge tree
  results/clinical_families.json    - {NORM,MI,STTC,CD,HYP} -> [stmt list]
  results/data_summary.json         - sanity stats (N records, fold sizes, etc.)
  figures/fig_likelihood_hist.png   - histogram (overall + 9 representative SCPs)
  figures/fig_fold_distribution.png - records per (fold, super-class)

This script needs ONLY the two CSV files (already downloaded):
  data/ptbxl_database.csv
  data/scp_statements.csv
"""

import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT = Path(os.environ.get("PTBXL_CAL_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "results"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

print("=" * 60)
print("Day 1: SCP-Soft Data Exploration")
print("=" * 60)

# 1) Load
print("\n[1/6] Loading data ...")
db = pd.read_csv(ROOT / "data/ptbxl_database.csv")
db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
scp = pd.read_csv(ROOT / "data/scp_statements.csv", index_col=0)
print(f"  ptbxl_database.csv: {len(db)} records, {db.columns.size} columns")
print(f"  scp_statements.csv: {len(scp)} statements")

# 2) Per-statement likelihood inventory
print("\n[2/6] Building per-statement likelihood inventory ...")
records = []
for _, row in db.iterrows():
    for stmt, lik in row["scp_codes"].items():
        records.append({"ecg_id": row["ecg_id"], "stmt": stmt, "likelihood": float(lik)})
ld = pd.DataFrame(records)
print(f"  Total (record, stmt) pairs: {len(ld)}")
print(f"  Unique statements appearing: {ld['stmt'].nunique()}")
print(f"  Unique likelihood values: {sorted(ld['likelihood'].unique())}")

hist = ld.pivot_table(index="stmt", columns="likelihood", values="ecg_id",
                     aggfunc="count", fill_value=0)
hist.to_csv(OUT / "likelihood_dist.csv")
print(f"  Saved: {OUT/'likelihood_dist.csv'}")

# Sanity: are likelihood values exactly {0,15,35,50,80,100}?
unique_liks = set(int(x) for x in ld["likelihood"].unique())
expected = {0, 15, 35, 50, 80, 100}
print(f"  Likelihood values match expected {expected}: {unique_liks == expected}")
if unique_liks != expected:
    print(f"  WARNING: extra/missing values: extra={unique_liks-expected}, missing={expected-unique_liks}")

# 3) Histogram figure (overall + 9 representative SCPs)
print("\n[3/6] Generating likelihood histogram figure ...")
positive = ld[ld["likelihood"] > 0]
top9 = positive["stmt"].value_counts().head(9).index.tolist()
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()
positive["likelihood"].hist(ax=axes[0], bins=[10, 20, 40, 55, 85, 105])
axes[0].set_title(f"All positive labels (n={len(positive)})")
axes[0].set_xlabel("likelihood")
axes[0].set_ylabel("count")
for i, stmt in enumerate(top9):
    sub = positive[positive["stmt"] == stmt]
    sub["likelihood"].hist(ax=axes[i+1], bins=[10, 20, 40, 55, 85, 105])
    axes[i+1].set_title(f"{stmt} (n={len(sub)})")
    axes[i+1].set_xlabel("likelihood")
plt.tight_layout()
plt.savefig(FIG / "fig_likelihood_hist.png", dpi=200)
plt.close()
print(f"  Saved: {FIG/'fig_likelihood_hist.png'}")

# 4) Build valid diagnostic hierarchy edges
print("\n[4/6] Building hierarchy edges ...")
diag = scp[scp["diagnostic"] == 1.0].copy()
print(f"  Diagnostic statements: {len(diag)} / {len(scp)}")
print(f"  Form statements: {(scp['form']==1.0).sum()}")
print(f"  Rhythm statements: {(scp['rhythm']==1.0).sum()}")

super_classes = sorted(diag["diagnostic_class"].dropna().unique().tolist())
sub_classes = sorted(diag["diagnostic_subclass"].dropna().unique().tolist())
diag_stmts = diag.index.tolist()
all_stmts = scp.index.tolist()
non_diag_stmts = scp[scp["diagnostic"] != 1.0].index.tolist()

print(f"  Super classes ({len(super_classes)}): {super_classes}")
print(f"  Sub classes ({len(sub_classes)}): {sub_classes}")

# super_to_sub mapping
super_to_sub = {}
for sub_c in sub_classes:
    super_c = diag[diag["diagnostic_subclass"] == sub_c]["diagnostic_class"].iloc[0]
    super_to_sub[sub_c] = super_c

# sub_to_stmts mapping (only diagnostic statements)
sub_to_stmts = {}
for sub_c in sub_classes:
    stmts = diag[diag["diagnostic_subclass"] == sub_c].index.tolist()
    sub_to_stmts[sub_c] = stmts

# build edge list
edges = []
for sub_c, super_c in super_to_sub.items():
    edges.append({"parent": super_c, "child": sub_c, "level": "super_to_sub"})
for sub_c, stmts in sub_to_stmts.items():
    for s in stmts:
        edges.append({"parent": sub_c, "child": s, "level": "sub_to_stmt"})

n_super_sub = sum(1 for e in edges if e["level"] == "super_to_sub")
n_sub_stmt = sum(1 for e in edges if e["level"] == "sub_to_stmt")
print(f"  Edges: {len(edges)} (super→sub: {n_super_sub}, sub→stmt: {n_sub_stmt})")

with open(OUT / "hierarchy_edges.json", "w") as f:
    json.dump({
        "super_classes": super_classes,
        "sub_classes": sub_classes,
        "diag_stmts": diag_stmts,
        "all_stmts": all_stmts,
        "non_diag_stmts": non_diag_stmts,
        "super_to_sub": super_to_sub,
        "sub_to_stmts": sub_to_stmts,
        "edges": edges,
    }, f, indent=2)
print(f"  Saved: {OUT/'hierarchy_edges.json'}")

# 5) Clinical family groups
print("\n[5/6] Clinical family groups ...")
families = {fam: diag[diag["diagnostic_class"] == fam].index.tolist()
           for fam in super_classes}
for fam, stmts in families.items():
    print(f"  {fam}: {len(stmts)} statements")
with open(OUT / "clinical_families.json", "w") as f:
    json.dump(families, f, indent=2)
print(f"  Saved: {OUT/'clinical_families.json'}")

# 6) Data summary
print("\n[6/6] Data summary ...")
fold_counts = db["strat_fold"].value_counts().sort_index().to_dict()
patient_counts = db["patient_id"].nunique()
patients_with_multi = (db.groupby("patient_id").size() > 1).sum()

# super-class distribution per fold
per_fold_super = {}
for fold, fold_db in db.groupby("strat_fold"):
    fold_super_counts = {fam: 0 for fam in super_classes}
    for _, row in fold_db.iterrows():
        super_in_record = set()
        for stmt, lik in row["scp_codes"].items():
            if lik > 0 and stmt in scp.index:
                sup = scp.loc[stmt].get("diagnostic_class")
                if isinstance(sup, str):
                    super_in_record.add(sup)
        for sup in super_in_record:
            if sup in fold_super_counts:
                fold_super_counts[sup] += 1
    per_fold_super[int(fold)] = fold_super_counts

summary = {
    "total_records": int(len(db)),
    "unique_patients": int(patient_counts),
    "patients_with_multi_records": int(patients_with_multi),
    "fold_counts": {int(k): int(v) for k, v in fold_counts.items()},
    "train_folds_1_8_count": int(db[db["strat_fold"].isin(range(1, 9))].shape[0]),
    "val_fold_9_count": int(db[db["strat_fold"] == 9].shape[0]),
    "test_fold_10_count": int(db[db["strat_fold"] == 10].shape[0]),
    "per_fold_super_class_counts": per_fold_super,
    "n_super_classes": len(super_classes),
    "n_sub_classes": len(sub_classes),
    "n_diagnostic_stmts": len(diag_stmts),
    "n_non_diagnostic_stmts": len(non_diag_stmts),
    "n_total_stmts": len(all_stmts),
    "n_hierarchy_edges": len(edges),
    "likelihood_values_present": sorted(int(x) for x in unique_liks),
}
with open(OUT / "data_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {OUT/'data_summary.json'}")
print(f"\nSummary:")
for k, v in summary.items():
    if not isinstance(v, dict):
        print(f"  {k}: {v}")

# 7) Fold distribution figure
fig, ax = plt.subplots(figsize=(10, 5))
folds_arr = sorted(per_fold_super.keys())
bottom = np.zeros(len(folds_arr))
for fam in super_classes:
    counts = [per_fold_super[f][fam] for f in folds_arr]
    ax.bar(folds_arr, counts, bottom=bottom, label=fam)
    bottom += np.array(counts)
ax.set_xlabel("Fold (1-8 train, 9 val, 10 test)")
ax.set_ylabel("Records per super-class")
ax.set_title("PTB-XL records per fold and super-class")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "fig_fold_distribution.png", dpi=200)
plt.close()
print(f"  Saved: {FIG/'fig_fold_distribution.png'}")

print("\n" + "=" * 60)
print("Day 1 data exploration COMPLETE.")
print("=" * 60)
print("\nNext step: wait for record100/ download to finish, then start Block A training.")
print(f"Current download progress: ~14% (3048 / ~21837 files)")
