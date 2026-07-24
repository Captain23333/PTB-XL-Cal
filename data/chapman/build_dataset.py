"""Build Chapman-Shaoxing dataset metadata CSV.

Parses all .hea files, maps SNOMED-CT codes to PTB-XL's 5 clinical families
(NORM, MI, STTC, CD, HYP), and writes a record-level CSV with binary labels.
A deterministic random 70/15/15 train/val/test split (seed 42) is added.

Output: data/chapman/chapman_metadata.csv
"""
import re, csv
from pathlib import Path
import numpy as np
import pandas as pd

CHAPMAN_DIR = Path(__file__).resolve().parent
ROOT = CHAPMAN_DIR.parent.parent

# SNOMED code → family mapping (45 codes from ConditionNames + standard additions)
SNOMED_TO_FAMILY = {
    # ── NORM ────────────────────────────────────────────────────────────────
    "426783006": "NORM",   # SR  Sinus Rhythm

    # ── MI ──────────────────────────────────────────────────────────────────
    "164865005": "MI",     # MI / MIBW / MIFW / MILW / MISW (shared code)

    # ── STTC ────────────────────────────────────────────────────────────────
    "428750005": "STTC",   # STTC  ST-T change (generic)
    "164930006": "STTC",   # STE   ST elevation
    "429622005": "STTC",   # STDD  ST drop-down
    "164931005": "STTC",   # STTU  ST tilt-up
    "164934002": "STTC",   # TWC   T-wave change
    "59931005":  "STTC",   # TWO   T-wave opposite
    "111975006": "STTC",   # QTIE  QT interval extension
    "428417006": "STTC",   # ERV   Early repolarisation
    "164917005": "STTC",   # AQW   Abnormal Q wave

    # ── CD (Conduction + Rhythm) ─────────────────────────────────────────────
    "270492004": "CD",     # 1AVB
    "195042002": "CD",     # 2AVB
    "54016002":  "CD",     # 2AVB1
    "28189009":  "CD",     # 2AVB2
    "27885002":  "CD",     # 3AVB
    "233917008": "CD",     # AVB  (generic AV block)
    "164909002": "CD",     # LBBB / LBBBB / LFBBB
    "59118001":  "CD",     # RBBB
    "698252002": "CD",     # IVB / IDC
    "74390002":  "CD",     # WPW
    "195060002": "CD",     # VPE  (ventricular pre-excitation)
    "164947007": "CD",     # PRIE (PR interval extension)
    "164889003": "CD",     # AFIB
    "164890007": "CD",     # AF   (Atrial flutter)
    "426761007": "CD",     # SVT
    "713422000": "CD",     # AT   (Atrial tachycardia)
    "233896004": "CD",     # AVNRT
    "233897008": "CD",     # AVRT
    "17338001":  "CD",     # VPB
    "75532003":  "CD",     # VEB
    "251180001": "CD",     # VET
    "11157007":  "CD",     # VB   (Ventricular bigeminy)
    "284470004": "CD",     # APB
    "251173003": "CD",     # ABI  (Atrial bigeminy)
    "426995002": "CD",     # JEB
    "251164006": "CD",     # JPT
    "195101003": "CD",     # WAVN / SAAWR
    "426177001": "CD",     # SB   (Sinus bradycardia)
    "427393009": "CD",     # SA   (Sinus irregularity)
    "427084000": "CD",     # ST   (Sinus tachycardia)

    # ── HYP ─────────────────────────────────────────────────────────────────
    "164873001": "HYP",    # LVH  (Chapman SNOMED)
    "55827005":  "HYP",    # LVH  (standard SNOMED; most-common ??? code = 5381 recs)
    "89792004":  "HYP",    # RVH
    "446358003": "HYP",    # RAH
}

FAMILIES = ["NORM", "MI", "STTC", "CD", "HYP"]

# ── Parse .hea files ─────────────────────────────────────────────────────────
print("Parsing .hea files...")
records = []
hea_dir = CHAPMAN_DIR / "WFDBRecords"
for hea_path in sorted(hea_dir.rglob("*.hea")):
    txt = hea_path.read_text(errors="ignore")
    lines = txt.splitlines()
    if not lines:
        continue

    # record path relative to WFDBRecords parent (for wfdb.rdsamp)
    rel = hea_path.relative_to(CHAPMAN_DIR)
    record_path = str(rel.with_suffix(""))   # e.g. WFDBRecords/01/010/JS00001

    rec_id = hea_path.stem

    age, sex = -1, "Unknown"
    dx_codes = []
    for line in lines:
        if line.startswith("#Age:"):
            try:
                age = int(line.split(":")[1].strip())
            except Exception:
                pass
        elif line.startswith("#Sex:"):
            sex = line.split(":")[1].strip()
        elif line.startswith("#Dx:"):
            raw = line.split(":")[1].strip()
            dx_codes = [c.strip() for c in raw.split(",") if c.strip()]

    # family labels
    fam_labels = {f: 0 for f in FAMILIES}
    for code in dx_codes:
        fam = SNOMED_TO_FAMILY.get(code)
        if fam:
            fam_labels[fam] = 1

    # skip if .mat file missing or corrupt
    if not (CHAPMAN_DIR / (record_path + ".mat")).exists():
        continue
    corrupt = {"JS01052","JS07313","JS19400","JS23074","JS26401",
               "JS30482","JS37401","JS42461","JS43400"}
    if rec_id in corrupt:
        continue

    records.append({
        "record_id": rec_id,
        "record_path": record_path,
        "age": age,
        "sex": sex,
        "dx_codes": ",".join(dx_codes),
        **fam_labels,
    })

df = pd.DataFrame(records)
print(f"Total records: {len(df)}")
print("\nFamily prevalence:")
for f in FAMILIES:
    n = df[f].sum()
    print(f"  {f}: {n} ({100*n/len(df):.1f}%)")

n_none = (df[FAMILIES].sum(axis=1) == 0).sum()
print(f"  No mapped label: {n_none} ({100*n_none/len(df):.1f}%)")

# ── Deterministic random recording-level split 70/15/15 ─────────────────────
rng = np.random.default_rng(42)
idx = rng.permutation(len(df))
n = len(df)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

split = np.empty(n, dtype=object)
split[idx[:n_train]]            = "train"
split[idx[n_train:n_train+n_val]] = "val"
split[idx[n_train+n_val:]]      = "test"
df["split"] = split

print("\nSplit sizes:", df["split"].value_counts().to_dict())

out_path = CHAPMAN_DIR / "chapman_metadata.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
