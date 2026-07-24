# Data

This project uses two publicly available ECG datasets. Neither is redistributed
in this repository -- both are already open-access on PhysioNet under their own
licenses, so we only document how to fetch them and where to place them.

## 1. PTB-XL (primary dataset, all main experiments)

- Source: PhysioNet, "PTB-XL, a large publicly available electrocardiography
  dataset", version 1.0.3.
- Page: https://physionet.org/content/ptb-xl/1.0.3/
- License: Open Data Commons Attribution License v1.0 (see the `LICENSE.txt`
  included in the PhysioNet download).
- Citation: Wagner et al., "PTB-XL, a large publicly available
  electrocardiography dataset", Scientific Data, 2020.

Download (no PhysioNet credentialing required for this dataset):

```bash
mkdir -p data
cd data
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
# then flatten physionet.org/files/ptb-xl/1.0.3/* up into data/
```

Expected layout under `data/` (matches the files PhysioNet serves):

```
data/
  ptbxl_database.csv
  scp_statements.csv
  RECORDS
  SHA256SUMS.txt
  LICENSE.txt
  records100/           # 100 Hz signal files (used by this project)
  records500/           # 500 Hz signal files (not used; can be skipped)
```

`src/data.py` reads `data/ptbxl_database.csv`, `data/scp_statements.csv`, and
signal files under `data/records100/` directly -- no renaming needed if you
download straight into `data/`.

## 2. Chapman-Shaoxing (external validation only)

- Source: PhysioNet, "A large scale 12-lead electrocardiogram database for
  arrhythmia study", version 1.0.0.
- Page: https://physionet.org/content/ecg-arrhythmia/1.0.0/
- License: Open Data Commons Attribution License v1.0.
- Citation: Zheng et al., "A 12-lead electrocardiogram database for arrhythmia
  research covering more than 10,000 patients", Scientific Data, 2020.

```bash
mkdir -p data/chapman
cd data/chapman
wget -r -N -c -np https://physionet.org/files/ecg-arrhythmia/1.0.0/
```

`src/data_chapman.py` expects the flattened records plus a
`chapman_metadata.csv` file under `data/chapman/` (built by
`experiments/train_chapman.py` / `experiments/calib_chapman.py` from the raw
PhysioNet metadata -- see the header comments in those scripts for the exact
label-mapping and train/test split logic used).

## Why data isn't included in this repository

Both datasets are already permanently and publicly archived on PhysioNet with
their own DOIs and licenses. Re-hosting them here would duplicate a public
resource, bloat the repository, and create a second copy that could drift out
of sync with the canonical version. Please cite the original dataset papers
(above) in addition to this repository if you use this code.
