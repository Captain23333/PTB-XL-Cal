# Data

This project uses two publicly available ECG datasets. Raw waveforms are not
redistributed in this repository because both datasets are already archived on
PhysioNet under their own licenses. We include only the compact, derived
Chapman-Shaoxing metadata needed to reproduce the published external-cohort
analysis.

## 1. PTB-XL (primary dataset, all main experiments)

- Source: PhysioNet, "PTB-XL, a large publicly available electrocardiography
  dataset", version 1.0.3.
- Page: https://physionet.org/content/ptb-xl/1.0.3/
- License: Creative Commons Attribution 4.0 International.
- DOI: https://doi.org/10.13026/kfzx-aw45
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
- License: Creative Commons Attribution 4.0 International.
- DOI: https://doi.org/10.13026/wgex-er52
- Citation: Zheng et al., "A 12-lead electrocardiogram database for arrhythmia
  research covering more than 10,000 patients", Scientific Data, 2020.

```bash
mkdir -p data/chapman
cd data/chapman
wget -r -N -c -np https://physionet.org/files/ecg-arrhythmia/1.0.0/
```

`src/data_chapman.py` expects the flattened records plus the exact
`chapman_metadata.csv` used by the experiment under `data/chapman/`. That
file must contain `record_path`, `split`, and the five binary columns `NORM`,
`MI`, `STTC`, `CD`, and `HYP`.

The release contains the exact experiment inputs:

- `chapman/chapman_metadata.csv`: 44,943 analyzable recordings and the frozen
  train/validation/test assignment (31,460/6,741/6,742);
- `chapman/build_dataset.py`: the 45-code SNOMED-CT-to-family mapping and the
  deterministic recording-level split generator (`numpy` seed 42);
- `chapman/ConditionNames_SNOMED-CT.csv`: the source diagnosis-code dictionary;
- `chapman/corrupt_records.txt`: the nine explicitly excluded record IDs.

After downloading and flattening the PhysioNet files, run:

```bash
python data/chapman/build_dataset.py
```

The frozen metadata shipped here has SHA-256
`a85635d26bfbb91ec476e44567ce1e3eb205483828ccfff0ce480e9a10fc690d`.
The split is a deterministic random **recording-level**, not label-stratified
or patient-level, 70/15/15 split. The released PhysioNet headers used by the
builder expose age, sex, and diagnosis codes but no patient identifier, so this
external-cohort analysis must not be described as a patient-independent split.

## Why raw waveforms aren't included in this repository

Both datasets are already permanently and publicly archived on PhysioNet with
their own DOIs and licenses. Re-hosting them here would duplicate a public
resource, bloat the repository, and create a second copy that could drift out
of sync with the canonical version. Please cite the original dataset papers
(above) in addition to this repository if you use this code.
