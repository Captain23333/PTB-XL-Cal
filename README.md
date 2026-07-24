# PTB-XL-Cal: Multi-label Calibration Benchmark for ECG Classification

Code release accompanying the paper "PTB-XL-Cal: A Calibration Benchmark for
Multi-Label ECG Classification, with a Prevalence-Associated Failure Mode".

This open-source package contains code, compact derived metadata, frozen PTB-XL
split files, and aggregate results. Raw waveforms and trained model checkpoints
are *not* included (see [Data](#data) and
[What's not included](#whats-not-included) below for why, and how to get them).

## What this benchmarks

Six post-hoc probability calibrators (C0 raw, C1 single-scalar temperature,
C2 per-label temperature, C3 per-label affine, C4 family-pooled affine, C5
per-label isotonic regression) were applied to eight SimpleResNet1D training
methods and an ECG-FM linear probe (150 primary evaluations). A hard-BCE
InceptionTime check added 18 architecture-transfer evaluations; it was not
run across all training methods. Chapman-Shaoxing was used only for the
external prevalence-associated check. Full protocol details are in the paper;
the locked metric suite and selection rule are implemented in
`src/calibration_metrics.py` and `src/calib_pipeline.py`.

## Repository layout

```
src/                    Core library
  data.py               PTB-XL dataset loader
  data_chapman.py       Chapman-Shaoxing dataset loader (external validation)
  models.py             SimpleResNet1D / InceptionTime backbones
  train.py              Training loop
  losses.py             Loss functions (hard-BCE, ASL, focal, label-smooth, etc.)
  baselines.py          CHMC-projection and other baseline-specific logic
  calibrators.py        The six post-hoc calibrators (C0..C5)
  calibration_metrics.py  Locked metric suite (ECE, Brier, slope/intercept,
                           threshold-transfer gap, patient-paired bootstrap)
  calib_pipeline.py     Selector + calibrator factory + clinical-family map
  evaluate.py           Evaluation-loop helpers

experiments/            One script per experiment block (see file headers for
                         what each one produces); *.sh are driver scripts for
                         running training/eval on a GPU box

eval_calibration.py     Release entry point: fit calibrators on val, evaluate
                         on test, emit a leaderboard-schema JSON

splits/                 Frozen PTB-XL fold assignments (val_fold9, test_fold10)
results/                Small summary CSV/JSON outputs (one row per
                         method x seed x calibrator x metric); this is what the
                         paper's tables and figures are built from. One compact
                         hard-BCE prediction fixture is included for self-test.
data/                   Download instructions plus frozen Chapman metadata,
                        mapping, exclusions, and split generator
```

## Environment

```bash
pip install -r requirements.txt
```

The reconstructed release environment uses Python 3.10 and the pinned package
versions in `requirements.txt`, including PyTorch 2.4.0. The evaluation
self-test is CPU-compatible; full model training can use a compatible CUDA
installation.

Optionally set these environment variables if you don't want to run from the
repository root:

```bash
export PTBXL_CAL_ROOT=/path/to/this/repo   # else inferred from __file__
export PTBXL_CAL_PYTHON=/path/to/python    # used by the *.sh driver scripts
```

## Data

Download PTB-XL and (optionally, for the external-validation experiments)
Chapman-Shaoxing from PhysioNet -- both are public, no special credentialing
needed. Full instructions, expected directory layout, and licensing/citation
requirements: [`data/README.md`](data/README.md).

## Reproducing the results

1. Download PTB-XL into `data/` (see above).
2. Train the base methods (this recreates the model checkpoints, which are
   **not** shipped in this release -- see below):
   ```bash
   bash experiments/run_block_a.sh          # 5 base methods x 3 seeds, SimpleResNet1D
   python experiments/train_inceptiontime_all_methods.py   # architecture-transfer check
   ```
3. Recompute val/test logits from the trained checkpoints:
   ```bash
   python experiments/calib_recompute_probs.py
   ```
   This populates `results/cal_probs/`, which is excluded from this release
   (regenerable, and large -- roughly 130 MB of per-method-seed logits).
4. Run the calibration/analysis blocks:
   ```bash
   python experiments/calib_block1_q1.py
   python experiments/calib_block2_q2.py
   python experiments/calib_block3_q3.py
   python experiments/calib_block4_threshold.py
   python experiments/calib_make_figures.py
   ```
5. Or use the single release entry point once `results/cal_probs/*.npz` exist:
   ```bash
   python eval_calibration.py \
       --predictions 'results/cal_probs/probs_*.npz' \
       --val-split splits/val_fold9.csv \
       --test-split splits/test_fold10.csv \
       --leaderboard leaderboard_worked_example.json
   ```
6. Self-test (no raw dataset download required):
   ```bash
   python eval_calibration.py --ci-test
   ```

For the Chapman-Shaoxing external validation and the InceptionTime /
prevalence-ablation / ECE-robustness / selector-stability supplementary
analyses, see `experiments/calib_chapman.py`, `experiments/train_chapman.py`,
`experiments/calib_inceptiontime*.py`, `experiments/calib_prevalence_ablation.py`,
`experiments/calib_ece_robustness.py`, and `experiments/calib_selector_stability.py`
respectively -- each has a docstring describing exactly what it computes.
The exact Chapman experiment metadata, 45-code mapping, exclusions, deterministic
split generator, and frozen 31,460/6,741/6,742 recording-level assignment are
included under `data/chapman/`. See `data/README.md` for the raw-waveform
download and the important limitation that this is not a patient-level split.

## What's not included

- **Raw waveform datasets**: public PhysioNet resources, not re-hosted here
  (see `data/README.md`). Compact derived Chapman metadata is included.
- **Model checkpoints** (`checkpoints/`): not included due to size; retrain
  with `experiments/run_block_a.sh` / the InceptionTime training scripts.
- **Most per-(method, seed) prediction logits** (`results/cal_probs/*.npz`,
  ~130 MB in the full working tree):
  regenerable from checkpoints via `experiments/calib_recompute_probs.py`;
  excluded to keep this repository small. The sole exception is
  `results/cal_probs/probs_hard_bce_seed42.npz` (about 1.6 MB), included as the
  worked-example self-test fixture. The small aggregated result tables
  (`results/*.csv`, `results/*.json`) that feed directly into the paper's
  tables and figures are also included.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
This does not apply to the third-party datasets described in
`data/README.md`, which retain their own licenses.

## Citation

If you use this code, please cite the accompanying paper (citation details to
be added on publication) as well as the original PTB-XL and Chapman-Shaoxing
dataset papers referenced in `data/README.md`.
