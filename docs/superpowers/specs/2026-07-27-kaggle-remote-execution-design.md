# Kaggle Remote Execution Design

## Purpose

Run the two CCI notebooks from the local workspace through the authenticated
Kaggle account `a210462khihng`, without manually importing notebooks or local
assets in the Kaggle UI.

## Data Sources

- Face images and semantic masks:
  `ipythonx/celebamaskhq`.
- Diffusion model:
  `sd2-community/stable-diffusion-2-1`, downloaded by Diffusers in the Kaggle
  runtime with notebook internet enabled.
- Private evaluator assets:
  upload the contents of local `models/` as
  `a210462khihng/cci-assets`.
- Project source:
  clone public repository `https://github.com/lokissdo/cci-diff.git` at the
  exact commit injected by the launcher.

No Hugging Face, GitHub, or Kaggle token is written to notebook source,
metadata, logs, or uploaded datasets.

## Remote Runs

Notebook 1 is pushed privately with an Nvidia T4 accelerator. Its metadata
attaches the public CelebAMask-HQ dataset and the private evaluator dataset.
The notebook clones the pinned public GitHub revision to
`/kaggle/working/cci-diff` and runs graph discovery.

Notebook 2 is independent of Notebook 1. It assumes the reviewed region policy
`mouth + upper_lip + lower_lip` for smile removal, selects its own eligible
cohort, and runs the raw-BLD, fixed-equal CCI, and adaptive CCI comparison. It
does not claim cohort disjointness unless an exclusion manifest is supplied in
a separate experiment.

Both remote notebooks currently run smile removal only. Their experiment
entry points execute inline in the notebook process through `runpy`, and
unbuffered timestamped messages expose stage progress in the active cell.
Repository cloning and dependency installation remain external setup commands.

## Local Interface

`scripts/run_kaggle_two_stage.py` supports:

- preparing the private model dataset;
- creating or versioning that dataset;
- preparing kernel metadata;
- pushing either notebook independently or both sequentially;
- polling terminal status;
- downloading outputs to `outputs/kaggle_remote`;
- `--prepare-only`, `--start-at`, and `--no-wait` controls.

All created Kaggle resources are private. Dataset, kernel, repository, and
commit references are explicit and deterministic. Existing datasets and
kernels are versioned rather than duplicated.

## Failure Handling

The launcher exits before upload when authentication, required local model
files, or notebook files are missing. It does not start Notebook 2 if Notebook
1 fails. Failed kernel status and log retrieval commands are printed without
including credentials. Re-running the launcher updates the same private
resources and resumes through the notebooks' existing artifact checks.
