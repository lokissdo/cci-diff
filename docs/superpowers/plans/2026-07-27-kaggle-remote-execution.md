# Kaggle Remote Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload the local CCI evaluator assets and project source, then run the two-stage CCI notebooks remotely through the authenticated Kaggle account.

**Architecture:** A tested Python launcher prepares the private evaluator dataset, pins the public GitHub source revision, emits deterministic kernel metadata, invokes the official Kaggle CLI, and downloads outputs. Each notebook clones that revision, resolves attached input paths, and uses the public Hugging Face SD2 model identifier; Notebook 2 assumes a fixed region policy and can run independently of Notebook 1.

**Tech Stack:** Python 3.10+, Kaggle CLI 2.2, Jupyter notebooks, pytest, Kaggle T4, Hugging Face Diffusers.

## Global Constraints

- Never upload or print Kaggle or Hugging Face tokens.
- Keep every created Kaggle dataset and kernel private.
- Use `ipythonx/celebamaskhq` for images and semantic masks.
- Use `sd2-community/stable-diffusion-2-1` for diffusion.
- Upload the contents of local `models/` as evaluator assets.
- Do not create a git commit.

---

### Task 1: Evaluator Packaging and Metadata

**Files:**
- Create: `src/cci_diff/kaggle_remote.py`
- Test: `tests/test_kaggle_remote.py`

**Interfaces:**
- Produces `prepare_model_dataset`, `kernel_metadata`, and terminal-status
  parsing helpers.
- Rejects missing evaluator model files.

- [ ] Write failing packaging and metadata tests.
- [ ] Run `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_kaggle_remote.py`.
- [ ] Implement deterministic package preparation and metadata generation.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Remote Launcher

**Files:**
- Create: `scripts/run_kaggle_two_stage.py`
- Test: `tests/test_kaggle_remote_cli.py`

**Interfaces:**
- CLI uses `.venv-kaggle/bin/kaggle` by default.
- Creates or versions `cci-assets`.
- Pushes `cci-global-graph-discovery` and
  `cci-raw-bld-fixed-adaptive` as independent kernels.
- Downloads completed outputs beneath `outputs/kaggle_remote`.

- [ ] Write failing command-construction and sequencing tests.
- [ ] Run the focused CLI test and verify failure.
- [ ] Implement prepare, upload, push, poll, and output download operations.
- [ ] Re-run the focused CLI tests and confirm they pass.

### Task 3: Kaggle-Aware Notebooks

**Files:**
- Modify: `notebooks/01_global_graph_discovery.ipynb`
- Modify: `notebooks/02_full_cci_fixed_vs_adaptive.ipynb`
- Modify: `tests/test_kaggle_notebooks.py`

**Interfaces:**
- Clone a launcher-injected Git commit from
  `https://github.com/lokissdo/cci-diff.git`.
- Resolve `/kaggle/input/cci-assets`.
- Use `sd2-community/stable-diffusion-2-1`.
- Notebook 2 has no kernel-source dependency and assumes the reviewed smile
  region policy.
- Both notebooks run smile removal only and execute experiment entry points
  inline with unbuffered, timestamped progress output.

- [ ] Add failing notebook-contract assertions.
- [ ] Modify configuration/bootstrap cells.
- [ ] Compile every notebook code cell.
- [ ] Run notebook tests and confirm they pass.

### Task 4: Verification and Remote Start

**Files:**
- Modify: `README.md`

- [ ] Document authentication, preparation, launch, status, and output paths.
- [ ] Run focused tests.
- [ ] Run the complete test suite.
- [ ] Run launcher `--prepare-only` and inspect generated metadata.
- [ ] Upload/version the two private datasets.
- [ ] Push Notebook 1 on Nvidia T4 and report its remote reference and status.
- [ ] Confirm no staged changes and no new commit.
