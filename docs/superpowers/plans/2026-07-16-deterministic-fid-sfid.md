# Deterministic FID and sFID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install `pytorch-fid`, compute reproducible FID and symmetric FID for all BLD/CCI 35/50-step smile/hair runs, and publish complete metric tables.

**Architecture:** Add one standalone evaluator that validates paired cohorts, caches InceptionV3 activations, computes standard and deterministic cross-split Frechet distances, and writes CSV/JSON/Markdown. Keep the existing ACE evaluator unchanged and merge its recorded metrics only at report time.

**Tech Stack:** Python 3.10, PyTorch, torchvision, NumPy, SciPy, Pillow, `pytorch-fid`, pytest.

## Global Constraints

- Work only under `/Users/hung.domodec.com/my-docs/cci-diff`.
- Install only into `.venv-ml` through `.venv-ml/bin/python -m pip`.
- Evaluate BLD/CCI at 35/50 steps, separately for 100 smile and 100 hair pairs.
- Use CPU, InceptionV3 2048-dimensional features, split seed 42, and 50/50 splits.
- Preserve both directional sFID values and report their arithmetic mean.
- Treat all 100-image FID/sFID values as exploratory.
- Do not modify the thesis evaluator.
- Do not stage or commit files.

---

### Task 1: Install and Verify `pytorch-fid`

**Files:**
- Modify environment only: `.venv-ml`

**Interfaces:**
- Produces: importable `pytorch_fid` package for Tasks 2-5.

- [ ] **Step 1: Verify interpreter-scoped pip**

Run `.venv-ml/bin/python -m pip --version` and confirm it reports the
`.venv-ml` site-packages path.

- [ ] **Step 2: Install the dependency**

Run `.venv-ml/bin/python -m pip install pytorch-fid`. Expected: installation
succeeds without modifying global Python.

- [ ] **Step 3: Verify imports and versions**

Run `.venv-ml/bin/python -c "import importlib.metadata as m; import pytorch_fid; print(m.version('pytorch-fid'))"`.
Expected: a concrete installed version is printed.

### Task 2: Deterministic Metric Core

**Files:**
- Create: `scripts/evaluate_fid_sfid.py`
- Create: `tests/test_evaluate_fid_sfid.py`

**Interfaces:**
- Produces: `split_indices(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]`.
- Produces: `fid_from_activations(source: np.ndarray, output: np.ndarray) -> float`.
- Produces: `fid_sfid_from_activations(source, output, *, seed) -> dict[str, float | int]`.

- [ ] **Step 1: Write deterministic split tests**

Assert seed 42 returns two disjoint 50-element arrays whose union is
`range(100)`, and repeated calls return equal arrays.

- [ ] **Step 2: Write synthetic Frechet tests**

Use fixed two-dimensional arrays. Assert identical arrays have FID
approximately zero and assert `sfid == (sfid_1 + sfid_2) / 2`.

- [ ] **Step 3: Verify RED**

Run `env PYTHONPATH=. .venv-ml/bin/pytest tests/test_evaluate_fid_sfid.py -q`.
Expected: import failure because the evaluator does not exist.

- [ ] **Step 4: Implement the metric core**

Use `numpy.random.default_rng(seed).permutation(count)` and
`pytorch_fid.fid_score.calculate_frechet_distance`. Reject odd counts, fewer
than four samples, unequal activation shapes, and non-finite activations.

- [ ] **Step 5: Verify GREEN**

Run the focused test command. Expected: all metric-core tests pass.

### Task 3: Cohort Validation and Activation Cache

**Files:**
- Modify: `scripts/evaluate_fid_sfid.py`
- Modify: `tests/test_evaluate_fid_sfid.py`

**Interfaces:**
- Produces: `load_experiment_rows(root: Path) -> dict[str, list[PairRow]]`.
- Produces: `validate_aligned_cohorts(experiments) -> None`.
- Produces: `activation_cache_path(cache_dir, key) -> Path`.
- Produces: `extract_or_load_activations(paths, cache_path, model, ...) -> np.ndarray`.

- [ ] **Step 1: Write validation tests**

Build temporary `ace_pair_metrics.csv` fixtures and assert acceptance of exactly
100 unique smile and 100 unique hair rows. Assert clear `ValueError` messages
for duplicate IDs, 99 rows, missing images, and cross-run cohort mismatch.

- [ ] **Step 2: Write cache tests**

Write a valid `.npz` cache containing paths, file sizes, modification times,
feature dimension, and activations. Assert a matching cache is reused and a
changed path fingerprint forces extraction.

- [ ] **Step 3: Verify RED**

Run the focused tests. Expected: failures for missing validation/cache APIs.

- [ ] **Step 4: Implement validation and extraction**

Use `pytorch_fid.inception.InceptionV3.BLOCK_INDEX_BY_DIM[2048]` and
`pytorch_fid.fid_score.get_activations`. Sort each task by numeric sample ID.
Cache source activations once per task and output activations once per
method/steps/task.

- [ ] **Step 5: Verify GREEN**

Run focused tests. Expected: all validation and cache tests pass.

### Task 4: CLI, Metric Reports, and Full Tables

**Files:**
- Modify: `scripts/evaluate_fid_sfid.py`
- Modify: `tests/test_evaluate_fid_sfid.py`

**Interfaces:**
- CLI inputs: four experiment roots, `--output-dir`, `--seed 42`,
  `--batch-size`, `--num-workers`, and `--device cpu`.
- Produces: `fid_sfid_metrics.csv`, `fid_sfid_metrics.json`,
  `fid_sfid_comparison.md`, `full_metrics.csv`, and `full_metrics.md`.

- [ ] **Step 1: Write report tests**

Feed synthetic metric rows and assert all eight method/steps/task combinations
appear. Assert `full_metrics.csv` includes method, steps, task, N, FID, sFID,
both directional sFID values, target accuracy, directional FR,
same-classifier FR at 0.5, strong target rate at 0.8, desired probability,
FVA, FS, MNAC, CD, locality fields, and median runtime.

- [ ] **Step 2: Verify RED**

Run focused tests. Expected: missing CLI/report behavior failures.

- [ ] **Step 3: Implement reports**

Read existing `ace_metrics.json`, `ace_pair_metrics.csv`, and
`pilot_summary.json`. Calculate true directional FR from `directional_flip`,
same-classifier FR at desired probability `>= 0.5`, and strong target rate at
`>= 0.8`. Label target-state accuracy separately so it is not called FR.

- [ ] **Step 4: Verify GREEN**

Run focused tests and inspect temporary Markdown output. Expected: all report
tests pass and no metric label conflates target accuracy with directional FR.

### Task 5: Real Evaluation and Verification

**Files:**
- Produce: `outputs/fid_sfid_bld_cci_steps35_50/*`

**Interfaces:**
- Consumes all APIs from Tasks 1-4.
- Produces the final eight-row FID/sFID and full metric tables.

- [ ] **Step 1: Run the complete test suite**

Run `env PYTHONPATH=. .venv-ml/bin/pytest -q`. Expected: all tests pass.

- [ ] **Step 2: Run the evaluator**

Run the CLI with the four approved experiment roots, CPU, seed 42, and output
directory `outputs/fid_sfid_bld_cci_steps35_50`.

- [ ] **Step 3: Verify output completeness**

Assert eight metric rows, four configurations, two tasks per configuration,
100 samples per row, finite FID/sFID values, and existing activation caches.

- [ ] **Step 4: Verify cache reproducibility**

Run the identical CLI a second time. Compare first and second CSV/JSON metric
values byte-for-byte and verify cache files were not rewritten.

- [ ] **Step 5: Review the final tables**

Check that lower FID/sFID values are not described as conclusive with only 100
samples and that the final recommendation prioritizes true directional FR.

- [ ] **Step 6: Confirm repository state**

Run `git status --short` and `git diff --cached --quiet`. Expected: evaluator,
tests, spec, and plan remain unstaged; no commit exists.
