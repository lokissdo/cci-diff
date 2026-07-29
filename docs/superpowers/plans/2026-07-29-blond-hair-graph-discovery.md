# Blond-Hair Graph Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing graph-discovery method for `Blond_Hair: 0 -> 1` without rerunning or overwriting smile.

**Architecture:** Introduce immutable task configurations in the existing Kaggle runner, then route cohort selection, screening, interventions, and graph freezing through the selected task. The notebook passes a hair-only task argument and writes to an independent output root.

**Tech Stack:** Python, pytest, nbformat JSON, PyTorch, Grad-CAM++, Diffusers, Kaggle CLI

## Global Constraints

- Preserve all existing smile behavior and artifacts.
- Use `examples/graphs/blond_hair_clean_cci.json`.
- Select not-blond sources and target `Blond_Hair: 0 -> 1`.
- Screen at most four semantic regions.
- Validate two samples before submitting 100 samples.
- Use seed 42 and 35 denoising steps.

---

### Task 1: Task-Aware Discovery Runner

**Files:**
- Modify: `scripts/run_kaggle_smile.py`
- Test: `tests/test_run_kaggle_smile.py`

**Interfaces:**
- Produces: `DiscoveryTask` configuration and `resolve_discovery_task(name)`.
- Consumes: existing cohort selection, screening, shard execution, merge, and graph-freezing helpers.

- [ ] **Step 1: Write failing tests**

Add tests asserting that `blond_hair` resolves to the blond graph, hair output
key, hair candidate regions, and pilot feature `hair`; assert that generated
commands and paths contain `blond_hair` and do not contain the smile template.

- [ ] **Step 2: Verify red**

Run:

```bash
pytest -q tests/test_run_kaggle_smile.py
```

Expected: failures because the task resolver and `--task` argument do not exist.

- [ ] **Step 3: Implement task routing**

Add an immutable task descriptor, `--task` with `smile` and `blond_hair`
choices, task-keyed cohort persistence, and task-specific template, candidate
regions, logging labels, and output paths. Leave evaluation smile-only and
reject `--mode evaluation --task blond_hair`.

- [ ] **Step 4: Verify green**

Run:

```bash
pytest -q tests/test_run_kaggle_smile.py
```

Expected: all tests pass.

### Task 2: Hair-Only Notebook

**Files:**
- Modify: `notebooks/01_global_graph_discovery.ipynb`
- Test: `tests/test_kaggle_notebooks.py`

**Interfaces:**
- Consumes: runner CLI `--task blond_hair`.
- Produces: a notebook parameter `TASK` and task-specific graph summary path.

- [ ] **Step 1: Write failing notebook test**

Assert the notebook defines `TASK = 'blond_hair'`, passes `--task`, and derives
the metrics path from `TASK`.

- [ ] **Step 2: Verify red**

Run:

```bash
pytest -q tests/test_kaggle_notebooks.py
```

Expected: failure because the notebook remains smile-only.

- [ ] **Step 3: Update notebook cells**

Set the standalone discovery notebook to hair-only, pass `--task blond_hair`,
and print hair-specific output paths and labels.

- [ ] **Step 4: Verify green**

Run:

```bash
pytest -q tests/test_kaggle_notebooks.py
```

Expected: all tests pass.

### Task 3: Verification and Kaggle Submission

**Files:**
- Verify: `scripts/run_kaggle_smile.py`
- Verify: `notebooks/01_global_graph_discovery.ipynb`

**Interfaces:**
- Consumes: committed hair-aware runner and notebook.
- Produces: two-sample smoke artifacts and a submitted 100-image Kaggle job.

- [ ] **Step 1: Run focused and full tests**

```bash
pytest -q tests/test_run_kaggle_smile.py tests/test_kaggle_notebooks.py
pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Commit and push**

Commit only files under `cci-diff`, push using the `lokissdo` GitHub identity,
and restore the original local Git configuration.

- [ ] **Step 3: Submit two-sample smoke**

Create a temporary Kaggle kernel payload from the notebook with
`SAMPLE_COUNT = 2`, submit it on two T4 GPUs, and require complete graph
artifacts with zero intervention failures.

- [ ] **Step 4: Submit 100-sample run**

After the smoke passes, submit the hair-only notebook with
`SAMPLE_COUNT = 100` and record the kernel slug for monitoring.

