# Kaggle Two-Stage CCI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dynamic, maximum-four-region graph-discovery pipeline and a
Kaggle-ready fixed-versus-adaptive full CCI pipeline with no allowed-change
role.

**Architecture:** Pure selection logic remains in `src/cci_diff`, while
scripts own resumable orchestration and artifact manifests. Two notebooks
configure and invoke those scripts without duplicating model or diffusion
logic.

**Tech Stack:** Python 3.10, PyTorch, diffusers, NumPy, Pillow, Jupyter,
pytest, CUDA.

## Global Constraints

- Do not commit or stage files.
- Cap screened candidates and intervention cardinality at four.
- Disable post-generation attacks during graph discovery.
- Use generation-classifier threshold 0.5 for raw discovery flips.
- Keep discovery and evaluation cohorts disjoint.
- Default Kaggle evaluation to 300 images per task.
- Load all models and datasets from user-configured local paths.

---

### Task 1: Dynamic Region Discovery

**Files:**
- Modify: `src/cci_diff/region_screening.py`
- Modify: `scripts/screen_counterfactual_regions.py`
- Modify: `scripts/run_counterfactual_region_interventions.py`
- Modify: `src/cci_diff/counterfactual_graph.py`
- Test: `tests/test_region_screening.py`
- Test: `tests/test_counterfactual_region_interventions.py`
- Test: `tests/test_counterfactual_graph.py`

**Interfaces:**
- Produce `select_saliency_covering_regions(..., max_regions=4)`.
- Produce cardinality-aware manifest fields `cardinality_results`,
  `stop_reason`, and `execution_complete`.
- Select unsupported fallbacks by target effect before minimal-change costs.

- [ ] Add failing tests for dynamic selection, four-region validation,
  target-effect-first fallback, and resumable cardinality manifests.
- [ ] Run focused tests and confirm the new expectations fail.
- [ ] Implement source-only subset selection and progressive cardinality
  execution with exact manifest provenance.
- [ ] Run focused tests and confirm they pass.

### Task 2: Remove Allowed Changes

**Files:**
- Modify: `src/cci_diff/concept_graph.py`
- Modify: `src/cci_diff/compilers/json_graph.py`
- Modify: `scripts/run_sd2_bld_cci.py`
- Modify: `examples/graphs/remove_smile_clean_cci.json`
- Modify: `examples/graphs/blond_hair_clean_cci.json`
- Modify: `paper/cci_conference_v1.tex`
- Test: `tests/test_concept_graph.py`
- Test: `tests/test_json_graph_compiler.py`
- Test: `tests/test_clean_cci_cli.py`

**Interfaces:**
- Accept only `target`, `constraint`, and `audit_only` node roles.
- Compute non-target drift by excluding only the target index.

- [ ] Add failing tests that reject `allowed_change` and include every
  non-target classifier attribute in drift.
- [ ] Remove the role, relation requirement, compiled-plan field, and metric
  exemption.
- [ ] Update example graphs and paper wording.
- [ ] Run graph/compiler/CLI tests and confirm they pass.

### Task 3: Explicit End-to-End Controller Modes

**Files:**
- Modify: `scripts/run_clean_cci_pilot.py`
- Create: `scripts/run_cci_end_to_end.py`
- Test: `tests/test_run_cci_end_to_end.py`

**Interfaces:**
- CLI `--controller_modes disabled fixed_equal feedback`.
- CLI `--features smile hair`.
- CLI path overrides for image root, mask root, SD2, classifier, identity,
  output root, sample count, seed, and device.
- Output one manifest and paired metric CSV keyed by feature, sample, and
  controller mode.

- [ ] Add failing command-construction and cohort-disjointness tests.
- [ ] Implement a resumable orchestrator using existing pilot command and
  audit extraction helpers.
- [ ] Run focused orchestration tests and confirm they pass.

### Task 4: Kaggle Notebooks

**Files:**
- Create: `notebooks/01_global_graph_discovery.ipynb`
- Create: `notebooks/02_full_cci_fixed_vs_adaptive.ipynb`
- Test: `tests/test_kaggle_notebooks.py`

**Interfaces:**
- First notebook invokes screening, interventions, and graph freezing for both
  targets with maximum four regions.
- Second notebook invokes raw BLD, fixed CCI, and adaptive CCI for 300
  disjoint images per task using fixed independent masks.

- [ ] Add failing notebook-schema tests for configuration, CUDA defaults,
  path validation, task definitions, and controller modes.
- [ ] Add concise notebooks with resumable command cells and result-display
  cells.
- [ ] Run notebook-schema tests and confirm they pass.

### Task 5: Verification

**Files:**
- Modify: `README.md`

- [ ] Document the two notebook workflow and imported Kaggle assets.
- [ ] Run focused tests for discovery, graph parsing, orchestration, and
  notebooks.
- [ ] Run the complete test suite.
- [ ] Inspect notebook JSON and Python syntax.
- [ ] Confirm git status contains no staged changes and no new commit.
