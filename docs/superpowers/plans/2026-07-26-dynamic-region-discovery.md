# Dynamic Region Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed top-three region screening with label-dependent
minimal-area saliency coverage and stop intervention discovery at the first
successful region-set cardinality.

**Architecture:** Add pure subset scoring and selection to
`region_screening.py`, expose it through the screening CLI, and add
cardinality-level stopping to the resumable intervention runner. The shell
wrapper consumes manifests rather than assuming a candidate count or expected
row count.

**Tech Stack:** Python 3.10, NumPy, Pillow, PyTorch, pytest, Bash.

## Global Constraints

- Work in `/Users/hung.domodec.com/my-docs/cci-diff`.
- Do not commit or stage files.
- Do not hardcode target-specific region identities.
- Do not fix candidate count across labels.
- Use source images only during screening.
- Require complete discovery-mask coverage for selected candidates.
- Use the generation classifier at threshold `0.5` for discovery flip rate.
- Preserve resumability and same-seed paired comparisons.
- Disable post-generation attacks during discovery.

---

### Task 1: Dynamic Saliency-Covering Subset

**Files:**
- Modify: `src/cci_diff/region_screening.py`
- Modify: `scripts/screen_counterfactual_regions.py`
- Modify: `tests/test_region_screening.py`
- Modify: `tests/test_counterfactual_graph_cli.py`

**Interfaces:**
- Produces `RegionSubsetEvidence` and
  `select_saliency_covering_regions(samples, ...)`.
- Screening manifest exposes `selected_candidate_regions`,
  `selection_status`, and `subset_metrics`.

- [ ] Write a failing synthetic test where one broad region and two compact
  regions meet equal coverage, and assert the compact pair wins by area.
- [ ] Run the focused tests and verify the missing interface fails.
- [ ] Implement exact subset enumeration over globally available regions,
  union saliency coverage, union area, passing/fallback selection, and strict
  threshold validation.
- [ ] Update the CLI to retain per-image saliency/masks, write subset metrics,
  and remove `--select_top_k`.
- [ ] Run screening tests and verify they pass.

### Task 2: Progressive Intervention Stopping

**Files:**
- Modify: `scripts/run_counterfactual_region_interventions.py`
- Modify: `tests/test_counterfactual_region_interventions.py`

**Interfaces:**
- New CLI option `--stop_flip_rate`.
- Manifest fields: `planned_region_sets`, `executed_region_sets`,
  `cardinality_results`, `stop_reason`, and `execution_complete`.

- [ ] Write a failing orchestration test where singleton FR is insufficient
  and pair FR reaches the threshold; assert triples are not executed.
- [ ] Reorder execution by cardinality while preserving same-seed audit
  reuse and incremental CSV writes.
- [ ] Compute stopping eligibility only from complete region sets.
- [ ] Record completion and stopping provenance.
- [ ] Run intervention-runner tests and verify they pass.

### Task 3: One-Command Wrapper

**Files:**
- Modify: `scripts/run_smile_graph_individual_100.sh`
- Modify: `tests/test_run_smile_graph_individual_100_sh.py`
- Modify: `README.md`

**Interfaces:**
- Wrapper reads selected regions from `screening_manifest.json`.
- Wrapper reads `execution_complete` rather than expecting `100*(2^K-1)`
  rows.

- [ ] Write a failing wrapper test rejecting `SCREEN_TOP_K` and requiring
  saliency/frequency thresholds plus `--stop_flip_rate`.
- [ ] Replace top-K environment settings with
  `SALIENCY_COVERAGE_THRESHOLD`, `SALIENCY_COHORT_FREQUENCY`, and
  `MAX_SELECTED_REGIONS`.
- [ ] Pass the dynamic screening and stopping options through both CLIs.
- [ ] Update help and README with the dynamic run-count explanation.
- [ ] Run shell syntax and wrapper tests.

### Task 4: Verification

**Files:**
- No production changes expected.

- [ ] Run a ten-image CPU screening smoke and inspect selected set/status.
- [ ] Run all focused graph, screening, runner, and wrapper tests.
- [ ] Run the complete pytest suite.
- [ ] Inspect git status and confirm no commit or staging occurred.
