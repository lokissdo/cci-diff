# Heatmap Ranking And Mask Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rank original semantic regions by robust Grad-CAM++ intensity and
skip diffusion interventions whose cohort union masks are exactly equivalent.

**Architecture:** Extend screening summaries with robust intensity, captured
saliency, and area statistics, then apply an explicit eligibility gate and
lexicographic ranking. Add a pure cohort-level union-signature planner to the
intervention runner so generation executes one canonical set per unique mask
treatment while preserving aliases in provenance.

**Tech Stack:** Python 3.10, NumPy, Pillow, hashlib, pytest, Bash.

## Global Constraints

- Work in `/Users/hung.domodec.com/my-docs/cci-diff`.
- Do not commit or stage files.
- Preserve original CelebAMask-HQ semantic masks, including full `skin`.
- Do not create a residual-skin mask.
- Skip only exact union-mask equivalence, never a strict superset.
- Preserve existing audit artifacts and resumability.
- Keep post-generation attacks disabled during graph discovery.

---

### Task 1: Robust Heatmap-Intensity Ranking

**Files:**
- Modify: `scripts/screen_counterfactual_regions.py`
- Modify: `src/cci_diff/region_screening.py`
- Modify: `tests/test_counterfactual_graph_cli.py`
- Modify: `tests/test_region_screening.py`

**Interfaces:**
- `aggregate_screening_rows(rows, sample_count)` adds median/mean captured
  mass, region density, and mask fraction.
- `select_screened_regions(summary, top_k,
  minimum_coverage_frequency, minimum_captured_saliency)` returns region names.

- [x] Add a failing test where broad `skin` has greater captured mass but
  lower median heatmap intensity than mouth and both lips; assert the three
  perioral regions are selected.
- [x] Run focused tests and verify the selector signature/ranking fails.
- [x] Implement robust aggregation and eligibility validation.
- [x] Sort eligible regions by median intensity, captured mass, availability,
  median area, and canonical name.
- [x] Add `--minimum_captured_saliency` to the screening CLI and manifest.
- [x] Run focused screening tests.

### Task 2: Cohort Union-Mask Deduplication

**Files:**
- Modify: `scripts/run_counterfactual_region_interventions.py`
- Modify: `tests/test_counterfactual_region_interventions.py`

**Interfaces:**
- `deduplicate_region_sets(region_sets, sample_ids, mask_root)` returns
  `(canonical_sets, aliases, signatures)`.
- `aliases` maps each skipped tuple to its canonical tuple.

- [x] Add a failing synthetic test where `skin` contains `mouth`, proving
  `skin + mouth` aliases to `skin`, while `mouth + lip` remains unique.
- [x] Run the focused test and verify the interface is absent.
- [x] Implement exact ordered per-image union comparison, SHA-256 provenance,
  and canonical representative selection.
- [x] Integrate canonical sets into orchestration and record requested sets,
  canonical sets, aliases, and signatures in the manifest.
- [x] Verify only canonical commands execute and existing canonical audits
  resume.
- [x] Run intervention-runner tests.

### Task 3: One-Command Wrapper

**Files:**
- Modify: `scripts/run_smile_graph_individual_100.sh`
- Modify: `tests/test_run_smile_graph_individual_100_sh.py`
- Modify: `README.md`

**Interfaces:**
- New environment variable `MINIMUM_CAPTURED_SALIENCY`, default `0.02`.
- A dry planning pass reads canonical set count from the intervention
  manifest before calculating expected discovery rows.

- [x] Add a failing wrapper test requiring the captured-saliency argument and
  manifest-derived canonical set count.
- [x] Pass the new screening threshold.
- [x] Run a dry intervention planning pass and calculate expected rows from
  `region_sets`.
- [x] Update help and README.
- [x] Run shell syntax and wrapper tests.

### Task 4: Verification

**Files:**
- No production changes expected.

- [x] Recompute the ranking from the existing 100-image screening rows and
  confirm the selected order is `mouth`, `lower_lip`, `upper_lip`.
- [x] Dry-plan the current cohort and confirm skin-containing aliases collapse
  to `skin` when skin is in the candidate pool.
- [x] Run all focused tests.
- [x] Run the complete pytest suite.
- [x] Confirm no files were committed or staged.
