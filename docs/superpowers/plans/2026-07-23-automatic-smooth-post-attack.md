# Automatic Smooth Post-Attack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in automatic smooth classifier-boundary correction to clean CCI runs while preserving the original CCI-BLD output and recording a separate corrected artifact.

**Architecture:** Move reusable attack primitives into `src/cci_diff/post_attack.py`, add tested grid and metric helpers there, and let `run_clean` orchestrate source saliency, per-candidate correction, PNG serialization, and audit metadata. Existing experimental scripts import the production module so standalone and automatic paths share one implementation.

**Tech Stack:** Python 3.10, PyTorch, torchvision, grad-cam, Pillow, NumPy.

## Global Constraints

- Preserve `sd2_bld_grid.png` byte-for-byte.
- Save corrected output as `sd2_bld_grid_corrected.png`.
- Default `--cci_post_attack` to `none`.
- Never fall back to historical sign-PGD.
- Keep clean-CCI version 1 `batch_size=1` validation; grid helpers must support multiple candidates.
- Compute final target and preservation metrics from reloaded PNG crops.
- Do not create a Git commit.

---

### Task 1: Productionize the smooth attack

**Files:**
- Create: `src/cci_diff/post_attack.py`
- Modify: `scripts/smooth_boundary_attack.py`
- Create: `tests/test_post_attack.py`

**Interfaces:**
- Produces: `soft_anatomical_mask`, `smooth_masked_gradient`, `refine_boundary`, `targeted_smooth_boundary_attack`, `gradcam_pp_saliency`, `split_horizontal_grid`, and `join_horizontal_grid`.

- [x] Write failing tests importing attack primitives from `cci_diff.post_attack`, checking grid round-trip order, and checking invalid grid widths.
- [x] Run `PYTEST_ADDOPTS='-p no:cacheprovider' .venv-ml/bin/python -m pytest tests/test_post_attack.py -q` and verify missing-module failures.
- [x] Move the tested primitive implementation into `src/cci_diff/post_attack.py`, add Grad-CAM++ and grid helpers, and replace `scripts/smooth_boundary_attack.py` with compatibility imports.
- [x] Run `tests/test_post_attack.py` and `tests/test_smooth_boundary_attack.py`; expect all tests to pass.

### Task 2: Add CLI ownership and validation

**Files:**
- Modify: `scripts/run_sd2_bld_cci.py`
- Modify: `tests/test_clean_cci_cli.py`

**Interfaces:**
- Produces parser fields `cci_post_attack`, `cci_post_attack_epsilon`, `cci_post_attack_step_size`, `cci_post_attack_max_steps`, `cci_post_attack_boundary_margin`, `cci_post_attack_boundary_steps`, `cci_post_attack_gaussian_kernel_size`, and `cci_post_attack_gaussian_sigma`.

- [x] Write failing tests for defaults, accepted smooth-boundary settings, invalid numerical settings, and rejection in legacy mode.
- [x] Run the focused CLI tests and confirm they fail on missing parser fields.
- [x] Add parser options and validate mode ownership, positive budgets/steps/sigma, odd positive Gaussian kernel size, and margin in `[0, 0.5)`.
- [x] Run `PYTEST_ADDOPTS='-p no:cacheprovider' .venv-ml/bin/python -m pytest tests/test_clean_cci_cli.py -q`; expect all tests to pass.

### Task 3: Add automatic correction and audit

**Files:**
- Modify: `src/cci_diff/post_attack.py`
- Modify: `scripts/run_sd2_bld_cci.py`
- Modify: `tests/test_post_attack.py`
- Modify: `tests/test_clean_cci_cli.py`

**Interfaces:**
- Produces `run_clean_post_attack(...) -> dict[str, Any] | None`.
- Writes `post_attack_soft_mask.png` and `sd2_bld_grid_corrected.png`.
- Adds `audit["cci"]["post_attack"]`.

- [x] Write failing unit tests proving disabled mode performs no work, raw output is unchanged, already-passing crops are copied, failed crops invoke the attack, corrected grid order is retained, and saved-PNG scores populate audit records.
- [x] Run focused tests and verify behavioral failures.
- [x] Implement soft-mask preparation from the source and semantic mask, per-candidate correction, corrected-grid serialization, reload-based classifier scoring, identity cosine, perturbation metrics, and per-candidate records.
- [x] Call `run_clean_post_attack` after raw post-run metrics but before writing `audit.json`; keep `run_clean` return value unchanged.
- [x] Run focused post-attack and clean-run tests; expect all tests to pass.

### Task 4: Verify standalone parity and repository health

**Files:**
- Modify: `scripts/compare_attack_masks.py`
- Modify: `scripts/compare_smooth_boundary_attack.py`
- Generate: `outputs/smooth_boundary_attack_10/`

**Interfaces:**
- Standalone comparison imports the production Grad-CAM++ and attack helpers.

- [x] Update experimental scripts to import shared production functions and rerun their focused tests.
- [x] Rerun the ten-image standalone comparison over existing A9 PNGs without regenerating diffusion outputs.
- [x] Confirm target pass rate remains `8/10`, no outside-FacePart leakage is introduced, and representative comparison images remain visually unchanged.
- [x] Run `PYTEST_ADDOPTS='-p no:cacheprovider' .venv-ml/bin/python -m pytest -q`; expect no failures.
- [x] Run scoped `git diff --check`; expect no output.
