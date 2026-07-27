# Grad-CAM++ vs FacePart Attack-Mask Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and run a reproducible matched-mask PGD comparison on five smile-removal examples.

**Architecture:** A standalone evaluation script loads existing generated
artifacts and the frozen CelebA classifier. Pure helpers generate Grad-CAM++
and FacePart masks, execute identical masked PGD, calculate paired metrics,
and materialize tabular and visual artifacts.

**Tech Stack:** Python 3.10, PyTorch, torchvision, grad-cam, Pillow, NumPy.

## Global Constraints

- Do not alter generation or CCI runtime behavior.
- Use Grad-CAM++ at `base_model.layer4[-1]`.
- Threshold normalized saliency at 0.4.
- Compare masks with identical attack settings and inputs.
- Do not commit repository changes.

---

### Task 1: Add tested comparison primitives

**Files:**
- Create: `scripts/compare_attack_masks.py`
- Create: `tests/test_compare_attack_masks.py`

- [x] Write failing tests for binary mask construction, masked PGD locality,
  and paired metric calculation.
- [x] Run the focused tests and verify they fail because the helpers do not
  exist.
- [x] Implement the minimal helpers and CLI validation.
- [x] Run the focused tests and verify they pass.

### Task 2: Run the five-sample experiment

**Files:**
- Generate: `outputs/gradcam_facepart_attack_5/`

- [x] Select five complete smile A9 samples from
  `outputs/clean_cci_component_ablation_10`.
- [x] Generate and save both masks and attacked outputs.
- [x] Write `results.csv`, `summary.json`, and comparison sheets.
- [x] Inspect the images and verify every pair uses the same source artifact.
- [x] Run the complete test suite and report target/locality trade-offs.
