# Minimal Smooth Boundary Attack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare ten existing CCI-BLD smile-removal images before and after a minimal smooth classifier attack targeted only to the `0.5` decision boundary.

**Architecture:** Add pure attack and metric helpers in a focused experimental module, then add a CLI that reuses existing A9 artifacts without rerunning diffusion. Each attacked PNG is reloaded before classifier, identity, locality, and residual metrics are recorded.

**Tech Stack:** Python 3.10, PyTorch, torchvision, grad-cam, Pillow, NumPy.

## Global Constraints

- Preserve historical PGD, diffusion, BLD, CCI, classifier, and evaluator behavior.
- Use the same classifier checkpoint and label index for paired before/after scoring.
- Use `epsilon=0.05`, `step_size=0.005`, target margin `0.01`, Gaussian kernel size `5`, and sigma `1.0`.
- Never change pixels outside the FacePart mouth-and-lips mask.
- Score the exact saved and reloaded PNG.
- Use all ten complete A9 samples: `0, 1, 3, 9, 14, 20, 24, 25, 27, 31`.
- Do not create a Git commit.

---

### Task 1: Add smooth attack primitives

**Files:**
- Create: `scripts/smooth_boundary_attack.py`
- Create: `tests/test_smooth_boundary_attack.py`

**Interfaces:**
- Produces: `soft_anatomical_mask(facepart_mask, saliency)`, `smooth_masked_gradient(gradient, mask, kernel_size, sigma, eps)`, `refine_boundary(model, failed, passed, label_index, desired_value, threshold, margin, max_steps)`, and `targeted_smooth_boundary_attack(...)`.

- [x] **Step 1: Write failing support, projection, and boundary tests**

```python
def test_soft_mask_has_no_support_outside_facepart():
    facepart = torch.tensor([[[[1.0, 0.0]]]])
    saliency = torch.tensor([[[[0.4, 1.0]]]])
    assert soft_anatomical_mask(facepart, saliency).tolist() == [[[[0.4, 0.0]]]]

def test_smooth_attack_respects_mask_and_epsilon():
    attacked, record = targeted_smooth_boundary_attack(
        ToyClassifier(), image, mask, label_index=0, desired_value=0,
        epsilon=0.05, step_size=0.005, max_steps=100,
    )
    assert torch.max(torch.abs(attacked - image)) <= 0.05 + 1e-6
    assert attacked[..., 1].equal(image[..., 1])
    assert "boundary_iterations" in record

def test_boundary_refinement_returns_smallest_successful_segment_point():
    refined, record = refine_boundary(
        ToyClassifier(), failed, passed, label_index=0, desired_value=0,
        threshold=0.5, margin=0.01, max_steps=12,
    )
    assert record["after_probability"] <= 0.49
    assert torch.linalg.vector_norm(refined - failed) < torch.linalg.vector_norm(passed - failed)
```

- [x] **Step 2: Run the focused tests and verify missing imports fail**

Run:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider' .venv-ml/bin/python -m pytest tests/test_smooth_boundary_attack.py -q
```

Expected: failure because `scripts.smooth_boundary_attack` does not exist.

- [x] **Step 3: Implement the minimal smooth projected update**

Implement a normalized Gaussian kernel with depthwise `conv2d`, multiply the
smoothed gradient by the soft mask, normalize active values by RMS, project
each candidate into `reference +/- epsilon`, and retain the last failed and
first successful endpoints.

- [x] **Step 4: Implement decision-boundary bisection**

For smile removal, define a robust successful endpoint as
`p(smile) <= 0.5 - margin`. Bisect the failed/successful line segment and
return the successful endpoint closest to the boundary. For attribute
addition, use `p(attribute) >= 0.5 + margin`.

- [x] **Step 5: Run focused tests**

Run the command from Step 2. Expected: all focused tests pass.

### Task 2: Add the ten-image paired evaluator

**Files:**
- Create: `scripts/compare_smooth_boundary_attack.py`
- Modify: `tests/test_smooth_boundary_attack.py`

**Interfaces:**
- Consumes: attack primitives from Task 1 and image, Grad-CAM++, identity, CSV, and PNG helpers from `scripts.compare_attack_masks`.
- Produces: `results.csv`, `summary.json`, per-sample masks, attacked PNGs, residual maps, and `comparison.jpg`.

- [x] **Step 1: Write failing aggregate-metric tests**

```python
def test_aggregate_comparison_reports_before_after_and_paired_deltas():
    summary = aggregate_comparison(rows)
    assert summary["without_attack"]["target_pass_rate"] == 0.0
    assert summary["with_attack"]["target_pass_rate"] == 1.0
    assert "identity_delta" in summary["paired"]
    assert "residual_tv" in summary["with_attack"]
```

- [x] **Step 2: Implement paired saved-PNG evaluation**

For every sample, load `sd2_bld_grid.png`, source, and semantic mask from the
A9 audit. Compute Grad-CAM++ from the source, build the soft anatomical mask,
attack the normalized CCI-BLD tensor, save and reload the PNG, and score both
the untouched baseline and attacked image.

- [x] **Step 3: Implement preservation and residual metrics**

Record target probability, desired probability, pass status, identity cosine,
mean absolute change, `L2`, `Linf`, changed fraction, residual total
variation, outside-FacePart MAE, iterations, and boundary iterations.

- [x] **Step 4: Add comparison panels and machine-readable outputs**

Each panel contains source, no-attack CCI-BLD, soft mask, attacked result,
amplified absolute residual, and a mouth-region crop. Write one row per
sample/method to `results.csv` and aggregate method and paired metrics to
`summary.json`.

- [x] **Step 5: Run focused tests**

Run:

```bash
PYTEST_ADDOPTS='-p no:cacheprovider' .venv-ml/bin/python -m pytest tests/test_smooth_boundary_attack.py -q
```

Expected: all focused tests pass.

### Task 3: Run and verify the experiment

**Files:**
- Generate: `outputs/smooth_boundary_attack_10/`

**Interfaces:**
- Consumes: all ten A9 samples and the frozen classifier/identity checkpoints.
- Produces: final quantitative and visual comparison.

- [x] **Step 1: Run all ten samples**

```bash
MPLCONFIGDIR=/private/tmp/matplotlib \
.venv-ml/bin/python scripts/compare_smooth_boundary_attack.py \
  --output_dir outputs/smooth_boundary_attack_10 \
  --sample_ids 0 1 3 9 14 20 24 25 27 31 \
  --device mps
```

Expected: exit `0`, ten comparison directories, `results.csv`, and
`summary.json`.

- [x] **Step 2: Inspect successful, failed, best, and worst visual cases**

Open at least four `comparison.jpg` files selected using target result,
identity delta, residual TV, and perturbation magnitude. Report visible
texture or boundary artifacts even when classifier success improves.

- [x] **Step 3: Run the complete test suite**

```bash
PYTEST_ADDOPTS='-p no:cacheprovider' .venv-ml/bin/python -m pytest -q
```

Expected: no failures.

- [x] **Step 4: Verify changed files**

```bash
git diff --check -- \
  scripts/smooth_boundary_attack.py \
  scripts/compare_smooth_boundary_attack.py \
  tests/test_smooth_boundary_attack.py \
  docs/superpowers/specs/2026-07-23-minimal-smooth-boundary-attack-design.md \
  docs/superpowers/plans/2026-07-23-minimal-smooth-boundary-attack.md
```

Expected: no output and exit `0`.
