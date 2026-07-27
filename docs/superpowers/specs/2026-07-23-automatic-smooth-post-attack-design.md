# Automatic Smooth Post-Attack Design

**Date:** 2026-07-23

## Objective

Add an optional automatic classifier-boundary correction after CCI-BLD image
generation. Preserve every original CCI-BLD artifact and save correction
results separately.

The feature is intended for controlled counterfactual evaluation. It must not
turn the corrected image into the only surviving output, silently apply the
historical destructive PGD configuration, or describe a same-classifier flip
as proof of perceptual realism.

## User Interface

Add the following clean-CCI runner options:

```text
--cci_post_attack {none,smooth_boundary}
--cci_post_attack_epsilon 0.05
--cci_post_attack_step_size 0.005
--cci_post_attack_max_steps 500
--cci_post_attack_boundary_margin 0.01
--cci_post_attack_boundary_steps 16
--cci_post_attack_gaussian_kernel_size 5
--cci_post_attack_gaussian_sigma 1.0
```

`--cci_post_attack` defaults to `none`, preserving historical runs. The
experiment shell runner may set `smooth_boundary` automatically.

## Output Contract

The original backend output remains:

```text
sd2_bld_grid.png
```

When smooth post-attack is enabled, save:

```text
sd2_bld_grid_corrected.png
post_attack_soft_mask.png
```

The main runner continues returning the original `sd2_bld_grid.png` path.
Consumers opt into the corrected artifact through the audit metadata. This
prevents an evaluator from silently mixing raw and corrected methods.

## Runtime Architecture

Move the reusable smooth attack primitives from the experimental script into:

```text
src/cci_diff/post_attack.py
```

The experiment script re-exports or imports these functions so existing
comparison commands remain valid. The production module owns:

- soft anatomical mask construction;
- Gaussian-smoothed, RMS-normalized masked gradients;
- projected low-budget updates;
- decision-boundary refinement;
- per-candidate attack records.

The runner owns artifact loading, candidate-grid splitting, classifier and
identity scoring, PNG serialization, and audit metadata.

## Automatic Decision Flow

For each generated candidate:

1. Score the exact crop using the configured target classifier.
2. Determine success with the intervention's desired value and the `0.5`
   classifier decision threshold.
3. If already successful with the configured safety margin, copy the crop
   unchanged.
4. Otherwise run the smooth boundary attack inside the soft anatomical mask.
5. Project every update into the configured normalized `L-inf` budget.
6. Save the candidate, rebuild the corrected horizontal grid, reload the PNG,
   and score the exact saved crop.
7. Report low-budget failure explicitly. Never invoke historical sign-PGD as
   fallback.

The corrected grid has the same dimensions and candidate order as the raw
grid. Single-image and multi-image grids use the same per-candidate path.

## Mask Construction

For the current CelebA smile workflow:

\[
m = m_s \odot H_{\mathrm{GradCAM++}},
\]

where `m_s` is the hard semantic FacePart mask and `H` is normalized
Grad-CAM++ saliency computed from the source image for the resolved target
attribute.

The hard semantic mask defines support, so no update may occur outside it.
Grad-CAM++ only changes update strength inside that support.

Post-attack requires:

- a resolved classifier target index;
- a semantic mask artifact;
- a source image;
- `batch_size` and candidate width consistent with the saved grid.

Missing requirements fail before generation rather than producing a partial
post-attack audit.

## Audit Contract

Add `cci.post_attack`:

```json
{
  "mode": "smooth_boundary",
  "raw_output_path": ".../sd2_bld_grid.png",
  "corrected_output_path": ".../sd2_bld_grid_corrected.png",
  "soft_mask_path": ".../post_attack_soft_mask.png",
  "configuration": {
    "decision_threshold": 0.5,
    "epsilon": 0.05,
    "step_size": 0.005,
    "max_steps": 500,
    "boundary_margin": 0.01,
    "boundary_steps": 16,
    "gaussian_kernel_size": 5,
    "gaussian_sigma": 1.0
  },
  "candidates": [
    {
      "index": 0,
      "before_probability": 0.8,
      "after_probability": 0.49,
      "already_successful": false,
      "target_pass": true,
      "margin_pass": true,
      "iterations": 7,
      "boundary_iterations": 16,
      "mean_abs_change": 0.0001,
      "linf": 0.01,
      "changed_fraction": 0.01,
      "outside_semantic_mae": 0.0,
      "identity_before": 0.95,
      "identity_after": 0.95
    }
  ]
}
```

Probabilities and image metrics are calculated from saved PNG crops.
Internal pre-serialization probabilities may be retained under explicitly
named fields for debugging.

## Failure Behavior

- Invalid numerical settings fail argument validation.
- Post-attack with a non-clean CCI mode fails validation in the initial
  implementation.
- Missing classifier, semantic mask, source, or identity model fails before
  generation.
- Grid dimensions inconsistent with `batch_size` fail without overwriting the
  raw result.
- A low-budget target miss still writes the corrected grid and records
  `target_pass=false`.
- The original CCI-BLD image and audit data are never deleted.

## Testing and Evaluation

Tests cover:

- argument defaults and validation;
- migration of attack primitives without behavioral regression;
- already-successful candidates remain pixel-identical;
- failed candidates invoke the smooth attack;
- grid split/reassembly preserves order and dimensions;
- raw output is never overwritten;
- saved-PNG probabilities populate the audit;
- perturbations remain inside the semantic mask;
- low-budget misses are explicit;
- disabled mode preserves the existing audit contract.

After implementation, rerun the ten-image smile comparison through the
automatic runner path and compare its metrics with
`outputs/smooth_boundary_attack_10/`. The automatic and standalone paths must
agree within PNG and floating-point tolerance.

## Repository Constraints

- Do not create a Git commit.
- Do not modify the historical PGD function or existing output artifacts.
- Do not change default CCI-BLD generation behavior.
