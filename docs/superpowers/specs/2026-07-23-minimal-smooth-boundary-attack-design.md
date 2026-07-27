# Minimal Smooth Boundary Attack Design

**Date:** 2026-07-23

## Objective

Replace the visually destructive comparison PGD with an optional experimental
attack that flips the smile classifier using the smallest practical,
anatomically localized perturbation. The historical PGD implementation and
all production diffusion and CCI behavior remain unchanged.

The attack targets the classifier decision boundary, not a high-confidence
non-smile prediction. For smile removal, success means a saved PNG has smile
probability at or below `0.5`.

## Why the Existing Attack Breaks the Image

The current normalized-space parameters use `step_size=0.5` and
`epsilon=0.3`. Because the step exceeds the perturbation budget, the first
sign-gradient update saturates the allowed change. The objective contains no
image prior, the binary mask has a hard edge, and no denoising follows the
pixel-space update. The result is classifier-effective high-frequency texture
around the mouth.

## Proposed Attack

### Soft anatomical attention mask

Let `m_f` be the binary FacePart mouth-and-lips mask and let `h` be the
normalized Grad-CAM++ saliency map computed from the source image. The attack
mask is

\[
m = m_f \odot h.
\]

The FacePart mask enforces anatomical support. Grad-CAM++ changes the update
strength inside that support without permitting changes elsewhere.

### Smooth continuous gradient

For classifier loss `L`, first spatially smooth its image gradient:

\[
\bar g = \operatorname{GaussianBlur}(\nabla_x L).
\]

Apply the soft mask and normalize the active gradient by its root-mean-square
magnitude:

\[
g_m = m \odot \bar g,\qquad
\hat g_m = \frac{g_m}{\operatorname{RMS}(g_m)+\varepsilon_g}.
\]

The update uses the continuous direction rather than `sign(g)`, avoiding an
equal maximum update at every selected pixel:

\[
x_{k+1} =
\Pi_{\lVert x-x_{\mathrm{ref}}\rVert_\infty\leq 0.05}
\left(x_k-\alpha\hat g_m\right),
\qquad \alpha=0.005.
\]

The attack never falls back to the historical large-budget update when the
small budget cannot flip a sample. Such cases are reported as failures.

### Boundary refinement

When an update first crosses the decision boundary, retain the previous
unsuccessful image and the new successful image. Binary search their line
segment for the smallest successful interpolation:

\[
x(\lambda)=(1-\lambda)x_{\mathrm{fail}}+\lambda x_{\mathrm{pass}}.
\]

Use a small safety margin for image quantization, aiming for smile probability
near `0.49` rather than a high-confidence non-smile result. Save and reload the
PNG before final scoring. If quantization moves the probability above `0.5`,
advance only enough toward the successful endpoint to recover the flip.

## Experiment

Run the new attack on the same five A9 smile-removal artifacts used by the
FacePart versus Grad-CAM++ comparison: IDs `0`, `1`, `3`, `9`, and `31`.

For each sample, save:

- source and pre-attack BLD image;
- FacePart mask, Grad-CAM++ saliency, and combined soft mask;
- smooth boundary attack output;
- a comparison panel against the existing FacePart and Grad-CAM++ PGD images;
- classifier probabilities before and after PNG quantization.

Aggregate:

- saved-PNG target pass rate;
- smile and desired-class probabilities;
- mean and maximum absolute perturbation;
- changed-pixel fraction;
- perturbation total variation;
- outside-FacePart leakage;
- identity cosine similarity;
- attack iterations and boundary-refinement iterations.

The comparison must distinguish classifier success from realism. Lower
perturbation and residual total variation, stronger identity, strict
anatomical locality, and visual inspection are evidence of improvement, but
are not treated as a standalone learned realism score.

## Implementation Boundaries

- Add the experimental helpers and CLI options to
  `scripts/compare_attack_masks.py`.
- Add focused tests to `tests/test_compare_attack_masks.py`.
- Preserve existing CLI defaults and historical result reproduction.
- Write new artifacts under `outputs/smooth_boundary_attack_5/`.
- Do not modify diffusion, BLD, CCI, classifier, or evaluator behavior.
- Do not create a Git commit.

## Verification

Tests must cover:

- soft-mask support remains inside the FacePart mask;
- continuous smoothed updates do not alter outside-mask pixels;
- the projected perturbation respects `epsilon`;
- boundary refinement returns a successful point closer to the boundary than
  the original crossing step;
- failed low-budget attacks remain explicit failures;
- saved output metrics are computed from the reloaded PNG.

Run focused tests, the five-sample experiment, visual inspection, and the full
test suite before reporting results.
