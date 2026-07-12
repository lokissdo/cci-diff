# Robust Smile Guidance Design

## Goal

Remove the `Smiling` attribute from `data/0.jpg` while keeping the face
photorealistic and preserving identity. Success requires the CelebA ResNet50
`Smiling` probability to fall below `0.5` without tooth artifacts, black/white
classifier shortcuts, or incoherent mouth geometry.

The current hard-mask run lowered the score only from `0.99921` to `0.99182`
and increased mouth-region MAE from `40.10` to `44.71`. The classifier altered
teeth pixels because the hard mask excluded lips, jaw, and cheek geometry.

## Non-Goals

- Do not optimize solely for the lowest classifier score.
- Do not replace the supplied ResNet50 or retrain it in this pass.
- Do not implement MaskDiME mask discovery.
- Do not add CLIP guidance yet.
- Do not change the binary mouth mask used for strict auditing.
- Do not require CUDA; all operations must run in float32 on Apple MPS.

## Approach

Combine four controls:

1. Keep the original binary mouth mask as `audit_mask`.
2. Form a hard semantic edit mask from the mouth, upper-lip, and lower-lip
   CelebAMask parts.
3. Feather that semantic union by only 3 pixels for gradient localization and
   blending, without morphological dilation.
4. Replace the single classifier view with deterministic multi-scale,
   low-pass views so high-frequency tooth patterns cannot cheaply reduce loss.
5. Apply separately normalized semantic and realism gradients only during the
   middle denoising steps.

The diffusion model remains the image prior. Guidance stops before late steps
so SD2 can resolve texture instead of receiving final-pixel adversarial pressure.

## Semantic Mask Union

Use the supplied aligned 512x512 CelebAMask parts:

```text
audit_mask = threshold(00000_mouth.png)
semantic_mask = audit_mask
              OR threshold(00000_u_lip.png)
              OR threshold(00000_l_lip.png)
generation_mask = gaussian_blur(semantic_mask, radius=3 px)
generation_mask = clamp(generation_mask, 0, 1)
```

The measured source-mask coverage is `2.17%` for mouth alone and `3.81%` for
mouth plus both lips. This adds only `1.64%` of the image and avoids the much
larger intervention region created by broad dilation.

The generation mask is downsampled to latent resolution with bilinear
interpolation. It is used for semantic gradient localization and soft BLD:

```text
z_blend = generation_mask * z_edit
        + (1 - generation_mask) * z_source_noised
```

Report metrics twice: the untouched mouth-only audit mask remains the strict
denominator used for comparison with earlier runs, while the hard semantic mask
measures whether changes stayed inside the explicitly allowed mouth-and-lips
region. Never use the feathered mask as an audit denominator.

Save both the hard semantic union and feathered generation mask beside each
output for inspection.

## Robust Smile Loss

Let `x = D(z)` be the decoded image and `C_31` the frozen `Smiling` logit. Build
three deterministic views:

```text
T_256(x) = resize(low_pass(x), 256) -> classifier input
T_384(x) = resize(low_pass(x), 384) -> classifier input
T_512(x) = low_pass(x)              -> classifier input
```

Use a differentiable Gaussian low-pass filter with kernel size 5 and sigma 1.0.
Each view is resized and ImageNet-normalized by the existing classifier adapter.

```text
L_smile = mean_k BCEWithLogits(C_31(T_k(x)), 0)
```

The views are deterministic so same-seed comparisons remain reproducible.
Agreement across scales makes isolated high-frequency teeth artifacts less
effective than a coherent expression change.

## Realism Losses

Construct a soft boundary ring around the semantic union and intersect it with
the generation mask. Preserve source appearance in this transition region:

```text
L_boundary = mean(boundary_ring * abs(x - x_source))
```

Penalize high-frequency edit residuals inside the generation mask:

```text
r = generation_mask * (x - x_source)
L_tv = mean(abs(r[:, :, 1:, :] - r[:, :, :-1, :]))
     + mean(abs(r[:, :, :, 1:] - r[:, :, :, :-1]))
```

Do not use full source reconstruction inside the semantic mask because that
would directly oppose closing or reshaping the smiling mouth.

## Separate Gradient Composition

The current adapter forms one weighted scalar and normalizes the final gradient.
For this experiment, compute term gradients separately:

```text
g_smile    = normalize(generation_mask * gradient_z(L_smile))
g_boundary = normalize(generation_mask * gradient_z(L_boundary))
g_tv       = normalize(generation_mask * gradient_z(L_tv))

g = 1.0 * g_smile
  + 0.3 * g_boundary
  + 0.05 * g_tv

z' = z - eta(t) * g
```

Per-term normalization makes the coefficients express priorities rather than
compensating for unrelated numeric loss scales. A zero-norm term contributes a
zero gradient without producing NaN values.

The existing single-loss adapter remains unchanged for backward compatibility.
Add a separate robust guidance function rather than changing hair behavior.

## Guidance Schedule

Use only middle denoising indices:

```text
start_step = 4
end_step = 16
every_n_steps = 2
base_step_size = 0.20
```

Apply a linear decay across the active interval:

```text
eta(t) = base_step_size * (end_step - t + 1)
                       / (end_step - start_step + 1)
```

The active indices are `4, 6, 8, 10, 12, 14, 16`. No classifier gradient is
applied during the final texture-refinement steps.

## Components

### Mask Utilities

Add focused utilities that validate and union aligned binary component masks,
then produce the strict audit mask, hard semantic mask, feathered image mask,
latent generation mask, and semantic boundary ring. Keep torch/Pillow imports
local so lightweight package imports remain available without ML dependencies.

### Robust Classifier Objective

Add functions for multi-scale classifier logits, boundary loss, residual TV,
and separate gradient composition. Each function must be independently testable
with small tensors and fake classifiers.

### SD2 Backend

Carry `audit_latent_mask`, `semantic_latent_mask`, and
`generation_latent_mask`. Use the generation mask for the robust hook and soft
blend; record the audit and semantic masks in state metadata. The existing hard
blend path remains the default for old commands.

### CLI

Add opt-in controls:

```text
--robust_classifier_guidance
--generation_mask_component data/00000_mouth.png
--generation_mask_component data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_u_lip.png
--generation_mask_component data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_l_lip.png
--generation_mask_feather 3
--classifier_scales 256,384,512
--classifier_blur_sigma 1.0
--boundary_weight 0.3
--tv_weight 0.05
--save_generation_mask
```

The existing `latent_classifier` mode remains reproducible without these flags.

## Audit

Record:

- strict audit, semantic component, hard union, and feathered mask paths;
- semantic-mask area and feather radius;
- classifier scales and low-pass sigma;
- applied step indices and effective step size per step;
- per-step semantic, boundary, and TV losses;
- per-term masked gradient norms;
- source and output smile probabilities;
- inside/outside MAE measured with both the strict audit mask and hard semantic
  union.

This data must reveal whether score reduction came from semantic change or an
unstable gradient spike.

## Error Handling

- Reject negative feather radius, sigma, and loss weights.
- Reject missing component masks or component masks with mismatched dimensions.
- Reject empty classifier scales and non-positive scale values.
- Fall back to the audit mask and record a warning if expansion yields an empty
  generation mask.
- Reject schedules where `start_step > end_step`.
- Replace zero-norm term gradients with zeros before composition.
- Keep every classifier and guidance tensor in float32 on MPS.
- Report MPS allocation failures without silently moving part of the graph to
  CPU.

## Testing

Use red-green TDD for:

- semantic union equals the logical OR of mouth, upper lip, and lower lip;
- semantic union coverage is greater than mouth-only coverage;
- 3-pixel feathering produces fractional values without broad expansion;
- soft blend equals hard blend for binary masks;
- low-pass multi-scale views preserve input gradients;
- multi-scale BCE averages the requested classifier views;
- boundary loss is zero when source and decoded images match;
- residual TV penalizes a checkerboard more than a smooth edit;
- separate gradient normalization handles zero gradients without NaN;
- schedule applies exactly at indices `4, 6, 8, 10, 12, 14, 16`;
- strict audit metrics still use the original mouth mask;
- allowed-region metrics use the hard semantic union, never the feathered mask;
- old no-hook, color, and classifier modes remain unchanged.

Run the complete unit suite, then one same-seed robust MPS run. Compare it with
the saved no-hook and hard-classifier outputs; do not select a different seed.

## Limitation

The semantic union permits coherent changes to the mouth interior and both lip
boundaries, but deliberately excludes smile-related cheeks and jaw geometry.
This is the intended minimality constraint. If the output stays realistic but
the smile probability remains above `0.5`, report an insufficient intervention;
do not silently expand the mask within the same experiment.

## Success Criteria

- Output `Smiling` probability is below `0.5`.
- The face reads visually as neutral or non-smiling.
- No duplicated teeth, black/white mouth patches, or broken lip boundary.
- Outside-semantic-mask MAE is no more than `1.25x` the no-hook baseline.
- Strict mouth-only metrics remain present for comparison with earlier runs.
- Identity, hair, eyes, and background remain visually stable.
- The audit records all seven intended robust guidance applications.

If the score falls but realism fails, classify the run as an adversarial
shortcut. If realism improves but the score stays above `0.5`, classify it as an
insufficient intervention. Do not hide either result with seed selection.

## Deferred Work

If robust single-classifier guidance still fails, the next experiment should
compare ResNet50-only, CLIP-only, and combined neutral-expression guidance. A
MaskDiME-like learned spatial mask remains a separate research direction.
