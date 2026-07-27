# Anisotropic Perioral Mask Experiment

## Goal

Test whether a horizontally biased soft generation mask removes the smile from CelebA-HQ image `00000` more realistically than the current mouth-and-lips mask. The experiment must cover the mouth corners and nearby smile creases while limiting changes to the nose, chin, and cheeks.

## Scope

The experiment changes generation-mask geometry only. It does not change the hard semantic audit mask, CCI loss, controller weights, prompt, diffusion model, inference-step count, or target threshold.

The hard semantic mask remains the union of `mouth`, `u_lip`, and `l_lip`. Spatial evaluation therefore continues to measure changes against the same semantic region used in earlier runs.

## Approaches Considered

1. **Anisotropic perioral dilation:** expand the generation mask more horizontally than vertically. This is selected because it can cover mouth corners without unnecessarily including a large lower-face region.
2. **Larger isotropic dilation:** use the existing scalar dilation with radii above eight pixels. This requires no new mask primitive but expands too far toward the nose and chin.
3. **Facial-landmark ellipse:** derive an adaptive perioral ellipse from detected landmarks. This is deferred because it adds a detector dependency before the simpler geometry has been validated.

## Mask Geometry

Extend generation-mask construction to accept independent non-negative integer horizontal and vertical dilation radii. Existing scalar dilation remains supported and is equivalent to setting both radii to the scalar value.

Rectangular binary dilation is applied to the semantic union before Gaussian feathering. The hard semantic mask is saved before dilation and must remain byte-identical across candidates.

The experiment evaluates these candidates:

| Candidate | Horizontal radius | Vertical radius | Feather radius |
|---|---:|---:|---:|
| Current control | 4 | 4 | 3 |
| Perioral small | 8 | 4 | 5 |
| Perioral medium | 12 | 6 | 7 |
| Perioral large | 16 | 8 | 9 |

## Controlled Run

- Feature: remove smile.
- Source: CelebA-HQ image `00000`.
- Variant: A3 feedback CCI with target projection.
- Seed: `42`.
- Inference steps: `35`.
- Target requested-state probability: `0.8`.
- Batch size: `1`.
- Model, scheduler, prompt, classifier, identity model, and semantic components remain identical to the existing A3 seed-42 run.

Each candidate is generated independently and saved in a separate directory. The experiment must preserve input/output pair artifacts and all existing audit fields. Candidate records add horizontal dilation, vertical dilation, and feather radius.

## Selection And Review

A candidate is target-successful when remove-smile probability is at least `0.8`. Failed candidates cannot outrank successful candidates.

Successful candidates are compared in this order:

1. Higher identity cosine.
2. Lower boundary discontinuity.
3. Lower full-image changed fraction at `5/255`.
4. Lower changed fraction outside the hard semantic mask at `5/255`.

Automatic ranking is evidence, not the final realism decision for this one-image experiment. Visual review rejects candidates with doubled lips, residual teeth, mouth ghosts, broken mouth corners, or an implausible transition into surrounding skin.

## Compatibility And Validation

- Existing calls with scalar dilation must produce the same masks as before.
- Negative or non-integer anisotropic radii are rejected.
- Horizontal-only expansion must increase support along the x-axis without changing the expected y-axis extent before feathering.
- The semantic mask must not change when generation dilation or feathering changes.
- CLI and pilot command construction must pass candidate geometry explicitly and record it in manifests and CSV outputs.
- Focused mask and command tests run before the MPS experiment, followed by the full repository test suite.

## Decision Rule

Adopt anisotropic perioral masks for the larger evaluation only if at least one candidate reaches the `0.8` target threshold and visibly improves the mouth over the current seed-42 control without a material identity or locality regression. Otherwise retain the current mask implementation and investigate a landmark-derived region or a mouth-specific realism objective.
