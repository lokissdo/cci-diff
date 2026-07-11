# Soft Mask CCI Design

## Goal

Improve the visual quality of CCI-guided hair edits by replacing hard binary
generation masks with cleaned, softened operational masks, while keeping the
original binary mask for strict CCI auditing.

The immediate failure case is the blond-hair CCI hook painting the exact
CelebAMask hair shape, including hard edges and small specks. The target result
is a smoother hair edit with less visible mask structure, without hiding outside
mask leakage in the audit.

## Non-Goals

- Do not implement MaskDiME-style learned mask optimization in this pass.
- Do not change the core CCI metric definitions.
- Do not replace the existing CelebAMask label-map mask generation.
- Do not require CUDA; the path must continue to run on Mac MPS with float32.

## Approach

Use a dual-mask design:

- `audit_mask`: the original binary mask generated from the CelebAMask label map.
- `generation_mask`: a cleaned and softened mask derived from `audit_mask`.

The generation mask is used during denoising for CCI loss weighting, gradient
masking, and optionally blended-latent restoration. The audit mask remains the
reference for target-region and outside-region measurements.

This is similar in spirit to masked diffusion guidance, including MaskDiME, but
it is not MaskDiME. MaskDiME can optimize the mask or counterfactual region with
classifier gradients. This design uses a fixed semantic mask and makes it safer
for generation.

## Components

### `cci_diff.masking`

Add a lightweight mask utility module with torch-local helpers:

- `clean_binary_mask(mask, min_area_px, close_kernel_px)`: removes tiny connected
  components and optionally closes small holes.
- `soften_mask(mask, blur_radius_px, edge_floor, edge_ceiling)`: converts a hard
  mask to a feathered float mask in `[0, 1]`.
- `prepare_generation_mask(mask, mode, ...)`: returns both the hard audit mask
  and the softened generation mask.

The module should keep numpy/Pillow imports optional or local so core tests can
still import without the ML stack.

### SD2 Backend

Extend `BlendedLatentDiffusionSD2Backend.edit_image` to carry two masks:

- `audit_latent_mask`: binary mask for state recording and later metrics.
- `generation_latent_mask`: soft mask used for denoising operations.

Current `blend_latents` uses boolean selection:

```python
noise_source_latents.where(~latent_mask.bool(), latents)
```

For soft blending, add:

```python
blend_soft_latents(edited, generation_mask, source)
```

with:

```python
generation_mask * edited + (1 - generation_mask) * source
```

Keep the old hard blend path available through `--mask_softness none` or
`--mask_softness loss`.

### CCI Hook

In `build_cci_latent_guidance_hook`, replace nearest-neighbor binary image mask
upsampling with the softened generation mask.

Use the generation mask for:

- average target color loss
- latent gradient mask
- outside preservation loss with `1 - generation_mask`

Use the audit mask only for reporting and comparison.

### CLI

Add explicit controls:

- `--mask_softness none|loss|blend|both`
- `--mask_blur_radius 6`
- `--mask_min_area 64`
- `--mask_close_radius 3`
- `--mask_edge_floor 0.0`
- `--mask_edge_ceiling 1.0`
- `--save_generation_mask`

Recommended default for the current sample:

```text
--mask_softness both
--mask_blur_radius 6
--mask_min_area 64
--mask_close_radius 3
```

Default behavior should remain backward-compatible unless the user opts in to
soft masking. The wrapper script may enable the recommended soft settings for
the hair experiment after the new path is verified.

## Data Flow

1. `run_sd2_bld_mps.sh` creates the binary hair mask from the label map.
2. `run_sd2_bld_cci.py` loads the CCI config and CLI mask-softness settings.
3. The SD2 backend reads the binary mask as `audit_mask`.
4. If mask softening is enabled, derive `generation_mask`.
5. The CCI hook uses `generation_mask` for differentiable guidance.
6. The BLD loop uses the selected hard or soft blend mode.
7. `audit.json` records both mask paths/settings and the state phases.
8. Comparison scripts compute metrics with the binary audit mask.

## Audit Behavior

`audit.json` should include:

```json
{
  "masking": {
    "audit_mask": "outputs/generated_masks/1_hair.png",
    "generation_mask": "outputs/generated_masks/1_hair_soft.png",
    "softness": "both",
    "blur_radius": 6,
    "min_area": 64,
    "close_radius": 3,
    "soft_blend_enabled": true
  }
}
```

The audit should never silently switch to measuring on the soft mask. The binary
mask remains the denominator for hair RGB, outside drift, leakage, and
hair/outside difference ratio.

## Error Handling

- Reject negative blur radius, min area, and close radius.
- Reject `edge_floor > edge_ceiling`.
- If soft mask generation produces an all-zero mask, fall back to the binary mask
  and record a warning in `audit.json`.
- If optional image-processing dependencies are missing, show a clear install
  message and leave the old hard-mask path usable.

## Testing

Add focused tests for:

- connected-component cleanup removes tiny specks.
- softening preserves shape but creates fractional edge values.
- soft blending equals hard blending when the mask is binary.
- CLI parses mask-softness options.
- `audit.json` records both binary and generation mask settings.
- CCI hook receives the generation mask while audit uses the binary mask.

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests
```

Then run A/B generation:

```bash
OUTPUT_DIR=outputs/sample_1_sd2_bld_hair_no_hook_mps \
NUM_INFERENCE_STEPS=35 \
./scripts/run_sd2_bld_mps.sh --cci_hook none

OUTPUT_DIR=outputs/sample_1_sd2_bld_hair_cci_soft_mps \
NUM_INFERENCE_STEPS=35 \
./scripts/run_sd2_bld_mps.sh \
  --cci_hook latent_color \
  --cci_step_size 2.0 \
  --cci_every_n_steps 1 \
  --cci_normalize_grad \
  --mask_softness both \
  --mask_blur_radius 6 \
  --mask_min_area 64 \
  --mask_close_radius 3 \
  --save_generation_mask
```

## Success Criteria

- The hook output is visibly less hard-edged than the current CCI output.
- Blond-distance inside the binary hair mask improves over the no-hook baseline.
- Outside-mask drift remains much lower than inside-mask change.
- Tiny black/bright specks from the hard mask are reduced.
- The audit clearly records that generation used a soft mask while metrics used
  the original binary mask.

## Future Work

After this pass, a separate MaskDiME-like project could add gradient-optimized
mask discovery. That should be a new spec because it changes the causal question
from "edit this known semantic region" to "discover the minimal causal region."
