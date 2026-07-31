# Final-Restoration Visual Ablation Design

## Goal

Produce an exact, reviewable visual comparison showing what A11's final latent
restoration changes. The comparison must exclude the later pixel-space attack
and keep every diffusion input identical except for enabling final restoration.

## Approaches Considered

1. Run A11 twice with the same sample and seed, toggling only
   `--cci_disable_final_correction`. This keeps production code unchanged and
   is the selected approach.
2. Modify the BLD backend to save the latent immediately before restoration.
   This would isolate the stage in one run, but it changes production code for
   a one-off diagnostic.
3. Compare A0 with A11. Existing artifacts permit this, but the comparison
   includes diffusion-time CCI guidance and therefore does not isolate final
   restoration.

## Controlled Experiment

Use CelebAMask-HQ sample `26811`, seed `42`, the mouth-only graph, dilation
`8`, 35 SD2 inference steps, MPS, and float32. Existing results show a large
restoration response for this sample, making it suitable for visual inspection.

Run two A11 cases:

- `without_final_restoration`: trust-region diffusion guidance remains enabled,
  but pass `--cci_disable_final_correction`.
- `with_final_restoration`: use the identical command without that flag.

Both cases must set `--cci_post_attack none`. The prompt, source image, masks,
graph, classifier, identity model, controller mode, model, and all diffusion
parameters must otherwise be identical.

## Script and Outputs

Add a separate orchestration script that:

1. Writes a self-contained mouth-only graph and sample binding under its output
   directory.
2. Runs the two controlled cases.
3. Validates both audits and saved images.
4. Calculates pixel MAE, maximum absolute difference, changed-pixel fraction,
   desired-target probability, identity cosine, non-target drift, and runtime.
5. Writes:
   - the two original output images;
   - a labelled side-by-side comparison;
   - an amplified absolute-difference heatmap;
   - JSON and Markdown metric reports.

Default output:

`outputs/final_restoration_ablation_26811`

## Reproducibility and Interpretation

The two executions use identical deterministic inputs and seed. The report will
record both commands and SHA-256 hashes for the source, graph, bindings, and
model checkpoints where available.

Because these are separate MPS executions, the script will check that the
enabled run's recorded `initial_probability` agrees closely with the disabled
run's final desired probability. A mismatch is reported rather than hidden.
The visual result is interpreted as restoration-only only when this consistency
check passes.

The amplified difference image is diagnostic and must not be presented as a
natural output image. The unmodified before/after images remain the primary
visual evidence.

## Error Handling and Tests

The script fails if an input is missing, either subprocess fails, an audit is
missing, post-attack data is present, the enabled run lacks a restoration
record, or comparison image dimensions differ.

Tests cover command parity, the single allowed flag difference, post-attack
exclusion, metric computation, report generation, and consistency validation.
The existing 300-image scheduler and production BLD implementation remain
unchanged.
