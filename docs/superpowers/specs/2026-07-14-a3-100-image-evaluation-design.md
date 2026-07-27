# A3 100-Image-Per-Task Evaluation Design

**Date:** 2026-07-14

## Objective

Run the full A3 clean-CCI configuration on 100 eligible CelebA-HQ source images for each of two counterfactual tasks, select a spatially minimal target-feasible edit from a deterministic mask search, preserve an easily browsable input for every selected output, and evaluate the resulting 200 source-counterfactual pairs with independent ACE checkpoints.

This is an absolute A3 evaluation. It does not establish a causal A3-versus-A0 improvement because the same 100-image cohort is not being generated with A0 in this experiment.

## Experiment Cohorts

The experiment has two independently selected cohorts:

1. `smile`: 100 unique sources classified as `Smiling >= 0.5`, with a detectable face and complete mouth, upper-lip, and lower-lip masks. The requested output value is `Smiling = 0`.
2. `hair`: 100 unique sources classified as `Blond_Hair < 0.5`, with a detectable face and a complete hair mask. The requested output value is `Blond_Hair = 1`.

A source may appear in both cohorts if it independently satisfies both eligibility rules. Within one task, each source image ID appears once.

Selection is deterministic by ascending CelebA-HQ image ID. The selector records every scanned image, source probability, file-completeness result, and face-detection result in the experiment manifest.

## Generation Configuration

Only variant A3 is generated:

- Hook: `clean_constraint`.
- Controller mode: `feedback`.
- Target-conflict projection: enabled.
- Device: MPS.
- Torch dtype: `float32`.
- Seed: `42` for every source.
- Inference steps: `35`.
- Guidance scale: `5.0`.
- Blending start: `0.25`.
- Batch size: `1`.
- Model: local SD2 checkpoint `checkpoints/sd2-1-base`.

The classifier, identity model, concept graph, masks, and sample bindings remain those used by the existing clean-CCI pilot.

## Spatial Minimality Policy

Spatial minimality means minimizing the support of visible pixel changes, not merely preserving classifier attributes. Other attributes remain measured for MNAC and CD, but no new non-target attribute constraint is introduced before this experiment because it could further suppress target success.

For every source, generate three deterministic A3 candidates from the same semantic component union:

1. `d0`: no binary dilation, followed by the graph's 3-pixel feather.
2. `d4`: 4-pixel binary dilation at 512-by-512 generation resolution, followed by the same 3-pixel feather.
3. `d8`: 8-pixel binary dilation at 512-by-512 generation resolution, followed by the same 3-pixel feather.

The hard semantic audit mask never expands. Only the soft generation mask changes. All candidates use the same source, graph, prompt, seed, scheduler settings, and controller configuration.

Candidate selection uses the generation classifier, not the independent ACE oracle. This prevents evaluation-model leakage. A candidate is target-feasible when its requested-state probability is at least 0.8.

If one or more candidates are target-feasible, select lexicographically by:

1. Lowest full-image changed-pixel fraction at threshold `5/255`.
2. Lowest changed-pixel fraction outside the hard semantic mask at `5/255`.
3. Highest requested-state probability.
4. Highest generation-time identity cosine.

If no candidate is target-feasible, select the candidate with the highest requested-state probability, then the smallest changed area. The result remains recorded as a target failure.

For source image `x` and candidate `x_cf`, changed area at threshold `delta` is:

```text
A_delta = mean(max_channel(abs(x_cf - x)) > delta)
```

Measure `A_delta` for `delta` equal to `1/255`, `5/255`, and `10/255`. Also measure the changed fraction outside the semantic mask, outside the generation mask, RGB L1 magnitude inside and outside both masks, generation-mask fraction, semantic-mask fraction, and boundary discontinuity.

## Runner Changes

Extend `scripts/run_clean_cci_pilot.py` conservatively:

- Add `--variants` with choices `A0` through `A4`. Its default remains all variants for backward compatibility.
- Iterate only the selected variants in generation, summaries, ranking, manifests, and contact sheets.
- Add the source path and comparison-artifact paths to every result row.
- Add deterministic `d0`, `d4`, and `d8` generation-mask candidates and preserve each candidate audit.
- Select one final result per source using the target-first spatial policy above.
- Record selected dilation, candidate count, candidate metric rows, and all spatial-area measurements.
- Keep audit-based resume behavior. A valid existing audit skips regeneration and still recreates missing comparison artifacts.
- Add continue-on-error behavior for the large run. Failures are appended to a structured failure report while later samples continue.
- Retry incomplete or failed samples once in a final resume pass with the same configuration and seed.
- Exit nonzero if fewer than 100 valid A3 audits exist for either task after the retry pass.

Each SD2 candidate generation remains a separate subprocess. This costs model-loading time but isolates MPS memory between runs and makes a long experiment safer to resume.

## Output Layout

The root is:

```text
outputs/clean_cci_a3_100/
```

Each result directory follows:

```text
<feature>/<sample_id>/A3/
  input.jpg
  sd2_bld_grid.png
  input_output.jpg
  audit.json
  semantic_mask.png
  generation_mask.png
  selected.json
  candidates/
    d0/
      sd2_bld_grid.png
      audit.json
      cci_trace.jsonl
      semantic_mask.png
      generation_mask.png
    d4/
      ...
    d8/
      ...
```

`input.jpg` is a portable copy of the exact source file used for generation. The root output, audit, and masks are copies of the selected candidate for compatibility with existing result readers. `selected.json` records the selected candidate and complete ordering evidence. `input_output.jpg` places the input on the left and selected counterfactual output on the right at equal displayed dimensions, with short labels and the feature, image ID, variant, and selected dilation below them. The renderer must not stretch either image.

The result CSV contains `source_path`, `input_copy_path`, `output_path`, and `comparison_path`, making every output traceable without parsing an audit file.

Paginated contact sheets are generated separately for smile and hair. Each tile contains the same input-output pair, and each page contains at most 20 pairs so Finder and VS Code can open it reliably.

## Execution Flow

1. Run two eligible samples per task into the final output root, generating all three mask candidates.
2. Verify 12 candidate audits, four selections, four input copies, four selected outputs, four pair comparisons, and their metric rows.
3. Re-run the same command with `--limit 100`. The first four valid audits are reused.
4. Continue until all possible samples have been attempted, recording failures instead of losing completed work.
5. Run one resume pass to retry missing or failed A3 results.
6. Require three complete candidate audits and one valid selection for each of exactly 100 smile and 100 hair sources before final aggregate reporting.
7. Run the independent ACE evaluation in bounded batches.
8. Generate the per-image CSV, task summaries, confidence intervals, contact sheets, and paper-comparison report.

## Independent ACE Evaluation

Create a reusable CCI-side evaluator rather than modifying the thesis evaluator in place. It accepts the experiment root and ACE root as arguments and uses:

- CelebA-HQ 40-attribute oracle: `evaluate/ACE/models/checkpoint.tar`.
- VGGFace2 ResNet-50: `evaluate/ACE/pretrained_models/resnet50_ft_weight.pkl`.
- SimSiam ResNet-50: `evaluate/ACE/pretrained_models/checkpoint_0099.pth.tar`.

The evaluator compares each source embedding to its corresponding counterfactual embedding. It must not repeat the local FVA bug that compares a counterfactual embedding with itself.

Per-image fields:

- Source and output target probabilities from the independent oracle.
- Probability of the requested target state.
- Directional target success and FR indicator at threshold 0.5.
- VGGFace2 cosine and FVA pass indicator.
- SimSiam FS cosine.
- MNAC across all 40 thresholded attributes.
- Target-excluded collateral flip count.
- Names of every changed attribute.
- Selected mask dilation and all candidate target probabilities.
- Full-image, outside-semantic-mask, and outside-generation-mask changed fractions at `1/255`, `5/255`, and `10/255`.
- RGB L1 residual magnitudes and mask-area fractions.

Per-task summaries:

- FR percentage.
- Desired-target probability mean and median.
- FVA percentage and mean VGGFace2 cosine.
- Mean FS.
- Mean MNAC and mean collateral flips.
- Unconditional preservation metrics for protocol visibility.
- Preservation metrics conditioned on successful target flips for target-first interpretation.
- Nonparametric 95% bootstrap confidence intervals using 10,000 task-level resamples with a fixed bootstrap seed.

## Population Metrics

Compute FID separately for the 100 smile sources versus their outputs and the 100 hair sources versus their outputs. The report labels these values exploratory because FID is sensitive to sample count and the paper may use a different population size.

Compute CD separately per task from the independent oracle's 40-attribute outputs over the 100 source-output pairs. The implementation and aggregation must follow the local ACE correlation-difference definition.

COUT remains unavailable. The current local wrapper incorrectly treats attribute 0 and attribute 31 as opposite classes for a binary attribute toggle. No COUT number will be fabricated or compared with the paper until a direction-aware binary protocol is independently specified and validated.

## Reports

The experiment root contains:

- `pilot_manifest.json`: selected cohorts, model hashes, variant definition, and runtime configuration.
- `pilot_results.csv`: generation-time audit metrics and all source/output paths.
- `candidate_results.csv`: all 600 candidate rows and spatial-selection measurements.
- `ace_pair_metrics.csv`: independent metrics for all 200 pairs.
- `ace_task_summary.csv`: one row for smile and one for hair.
- `ace_metrics.json`: full protocol metadata, confidence intervals, and changed-attribute lists.
- `ace_paper_comparison.md`: task-level results beside prior-paper values with sample-size and protocol caveats.
- `failures.jsonl`: structured generation failures, empty when the run is complete.
- `contact_sheets/smile_*.jpg` and `contact_sheets/hair_*.jpg`: paginated pair comparisons.

The report ranks target success first. FS, FVA, MNAC, and collateral changes are interpreted as quality evidence only after the requested target succeeds.

## Failure Handling

- A missing image or mask is an eligibility rejection, not a generation failure.
- A face-detection failure is recorded and skipped during cohort selection.
- A subprocess failure records feature, sample ID, variant, command exit status, and expected audit path.
- A malformed or incomplete audit does not count as completed and is retried.
- Missing comparison images are recreated without regenerating a valid counterfactual.
- Metric evaluation is batch-bounded so 200 images do not need to reside on the accelerator simultaneously.
- Final reporting refuses to label the run complete unless both task counts equal 100.

## Tests

Add focused tests for:

- `--variants A3` generates commands only for A3 while the default still includes A0-A4.
- A valid audit is resumed without invoking generation.
- A missing pair artifact is recreated from the recorded source and output.
- Pair rendering preserves aspect ratio and places input left of output.
- Mask candidates apply exactly 0, 4, and 8 pixels of binary dilation while preserving the hard semantic audit mask.
- A successful smaller-area candidate wins over a successful larger-area candidate regardless of surplus target confidence above 0.8.
- A failed small candidate loses to a successful larger candidate.
- If every candidate fails, the highest requested-state probability wins and remains marked failed.
- Changed-area thresholds produce known fractions for synthetic residual patterns.
- Failure logging continues to later samples and a resumed run retries incomplete samples.
- FVA compares source with counterfactual, demonstrated with unequal synthetic embeddings.
- Directional FR handles desired values 0 and 1 correctly.
- MNAC and collateral MNAC differ only by the target flip.
- Task summaries and bootstrap intervals use exactly the task's 100 rows.
- Final completeness validation rejects 99 results and accepts 100 results per task.

## Acceptance Criteria

The experiment is complete only when:

- Smile has 100 valid A3 source-output pairs.
- Hair has 100 valid A3 source-output pairs.
- All 200 result directories contain `input.jpg`, selected `sd2_bld_grid.png`, selected `audit.json`, selected masks, `input_output.jpg`, `selected.json`, and three parseable candidate audits.
- Exactly 600 candidate rows exist, with three candidates for every selected result.
- All 200 rows have independent FR, FVA, FS, MNAC, and collateral metrics.
- All 200 selected rows and 600 candidate rows contain the spatial-area measurements required by the selection policy.
- FID and CD are reported separately for both tasks or explicitly record a reproducible evaluator dependency failure.
- Contact sheets and the paper-comparison report are generated.
- The failure report contains no unresolved generation failure.
- No git commit is created.

## Expected Cost

The three-candidate isolated-process MPS run can require roughly three to six hours, depending on checkpoint load time and thermal throttling. A complete run contains exactly 600 candidate generations and 200 selected results. Output images, traces, copied inputs, and comparison images may consume one to several gigabytes. The run is designed to survive interruption and continue from valid candidate audits.
