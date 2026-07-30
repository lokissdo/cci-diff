# Attacked Region 300-Image Evaluation Design

## Goal

Run two sequential, restartable, end-to-end smile-removal evaluations on the
same 300 seed-42 CelebA-HQ sources:

1. mouth region only;
2. mouth, upper lip, and lower lip.

Each job compares raw BLD (`A0`, controller disabled) with adaptive CCI
(`A11`, trust-region controller), applies the smooth-boundary post-attack to
both methods, and reports:

| Method | FID down | sFID down | FVA up | FS up | MNAC down | CD down | COUT up | FR (%) up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Experiment Contract

- Task: remove `Smiling`.
- Sample count: exactly 300 unique eligible images.
- Sampling: deterministic random seed 42.
- Cohort: identical image IDs for both methods and both region jobs.
- Diffusion: SD2.1 base, 35 inference steps, MPS, float32.
- Generation-mask dilation: 8 pixels.
- Methods: `A0` and `A11` only.
- Attack: adaptive smooth-boundary post-attack using the existing epsilon
  schedule `0.05,0.08,0.10,0.30,0.50` and boundary margin `0.03`.
- Metric input: the final saved post-attack output, never the pre-attack image.
- Classifier source of truth:
  `models/resnet50_multilabel_model.pth`.
- The same classifier is intentionally used for CCI guidance, attack
  optimization, COUT, FR, MNAC, and CD.
- The report must disclose that the classifier-dependent metrics are not
  independent-oracle measurements.

Each region job produces 300 A0 outputs and 300 A11 outputs. The full
scheduler therefore produces 1,200 final outputs.

## Region Configuration

The pilot runner gains a user-facing region override:

```text
--region_components mouth
--region_components mouth upper_lip lower_lip
```

The canonical names map to CelebAMask-HQ annotations:

- `mouth` -> `mouth`
- `upper_lip` -> `u_lip`
- `lower_lip` -> `l_lip`

The override controls all region-dependent behavior:

- the graph region component list;
- the semantic union mask;
- the generation mask;
- the locality audit;
- the smooth-boundary attack support.

The scheduler records the resolved canonical and annotation component names in
each pilot manifest. Both jobs use the 300 IDs selected by the first job. The
second job does not resample.

## Correct Binary COUT

The existing thesis wrapper is invalid for smile removal because it treats
CelebA attribute 0 (`5_o_Clock_Shadow`) and attribute 31 (`Smiling`) as
mutually exclusive classes. CelebA attributes are independent sigmoid
outputs.

The corrected COUT implementation uses only the classifier's `Smiling`
probability at index 31. For every transition image:

```text
p_source  = p_smiling
p_desired = 1 - p_smiling
```

For each aligned source/output pair:

1. compute per-pixel RGB absolute change magnitude;
2. sort pixels from largest to smallest change;
3. construct 50 transition points by progressively replacing source pixels
   with output pixels;
4. evaluate `p_source` and `p_desired` at every point;
5. integrate each curve with the trapezoidal rule;
6. calculate:

```text
COUT = AUPC(p_desired) - AUPC(p_source)
```

The task score is the arithmetic mean of the 300 per-image scores. Higher is
better. The evaluator also writes each per-image COUT value and the evaluated
count. Because cohort selection requires the source to be classified as
smiling by this same classifier, all 300 rows are eligible for COUT and FR.

The evaluator consumes probabilities directly. It must not apply sigmoid a
second time to the classifier's existing sigmoid outputs.

## Remaining Metric Definitions

- **FID:** Frechet distance between all 300 source and final-output InceptionV3
  activations.
- **sFID:** deterministic symmetric FID using seed 42, two disjoint 150-image
  splits, both cross-split directions, and their arithmetic mean.
- **FVA:** percentage of paired source/output VGGFace2 cosine similarities
  greater than `0.5`.
- **FS:** mean paired source/output SimSiam cosine similarity.
- **MNAC:** mean number of classifier binary attributes that change at
  threshold `0.5`, excluding `Smiling`.
- **CD:** sum of absolute correlations between each target-excluded attribute
  change vector and the `Smiling` change vector. Constant vectors contribute
  zero.
- **FR (%):** percentage of the 300 sources whose classifier label changes
  from smiling to not smiling in the final post-attack output.

FID, sFID, FVA, and FS use their existing evaluator models. COUT, FR, MNAC,
and CD use the configured local multi-label classifier.

## Data Flow

For each region job:

```text
fixed source IDs
  -> A0 and A11 SD2 generation
  -> smooth-boundary attack on each result
  -> selected final attacked output
  -> classifier metrics: FR, MNAC, CD, COUT
  -> embedding metrics: FVA, FS
  -> Inception metrics: FID, sFID
  -> one method-comparison table
```

The mouth-plus-lips job starts only after the mouth-only generation,
completeness validation, and metrics finish successfully.

## Scheduler and Restart Behavior

Create:

```text
scripts/run_attacked_region_300.sh
```

The script:

- resolves the repository and virtual-environment paths without depending on
  the caller's working directory;
- runs under `caffeinate -dimsu`;
- uses `set -eu`;
- writes separate logs and outputs for the two region jobs;
- reuses valid completed candidate audits when restarted;
- validates exactly 300 A0 and 300 A11 selected rows before metrics;
- stops immediately if generation, attack, completeness validation, or metric
  evaluation fails;
- writes a final combined Markdown and CSV report containing four rows:
  A0/A11 for mouth-only and A0/A11 for mouth-plus-lips.

Output layout:

```text
outputs/attacked_a0_a11_smile300_seed42/
  mouth/
    pilot_results.csv
    classifier_pair_metrics.csv
    metric_summary.csv
    metric_summary.md
    ...
  mouth_upper_lower_lip/
    pilot_results.csv
    classifier_pair_metrics.csv
    metric_summary.csv
    metric_summary.md
    ...
  combined_metrics.csv
  combined_metrics.md
  scheduler.log
```

## Failure Handling

- An unavailable source image, required annotation, or detectable face makes
  an ID ineligible during initial cohort selection.
- A generation or attack subprocess failure leaves completed rows intact and
  exits nonzero. Rerunning the scheduler resumes from valid audits.
- Metrics do not start until both methods have exactly the same 300 IDs.
- Non-finite activations, probabilities, transition curves, or aggregate
  values fail the job instead of producing placeholders.
- FID/sFID cache entries are reused only when their complete image-path
  fingerprint matches.

## Testing

Add focused tests before implementation for:

- region override parsing and canonical-to-annotation mapping;
- graph/binding generation for mouth-only and mouth-plus-lips;
- A0 and A11 both receiving the post-attack arguments;
- identical cohort enforcement between region jobs;
- binary COUT using `1 - p_smiling`, with no second sigmoid;
- COUT directionality and trapezoidal integration;
- MNAC excluding index 31;
- FR requiring a smiling-to-not-smiling transition;
- CD handling constant vectors;
- smile-only 300-row FID/sFID evaluation;
- scheduler command order, completeness gates, and final report columns.

Run focused tests, then the complete repository `tests/` suite. No 300-image
generation is part of automated tests.
