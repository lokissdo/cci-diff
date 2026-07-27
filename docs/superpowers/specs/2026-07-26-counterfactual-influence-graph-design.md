# Counterfactual Influence Graph Design

## Purpose

Replace hand-authored target-to-region policy with a classifier-specific,
empirically verified counterfactual influence graph. The graph answers:

> Which semantic regions must be allowed to change to flip one classifier
> label with the smallest spatial and non-target cost?

The graph describes model-level intervention effects. It does not claim that a
classifier label biologically causes a facial component.

## Scope

Version one supports binary CelebA classifier targets and CelebAMask-HQ
semantic regions. It produces reusable graph evidence and a selected region
set for the existing clean-CCI pipeline.

The implementation contains three stages:

1. Grad-CAM++ screening ranks semantic regions as inexpensive proposals.
2. Same-seed masked diffusion interventions measure actual counterfactual
   effects for singleton and joint region sets.
3. An analyzer estimates effects and confidence, measures region synergy, and
   selects the smallest region set meeting a requested flip-rate threshold.

Grad-CAM++ may propose a region, but only measured diffusion interventions can
create a verified graph edge.

## Non-Goals

- Do not infer real-world or anatomical causality.
- Do not derive edges from CelebA co-occurrence alone.
- Do not use graph edge strengths as diffusion loss coefficients.
- Do not replace online residual-driven dual feedback.
- Do not tune on the final evaluation cohort.
- Do not require a new learned neural graph model in version one.

## Terminology

- `target`: classifier output being flipped, such as `Smiling`.
- `desired_value`: requested binary target value.
- `region`: one registered semantic component, such as `upper_lip`.
- `region_set`: one or more regions enabled in a paired intervention.
- `proposal edge`: target-to-region edge supported only by localization.
- `verified edge`: target-to-region edge with positive intervention evidence
  and a bootstrap confidence interval.
- `counterfactual influence graph`: versioned output containing measured
  model-level effects, costs, confidence, interactions, and the selected
  region set.

## Inputs

### Screening

- frozen CelebA classifier checkpoint and output index;
- source images;
- aligned CelebAMask-HQ component masks;
- candidate semantic regions;
- Grad-CAM++ saliency maps.

### Intervention Generation

- a validated clean-CCI graph used as a controller template;
- selected source image IDs;
- candidate semantic regions;
- one or more seeds shared across every region set;
- SD2 checkpoint, classifier checkpoint, and identity checkpoint;
- fixed generation settings shared across every intervention.

### Analysis

Each intervention row contains:

```text
target
desired_value
sample_id
seed
regions
source_probability
output_probability
target_pass
mask_fraction
identity_cosine
non_target_drift
outside_l1
changed_fraction
output_path
audit_path
```

Rows with the same sample and seed differ only in their allowed region set.

## Region Screening

For normalized saliency map `A` and binary component mask `M_r`, compute:

```text
captured_mass_r = sum(A * M_r) / max(sum(A), eps)
region_density_r = sum(A * M_r) / max(sum(M_r), eps)
mask_fraction_r = mean(M_r)
proposal_score_r = captured_mass_r * region_density_r
```

Aggregate scores by sample. Rank regions by mean proposal score and report
coverage frequency. Screening must not mark an edge as verified.

## Paired Interventions

For every requested sample and seed:

1. Generate every singleton region set.
2. Generate combinations up to `max_set_size`.
3. Use identical diffusion settings and random seed for every region set.
4. Build a temporary execution graph whose semantic components equal the
   active region set.
5. Build a hard union mask named `target_region` for strict auditing.
6. Disable post-generation adversarial attack during discovery so the graph
   measures the diffusion intervention rather than a pixel attack.
7. Resume completed interventions by validating their audit artifacts.

Candidate region count is limited to eight. The runner rejects a combinatorial
grid larger than 255 region sets.

## Counterfactual Effect

Convert classifier probability to desired-class probability:

```text
p_desired(p, y*) = p        if y* = 1
                   1 - p    if y* = 0
```

For sample `i`, seed `s`, and region set `R`:

```text
delta_(i,s,R) =
    p_desired(output_probability, y*)
  - p_desired(source_probability, y*)
```

For each region set report:

- row count and distinct sample count;
- generation-classifier flip rate at threshold `0.5`;
- mean and median desired-probability change;
- cluster-bootstrap 95% confidence interval, sampling image IDs so multiple
  seeds from one image are not treated as independent identities;
- mean mask fraction;
- mean identity cosine;
- mean non-target drift;
- mean outside-mask L1;
- mean changed fraction.

## Verified Edges

A singleton target-to-region edge is verified when:

```text
mean_effect > 0
and confidence_interval_low > 0
and sample_count >= minimum_samples
```

Otherwise the edge remains a proposal with its evidence recorded.

## Region Interaction

For a pair `{a,b}`:

```text
synergy(a,b) =
    mean_effect({a,b})
  - mean_effect({a})
  - mean_effect({b})
```

The same definition extends descriptively to larger sets by subtracting
singleton effects. Synergy is reported as an interaction statistic, not as an
independent causal claim.

## Region-Set Selection

Selection is lexicographic and target-first.

If one or more region sets reach `minimum_flip_rate`:

```text
minimize:
1. mean mask fraction
2. mean outside L1
3. mean changed fraction
4. mean non-target drift
5. negative mean identity cosine
6. number of regions
7. canonical region tuple
```

If no set reaches the threshold:

```text
maximize flip rate
then maximize mean target effect
then apply the same minimal-change ordering
```

The default minimum flip rate is `0.95`.

## Graph Output

The analyzer writes versioned JSON:

```json
{
  "version": 1,
  "graph_type": "classifier_counterfactual_influence",
  "target": {
    "attribute": "Smiling",
    "label_index": 31,
    "desired_value": 0,
    "decision_threshold": 0.5
  },
  "selection": {
    "minimum_flip_rate": 0.95,
    "selected_regions": ["mouth", "upper_lip", "lower_lip"],
    "threshold_reached": true
  },
  "edges": [],
  "region_sets": [],
  "provenance": {}
}
```

Every edge records effect, confidence interval, success probability, costs,
sample count, and verification status. Provenance includes input digests,
seeds, generation settings, and source cohort IDs.

## CCI Integration

The discovered graph compiles to the existing execution policy by replacing
only:

- `region.audit_role` with `target_region`;
- `region.components` with `selected_regions`;
- image bindings with the selected component masks and their hard union.

The controller remains unchanged:

```text
lambda_k(0) = 0
constraint coefficients come from normalized residuals and dual updates
```

The influence graph chooses what may change. The controller determines how
much pressure is required at each denoising step.

## Artifacts

The screening stage writes:

```text
screening_rows.csv
screening_summary.csv
screening_manifest.json
```

The intervention stage writes:

```text
intervention_manifest.json
intervention_results.csv
policies/<sample>/<region-set>/graph.json
policies/<sample>/<region-set>/binding.json
policies/<sample>/<region-set>/target_region.png
runs/<sample>/<seed>/<region-set>/
```

The analysis stage writes:

```text
influence_graph.json
region_set_metrics.csv
interactions.csv
selected_execution_graph.json
discovery_report.md
```

## Validation

Unit tests cover:

- desired-probability conversion for both binary directions;
- deterministic cluster bootstrap;
- positive and unsupported confidence intervals;
- synergy calculation;
- target-first minimal-set selection;
- no-passing-set fallback;
- canonical region serialization;
- Grad-CAM mask overlap scores;
- semantic union-mask construction;
- temporary graph and binding generation;
- audit-to-observation extraction;
- resumable intervention validation;
- CLI validation and direct execution.

An integration smoke test uses synthetic audit files and masks. It does not
load diffusion checkpoints.

The learned graph must be evaluated on a held-out cohort against:

- the existing manual graph;
- Grad-CAM-only region selection;
- singleton-only discovery;
- joint-region discovery.

Primary evaluation is generation-classifier flip rate. Secondary metrics are
FID, FVA, FS, MNAC, CD, changed area, identity cosine, non-target drift, and
outside-mask change.

## Research Claim

The supported claim is:

> The method automatically discovers a classifier-specific semantic
> intervention policy from paired masked counterfactual experiments.

The unsupported claim is:

> The learned edges represent biological or real-world causes of facial
> attributes.
