# Individual Grad-CAM Region Selection Design

## Goal

Use a frozen classifier-specific counterfactual influence graph to choose the
smallest likely sufficient semantic region set for each held-out source image,
then run exactly one CCI-BLD generation.

The selector uses only the source image, its Grad-CAM++ map, its semantic
component masks, and evidence already stored in the frozen global graph. It
does not inspect generated candidates, retry failed images, or update the
global graph.

## Experimental Separation

The discovery cohort and held-out evaluation cohort have different roles:

1. Discovery images build and verify the global class-to-region graph.
2. The global graph and individual-selection threshold are frozen.
3. Held-out images use that frozen policy without modifying it.

All region-set choices for a test image are made before diffusion. A failed
output remains a failed output.

## Inputs

The individual selector consumes:

- a source image;
- one frozen `classifier_counterfactual_influence` graph;
- the target concept and desired binary value from the template CCI graph;
- one Grad-CAM++ saliency map for the source classifier decision;
- image-specific semantic masks for globally verified regions;
- a frozen saliency-coverage threshold, default `0.80`.

The frozen influence graph provides:

- verified singleton target-to-region edges;
- the globally selected fallback region set;
- measured global effect evidence for region sets;
- classifier and discovery provenance.

## Source-Decision Saliency

Let the classifier probability for class `c` be `p_c(x)`. The source image
must lie on the opposite side of the desired decision:

```text
desired_value = 0: p_c(x) >= 0.5
desired_value = 1: p_c(x) < 0.5
```

Grad-CAM++ explains the current source decision. For positive sources it
targets `p_c`; for negative sources it targets `1 - p_c`, matching the
existing multi-label Grad-CAM adapter.

## Per-Image Region Scores

For saliency `A_i`, semantic mask `M_(i,r)`, and the union `M_(i,G)` of all
available globally verified regions:

```text
importance_(i,r) =
  sum(A_i * M_(i,r)) / max(sum(A_i * M_(i,G)), eps)
```

For each non-empty subset `R` of the available verified regions:

```text
coverage_i(R) =
  sum(A_i * union_mask_i(R)) / max(sum(A_i * M_(i,G)), eps)
```

The exact union prevents overlapping masks from double-counting saliency.

## Selection

Feasible sets satisfy:

```text
coverage_i(R) >= coverage_threshold
```

Select lexicographically:

1. minimum union-mask fraction;
2. minimum region count;
3. maximum global measured effect for the exact region set;
4. canonical region tuple.

Because the full available verified-region union has coverage `1.0`, at least
one set is feasible when saliency exists inside the global union.

If saliency inside the global union is numerically zero, use the frozen global
selected region set as a deterministic fallback. If some fallback masks are
unavailable, use the available fallback components. If none are available,
the sample is invalid and generation must not run.

## Generation Masks

The selected component masks produce:

- `target_region.png`: exact hard semantic union for auditing and constraints;
- the backend generation mask: dilated/feathered from the selected components
  for BLD blending and latent-gradient localization.

The temporary execution graph retains the frozen target, constraints,
controller, and edges, while replacing:

```text
region.audit_role = "target_region"
region.components = selected_regions
```

## One-Pass Runner

For every held-out image:

1. validate source eligibility and required mask files;
2. compute the source classifier probability and Grad-CAM++;
3. select one semantic region set;
4. write the selection and temporary execution policy;
5. run one clean-CCI feedback generation;
6. validate the output audit;
7. write target, identity, drift, locality, and spatial-change metrics.

There is no candidate grid, output-based reranking, post-generation attack,
retry, or region escalation.

## Artifacts

```text
individual_policy.json
individual_manifest.json
individual_selections.csv
individual_results.csv
policies/<sample>/graph.json
policies/<sample>/binding.json
policies/<sample>/target_region.png
runs/<sample>/
```

Each selection row records:

- sample ID and source probability;
- available and missing global regions;
- per-region importance;
- selected regions;
- selected coverage and mask fraction;
- threshold and fallback status;
- global graph digest.

## Evaluation

Compare the same held-out images and seeds under:

1. raw/manual BLD;
2. frozen global-region CCI;
3. individual source-only Grad-CAM CCI.

Primary target metrics are generation-classifier FR and desired-probability
change. Minimality metrics are semantic mask fraction, changed fraction,
outside-mask change, non-target drift, and identity cosine. Cohort quality
metrics remain FID, sFID, FVA, FS, MNAC, CD, and COUT where available.

The individual method is supported when it retains competitive target success
while reducing intervention area or collateral change relative to the frozen
global-region policy.

## Claim Boundary

The global graph contains intervention-verified, classifier-specific influence
edges. Grad-CAM++ chooses among those verified regions for each source image.
Grad-CAM importance itself is not treated as causal evidence.
