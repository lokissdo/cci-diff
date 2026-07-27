# Heatmap Ranking And Mask Deduplication Design

## Purpose

Improve classifier-only semantic-region screening without changing the
meaning of CelebAMask-HQ components, and avoid diffusion runs whose actual
semantic union masks are identical.

## Semantic Masks

All registered masks retain their original meaning. In particular, `skin`
remains the supplied full skin mask. The workflow does not create or evaluate
`skin_without_perioral` or any other residual region.

## Region Statistics

For normalized Grad-CAM++ map \(A_i\) and semantic mask \(M_{ir}\), compute:

\[
D_{ir}=
\frac{\sum_p A_i(p)M_{ir}(p)}
     {\max(\sum_p M_{ir}(p),\epsilon)}
\]

\[
C_{ir}=
\frac{\sum_p A_i(p)M_{ir}(p)}
     {\max(\sum_p A_i(p),\epsilon)}.
\]

Aggregate each region across discovery images using:

- median heatmap intensity \(D_r=\operatorname{median}_i D_{ir}\);
- median captured saliency \(C_r=\operatorname{median}_i C_{ir}\);
- mask availability frequency;
- median mask fraction.

## Eligibility And Ranking

A region is eligible when:

- its mask is available in at least `minimum_coverage_frequency`, default
  `0.90`;
- its median captured saliency is at least
  `minimum_captured_saliency`, default `0.02`.

Eligible regions are ranked lexicographically by:

1. descending median heatmap intensity;
2. descending median captured saliency;
3. descending availability frequency;
4. ascending median mask fraction;
5. canonical region name.

The first `top_k` regions form the intervention candidate pool. `top_k`
remains a configurable compute budget in this version; the separate dynamic
region-count design remains future work.

## Exact Union Deduplication

For candidate region set \(S\) and discovery image \(i\), construct the hard
semantic union:

\[
U_i(S)=\bigvee_{r\in S}M_{ir}.
\]

Two region sets are equivalent only when:

\[
U_i(S_1)=U_i(S_2)
\quad\text{for every discovery image }i.
\]

Equivalent sets share one cohort signature made from ordered per-image mask
hashes. Select the canonical representative by:

1. fewest region labels;
2. canonical region tuple.

Run diffusion only for canonical representatives. Record every skipped set as
an alias of its representative in `intervention_manifest.json`. Exact
equivalence is safe to skip. Strict superset relations are not skipped.

## Analysis Semantics

Aliases do not create additional intervention observations and must not be
used to estimate interactions. A `skin + mouth` alias of `skin` is one
treatment, not evidence that mouth has zero interaction with skin.

## Resumability

Existing audits under canonical region-set directories remain reusable.
No output directory is deleted. A dry planning pass may consolidate existing
canonical observations before additional generation.

## Provenance

The screening manifest records aggregation fields, eligibility thresholds,
ranking order, and selected candidates. The intervention manifest records
requested sets, executed canonical sets, cohort signatures, and aliases.
