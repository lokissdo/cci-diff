# Pareto Region Selection Design

## Objective

Select a localized semantic generation mask from counterfactual intervention
evidence without requiring a fixed flip-rate threshold. Broad masks such as
`skin` remain visible in the discovery evidence but must not win solely because
they make classifier flipping easier.

## Scope

This change affects only the final global region-set selection. Grad-CAM++
proposal generation, masked diffusion interventions, bootstrap aggregation,
interaction reporting, and the downstream CCI pipeline remain unchanged.

## Evidence

For every tested region set \(S\), discovery already measures:

- \(E(S)\): mean desired-class probability change;
- \(F(S)\): generation-classifier flip rate;
- \(A(S)\): mean semantic mask fraction;
- preservation measurements including identity, outside-mask change,
  changed-pixel fraction, and non-target drift.

Only sets with complete finite target-effect, flip-rate, and mask-area
measurements participate in automatic generation-mask selection.

## Pareto Filtering

A set \(S_a\) dominates \(S_b\) when:

\[
E(S_a) \ge E(S_b), \qquad
F(S_a) \ge F(S_b), \qquad
A(S_a) \le A(S_b),
\]

with at least one strict inequality. Dominated sets remain in exported evidence
but cannot become the selected generation mask.

## Final Selection

Among non-dominated sets with positive mean target effect, rank by:

1. descending target efficiency \(E(S) / \max(A(S), \epsilon)\);
2. descending mean target effect;
3. descending flip rate;
4. ascending mask fraction;
5. lower outside-mask change and non-target drift;
6. higher identity similarity;
7. fewer semantic components and deterministic region name order.

The small numerical \(\epsilon\) only prevents division by zero and is not a
model-selection weight.

If no set has positive mean target effect, select the non-dominated set with
the highest mean target effect, then highest flip rate, then smallest area.
Mark the result as `fallback_nonpositive_effect`.

The graph selection status becomes:

- `pareto_efficient`: selected from positive-effect Pareto candidates;
- `fallback_nonpositive_effect`: no measured set moved toward the target.

`required_flip_rate` remains in legacy artifacts only for compatibility and is
not used to select the generation mask.

## Outputs

The influence graph and region metrics add:

- `pareto_optimal`;
- `target_efficiency`;
- `dominated_by`;
- the selection rule and status in graph provenance.

The complete evidence table remains available so broad masks such as `skin`
can be reported as classifier-sensitive audit regions even when they are not
selected for generation.

## Testing

Unit tests cover:

- removal of a strictly dominated set;
- preservation of effect-area trade-offs on the Pareto frontier;
- selection by target effect per area;
- deterministic tie-breaking;
- nonpositive-effect fallback;
- missing or zero-area validation;
- serialization of Pareto evidence and selection provenance.

An integration test verifies that graph discovery no longer selects a broad
mask merely because it is the only set above a fixed flip-rate threshold.
