"""Metrics for causal concept intervention counterfactuals.

These helpers intentionally avoid ML-framework dependencies so the audit logic
can be tested locally and reused around different diffusion implementations.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable, Mapping, Sequence

ConceptScores = Mapping[str, float]


def concept_delta(
    original: ConceptScores,
    counterfactual: ConceptScores,
    concept: str,
) -> float:
    """Return absolute score movement for one concept."""

    return abs(float(counterfactual[concept]) - float(original[concept]))


def target_concept_success(
    counterfactual: ConceptScores,
    concept: str,
    *,
    desired_value: int,
    threshold: float = 0.5,
) -> bool:
    """Check whether a concept detector reached the requested binary value."""

    score = float(counterfactual[concept])
    if desired_value == 1:
        return score >= threshold
    if desired_value == 0:
        return score < threshold
    raise ValueError("desired_value must be 0 or 1")


def preservation_score(
    original: ConceptScores,
    counterfactual: ConceptScores,
    preserved_concepts: Iterable[str],
) -> float:
    """Return 1 minus mean absolute change across preserved concepts."""

    deltas = [concept_delta(original, counterfactual, c) for c in preserved_concepts]
    if not deltas:
        return 1.0
    return max(0.0, 1.0 - mean(deltas))


def concept_leakage(
    original: ConceptScores,
    counterfactual: ConceptScores,
    non_target_concepts: Iterable[str],
) -> float:
    """Return mean unintended movement over non-target concepts."""

    deltas = [concept_delta(original, counterfactual, c) for c in non_target_concepts]
    if not deltas:
        return 0.0
    return mean(deltas)


def counterfactual_purity(target_delta: float, leakage_deltas: Iterable[float]) -> float:
    """Measure how much of the semantic movement belongs to the target concept."""

    leakage_total = sum(float(delta) for delta in leakage_deltas)
    target_delta = float(target_delta)
    denominator = target_delta + leakage_total
    if denominator == 0:
        return 0.0
    return target_delta / denominator


def causal_concept_effect(before_scores: Sequence[float], after_scores: Sequence[float]) -> float:
    """Return mean classifier score change after a controlled intervention."""

    if len(before_scores) != len(after_scores):
        raise ValueError("before_scores and after_scores must have the same length")
    if not before_scores:
        return 0.0
    deltas = [float(after) - float(before) for before, after in zip(before_scores, after_scores)]
    return mean(deltas)


def bias_audit_matrix(records: Iterable[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    """Group mean score changes by intervention concept and target classifier."""

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        intervention = str(record["intervention"])
        classifier = str(record["classifier"])
        delta = float(record["after"]) - float(record["before"])
        grouped[intervention][classifier].append(delta)

    return {
        intervention: {
            classifier: mean(deltas)
            for classifier, deltas in classifier_deltas.items()
        }
        for intervention, classifier_deltas in grouped.items()
    }
