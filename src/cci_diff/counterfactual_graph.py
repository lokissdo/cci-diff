"""Evidence and selection for classifier-specific counterfactual influence graphs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np


RegionTuple = tuple[str, ...]


@dataclass(frozen=True)
class InterventionObservation:
    """One same-seed masked counterfactual intervention result."""

    target: str
    desired_value: int
    sample_id: int
    seed: int
    regions: RegionTuple
    source_probability: float
    output_probability: float
    mask_fraction: float | None = None
    identity_cosine: float | None = None
    non_target_drift: float | None = None
    outside_l1: float | None = None
    changed_fraction: float | None = None
    output_path: str | None = None
    audit_path: str | None = None

    def __post_init__(self) -> None:
        target = self.target.strip()
        if not target:
            raise ValueError("target must be non-empty")
        if isinstance(self.desired_value, bool) or self.desired_value not in (0, 1):
            raise ValueError("desired_value must be 0 or 1")
        if isinstance(self.sample_id, bool) or not isinstance(self.sample_id, int):
            raise ValueError("sample_id must be an integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        regions = tuple(sorted({str(region).strip() for region in self.regions}))
        if not regions or any(not region for region in regions):
            raise ValueError("regions must contain at least one non-empty region")
        _validate_probability("source_probability", self.source_probability)
        _validate_probability("output_probability", self.output_probability)
        for name in (
            "mask_fraction",
            "identity_cosine",
            "non_target_drift",
            "outside_l1",
            "changed_fraction",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "regions", regions)

    @property
    def source_desired_probability(self) -> float:
        return _desired_probability(self.source_probability, self.desired_value)

    @property
    def output_desired_probability(self) -> float:
        return _desired_probability(self.output_probability, self.desired_value)

    @property
    def target_effect(self) -> float:
        return self.output_desired_probability - self.source_desired_probability

    @property
    def target_pass(self) -> bool:
        return self.output_desired_probability >= 0.5


@dataclass(frozen=True)
class RegionSetEvidence:
    """Aggregated evidence for one semantic region set."""

    regions: RegionTuple
    row_count: int
    sample_count: int
    flip_rate: float
    mean_effect: float
    median_effect: float
    effect_ci_low: float
    effect_ci_high: float
    mean_mask_fraction: float | None = None
    mean_identity_cosine: float | None = None
    mean_non_target_drift: float | None = None
    mean_outside_l1: float | None = None
    mean_changed_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regions"] = list(self.regions)
        return payload


@dataclass(frozen=True)
class RegionInteraction:
    """Additive interaction evidence for a pair of semantic regions."""

    regions: RegionTuple
    joint_effect: float
    singleton_effect_sum: float
    synergy: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["regions"] = list(self.regions)
        return payload


@dataclass(frozen=True)
class InfluenceGraphResult:
    """Serializable result of counterfactual region discovery."""

    target: str
    desired_value: int
    selected_regions: RegionTuple
    selection_status: str
    required_flip_rate: float
    minimum_samples: int
    verified_edges: tuple[tuple[str, str, str], ...]
    evidence: tuple[RegionSetEvidence, ...]
    interactions: tuple[RegionInteraction, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "type": "classifier_counterfactual_influence",
            "graph_type": "classifier_counterfactual_influence",
            "target": self.target,
            "desired_value": self.desired_value,
            "selected_regions": list(self.selected_regions),
            "selection_status": self.selection_status,
            "required_flip_rate": self.required_flip_rate,
            "minimum_samples": self.minimum_samples,
            "verified_edges": [
                {"source": source, "target": target, "relation": relation}
                for source, target, relation in self.verified_edges
            ],
            "region_set_evidence": [item.to_dict() for item in self.evidence],
            "interactions": [item.to_dict() for item in self.interactions],
            "provenance": dict(self.provenance),
        }


def aggregate_region_sets(
    observations: Iterable[InterventionObservation],
    *,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    random_seed: int = 0,
) -> dict[RegionTuple, RegionSetEvidence]:
    """Aggregate interventions with image-level clustered uncertainty."""

    rows = tuple(observations)
    if not rows:
        raise ValueError("At least one intervention observation is required")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    targets = {(row.target, row.desired_value) for row in rows}
    if len(targets) != 1:
        raise ValueError("All observations must share target and desired_value")
    already_satisfied = [
        (row.sample_id, row.seed)
        for row in rows
        if row.source_desired_probability >= 0.5
    ]
    if already_satisfied:
        raise ValueError(
            "Source already satisfies the desired target for interventions: "
            f"{already_satisfied[:5]}"
        )
    keys = [(row.sample_id, row.seed, row.regions) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate intervention for sample, seed, and regions")

    grouped: dict[RegionTuple, list[InterventionObservation]] = {}
    for row in rows:
        grouped.setdefault(row.regions, []).append(row)

    result: dict[RegionTuple, RegionSetEvidence] = {}
    for group_index, regions in enumerate(sorted(grouped)):
        group = grouped[regions]
        sample_groups: dict[int, list[InterventionObservation]] = {}
        for row in group:
            sample_groups.setdefault(row.sample_id, []).append(row)

        sample_effects = np.asarray(
            [
                np.mean([row.target_effect for row in sample_rows])
                for _, sample_rows in sorted(sample_groups.items())
            ],
            dtype=np.float64,
        )
        sample_flips = np.asarray(
            [
                np.mean([float(row.target_pass) for row in sample_rows])
                for _, sample_rows in sorted(sample_groups.items())
            ],
            dtype=np.float64,
        )
        ci_low, ci_high = _cluster_bootstrap_interval(
            sample_effects,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            random_seed=random_seed + group_index,
        )
        result[regions] = RegionSetEvidence(
            regions=regions,
            row_count=len(group),
            sample_count=len(sample_groups),
            flip_rate=float(np.mean(sample_flips)),
            mean_effect=float(np.mean(sample_effects)),
            median_effect=float(np.median(sample_effects)),
            effect_ci_low=ci_low,
            effect_ci_high=ci_high,
            mean_mask_fraction=_clustered_optional_mean(
                sample_groups, "mask_fraction"
            ),
            mean_identity_cosine=_clustered_optional_mean(
                sample_groups, "identity_cosine"
            ),
            mean_non_target_drift=_clustered_optional_mean(
                sample_groups, "non_target_drift"
            ),
            mean_outside_l1=_clustered_optional_mean(
                sample_groups, "outside_l1"
            ),
            mean_changed_fraction=_clustered_optional_mean(
                sample_groups, "changed_fraction"
            ),
        )
    return result


def compute_interactions(
    evidence_by_regions: Mapping[RegionTuple, RegionSetEvidence],
) -> tuple[RegionInteraction, ...]:
    """Compute additive pair synergy where both singleton effects exist."""

    interactions = []
    for regions, joint in sorted(evidence_by_regions.items()):
        if len(regions) != 2:
            continue
        left = evidence_by_regions.get((regions[0],))
        right = evidence_by_regions.get((regions[1],))
        if left is None or right is None:
            continue
        singleton_sum = left.mean_effect + right.mean_effect
        interactions.append(
            RegionInteraction(
                regions=regions,
                joint_effect=joint.mean_effect,
                singleton_effect_sum=singleton_sum,
                synergy=joint.mean_effect - singleton_sum,
            )
        )
    return tuple(interactions)


def select_region_set(
    evidence_by_regions: Mapping[RegionTuple, RegionSetEvidence],
    *,
    required_flip_rate: float = 0.95,
) -> RegionSetEvidence:
    """Choose target-feasible regions first, then minimize intervention cost."""

    if not evidence_by_regions:
        raise ValueError("At least one region-set evidence item is required")
    _validate_probability("required_flip_rate", required_flip_rate)
    candidates = tuple(evidence_by_regions.values())
    passing = tuple(
        item for item in candidates if item.flip_rate >= required_flip_rate
    )
    if passing:
        return min(passing, key=_intervention_cost_key)
    return min(
        candidates,
        key=lambda item: (
            -item.mean_effect,
            -item.flip_rate,
            *_intervention_cost_key(item),
        ),
    )


def build_influence_graph(
    *,
    target: str,
    desired_value: int,
    evidence_by_regions: Mapping[RegionTuple, RegionSetEvidence],
    required_flip_rate: float = 0.95,
    minimum_samples: int = 20,
    provenance: Mapping[str, Any] | None = None,
) -> InfluenceGraphResult:
    """Build a reviewed classifier-specific influence graph from evidence."""

    if not target.strip():
        raise ValueError("target must be non-empty")
    if isinstance(desired_value, bool) or desired_value not in (0, 1):
        raise ValueError("desired_value must be 0 or 1")
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    selected = select_region_set(
        evidence_by_regions, required_flip_rate=required_flip_rate
    )
    verified_edges = tuple(
        (
            target,
            item.regions[0],
            "classifier_counterfactual_influence",
        )
        for item in sorted(
            evidence_by_regions.values(), key=lambda value: value.regions
        )
        if len(item.regions) == 1
        and item.sample_count >= minimum_samples
        and item.mean_effect > 0.0
        and item.effect_ci_low > 0.0
    )
    evidence = tuple(
        sorted(evidence_by_regions.values(), key=lambda value: value.regions)
    )
    return InfluenceGraphResult(
        target=target,
        desired_value=desired_value,
        selected_regions=selected.regions,
        selection_status=(
            "meets_requirement"
            if selected.flip_rate >= required_flip_rate
            else "fallback"
        ),
        required_flip_rate=required_flip_rate,
        minimum_samples=minimum_samples,
        verified_edges=verified_edges,
        evidence=evidence,
        interactions=compute_interactions(evidence_by_regions),
        provenance=dict(provenance or {}),
    )


def _desired_probability(probability: float, desired_value: int) -> float:
    return probability if desired_value == 1 else 1.0 - probability


def _validate_probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")


def _cluster_bootstrap_interval(
    sample_values: np.ndarray,
    *,
    bootstrap_samples: int,
    confidence: float,
    random_seed: int,
) -> tuple[float, float]:
    if sample_values.size == 1:
        value = float(sample_values[0])
        return value, value
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(
        0,
        sample_values.size,
        size=(bootstrap_samples, sample_values.size),
    )
    means = np.mean(sample_values[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, (alpha, 1.0 - alpha))
    return float(low), float(high)


def _clustered_optional_mean(
    sample_groups: Mapping[int, list[InterventionObservation]],
    field_name: str,
) -> float | None:
    sample_means = []
    for _, rows in sorted(sample_groups.items()):
        values = [
            float(value)
            for row in rows
            if (value := getattr(row, field_name)) is not None
        ]
        if values:
            sample_means.append(float(np.mean(values)))
    return float(np.mean(sample_means)) if sample_means else None


def _optional_cost(value: float | None) -> float:
    return float(value) if value is not None else math.inf


def _identity_cost(value: float | None) -> float:
    return -float(value) if value is not None else math.inf


def _intervention_cost_key(item: RegionSetEvidence) -> tuple[Any, ...]:
    return (
        _optional_cost(item.mean_mask_fraction),
        _optional_cost(item.mean_outside_l1),
        _optional_cost(item.mean_changed_fraction),
        _optional_cost(item.mean_non_target_drift),
        _identity_cost(item.mean_identity_cosine),
        len(item.regions),
        item.regions,
    )
