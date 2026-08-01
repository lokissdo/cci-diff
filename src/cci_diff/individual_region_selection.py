"""Source-only semantic region selection from a frozen influence graph."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from cci_diff.concept_graph import sha256_file
from cci_diff.region_screening import canonical_region_sets


RegionTuple = tuple[str, ...]


@dataclass(frozen=True)
class FrozenRegionSetEvidence:
    """Finite discovery statistics for one selector candidate."""

    mean_effect: float
    flip_rate: float
    effect_ci_low: float
    mean_mask_fraction: float

    def __post_init__(self) -> None:
        for name in (
            "mean_effect",
            "flip_rate",
            "effect_ci_low",
            "mean_mask_fraction",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.mean_mask_fraction <= 0.0:
            raise ValueError("mean_mask_fraction must be positive")


@dataclass(frozen=True)
class FrozenInfluencePolicy:
    """Immutable class-level region evidence used during held-out inference."""

    target: str
    desired_value: int
    verified_regions: RegionTuple
    fallback_regions: RegionTuple
    region_set_effects: Mapping[RegionTuple, float]
    graph_path: str
    graph_sha256: str
    candidate_region_sets: tuple[RegionTuple, ...] = ()
    region_set_evidence: Mapping[
        RegionTuple, FrozenRegionSetEvidence
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target = self.target.strip()
        verified = _canonical_regions(self.verified_regions)
        fallback = _canonical_regions(self.fallback_regions)
        candidates = tuple(
            sorted(
                {
                    _canonical_regions(regions)
                    for regions in self.candidate_region_sets
                }
            )
        )
        if not target:
            raise ValueError("target must be non-empty")
        if isinstance(self.desired_value, bool) or self.desired_value not in (0, 1):
            raise ValueError("desired_value must be 0 or 1")
        if not verified:
            raise ValueError("verified_regions must be non-empty")
        if len(verified) > 8:
            raise ValueError("verified_regions cannot contain more than eight regions")
        if not fallback or not set(fallback).issubset(verified):
            raise ValueError("fallback regions must be non-empty and verified")
        if any(
            not regions or not set(regions).issubset(verified)
            for regions in candidates
        ):
            raise ValueError("candidate region sets must be non-empty and verified")
        effects = {}
        for regions, effect in self.region_set_effects.items():
            canonical = _canonical_regions(regions)
            value = float(effect)
            if not canonical or not set(canonical).issubset(verified):
                raise ValueError("region-set effects must use verified regions")
            if not math.isfinite(value):
                raise ValueError("region-set effects must be finite")
            if canonical in effects:
                raise ValueError(f"Duplicate region-set effect: {canonical}")
            effects[canonical] = value
        evidence = {}
        for regions, item in self.region_set_evidence.items():
            canonical = _canonical_regions(regions)
            if not canonical or not set(canonical).issubset(verified):
                raise ValueError("region-set evidence must use verified regions")
            if not isinstance(item, FrozenRegionSetEvidence):
                raise TypeError("region_set_evidence values must be frozen evidence")
            if canonical in evidence:
                raise ValueError(f"Duplicate frozen region-set evidence: {canonical}")
            evidence[canonical] = item
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "verified_regions", verified)
        object.__setattr__(self, "fallback_regions", fallback)
        object.__setattr__(self, "candidate_region_sets", candidates)
        object.__setattr__(
            self,
            "region_set_effects",
            MappingProxyType(effects),
        )
        object.__setattr__(
            self,
            "region_set_evidence",
            MappingProxyType(evidence),
        )

    def global_effect(self, regions: RegionTuple) -> float:
        """Return exact-set evidence, falling back to singleton effect sum."""

        canonical = _canonical_regions(regions)
        exact = self.region_set_effects.get(canonical)
        if exact is not None:
            return exact
        return sum(
            self.region_set_effects.get((region,), 0.0)
            for region in canonical
        )


@dataclass(frozen=True)
class IndividualRegionSelection:
    """One deterministic source-only region decision."""

    selected_regions: RegionTuple
    available_regions: RegionTuple
    missing_regions: RegionTuple
    coverage: float
    mask_fraction: float
    coverage_threshold: float
    fallback_used: bool
    fallback_reason: str | None
    region_importance: Mapping[str, float]
    candidate_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "region_importance",
            MappingProxyType(dict(self.region_importance)),
        )


def load_frozen_influence_policy(
    path: str | Path,
) -> FrozenInfluencePolicy:
    """Load globally verified regions and effect evidence from discovery."""

    graph_path = Path(path)
    if not graph_path.is_file():
        raise FileNotFoundError(f"Influence graph not found: {graph_path}")
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_types = {
        payload[key]
        for key in ("type", "graph_type")
        if payload.get(key) is not None
    }
    expected_type = "classifier_counterfactual_influence"
    if not graph_types or graph_types != {expected_type}:
        raise ValueError(
            "Unsupported influence graph type; expected "
            f"{expected_type!r}"
        )
    desired_value = payload.get("desired_value")
    if isinstance(desired_value, bool) or desired_value not in (0, 1):
        raise ValueError("desired_value must be 0 or 1")

    verified = []
    for edge in payload.get("verified_edges") or []:
        if edge.get("relation") != "classifier_counterfactual_influence":
            continue
        if edge.get("source") != payload.get("target"):
            raise ValueError("Verified edge source must match graph target")
        region = str(edge.get("target", "")).strip()
        if region:
            verified.append(region)
    explicit_verified = payload.get("verified_regions")
    audit_regions = _canonical_regions(
        explicit_verified if explicit_verified is not None else verified
    )
    if not audit_regions:
        raise ValueError("Influence graph has no verified regions")
    generation_regions = payload.get("generation_regions")
    verified_regions = audit_regions
    raw_candidates = payload.get("candidate_region_sets")
    if raw_candidates is None:
        legacy_candidate = generation_regions or payload.get("selected_regions")
        raw_candidates = [legacy_candidate] if legacy_candidate else []
    candidate_region_sets = tuple(
        sorted({_canonical_regions(regions) for regions in raw_candidates})
    )
    fallback_regions = _canonical_regions(
        payload.get("fallback_regions")
        or generation_regions
        or payload.get("selected_regions")
        or ()
    )

    effects = {}
    frozen_evidence = {}
    for item in payload.get("region_set_evidence") or []:
        try:
            regions = _canonical_regions(item["regions"])
            effect = float(item["mean_effect"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed region-set evidence") from error
        if not regions or not math.isfinite(effect):
            raise ValueError("Malformed region-set evidence")
        if not set(regions).issubset(verified_regions):
            continue
        if regions in effects:
            raise ValueError(f"Duplicate region-set evidence: {regions}")
        effects[regions] = effect
        complete_fields = (
            "flip_rate",
            "effect_ci_low",
            "mean_mask_fraction",
        )
        if all(item.get(name) is not None for name in complete_fields):
            try:
                frozen_evidence[regions] = FrozenRegionSetEvidence(
                    mean_effect=effect,
                    flip_rate=float(item["flip_rate"]),
                    effect_ci_low=float(item["effect_ci_low"]),
                    mean_mask_fraction=float(item["mean_mask_fraction"]),
                )
            except (TypeError, ValueError) as error:
                raise ValueError("Malformed region-set evidence") from error

    return FrozenInfluencePolicy(
        target=str(payload.get("target", "")),
        desired_value=desired_value,
        verified_regions=verified_regions,
        fallback_regions=fallback_regions,
        region_set_effects=effects,
        graph_path=str(graph_path),
        graph_sha256=sha256_file(graph_path),
        candidate_region_sets=candidate_region_sets,
        region_set_evidence=frozen_evidence,
    )


def select_individual_region_set(
    saliency: np.ndarray,
    component_masks: Mapping[str, np.ndarray],
    policy: FrozenInfluencePolicy,
    *,
    coverage_threshold: float = 0.80,
    eps: float = 1e-12,
) -> IndividualRegionSelection:
    """Select the minimum-area verified subset covering source saliency."""

    if not 0.0 < coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold must be in (0, 1]")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    saliency_array = np.asarray(saliency, dtype=np.float64)
    if saliency_array.ndim != 2:
        raise ValueError("saliency must be a two-dimensional array")
    if not np.all(np.isfinite(saliency_array)):
        raise ValueError("saliency must contain only finite values")
    if np.any(saliency_array < 0):
        raise ValueError("saliency must be non-negative")

    masks: dict[str, np.ndarray] = {}
    for region in policy.verified_regions:
        if region not in component_masks:
            continue
        mask = np.asarray(component_masks[region])
        if mask.shape != saliency_array.shape:
            raise ValueError("saliency and component masks must have identical shapes")
        binary = mask > 0
        if np.any(binary):
            masks[region] = binary
    available = tuple(sorted(masks))
    missing = tuple(
        region for region in policy.verified_regions if region not in masks
    )
    if not available:
        raise ValueError("No verified region has an available non-empty mask")

    region_sets = canonical_region_sets(
        available,
        max_set_size=len(available),
    )
    global_union = np.logical_or.reduce([masks[region] for region in available])
    support = float(np.sum(saliency_array[global_union]))
    importance = {
        region: (
            float(np.sum(saliency_array[masks[region]]) / support)
            if support > eps
            else 0.0
        )
        for region in policy.verified_regions
    }

    if support <= eps:
        fallback = tuple(
            region
            for region in policy.fallback_regions
            if region in masks
        )
        if not fallback:
            raise ValueError(
                "No available fallback region for zero-saliency selection"
            )
        union = _union_for_regions(masks, fallback)
        return IndividualRegionSelection(
            selected_regions=fallback,
            available_regions=available,
            missing_regions=missing,
            coverage=0.0,
            mask_fraction=float(np.mean(union)),
            coverage_threshold=coverage_threshold,
            fallback_used=True,
            fallback_reason="zero_saliency_in_verified_union",
            region_importance=importance,
            candidate_count=len(region_sets),
        )

    feasible = []
    for regions in region_sets:
        union = _union_for_regions(masks, regions)
        coverage = float(np.sum(saliency_array[union]) / support)
        if coverage + eps < coverage_threshold:
            continue
        feasible.append(
            (
                float(np.mean(union)),
                len(regions),
                -policy.global_effect(regions),
                regions,
                coverage,
            )
        )
    if not feasible:
        raise ValueError(
            "No region set reaches coverage_threshold despite non-zero support"
        )
    mask_fraction, _, _, selected, coverage = min(feasible)
    return IndividualRegionSelection(
        selected_regions=selected,
        available_regions=available,
        missing_regions=missing,
        coverage=coverage,
        mask_fraction=mask_fraction,
        coverage_threshold=coverage_threshold,
        fallback_used=False,
        fallback_reason=None,
        region_importance=importance,
        candidate_count=len(region_sets),
    )


def _canonical_regions(regions) -> RegionTuple:
    return tuple(sorted({str(region).strip() for region in regions if str(region).strip()}))


def _union_for_regions(
    masks: Mapping[str, np.ndarray],
    regions: RegionTuple,
) -> np.ndarray:
    return np.logical_or.reduce([masks[region] for region in regions])
