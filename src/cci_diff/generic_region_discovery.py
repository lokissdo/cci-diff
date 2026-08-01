"""Target-generic semantic-region proposals for bounded discovery runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from cci_diff.counterfactual_graph import (
    RegionSetEvidence,
    eligible_candidate_region_sets,
)


RegionTuple = tuple[str, ...]


@dataclass(frozen=True)
class BeamSearchConfig:
    """Budgets for source screening and intervention-guided beam search."""

    atomic_shortlist_size: int = 6
    beam_width: int = 4
    level_evaluation_budget: int = 6
    max_components: int = 3
    minimum_samples: int = 1

    def __post_init__(self) -> None:
        for name in (
            "atomic_shortlist_size",
            "beam_width",
            "level_evaluation_budget",
            "minimum_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_components, bool)
            or not isinstance(self.max_components, int)
            or not 1 <= self.max_components <= 3
        ):
            raise ValueError("max_components must be an integer from 1 to 3")


def shortlist_atomic_components(
    screening_rows: Iterable[Mapping[str, Any]],
    config: BeamSearchConfig,
) -> RegionTuple:
    """Rank atomic components using source-only Grad-CAM screening evidence."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    sample_ids: set[Any] = set()
    for row in screening_rows:
        region = str(row.get("region", "")).strip()
        if not region:
            raise ValueError("Every screening row must name a region")
        grouped.setdefault(region, []).append(row)
        if "sample_id" in row:
            sample_ids.add(row["sample_id"])
    if not grouped:
        raise ValueError("At least one screening row is required")

    total_samples = max(len(sample_ids), max(len(rows) for rows in grouped.values()))

    def rank(region: str) -> tuple[Any, ...]:
        rows = grouped[region]
        density = _median_field(rows, "region_density")
        mass = _median_field(rows, "captured_mass")
        proposal = _median_field(rows, "proposal_score")
        area = _median_field(rows, "mask_fraction")
        coverage = len({row.get("sample_id", index) for index, row in enumerate(rows)})
        return (
            -density,
            -mass,
            -proposal,
            -(coverage / total_samples),
            area,
            region,
        )

    return tuple(sorted(grouped, key=rank)[: config.atomic_shortlist_size])


def propose_region_sets(
    atomic_shortlist: Sequence[str],
    beam: Sequence[RegionTuple],
    cardinality: int,
    config: BeamSearchConfig,
) -> tuple[RegionTuple, ...]:
    """Propose a deterministic, budgeted beam level of semantic region sets."""

    if not 1 <= cardinality <= config.max_components:
        raise ValueError("cardinality must be within the configured component limit")
    atoms = tuple(dict.fromkeys(str(item).strip() for item in atomic_shortlist))
    if not atoms or any(not item for item in atoms):
        raise ValueError("atomic_shortlist must contain non-empty region names")
    atom_rank = {region: index for index, region in enumerate(atoms)}

    if cardinality == 1:
        candidates = {(region,) for region in atoms}
    else:
        candidates = {
            tuple(sorted((*regions, atom)))
            for regions in beam
            for atom in atoms
            if atom not in regions
            and len(set(regions)) == cardinality - 1
        }

    def proposal_rank(regions: RegionTuple) -> tuple[Any, ...]:
        return (
            sum(atom_rank.get(region, len(atoms)) for region in regions),
            tuple(atom_rank.get(region, len(atoms)) for region in regions),
            regions,
        )

    return tuple(
        sorted(candidates, key=proposal_rank)[: config.level_evaluation_budget]
    )


def advance_beam(
    evidence: Iterable[RegionSetEvidence],
    config: BeamSearchConfig,
) -> tuple[RegionTuple, ...]:
    """Advance using supported Pareto evidence, with a deterministic fallback."""

    rows = tuple(evidence)
    if not rows:
        raise ValueError("At least one evidence item is required")
    supported_regions = set(
        eligible_candidate_region_sets(rows, config.minimum_samples)
    )
    candidates = tuple(row for row in rows if row.regions in supported_regions)
    if not candidates:
        candidates = tuple(row for row in rows if _has_complete_evidence(row))
    if not candidates:
        raise ValueError("No evidence item has complete finite selection metrics")

    ranked = sorted(
        candidates,
        key=lambda row: (
            -row.flip_rate,
            -row.mean_effect,
            float(row.mean_mask_fraction),
            len(row.regions),
            row.regions,
        ),
    )
    return tuple(row.regions for row in ranked[: config.beam_width])


def _median_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Screening field {field!r} must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"Screening field {field!r} must be finite")
        values.append(value)
    return float(median(values))


def _has_complete_evidence(item: RegionSetEvidence) -> bool:
    area = item.mean_mask_fraction
    return (
        item.sample_count > 0
        and math.isfinite(item.flip_rate)
        and math.isfinite(item.mean_effect)
        and area is not None
        and math.isfinite(area)
        and area > 0.0
    )
