"""Semantic region screening and union-mask helpers."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


CELEBAMASK_COMPONENT_SUFFIXES = {
    "background": "background",
    "skin": "skin",
    "nose": "nose",
    "eye_glasses": "eye_g",
    "left_eye": "l_eye",
    "right_eye": "r_eye",
    "left_brow": "l_brow",
    "right_brow": "r_brow",
    "left_ear": "l_ear",
    "right_ear": "r_ear",
    "mouth": "mouth",
    "upper_lip": "u_lip",
    "lower_lip": "l_lip",
    "hair": "hair",
    "hat": "hat",
    "ear_ring": "ear_r",
    "necklace": "neck_l",
    "neck": "neck",
    "cloth": "cloth",
}


@dataclass(frozen=True)
class RegionScreenScore:
    region: str
    captured_mass: float
    region_density: float
    mask_fraction: float
    proposal_score: float


@dataclass(frozen=True)
class RegionSubsetEvidence:
    regions: tuple[str, ...]
    mean_saliency_coverage: float
    cohort_frequency: float
    mean_mask_fraction: float
    passes: bool


def score_region_masks(
    saliency: np.ndarray,
    component_masks: Mapping[str, np.ndarray],
    *,
    eps: float = 1e-12,
) -> tuple[RegionScreenScore, ...]:
    """Rank component masks by captured saliency mass and density."""

    saliency_array = np.asarray(saliency, dtype=np.float64)
    if saliency_array.ndim != 2:
        raise ValueError("saliency must be a two-dimensional array")
    if not np.all(np.isfinite(saliency_array)):
        raise ValueError("saliency must contain only finite values")
    if np.any(saliency_array < 0):
        raise ValueError("saliency must be non-negative")
    if not component_masks:
        raise ValueError("component_masks must be non-empty")
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be positive and finite")

    saliency_sum = max(float(np.sum(saliency_array)), eps)
    scores = []
    for region, mask in component_masks.items():
        canonical = str(region).strip()
        if not canonical:
            raise ValueError("region names must be non-empty")
        mask_array = np.asarray(mask)
        if mask_array.shape != saliency_array.shape:
            raise ValueError("saliency and masks must have identical shapes")
        binary = mask_array > 0
        pixel_count = int(np.count_nonzero(binary))
        if pixel_count == 0:
            raise ValueError(f"Mask for region {canonical!r} must be non-empty")
        overlap_sum = float(np.sum(saliency_array[binary]))
        captured_mass = overlap_sum / saliency_sum
        region_density = overlap_sum / max(pixel_count, eps)
        mask_fraction = pixel_count / binary.size
        scores.append(
            RegionScreenScore(
                region=canonical,
                captured_mass=captured_mass,
                region_density=region_density,
                mask_fraction=mask_fraction,
                proposal_score=captured_mass * region_density,
            )
        )
    return tuple(
        sorted(scores, key=lambda item: (-item.proposal_score, item.region))
    )


def select_screened_regions(
    summary: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    minimum_coverage_frequency: float,
    minimum_captured_saliency: float = 0.0,
) -> tuple[str, ...]:
    """Select concentrated, globally supported Grad-CAM proposals."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if top_k > 8:
        raise ValueError("top_k cannot exceed the eight-region grid limit")
    if not 0.0 <= minimum_coverage_frequency <= 1.0:
        raise ValueError("minimum_coverage_frequency must be in [0, 1]")
    if not 0.0 <= minimum_captured_saliency <= 1.0:
        raise ValueError("minimum_captured_saliency must be in [0, 1]")
    eligible = [
        row
        for row in summary
        if float(row["coverage_frequency"]) >= minimum_coverage_frequency
        and float(row["median_captured_mass"])
        >= minimum_captured_saliency
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(row["median_region_density"]),
            -float(row["median_captured_mass"]),
            -float(row["coverage_frequency"]),
            float(row["median_mask_fraction"]),
            str(row["region"]),
        ),
    )
    if len(ranked) < top_k:
        raise ValueError(
            f"Only {len(ranked)} regions meet minimum coverage "
            f"{minimum_coverage_frequency} and captured saliency "
            f"{minimum_captured_saliency}; required {top_k}"
        )
    return tuple(str(row["region"]) for row in ranked[:top_k])


def select_saliency_covering_regions(
    rows: Sequence[Mapping[str, Any]],
    *,
    saliency_coverage_threshold: float,
    cohort_frequency_threshold: float,
    max_regions: int = 4,
) -> tuple[
    tuple[str, ...],
    tuple[RegionSubsetEvidence, ...],
    str,
]:
    """Choose the smallest disjoint semantic union covering source saliency."""

    if not rows:
        raise ValueError("rows must be non-empty")
    if (
        isinstance(max_regions, bool)
        or not isinstance(max_regions, int)
        or max_regions <= 0
    ):
        raise ValueError("max_regions must be a positive integer")
    if max_regions > 4:
        raise ValueError("max_regions cannot exceed 4")
    for name, value in (
        ("saliency_coverage_threshold", saliency_coverage_threshold),
        ("cohort_frequency_threshold", cohort_frequency_threshold),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")

    sample_ids = tuple(sorted({int(row["sample_id"]) for row in rows}))
    regions = tuple(sorted({str(row["region"]) for row in rows}))
    by_sample_region = {
        (int(row["sample_id"]), str(row["region"])): row for row in rows
    }
    eligible_regions = tuple(
        region
        for region in regions
        if sum(
            (sample_id, region) in by_sample_region for sample_id in sample_ids
        )
        / len(sample_ids)
        >= cohort_frequency_threshold
    )
    if not eligible_regions:
        raise ValueError("No regions meet cohort_frequency_threshold")

    evidence = []
    for size in range(1, min(max_regions, len(eligible_regions)) + 1):
        for subset in itertools.combinations(eligible_regions, size):
            coverages = []
            areas = []
            for sample_id in sample_ids:
                subset_rows = [
                    by_sample_region[(sample_id, region)]
                    for region in subset
                    if (sample_id, region) in by_sample_region
                ]
                coverages.append(
                    min(
                        1.0,
                        sum(float(row["captured_mass"]) for row in subset_rows),
                    )
                )
                areas.append(
                    min(
                        1.0,
                        sum(float(row["mask_fraction"]) for row in subset_rows),
                    )
                )
            frequency = sum(
                value >= saliency_coverage_threshold for value in coverages
            ) / len(coverages)
            evidence.append(
                RegionSubsetEvidence(
                    regions=subset,
                    mean_saliency_coverage=sum(coverages) / len(coverages),
                    cohort_frequency=frequency,
                    mean_mask_fraction=sum(areas) / len(areas),
                    passes=frequency >= cohort_frequency_threshold,
                )
            )
    ordered = tuple(
        sorted(
            evidence,
            key=lambda item: (
                not item.passes,
                item.mean_mask_fraction if item.passes else -item.cohort_frequency,
                len(item.regions) if item.passes else -item.mean_saliency_coverage,
                -item.mean_saliency_coverage,
                item.regions,
            ),
        )
    )
    selected = ordered[0]
    status = "meets_coverage" if selected.passes else "fallback"
    return selected.regions, ordered, status


def canonical_region_sets(
    regions: list[str] | tuple[str, ...],
    *,
    max_set_size: int | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate deterministic non-empty region combinations."""

    canonical = tuple(sorted({str(region).strip() for region in regions}))
    if not canonical or any(not region for region in canonical):
        raise ValueError("regions must contain at least one non-empty name")
    if len(canonical) > 4:
        raise ValueError("At most four candidate regions are supported")
    maximum = len(canonical) if max_set_size is None else max_set_size
    if maximum <= 0 or maximum > len(canonical):
        raise ValueError("max_set_size must be between 1 and region count")
    combinations = tuple(
        combination
        for size in range(1, maximum + 1)
        for combination in itertools.combinations(canonical, size)
    )
    return combinations


def build_union_mask(
    component_masks: Mapping[str, np.ndarray | str | Path],
    regions: tuple[str, ...] | list[str],
    *,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Build an exact binary union for the requested semantic components."""

    canonical = tuple(sorted({str(region).strip() for region in regions}))
    if not canonical or any(not region for region in canonical):
        raise ValueError("regions must contain at least one non-empty name")
    missing = [region for region in canonical if region not in component_masks]
    if missing:
        raise ValueError(f"Missing component masks: {missing}")

    arrays = [_load_mask(component_masks[region]) for region in canonical]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError("component masks must have identical shapes")
    union = np.logical_or.reduce([array > 0 for array in arrays])
    if not np.any(union):
        raise ValueError("Union mask must be non-empty")
    result = union.astype(np.uint8) * 255
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result, mode="L").save(destination)
    return result


def celebamask_component_path(
    mask_root: str | Path,
    sample_id: int,
    region: str,
) -> Path:
    """Resolve a canonical region to the CelebAMask-HQ component filename."""

    if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0:
        raise ValueError("sample_id must be a non-negative integer")
    try:
        suffix = CELEBAMASK_COMPONENT_SUFFIXES[region]
    except KeyError as exc:
        raise ValueError(f"Unknown CelebAMask region: {region}") from exc
    return (
        Path(mask_root)
        / str(sample_id // 2000)
        / f"{sample_id:05d}_{suffix}.png"
    )


def _load_mask(value: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
    else:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Component mask not found: {path}")
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"))
    if array.ndim != 2:
        raise ValueError("component masks must be two-dimensional")
    return array
