"""Spatial change measurements and target-first candidate selection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _weighted_mean(values, weights) -> float:
    import numpy as np

    total = float(np.sum(weights))
    if total == 0:
        return 0.0
    return float(np.sum(values * weights) / total)


def measure_spatial_change(
    source_path: str | Path,
    output_path: str | Path,
    semantic_mask_path: str | Path,
    generation_mask_path: str | Path,
) -> dict[str, float]:
    """Measure changed area and magnitude relative to semantic mask support."""

    import numpy as np
    from PIL import Image

    with Image.open(source_path) as image:
        source_image = image.convert("RGB")
    with Image.open(output_path) as image:
        output_image = image.convert("RGB")
    if source_image.size != output_image.size:
        source_image = source_image.resize(
            output_image.size,
            Image.Resampling.BILINEAR,
        )
    source = np.asarray(source_image, dtype=np.float32) / 255.0
    output = np.asarray(output_image, dtype=np.float32) / 255.0

    size = source_image.size
    with Image.open(semantic_mask_path) as image:
        semantic_image = image.convert("L").resize(size, Image.Resampling.NEAREST)
        semantic_values = np.asarray(semantic_image, dtype=np.float32) / 255.0
    with Image.open(generation_mask_path) as image:
        generation_image = image.convert("L").resize(
            size,
            Image.Resampling.BILINEAR,
        )
        generation = np.asarray(generation_image, dtype=np.float32) / 255.0

    semantic = semantic_values >= 0.5
    delta = np.max(np.abs(output - source), axis=2)
    pixel_l1 = np.mean(np.abs(output - source), axis=2)
    semantic_weights = semantic.astype(np.float32)
    outside_semantic_weights = 1.0 - semantic_weights
    outside_generation = 1.0 - generation

    metrics = {
        "semantic_mask_fraction": float(semantic.mean()),
        "generation_mask_fraction": float(generation.mean()),
        "inside_semantic_l1": _weighted_mean(pixel_l1, semantic_weights),
        "outside_semantic_l1": _weighted_mean(
            pixel_l1,
            outside_semantic_weights,
        ),
        "inside_generation_l1": _weighted_mean(pixel_l1, generation),
        "outside_generation_l1": _weighted_mean(
            pixel_l1,
            outside_generation,
        ),
    }
    for level in (1, 5, 10):
        changed = delta > level / 255.0
        metrics[f"changed_fraction_{level}"] = float(changed.mean())
        metrics[f"outside_semantic_fraction_{level}"] = float(
            np.logical_and(changed, ~semantic).mean()
        )
        metrics[f"outside_generation_fraction_{level}"] = float(
            np.mean(changed * outside_generation)
        )
    return metrics


def _finite_number(row: Mapping[str, Any], field: str) -> float:
    if field not in row:
        raise ValueError(f"Candidate row requires {field!r}")
    try:
        value = float(row[field])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Candidate field {field!r} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"Candidate field {field!r} must be finite")
    return value


def _optional_score(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None:
        return float("-inf")
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return score if math.isfinite(score) else float("-inf")


def select_spatial_candidate(
    rows: Sequence[Mapping[str, Any]],
    target_probability: float = 0.8,
) -> Mapping[str, Any]:
    """Select target success first, then minimize changed spatial support."""

    if not rows:
        raise ValueError("Candidate selection requires at least one row")
    if not 0 <= target_probability <= 1:
        raise ValueError("target_probability must be between 0 and 1")

    probabilities = [_finite_number(row, "desired_probability") for row in rows]
    areas = [_finite_number(row, "changed_fraction_5") for row in rows]
    passing_indices = [
        index
        for index, probability in enumerate(probabilities)
        if probability >= target_probability
    ]
    if passing_indices:
        return min(
            (rows[index] for index in passing_indices),
            key=lambda row: (
                _finite_number(row, "changed_fraction_5"),
                _finite_number(row, "outside_semantic_fraction_5"),
                -_finite_number(row, "desired_probability"),
                -_optional_score(row, "identity_cosine"),
            ),
        )
    return min(
        zip(rows, probabilities, areas),
        key=lambda item: (-item[1], item[2]),
    )[0]
