"""Calibration-frozen matched-success preservation comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class FrontierPoint:
    effort: str
    success: float
    drift: float


@dataclass(frozen=True)
class Interpolation:
    left_effort: str
    right_effort: str
    left_weight: float
    right_weight: float

    def to_weights(self) -> dict[str, float]:
        if self.left_effort == self.right_effort:
            return {self.left_effort: 1.0}
        return {
            self.left_effort: self.left_weight,
            self.right_effort: self.right_weight,
        }


@dataclass(frozen=True)
class FrozenOperatingPoints:
    grid: tuple[float, ...]
    weights: dict[str, dict[float, dict[str, float]]]


def calibration_frontier(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity_floor: float,
    locality_ceiling: float,
) -> list[FrontierPoint]:
    """Aggregate safe efforts and retain only non-dominated operating points."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["effort"]), []).append(row)
    candidates = []
    for effort, values in grouped.items():
        success = _mean(values, "target_success")
        drift = _mean(values, "independent_non_target_drift")
        identity = _mean(values, "identity_cosine")
        locality = _mean(values, "outside_semantic_l1")
        if identity >= identity_floor and locality <= locality_ceiling:
            candidates.append(FrontierPoint(effort, success, drift))
    candidates.sort(key=lambda item: (item.success, item.drift, item.effort))

    frontier = []
    for candidate in candidates:
        dominated = any(
            other.success >= candidate.success
            and other.drift <= candidate.drift
            and (
                other.success > candidate.success
                or other.drift < candidate.drift
            )
            for other in candidates
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def freeze_common_operating_points(
    frontiers: Mapping[str, Sequence[FrontierPoint]],
    *,
    step: float,
) -> FrozenOperatingPoints:
    """Freeze interpolation weights on the shared calibration success range."""

    if step <= 0 or not math.isfinite(step):
        raise ValueError("step must be positive and finite")
    if len(frontiers) < 2 or any(not points for points in frontiers.values()):
        raise ValueError("at least two non-empty frontiers are required")
    ordered = {
        variant: sorted(points, key=lambda item: item.success)
        for variant, points in frontiers.items()
    }
    lower = max(points[0].success for points in ordered.values())
    upper = min(points[-1].success for points in ordered.values())
    if lower > upper + 1e-12:
        raise ValueError("frontiers have no common success range")

    grid = []
    current = math.ceil((lower - 1e-12) / step) * step
    if current > upper + 1e-12:
        grid = [lower] if math.isclose(lower, upper) else [lower, upper]
    else:
        while current <= upper + 1e-12:
            grid.append(round(current, 12))
            current += step
        if grid and grid[0] < lower - 1e-10:
            grid.pop(0)
    if not grid:
        grid = [lower]

    weights: dict[str, dict[float, dict[str, float]]] = {}
    for variant, points in ordered.items():
        weights[variant] = {}
        for target in grid:
            weights[variant][target] = _interpolation(
                points,
                target,
            ).to_weights()
    return FrozenOperatingPoints(tuple(grid), weights)


def normalized_auc(points: Sequence[tuple[float, float]]) -> float:
    """Trapezoidal drift area divided by the covered success width."""

    ordered = sorted((float(x), float(y)) for x, y in points)
    if not ordered:
        raise ValueError("at least one point is required")
    if len(ordered) == 1:
        return ordered[0][1]
    width = ordered[-1][0] - ordered[0][0]
    if width <= 0:
        raise ValueError("success points must cover a positive range")
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in zip(
            ordered,
            ordered[1:],
        )
    )
    return area / width


def matched_estimates(
    rows: Sequence[Mapping[str, Any]],
    frozen: FrozenOperatingPoints,
    *,
    fixed_variant: str,
    adaptive_variant: str,
) -> dict[str, Any]:
    """Apply frozen weights and return paired source-level estimates."""

    contributions = _paired_contributions(
        rows,
        frozen,
        fixed_variant=fixed_variant,
        adaptive_variant=adaptive_variant,
    )
    intervals = {}
    for metric, by_cluster in contributions.items():
        values = [
            value
            for cluster_values in by_cluster.values()
            for value in cluster_values
        ]
        intervals[metric] = sum(values) / len(values)
    curves = _source_curves(rows, frozen)
    highest = frozen.grid[-1]

    variant_estimates = {}
    for variant in (fixed_variant, adaptive_variant):
        selected = [
            record
            for record in curves
            if record["variant"] == variant and record["success_grid"] == highest
        ]
        variant_estimates[variant] = {
            "drift": _record_mean(selected, "drift"),
            "target_success": _record_mean(selected, "target_success"),
            "safety_pass_rate": _record_mean(selected, "safety_pass_rate"),
            "normalized_auc": _mean_source_auc(
                curves,
                variant,
                frozen.grid,
            ),
        }
    return {
        "highest_common_success": highest,
        "variants": variant_estimates,
        "adaptive_minus_fixed": intervals,
    }


def paired_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    frozen: FrozenOperatingPoints,
    *,
    fixed_variant: str,
    adaptive_variant: str,
    seed: int,
    samples: int = 10_000,
) -> dict[str, dict[str, float]]:
    """Percentile intervals for adaptive-minus-fixed paired cluster effects."""

    import numpy as np

    if samples <= 0:
        raise ValueError("samples must be positive")
    contributions = _paired_contributions(
        rows,
        frozen,
        fixed_variant=fixed_variant,
        adaptive_variant=adaptive_variant,
    )
    cluster_names = sorted(
        set.intersection(
            *(set(values) for values in contributions.values())
        )
    )
    if not cluster_names:
        raise ValueError("no paired identity clusters are available")
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(cluster_names),
        size=(samples, len(cluster_names)),
    )
    result = {}
    for metric, by_cluster in contributions.items():
        estimate_values = [
            value
            for cluster in cluster_names
            for value in by_cluster[cluster]
        ]
        bootstrapped = []
        for draw in draws:
            values = [
                value
                for index in draw
                for value in by_cluster[cluster_names[int(index)]]
            ]
            bootstrapped.append(sum(values) / len(values))
        result[metric] = {
            "estimate": float(sum(estimate_values) / len(estimate_values)),
            "low": float(np.quantile(bootstrapped, 0.025)),
            "high": float(np.quantile(bootstrapped, 0.975)),
        }
    return result


def acceptance_flags(
    estimates: Mapping[str, Any],
    intervals: Mapping[str, Mapping[str, float]],
    *,
    fixed_variant: str,
    adaptive_variant: str,
) -> dict[str, bool]:
    fixed = estimates["variants"][fixed_variant]
    adaptive = estimates["variants"][adaptive_variant]
    target = intervals["target_success"]
    safety = intervals["safety_pass_rate"]
    drift = intervals["drift"]
    auc = intervals["normalized_auc"]
    flags = {
        "target_equivalent": target["low"] >= -0.05
        and target["high"] <= 0.05,
        "drift_reduction_10_percent": adaptive["drift"]
        <= 0.9 * fixed["drift"]
        and drift["high"] < 0,
        "frontier_auc_lower": auc["high"] < 0,
        "joint_safety_95_percent": min(
            fixed["safety_pass_rate"],
            adaptive["safety_pass_rate"],
        )
        >= 0.95,
        "safety_noninferior": safety["low"] >= -0.02,
    }
    flags["supported"] = all(flags.values())
    return flags


def _interpolation(
    points: Sequence[FrontierPoint],
    target: float,
) -> Interpolation:
    for point in points:
        if math.isclose(point.success, target, abs_tol=1e-12):
            return Interpolation(point.effort, point.effort, 1.0, 0.0)
    for left, right in zip(points, points[1:]):
        if left.success <= target <= right.success:
            width = right.success - left.success
            if width <= 0:
                continue
            right_weight = (target - left.success) / width
            return Interpolation(
                left.effort,
                right.effort,
                1.0 - right_weight,
                right_weight,
            )
    raise ValueError("cannot extrapolate beyond a calibration frontier")


def _source_curves(
    rows: Sequence[Mapping[str, Any]],
    frozen: FrozenOperatingPoints,
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, str, str, str],
        list[Mapping[str, Any]],
    ] = {}
    clusters: dict[tuple[str, str], str] = {}
    for row in rows:
        variant = str(row["variant"])
        if variant not in frozen.weights:
            continue
        source = str(row["source_id"])
        effort = str(row["effort"])
        key = (str(row["identity_cluster"]), source, variant, effort)
        grouped.setdefault(key, []).append(row)
        clusters[(variant, source)] = str(row["identity_cluster"])

    effort_values = {}
    for key, values in grouped.items():
        effort_values[key] = {
            "drift": _mean(values, "independent_non_target_drift"),
            "target_success": _mean(values, "target_success"),
            "safety_pass_rate": sum(
                float(row["identity_cosine"]) >= 0.90
                and float(row["outside_semantic_l1"]) <= 0.02
                for row in values
            )
            / len(values),
        }

    records = []
    sources = sorted({(key[0], key[1], key[2]) for key in grouped})
    for cluster, source, variant in sources:
        for grid in frozen.grid:
            weights = frozen.weights[variant][grid]
            selected = {
                effort: effort_values.get((cluster, source, variant, effort))
                for effort in weights
            }
            if any(value is None for value in selected.values()):
                continue
            records.append(
                {
                    "identity_cluster": cluster,
                    "source_id": source,
                    "variant": variant,
                    "success_grid": grid,
                    **{
                        metric: sum(
                            weights[effort] * selected[effort][metric]
                            for effort in weights
                        )
                        for metric in (
                            "drift",
                            "target_success",
                            "safety_pass_rate",
                        )
                    },
                }
            )
    return records


def _paired_contributions(
    rows: Sequence[Mapping[str, Any]],
    frozen: FrozenOperatingPoints,
    *,
    fixed_variant: str,
    adaptive_variant: str,
) -> dict[str, dict[str, list[float]]]:
    curves = _source_curves(rows, frozen)
    by_key = {
        (row["identity_cluster"], row["source_id"], row["variant"], row["success_grid"]): row
        for row in curves
    }
    clusters_sources = sorted(
        {
            (row["identity_cluster"], row["source_id"])
            for row in curves
        }
    )
    highest = frozen.grid[-1]
    contributions = {
        "drift": {},
        "target_success": {},
        "safety_pass_rate": {},
        "normalized_auc": {},
    }
    for cluster, source in clusters_sources:
        fixed_high = by_key.get((cluster, source, fixed_variant, highest))
        adaptive_high = by_key.get(
            (cluster, source, adaptive_variant, highest)
        )
        if fixed_high is None or adaptive_high is None:
            continue
        for metric in ("drift", "target_success", "safety_pass_rate"):
            contributions[metric].setdefault(cluster, []).append(
                adaptive_high[metric] - fixed_high[metric]
            )
        fixed_curve = [
            by_key.get((cluster, source, fixed_variant, grid))
            for grid in frozen.grid
        ]
        adaptive_curve = [
            by_key.get((cluster, source, adaptive_variant, grid))
            for grid in frozen.grid
        ]
        if any(item is None for item in (*fixed_curve, *adaptive_curve)):
            continue
        fixed_auc = normalized_auc(
            [
                (item["success_grid"], item["drift"])
                for item in fixed_curve
            ]
        )
        adaptive_auc = normalized_auc(
            [
                (item["success_grid"], item["drift"])
                for item in adaptive_curve
            ]
        )
        contributions["normalized_auc"].setdefault(cluster, []).append(
            adaptive_auc - fixed_auc
        )
    if any(not values for values in contributions.values()):
        raise ValueError("test rows do not cover all frozen paired points")
    return contributions


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return sum(values) / len(values)


def _record_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    if not rows:
        raise ValueError("no rows available for matched estimate")
    return sum(float(row[field]) for row in rows) / len(rows)


def _mean_source_auc(
    curves: Sequence[Mapping[str, Any]],
    variant: str,
    grid: Sequence[float],
) -> float:
    sources = sorted(
        {
            (str(row["identity_cluster"]), str(row["source_id"]))
            for row in curves
            if row["variant"] == variant
        }
    )
    values = []
    for cluster, source in sources:
        selected = [
            row
            for row in curves
            if row["variant"] == variant
            and row["identity_cluster"] == cluster
            and row["source_id"] == source
        ]
        by_grid = {row["success_grid"]: row for row in selected}
        if all(value in by_grid for value in grid):
            values.append(
                normalized_auc(
                    [(value, by_grid[value]["drift"]) for value in grid]
                )
            )
    if not values:
        raise ValueError("no complete source frontier is available")
    return sum(values) / len(values)
