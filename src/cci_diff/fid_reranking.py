"""Deterministic distribution-aware candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ReferenceProjection:
    """PCA state and projected reference statistics."""

    mean: np.ndarray
    components: np.ndarray
    projected_reference: np.ndarray
    projected_mean: np.ndarray
    projected_variance: np.ndarray


@dataclass(frozen=True)
class SelectionResult:
    """One selected candidate per source plus optimization provenance."""

    rows: tuple[Mapping[str, Any], ...]
    indices: tuple[int, ...]
    initial_fid: float
    final_fid: float
    target_passes: int
    accepted_swaps: int
    passes: int


def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _number(row: Mapping[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"candidate field {field!r} must be numeric") from error
    if not np.isfinite(value):
        raise ValueError(f"candidate field {field!r} must be finite")
    return value


def _truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def select_reference_ids(
    image_root: str | Path,
    *,
    count: int,
    excluded_ids: Collection[int],
) -> tuple[tuple[int, Path], ...]:
    """Select the first numeric image IDs outside the evaluation cohort."""

    if count <= 0:
        raise ValueError("reference count must be positive")
    excluded = {int(value) for value in excluded_ids}
    candidates = [
        (int(path.stem), path)
        for path in Path(image_root).glob("*.jpg")
        if path.stem.isdigit() and int(path.stem) not in excluded
    ]
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < count:
        raise ValueError(
            f"reference cohort requires {count} images; found {len(candidates)}"
        )
    return tuple(candidates[:count])


def _canonicalize_component_signs(components: np.ndarray) -> np.ndarray:
    canonical = np.asarray(components, dtype=np.float64).copy()
    for index, component in enumerate(canonical):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            canonical[index] *= -1
    return canonical


def fit_reference_projection(
    activations: np.ndarray,
    *,
    dimensions: int = 64,
) -> ReferenceProjection:
    """Fit deterministic PCA using only reference activations."""

    values = _finite_matrix(activations, "reference activations")
    if len(values) < 2:
        raise ValueError("reference projection requires at least two activations")
    if dimensions <= 0:
        raise ValueError("projection dimensions must be positive")
    rank = min(int(dimensions), len(values) - 1, values.shape[1])
    mean = values.mean(axis=0)
    _, _, right = np.linalg.svd(values - mean, full_matrices=False)
    components = _canonicalize_component_signs(right[:rank])
    projected = (values - mean) @ components.T
    return ReferenceProjection(
        mean=mean,
        components=components,
        projected_reference=projected,
        projected_mean=projected.mean(axis=0),
        projected_variance=np.maximum(projected.var(axis=0), 1e-12),
    )


def project_features(
    activations: np.ndarray,
    projection: ReferenceProjection,
) -> np.ndarray:
    """Project aligned Inception activations into reference PCA space."""

    values = _finite_matrix(activations, "activations")
    if values.shape[1] != projection.mean.shape[0]:
        raise ValueError("activation input dimension does not match projection")
    return (values - projection.mean) @ projection.components.T


def diagonal_mahalanobis(
    projected: np.ndarray,
    projection: ReferenceProjection,
) -> np.ndarray:
    """Return squared diagonal Mahalanobis distance for each projected row."""

    values = _finite_matrix(projected, "projected features")
    if values.shape[1] != projection.projected_mean.shape[0]:
        raise ValueError("projected feature dimension does not match reference")
    centered = values - projection.projected_mean
    return np.sum(centered * centered / projection.projected_variance, axis=1)


def frechet_distance(
    reference: np.ndarray,
    selected: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Calculate regularized Frechet distance for unequal cohort sizes."""

    from scipy.linalg import sqrtm

    reference = _finite_matrix(reference, "reference features")
    selected = _finite_matrix(selected, "selected features")
    if len(reference) < 2 or len(selected) < 2:
        raise ValueError("Frechet distance requires at least two feature rows")
    if reference.shape[1] != selected.shape[1]:
        raise ValueError("reference and selected features require equal dimensions")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    mean_reference = reference.mean(axis=0)
    mean_selected = selected.mean(axis=0)
    covariance_reference = np.atleast_2d(np.cov(reference, rowvar=False))
    covariance_selected = np.atleast_2d(np.cov(selected, rowvar=False))
    identity = np.eye(reference.shape[1], dtype=np.float64)
    regularized_reference = covariance_reference + epsilon * identity
    regularized_selected = covariance_selected + epsilon * identity
    covariance_root = sqrtm(regularized_reference @ regularized_selected)
    if not np.isfinite(covariance_root).all():
        raise ValueError("Frechet covariance product produced non-finite values")
    if np.iscomplexobj(covariance_root):
        covariance_root = covariance_root.real
    value = np.sum((mean_reference - mean_selected) ** 2) + np.trace(
        regularized_reference
        + regularized_selected
        - 2.0 * covariance_root
    )
    return max(float(value), 0.0)


def candidate_is_eligible(
    row: Mapping[str, Any],
    *,
    identity_minimum: float = 0.80,
    outside_l1_maximum: float = 0.03,
) -> bool:
    """Require target success, identity, and locality."""

    return (
        _number(row, "desired_probability") >= 0.5
        and _number(row, "identity_cosine") >= identity_minimum
        and _number(row, "outside_semantic_l1") <= outside_l1_maximum
    )


def _preservation_valid(
    row: Mapping[str, Any],
    *,
    identity_minimum: float,
    outside_l1_maximum: float,
) -> bool:
    return (
        _number(row, "identity_cosine") >= identity_minimum
        and _number(row, "outside_semantic_l1") <= outside_l1_maximum
    )


def _target_pass(row: Mapping[str, Any]) -> bool:
    return _number(row, "desired_probability") >= 0.5


def _group_indices(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    seen: set[tuple[int, int]] = set()
    for index, row in enumerate(rows):
        sample_id = int(row["sample_id"])
        seed = int(row["seed"])
        key = (sample_id, seed)
        if key in seen:
            raise ValueError(f"duplicate candidate for sample/seed {key}")
        seen.add(key)
        groups.setdefault(sample_id, []).append(index)
    if not groups:
        raise ValueError("candidate selection requires at least one row")
    for indices in groups.values():
        indices.sort(key=lambda index: int(rows[index]["seed"]))
    return dict(sorted(groups.items()))


def _validate_inputs(
    rows: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]]]:
    features = _finite_matrix(features, "candidate features")
    reference = _finite_matrix(reference, "reference features")
    if len(features) != len(rows):
        raise ValueError("candidate rows and features must align")
    if features.shape[1] != reference.shape[1]:
        raise ValueError("candidate and reference feature dimensions must align")
    return features, reference, _group_indices(rows)


def _target_pass_count(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> int:
    return sum(_target_pass(rows[index]) for index in indices)


def _make_result(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    features: np.ndarray,
    reference: np.ndarray,
    *,
    initial_fid: float | None = None,
    accepted_swaps: int = 0,
    passes: int = 0,
) -> SelectionResult:
    ordered = tuple(
        sorted(
            (int(index) for index in indices),
            key=lambda index: int(rows[index]["sample_id"]),
        )
    )
    final_fid = frechet_distance(reference, features[list(ordered)])
    return SelectionResult(
        rows=tuple(rows[index] for index in ordered),
        indices=ordered,
        initial_fid=final_fid if initial_fid is None else initial_fid,
        final_fid=final_fid,
        target_passes=_target_pass_count(rows, ordered),
        accepted_swaps=accepted_swaps,
        passes=passes,
    )


def select_single_seed(
    rows: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int = 42,
) -> SelectionResult:
    """Select one fixed seed for every source."""

    features, reference, groups = _validate_inputs(rows, features, reference)
    indices = []
    for sample_id, candidates in groups.items():
        matching = [
            index for index in candidates if int(rows[index]["seed"]) == int(seed)
        ]
        if len(matching) != 1:
            raise ValueError(f"sample {sample_id} requires exactly one seed {seed}")
        indices.append(matching[0])
    return _make_result(rows, indices, features, reference)


def select_random_candidates(
    rows: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    reference: np.ndarray,
    *,
    selector_seed: int = 20260725,
) -> SelectionResult:
    """Select one candidate per source with a fixed random generator."""

    features, reference, groups = _validate_inputs(rows, features, reference)
    generator = np.random.default_rng(selector_seed)
    indices = [
        candidates[int(generator.integers(0, len(candidates)))]
        for candidates in groups.values()
    ]
    return _make_result(rows, indices, features, reference)


def _fallback_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        -_number(row, "desired_probability"),
        -_number(row, "identity_cosine"),
        _number(row, "outside_semantic_l1"),
        _number(row, "post_attack_linf"),
        int(row["seed"]),
    )


def _eligible_key(
    row: Mapping[str, Any],
    distance: float,
) -> tuple[float, ...]:
    return (
        float(distance),
        0.0 if _truth(row.get("raw_target_pass", False)) else 1.0,
        _number(row, "post_attack_linf"),
        int(row["seed"]),
    )


def _independent_indices(
    rows: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    reference: np.ndarray,
    *,
    identity_minimum: float,
    outside_l1_maximum: float,
) -> list[int]:
    projection = fit_reference_projection(
        reference,
        dimensions=min(64, reference.shape[1]),
    )
    projected = project_features(features, projection)
    distances = diagonal_mahalanobis(projected, projection)
    groups = _group_indices(rows)
    selected = []
    for candidates in groups.values():
        eligible = [
            index
            for index in candidates
            if candidate_is_eligible(
                rows[index],
                identity_minimum=identity_minimum,
                outside_l1_maximum=outside_l1_maximum,
            )
        ]
        if eligible:
            selected.append(
                min(
                    eligible,
                    key=lambda index: _eligible_key(
                        rows[index],
                        distances[index],
                    ),
                )
            )
        else:
            selected.append(min(candidates, key=lambda index: _fallback_key(rows[index])))
    return selected


def select_independent_candidates(
    rows: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    reference: np.ndarray,
    *,
    identity_minimum: float = 0.80,
    outside_l1_maximum: float = 0.03,
) -> SelectionResult:
    """Select each source independently by reference-distribution distance."""

    features, reference, _ = _validate_inputs(rows, features, reference)
    indices = _independent_indices(
        rows,
        features,
        reference,
        identity_minimum=identity_minimum,
        outside_l1_maximum=outside_l1_maximum,
    )
    return _make_result(rows, indices, features, reference)


def select_global_fid_candidates(
    rows: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    reference: np.ndarray,
    *,
    minimum_passes: int,
    maximum_passes: int = 8,
    epsilon: float = 1e-6,
    identity_minimum: float = 0.80,
    outside_l1_maximum: float = 0.03,
) -> SelectionResult:
    """Minimize set-level FID while preserving target and preservation bounds."""

    features, reference, groups = _validate_inputs(rows, features, reference)
    if minimum_passes < 0 or minimum_passes > len(groups):
        raise ValueError("minimum_passes must be between zero and source count")
    if maximum_passes <= 0:
        raise ValueError("maximum_passes must be positive")
    current = _independent_indices(
        rows,
        features,
        reference,
        identity_minimum=identity_minimum,
        outside_l1_maximum=outside_l1_maximum,
    )
    if _target_pass_count(rows, current) < minimum_passes:
        raise ValueError("initial selection cannot satisfy minimum_passes")
    initial_fid = frechet_distance(reference, features[current], epsilon=epsilon)
    current_fid = initial_fid
    accepted_swaps = 0
    completed_passes = 0

    for completed_passes in range(1, maximum_passes + 1):
        changed = False
        for sample_id, candidates in groups.items():
            current_for_sample = next(
                index
                for index in current
                if int(rows[index]["sample_id"]) == sample_id
            )
            alternatives = sorted(
                (index for index in candidates if index != current_for_sample),
                key=lambda index: (
                    0
                    if _preservation_valid(
                        rows[index],
                        identity_minimum=identity_minimum,
                        outside_l1_maximum=outside_l1_maximum,
                    )
                    else 1,
                    0 if _target_pass(rows[index]) else 1,
                    0 if _truth(rows[index].get("raw_target_pass", False)) else 1,
                    _number(rows[index], "post_attack_linf"),
                    int(rows[index]["seed"]),
                ),
            )
            for alternative in alternatives:
                if not _preservation_valid(
                    rows[alternative],
                    identity_minimum=identity_minimum,
                    outside_l1_maximum=outside_l1_maximum,
                ):
                    continue
                proposal = [
                    alternative if index == current_for_sample else index
                    for index in current
                ]
                if _target_pass_count(rows, proposal) < minimum_passes:
                    continue
                proposal_fid = frechet_distance(
                    reference,
                    features[proposal],
                    epsilon=epsilon,
                )
                if proposal_fid < current_fid - 1e-8:
                    current = proposal
                    current_fid = proposal_fid
                    accepted_swaps += 1
                    changed = True
                    break
        if not changed:
            break

    return _make_result(
        rows,
        current,
        features,
        reference,
        initial_fid=initial_fid,
        accepted_swaps=accepted_swaps,
        passes=completed_passes,
    )
