from __future__ import annotations

import math

import numpy as np
import pytest

from cci_diff.fid_reranking import (
    candidate_is_eligible,
    diagonal_mahalanobis,
    fit_reference_projection,
    frechet_distance,
    project_features,
    select_global_fid_candidates,
    select_independent_candidates,
    select_random_candidates,
    select_reference_ids,
    select_single_seed,
)


def _candidate(
    sample_id: int,
    seed: int,
    *,
    desired: float = 0.9,
    identity: float = 0.9,
    outside: float = 0.01,
    raw_pass: bool = True,
    attack_linf: float = 0.0,
):
    return {
        "sample_id": sample_id,
        "seed": seed,
        "desired_probability": desired,
        "identity_cosine": identity,
        "outside_semantic_l1": outside,
        "raw_target_pass": raw_pass,
        "post_attack_linf": attack_linf,
    }


def test_reference_ids_are_sorted_complete_and_exclude_evaluation_ids(tmp_path):
    for image_id in (5, 0, 3, 1, 4, 2):
        (tmp_path / f"{image_id}.jpg").write_bytes(b"x")
    (tmp_path / "not-an-id.jpg").write_bytes(b"x")

    selected = select_reference_ids(
        tmp_path,
        count=3,
        excluded_ids={1, 3},
    )

    assert [image_id for image_id, _ in selected] == [0, 2, 4]


def test_reference_ids_require_enough_images(tmp_path):
    (tmp_path / "0.jpg").write_bytes(b"x")

    with pytest.raises(ValueError, match="requires 2"):
        select_reference_ids(tmp_path, count=2, excluded_ids=set())


def test_reference_projection_is_deterministic_and_limited_by_rank():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(10, 6))

    first = fit_reference_projection(values, dimensions=4)
    second = fit_reference_projection(values, dimensions=4)

    np.testing.assert_allclose(first.mean, second.mean)
    np.testing.assert_allclose(first.components, second.components)
    assert first.components.shape == (4, 6)
    assert first.projected_reference.shape == (10, 4)


def test_projection_rejects_nonfinite_or_singleton_reference():
    with pytest.raises(ValueError, match="at least two"):
        fit_reference_projection(np.zeros((1, 4)), dimensions=2)
    invalid = np.zeros((2, 4))
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_reference_projection(invalid, dimensions=2)


def test_project_features_requires_matching_input_dimension():
    projection = fit_reference_projection(np.arange(24).reshape(6, 4), dimensions=2)

    with pytest.raises(ValueError, match="input dimension"):
        project_features(np.zeros((2, 3)), projection)


def test_frechet_distance_accepts_unequal_cohort_sizes():
    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    selected = np.array([[-math.sqrt(2 / 3)], [math.sqrt(2 / 3)]])

    assert frechet_distance(reference, selected, epsilon=1e-6) == pytest.approx(
        0.0, abs=1e-5
    )


def test_frechet_distance_is_zero_for_identical_multivariate_features():
    rng = np.random.default_rng(12)
    values = rng.normal(size=(20, 4))

    assert frechet_distance(values, values) == pytest.approx(0.0, abs=1e-8)


def test_frechet_distance_rejects_invalid_matrices():
    with pytest.raises(ValueError, match="dimensions"):
        frechet_distance(np.zeros((4, 2)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="at least two"):
        frechet_distance(np.zeros((1, 2)), np.zeros((4, 2)))


def test_candidate_eligibility_requires_target_identity_and_locality():
    valid = _candidate(0, 42, desired=0.51, identity=0.81, outside=0.02)

    assert candidate_is_eligible(valid)
    assert not candidate_is_eligible(dict(valid, desired_probability=0.49))
    assert not candidate_is_eligible(dict(valid, identity_cosine=0.79))
    assert not candidate_is_eligible(dict(valid, outside_semantic_l1=0.031))


def test_diagonal_mahalanobis_uses_reference_projection_statistics():
    reference = np.array([[-2.0], [0.0], [2.0]])
    projection = fit_reference_projection(reference, dimensions=1)
    center = project_features(np.array([[0.0]]), projection)
    edge = project_features(np.array([[4.0]]), projection)

    assert diagonal_mahalanobis(center, projection)[0] == pytest.approx(0.0)
    assert diagonal_mahalanobis(edge, projection)[0] > 0


def test_single_seed_selection_requires_one_row_per_source():
    rows = [
        _candidate(0, 42),
        _candidate(0, 43),
        _candidate(1, 42),
        _candidate(1, 43),
    ]
    features = np.arange(4, dtype=float).reshape(-1, 1)
    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])

    selected = select_single_seed(rows, features, reference, seed=42)

    assert [(row["sample_id"], row["seed"]) for row in selected.rows] == [
        (0, 42),
        (1, 42),
    ]


def test_single_seed_selection_rejects_missing_seed():
    rows = [_candidate(0, 42), _candidate(1, 43)]
    with pytest.raises(ValueError, match="seed 42"):
        select_single_seed(
            rows,
            np.zeros((2, 1)),
            np.array([[-1.0], [1.0]]),
            seed=42,
        )


def test_random_selection_is_deterministic():
    rows = [
        _candidate(sample_id, seed)
        for sample_id in range(5)
        for seed in (42, 43, 44, 45)
    ]
    features = np.arange(len(rows), dtype=float).reshape(-1, 1)
    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])

    first = select_random_candidates(
        rows, features, reference, selector_seed=20260725
    )
    second = select_random_candidates(
        rows, features, reference, selector_seed=20260725
    )

    assert first.indices == second.indices
    assert len(first.rows) == 5


def test_independent_selector_prefers_valid_near_reference_candidate():
    rows = [
        _candidate(0, 42, desired=0.99),
        _candidate(0, 43, desired=0.51),
        _candidate(1, 42, desired=0.49),
        _candidate(1, 43, desired=0.8),
    ]
    features = np.array([[8.0], [0.0], [0.0], [1.0]])
    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])

    selected = select_independent_candidates(rows, features, reference)

    assert [(row["sample_id"], row["seed"]) for row in selected.rows] == [
        (0, 43),
        (1, 43),
    ]


def test_independent_selector_fallback_prioritizes_target_then_identity():
    rows = [
        _candidate(0, 42, desired=0.4, identity=0.95),
        _candidate(0, 43, desired=0.45, identity=0.7),
        _candidate(1, 42, desired=0.9),
    ]
    features = np.array([[0.0], [1.0], [-1.0]])
    reference = np.array([[-1.0], [1.0]])

    selected = select_independent_candidates(rows, features, reference)

    assert selected.rows[0]["seed"] == 43


def test_global_selector_reduces_fid_without_dropping_required_passes():
    rows = [
        _candidate(0, 42),
        _candidate(0, 43),
        _candidate(1, 42),
        _candidate(1, 43),
    ]
    features = np.array([[5.0], [-1.0], [5.0], [1.0]])
    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])

    selected = select_global_fid_candidates(
        rows,
        features,
        reference,
        minimum_passes=2,
        maximum_passes=8,
    )

    assert [(row["sample_id"], row["seed"]) for row in selected.rows] == [
        (0, 43),
        (1, 43),
    ]
    assert selected.target_passes == 2
    assert selected.final_fid <= selected.initial_fid


def test_global_selector_rejects_lower_fid_swap_that_breaks_fr_constraint():
    rows = [
        _candidate(0, 42, desired=0.9),
        _candidate(0, 43, desired=0.1),
        _candidate(1, 42, desired=0.9),
        _candidate(1, 43, desired=0.1),
    ]
    features = np.array([[10.0], [-1.0], [10.0], [1.0]])
    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])

    selected = select_global_fid_candidates(
        rows,
        features,
        reference,
        minimum_passes=2,
    )

    assert selected.target_passes == 2
    assert {row["seed"] for row in selected.rows} == {42}


def test_global_selector_validates_feature_alignment_and_minimum_passes():
    rows = [_candidate(0, 42), _candidate(1, 42)]
    reference = np.array([[-1.0], [1.0]])
    with pytest.raises(ValueError, match="align"):
        select_global_fid_candidates(
            rows,
            np.zeros((1, 1)),
            reference,
            minimum_passes=2,
        )
    with pytest.raises(ValueError, match="minimum_passes"):
        select_global_fid_candidates(
            rows,
            np.zeros((2, 1)),
            reference,
            minimum_passes=3,
        )
