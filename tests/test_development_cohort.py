import pytest

from cci_diff.development_cohort import (
    DevelopmentCounts,
    allocate_development_counts,
    assign_development_cohort,
)


def test_standard_sizes_use_two_five_eight_ratio():
    assert allocate_development_counts(30) == DevelopmentCounts(4, 10, 16)
    assert allocate_development_counts(300) == DevelopmentCounts(40, 100, 160)


def test_largest_remainder_is_deterministic_for_arbitrary_size():
    counts = allocate_development_counts(31)

    assert counts.discovery + counts.fit + counts.calibration == 31
    assert counts == DevelopmentCounts(4, 10, 17)


def test_larger_cohort_preserves_ids_in_each_role():
    eligible = tuple(range(10_000))
    evaluation = {1, 2, 3}

    small = assign_development_cohort(eligible, evaluation, 30, 42)
    large = assign_development_cohort(eligible, evaluation, 300, 42)

    for role in ("discovery", "fit", "calibration"):
        assert set(getattr(small, role)).issubset(getattr(large, role))
    assert small.all_ids.isdisjoint(evaluation)
    assert large.all_ids.isdisjoint(evaluation)


def test_roles_are_pairwise_disjoint_and_serializable():
    cohort = assign_development_cohort(range(1_000), {900}, 30, 7)

    assert set(cohort.discovery).isdisjoint(cohort.fit)
    assert set(cohort.discovery).isdisjoint(cohort.calibration)
    assert set(cohort.fit).isdisjoint(cohort.calibration)
    assert cohort.to_dict()["counts"] == {
        "discovery": 4,
        "fit": 10,
        "calibration": 16,
    }


@pytest.mark.parametrize("value", [True, 0, 14])
def test_data_size_must_be_at_least_fifteen(value):
    with pytest.raises(ValueError, match="at least 15"):
        allocate_development_counts(value)


def test_duplicate_eligible_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        assign_development_cohort([1, 1, 2], set(), 15, 42)


def test_insufficient_role_bucket_reports_required_and_available():
    with pytest.raises(ValueError, match="required=.*available="):
        assign_development_cohort(range(15), set(), 30, 42)
