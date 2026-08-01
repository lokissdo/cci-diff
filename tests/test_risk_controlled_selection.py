import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cci_diff.individual_region_selection import (
    FrozenInfluencePolicy,
    FrozenRegionSetEvidence,
)
from cci_diff.risk_controlled_selection import (
    FEATURE_NAMES,
    CandidateFeatureRow,
    FrozenSelectorArtifact,
    LogisticModel,
    PlattCalibrator,
    RiskThreshold,
    SafeSuccessThresholds,
    choose_grouped_l2,
    choose_risk_threshold,
    extract_candidate_feature_rows,
    fit_logistic_newton,
    safe_success_label,
    select_risk_controlled_regions,
    source_feature_signature,
    wilson_failure_upper_bound,
)


MOUTH = ("mouth",)
PERIORAL = ("lower_lip", "mouth", "upper_lip")


def make_policy(desired_value=0):
    return FrozenInfluencePolicy(
        target="Smiling",
        desired_value=desired_value,
        verified_regions=PERIORAL,
        fallback_regions=PERIORAL,
        region_set_effects={MOUTH: 0.20, PERIORAL: 0.35},
        graph_path="influence_graph.json",
        graph_sha256="a" * 64,
        candidate_region_sets=(MOUTH, PERIORAL),
        region_set_evidence={
            MOUTH: FrozenRegionSetEvidence(0.20, 0.70, 0.05, 0.02),
            PERIORAL: FrozenRegionSetEvidence(0.35, 0.97, 0.15, 0.05),
        },
    )


def make_masks():
    return {
        "mouth": np.array([[1, 0], [0, 0]], dtype=np.uint8),
        "upper_lip": np.array([[0, 1], [0, 0]], dtype=np.uint8),
        "lower_lip": np.array([[1, 0], [1, 0]], dtype=np.uint8),
    }


def make_model(*, coverage_coefficient=0.0, intercept=4.0):
    coefficients = [0.0] * len(FEATURE_NAMES)
    coefficients[FEATURE_NAMES.index("coverage")] = coverage_coefficient
    return LogisticModel(
        mean=(0.0,) * len(FEATURE_NAMES),
        scale=(1.0,) * len(FEATURE_NAMES),
        intercept=intercept,
        coefficients=tuple(coefficients),
        l2=0.01,
        iterations=1,
    )


def make_artifact(policy, *, model=None, risk_threshold=0.8, fallback_only=False):
    return FrozenSelectorArtifact(
        protocol_version=1,
        target=policy.target,
        desired_value=policy.desired_value,
        graph_sha256=policy.graph_sha256,
        candidate_region_sets=policy.candidate_region_sets,
        fallback_regions=policy.fallback_regions,
        feature_names=FEATURE_NAMES,
        feature_signature="b" * 64,
        classifier_sha256="c" * 64,
        generation_policy_signature="d" * 64,
        model=model or make_model(),
        calibrator=PlattCalibrator(intercept=0.0, slope=1.0, iterations=1),
        risk_calibration=RiskThreshold(
            threshold=risk_threshold,
            accepted=60,
            failures=0,
            failure_upper_bound=0.04,
            fallback_only=fallback_only,
        ),
        coverage_threshold=0.80,
        safe_success_thresholds=SafeSuccessThresholds(),
        fit_sample_ids=(1, 2),
        calibration_sample_ids=(3, 4),
    )


def make_row(regions, *, coverage, area, effect):
    values = [0.0] * len(FEATURE_NAMES)
    values[FEATURE_NAMES.index("coverage")] = coverage
    values[FEATURE_NAMES.index("mask_fraction")] = area
    values[FEATURE_NAMES.index("global_mean_effect")] = effect
    return CandidateFeatureRow(regions=regions, values=tuple(values))


@pytest.mark.parametrize(
    ("desired_value", "source_probability", "expected"),
    (
        (0, 0.90, math.log(9.0)),
        (1, 0.10, math.log(9.0)),
        (0, 0.10, -math.log(9.0)),
        (1, 0.90, -math.log(9.0)),
    ),
)
def test_feature_rows_use_generic_signed_logit_difficulty(
    desired_value, source_probability, expected
):
    rows = extract_candidate_feature_rows(
        source_probability,
        np.array([[4.0, 1.0], [0.0, 0.0]]),
        make_masks(),
        make_policy(desired_value),
    )

    assert rows[0].difficulty == pytest.approx(expected)


def test_feature_rows_use_exact_union_and_verified_union_denominator():
    rows = extract_candidate_feature_rows(
        0.90,
        np.array([[4.0, 1.0], [0.0, 5.0]]),
        make_masks(),
        make_policy(),
    )

    by_regions = {row.regions: row for row in rows}
    assert by_regions[MOUTH].coverage == pytest.approx(4 / 5)
    assert by_regions[MOUTH].saliency_density == pytest.approx(4.0)
    assert by_regions[MOUTH].mask_fraction == pytest.approx(1 / 4)
    assert by_regions[PERIORAL].coverage == pytest.approx(1.0)
    assert by_regions[PERIORAL].mask_fraction == pytest.approx(3 / 4)
    assert by_regions[PERIORAL].component_count == 3.0


def test_feature_rows_reject_missing_component_mask():
    masks = make_masks()
    masks.pop("upper_lip")

    with pytest.raises(ValueError, match="upper_lip"):
        extract_candidate_feature_rows(
            0.9, np.ones((2, 2)), masks, make_policy()
        )


@pytest.mark.parametrize(
    ("probability", "identity", "outside", "expected"),
    (
        (0.53, 0.08, 0.02, 1),
        (0.529, 0.08, 0.02, 0),
        (0.80, 0.081, 0.02, 0),
        (0.80, 0.08, 0.021, 0),
    ),
)
def test_safe_success_label_uses_frozen_boundaries(
    probability, identity, outside, expected
):
    assert safe_success_label(probability, identity, outside) == expected


def test_source_feature_signature_is_canonical_and_sensitive():
    first = source_feature_signature({"classifier": "abc", "size": 512})
    reordered = source_feature_signature({"size": 512, "classifier": "abc"})
    changed = source_feature_signature({"classifier": "xyz", "size": 512})

    assert first == reordered
    assert first != changed
    assert len(first) == 64


def test_newton_logistic_converges_and_is_deterministic():
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)

    first = fit_logistic_newton(x, y, l2=0.01)
    second = fit_logistic_newton(x, y, l2=0.01)

    assert first == second
    assert np.all(np.diff(first.predict_probability(x)) > 0)
    assert first.iterations <= 200


def test_grouped_l2_never_splits_one_sample_across_folds():
    sample_ids = np.repeat(np.arange(10), 2)
    x = np.column_stack(
        [sample_ids.astype(np.float64), np.tile([0.0, 1.0], 10)]
    )
    y = (sample_ids >= 5).astype(np.float64)

    model, audit = choose_grouped_l2(x, y, sample_ids, folds=5)

    assert model.l2 in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    assert audit.l2 == model.l2
    for fold in audit.folds:
        assert set(fold.train_sample_ids).isdisjoint(fold.validation_sample_ids)


def test_model_json_round_trip_preserves_scores():
    x = np.array([[-1.0], [1.0]], dtype=np.float64)
    model = fit_logistic_newton(x, np.array([0.0, 1.0]), l2=0.1)

    restored = LogisticModel.from_dict(
        json.loads(json.dumps(model.to_dict(), sort_keys=True))
    )

    np.testing.assert_array_equal(
        restored.predict_probability(x), model.predict_probability(x)
    )


def test_wilson_threshold_is_lowest_safe_score_with_minimum_support():
    scores = np.array([0.95] * 60 + [0.50] * 40)
    labels = np.array([1] * 60 + [0] * 40)

    result = choose_risk_threshold(
        scores, labels, min_accepted=60, max_failure_ucb=0.05
    )

    assert result.threshold == pytest.approx(0.95)
    assert result.accepted == 60
    assert result.failure_upper_bound <= 0.05
    assert wilson_failure_upper_bound(0, 60) == pytest.approx(
        result.failure_upper_bound
    )


def test_risk_threshold_falls_back_when_no_score_has_safe_support():
    result = choose_risk_threshold(
        np.array([0.9] * 59), np.ones(59), min_accepted=60
    )

    assert result.fallback_only
    assert result.accepted == 0


def test_selector_chooses_minimum_area_feasible_candidate():
    policy = make_policy()
    rows = (
        make_row(MOUTH, coverage=0.85, area=0.02, effect=0.20),
        make_row(PERIORAL, coverage=0.99, area=0.05, effect=0.35),
    )

    selection = select_risk_controlled_regions(
        rows, policy, make_artifact(policy)
    )

    assert selection.selected_regions == MOUTH
    assert not selection.fallback_used


def test_selector_falls_back_when_small_mask_fails_coverage_and_risk():
    policy = make_policy()
    rows = (
        make_row(MOUTH, coverage=0.79, area=0.02, effect=0.20),
        make_row(PERIORAL, coverage=0.99, area=0.05, effect=0.35),
    )
    artifact = make_artifact(
        policy,
        model=make_model(coverage_coefficient=0.0, intercept=0.0),
        risk_threshold=0.80,
    )

    selection = select_risk_controlled_regions(rows, policy, artifact)

    assert selection.selected_regions == PERIORAL
    assert selection.fallback_used
    assert selection.fallback_reason == "no_candidate_passed_coverage_and_risk"


def test_selector_artifact_round_trip_and_digest_validation():
    policy = make_policy()
    artifact = make_artifact(policy)
    restored = FrozenSelectorArtifact.from_dict(artifact.to_dict())

    assert restored == artifact
    with pytest.raises(ValueError, match="graph"):
        select_risk_controlled_regions(
            (make_row(MOUTH, coverage=1.0, area=0.02, effect=0.2),),
            policy,
            replace(restored, graph_sha256="f" * 64),
        )


def test_selector_accepts_predeclared_subset_of_graph_candidates():
    base = make_policy()
    upper = ("upper_lip",)
    policy = FrozenInfluencePolicy(
        target=base.target,
        desired_value=base.desired_value,
        verified_regions=base.verified_regions,
        fallback_regions=base.fallback_regions,
        region_set_effects={**base.region_set_effects, upper: 0.10},
        graph_path=base.graph_path,
        graph_sha256=base.graph_sha256,
        candidate_region_sets=(MOUTH, upper, PERIORAL),
        region_set_evidence={
            **base.region_set_evidence,
            upper: FrozenRegionSetEvidence(0.10, 0.40, 0.02, 0.01),
        },
    )
    artifact = replace(
        make_artifact(base), candidate_region_sets=(MOUTH, PERIORAL)
    )

    selection = select_risk_controlled_regions(
        (
            make_row(MOUTH, coverage=0.90, area=0.02, effect=0.20),
            make_row(PERIORAL, coverage=1.0, area=0.05, effect=0.35),
        ),
        policy,
        artifact,
    )

    assert selection.selected_regions == MOUTH


def test_readme_documents_source_only_selector_and_oracle_boundary():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Risk-controlled source-only mask selection" in text
    assert "Oracle metrics are evaluation-only" in text
    assert "prepare_adaptive_replay_data.py" in text
    assert "materialize_adaptive_region_cohort.py" in text
    assert "NOT HELD-OUT" in text
