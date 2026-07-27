import math

import pytest

from cci_diff.counterfactual_graph import (
    InterventionObservation,
    RegionSetEvidence,
    aggregate_region_sets,
    build_influence_graph,
    compute_interactions,
    select_region_set,
)


def observation(
    *,
    sample_id,
    regions,
    source_probability,
    output_probability,
    seed=42,
    desired_value=0,
    **metrics,
):
    return InterventionObservation(
        target="Smiling",
        desired_value=desired_value,
        sample_id=sample_id,
        seed=seed,
        regions=regions,
        source_probability=source_probability,
        output_probability=output_probability,
        **metrics,
    )


def evidence(
    regions,
    *,
    flip_rate,
    mean_effect,
    mask_fraction,
    outside_l1=0.02,
    changed_fraction=0.03,
    non_target_drift=0.01,
    identity_cosine=0.95,
    ci_low=0.01,
):
    return RegionSetEvidence(
        regions=tuple(regions),
        row_count=20,
        sample_count=20,
        flip_rate=flip_rate,
        mean_effect=mean_effect,
        median_effect=mean_effect,
        effect_ci_low=ci_low,
        effect_ci_high=mean_effect + 0.05,
        mean_mask_fraction=mask_fraction,
        mean_identity_cosine=identity_cosine,
        mean_non_target_drift=non_target_drift,
        mean_outside_l1=outside_l1,
        mean_changed_fraction=changed_fraction,
    )


def test_observation_converts_smile_removal_to_desired_probability():
    row = observation(
        sample_id=1,
        regions=("mouth", "lower_lip", "mouth"),
        source_probability=0.9,
        output_probability=0.3,
    )

    assert row.regions == ("lower_lip", "mouth")
    assert row.source_desired_probability == pytest.approx(0.1)
    assert row.output_desired_probability == pytest.approx(0.7)
    assert row.target_effect == pytest.approx(0.6)
    assert row.target_pass


def test_observation_uses_positive_probability_for_desired_value_one():
    row = observation(
        sample_id=2,
        regions=("hair",),
        source_probability=0.2,
        output_probability=0.75,
        desired_value=1,
    )

    assert row.source_desired_probability == pytest.approx(0.2)
    assert row.output_desired_probability == pytest.approx(0.75)
    assert row.target_effect == pytest.approx(0.55)
    assert row.target_pass


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"desired_value": 2}, "desired_value"),
        ({"source_probability": math.nan}, "source_probability"),
        ({"output_probability": 1.1}, "output_probability"),
        ({"regions": ()}, "regions"),
    ),
)
def test_observation_rejects_invalid_values(updates, message):
    kwargs = {
        "sample_id": 1,
        "regions": ("mouth",),
        "source_probability": 0.9,
        "output_probability": 0.3,
    }
    kwargs.update(updates)

    with pytest.raises(ValueError, match=message):
        observation(**kwargs)


def test_aggregation_bootstraps_by_image_and_is_deterministic():
    rows = [
        observation(
            sample_id=1,
            seed=1,
            regions=("mouth",),
            source_probability=0.9,
            output_probability=0.5,
            mask_fraction=0.10,
        ),
        observation(
            sample_id=1,
            seed=2,
            regions=("mouth",),
            source_probability=0.9,
            output_probability=0.3,
            mask_fraction=0.12,
        ),
        observation(
            sample_id=2,
            seed=1,
            regions=("mouth",),
            source_probability=0.8,
            output_probability=0.7,
            mask_fraction=0.08,
        ),
        observation(
            sample_id=2,
            seed=2,
            regions=("mouth",),
            source_probability=0.8,
            output_probability=0.5,
            mask_fraction=0.10,
        ),
    ]

    first = aggregate_region_sets(rows, bootstrap_samples=500, random_seed=9)
    second = aggregate_region_sets(rows, bootstrap_samples=500, random_seed=9)
    result = first[("mouth",)]

    assert result.row_count == 4
    assert result.sample_count == 2
    assert result.mean_effect == pytest.approx(0.35)
    assert result.mean_mask_fraction == pytest.approx(0.10)
    assert result.effect_ci_low == second[("mouth",)].effect_ci_low
    assert result.effect_ci_high == second[("mouth",)].effect_ci_high


def test_aggregation_ignores_missing_optional_metrics_and_rejects_duplicates():
    row = observation(
        sample_id=1,
        regions=("mouth",),
        source_probability=0.9,
        output_probability=0.3,
    )
    result = aggregate_region_sets([row], bootstrap_samples=10)[("mouth",)]

    assert result.mean_identity_cosine is None
    assert result.mean_outside_l1 is None
    with pytest.raises(ValueError, match="Duplicate intervention"):
        aggregate_region_sets([row, row], bootstrap_samples=10)


def test_aggregation_rejects_sources_that_already_satisfy_target():
    row = observation(
        sample_id=1,
        regions=("mouth",),
        source_probability=0.2,
        output_probability=0.1,
        desired_value=0,
    )

    with pytest.raises(ValueError, match="already satisfies"):
        aggregate_region_sets([row], bootstrap_samples=10)


def test_pair_interaction_is_joint_effect_minus_singleton_effects():
    region_evidence = {
        ("lower_lip",): evidence(
            ("lower_lip",), flip_rate=0.7, mean_effect=0.2, mask_fraction=0.03
        ),
        ("mouth",): evidence(
            ("mouth",), flip_rate=0.8, mean_effect=0.3, mask_fraction=0.05
        ),
        ("lower_lip", "mouth"): evidence(
            ("lower_lip", "mouth"),
            flip_rate=0.95,
            mean_effect=0.8,
            mask_fraction=0.07,
        ),
    }

    interactions = compute_interactions(region_evidence)

    assert len(interactions) == 1
    assert interactions[0].regions == ("lower_lip", "mouth")
    assert interactions[0].synergy == pytest.approx(0.3)


def test_selection_prefers_minimal_passing_set_before_higher_flip_rate():
    small = evidence(
        ("mouth",), flip_rate=0.95, mean_effect=0.5, mask_fraction=0.04
    )
    large = evidence(
        ("lower_lip", "mouth"),
        flip_rate=1.0,
        mean_effect=0.7,
        mask_fraction=0.09,
    )
    failing = evidence(
        ("upper_lip",), flip_rate=0.90, mean_effect=0.8, mask_fraction=0.02
    )

    selected = select_region_set(
        {
            small.regions: small,
            large.regions: large,
            failing.regions: failing,
        },
        required_flip_rate=0.95,
    )

    assert selected.regions == ("mouth",)


def test_selection_fallback_maximizes_effect_before_flip_rate():
    weaker = evidence(
        ("mouth",), flip_rate=0.90, mean_effect=0.6, mask_fraction=0.04
    )
    stronger = evidence(
        ("lower_lip",), flip_rate=0.80, mean_effect=0.7, mask_fraction=0.05
    )

    selected = select_region_set(
        {weaker.regions: weaker, stronger.regions: stronger},
        required_flip_rate=0.95,
    )

    assert selected.regions == ("lower_lip",)


def test_graph_serialization_only_verifies_supported_positive_singletons():
    supported = evidence(
        ("mouth",),
        flip_rate=0.95,
        mean_effect=0.5,
        mask_fraction=0.04,
        ci_low=0.2,
    )
    unsupported = evidence(
        ("upper_lip",),
        flip_rate=0.95,
        mean_effect=0.4,
        mask_fraction=0.05,
        ci_low=-0.01,
    )
    result = build_influence_graph(
        target="Smiling",
        desired_value=0,
        evidence_by_regions={
            supported.regions: supported,
            unsupported.regions: unsupported,
        },
        required_flip_rate=0.95,
        minimum_samples=10,
        provenance={"classifier": "resnet50_multilabel_model.pth"},
    )

    payload = result.to_dict()

    assert payload["type"] == "classifier_counterfactual_influence"
    assert payload["selected_regions"] == ["mouth"]
    assert payload["selection_status"] == "meets_requirement"
    assert payload["verified_edges"] == [
        {
            "source": "Smiling",
            "target": "mouth",
            "relation": "classifier_counterfactual_influence",
        }
    ]
    assert payload["provenance"]["classifier"] == "resnet50_multilabel_model.pth"
