import json

import numpy as np
import pytest

from cci_diff.individual_region_selection import (
    FrozenInfluencePolicy,
    FrozenRegionSetEvidence,
    load_frozen_influence_policy,
    select_individual_region_set,
)
from cci_diff.counterfactual_graph import RegionSetEvidence, build_influence_graph


def influence_payload():
    return {
        "version": 1,
        "type": "classifier_counterfactual_influence",
        "graph_type": "classifier_counterfactual_influence",
        "target": "Smiling",
        "desired_value": 0,
        "selected_regions": ["mouth", "upper_lip", "lower_lip"],
        "generation_regions": ["mouth", "upper_lip", "lower_lip"],
        "candidate_region_sets": [
            ["mouth"],
            ["mouth", "upper_lip", "lower_lip"],
        ],
        "fallback_regions": ["mouth", "upper_lip", "lower_lip"],
        "verified_edges": [
            {
                "source": "Smiling",
                "target": "mouth",
                "relation": "classifier_counterfactual_influence",
            },
            {
                "source": "Smiling",
                "target": "upper_lip",
                "relation": "classifier_counterfactual_influence",
            },
            {
                "source": "Smiling",
                "target": "lower_lip",
                "relation": "classifier_counterfactual_influence",
            },
        ],
        "region_set_evidence": [
            {
                "regions": ["mouth"],
                "mean_effect": 0.30,
                "flip_rate": 0.70,
                "effect_ci_low": 0.10,
                "mean_mask_fraction": 0.02,
            },
            {"regions": ["lower_lip"], "mean_effect": 0.20},
            {"regions": ["upper_lip"], "mean_effect": 0.10},
            {
                "regions": ["mouth", "lower_lip"],
                "mean_effect": 0.42,
            },
            {
                "regions": ["mouth", "upper_lip", "lower_lip"],
                "mean_effect": 0.55,
                "flip_rate": 0.97,
                "effect_ci_low": 0.20,
                "mean_mask_fraction": 0.05,
            },
        ],
    }


def test_discovered_joint_candidate_round_trips_without_singleton_lip_edges(
    tmp_path,
):
    def evidence(regions, effect, flip_rate, area):
        return RegionSetEvidence(
            regions=regions,
            row_count=40,
            sample_count=40,
            flip_rate=flip_rate,
            mean_effect=effect,
            median_effect=effect,
            effect_ci_low=effect / 2.0,
            effect_ci_high=effect * 1.5,
            mean_mask_fraction=area,
        )

    mouth = evidence(("mouth",), 0.20, 0.70, 0.02)
    perioral = evidence(
        ("lower_lip", "mouth", "upper_lip"), 0.35, 0.97, 0.05
    )
    graph = build_influence_graph(
        target="Smiling",
        desired_value=0,
        evidence_by_regions={
            mouth.regions: mouth,
            perioral.regions: perioral,
        },
        minimum_samples=20,
    )
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph.to_dict()), encoding="utf-8")

    policy = load_frozen_influence_policy(graph_path)

    assert policy.verified_regions == ("lower_lip", "mouth", "upper_lip")
    assert policy.candidate_region_sets == (
        ("lower_lip", "mouth", "upper_lip"),
        ("mouth",),
    )
    assert policy.fallback_regions == ("lower_lip", "mouth", "upper_lip")


def make_policy(**updates):
    values = {
        "target": "Smiling",
        "desired_value": 0,
        "verified_regions": ("lower_lip", "mouth", "upper_lip"),
        "fallback_regions": ("lower_lip", "mouth", "upper_lip"),
        "candidate_region_sets": (
            ("mouth",),
            ("lower_lip", "mouth", "upper_lip"),
        ),
        "region_set_effects": {
            ("mouth",): 0.30,
            ("lower_lip",): 0.20,
            ("upper_lip",): 0.10,
            ("lower_lip", "mouth"): 0.42,
        },
        "region_set_evidence": {
            ("mouth",): FrozenRegionSetEvidence(0.30, 0.70, 0.10, 0.02),
            ("lower_lip", "mouth", "upper_lip"): FrozenRegionSetEvidence(
                0.55, 0.97, 0.20, 0.05
            ),
        },
        "graph_path": "influence_graph.json",
        "graph_sha256": "a" * 64,
    }
    if "verified_regions" in updates and "candidate_region_sets" not in updates:
        values["candidate_region_sets"] = ()
        values["region_set_evidence"] = {}
    values.update(updates)
    return FrozenInfluencePolicy(**values)


def test_load_frozen_influence_policy_reads_verified_regions_and_effects(
    tmp_path,
):
    path = tmp_path / "influence_graph.json"
    path.write_text(json.dumps(influence_payload()), encoding="utf-8")

    policy = load_frozen_influence_policy(path)

    assert policy.target == "Smiling"
    assert policy.desired_value == 0
    assert policy.verified_regions == ("lower_lip", "mouth", "upper_lip")
    assert policy.fallback_regions == (
        "lower_lip",
        "mouth",
        "upper_lip",
    )
    assert policy.candidate_region_sets == (
        ("lower_lip", "mouth", "upper_lip"),
        ("mouth",),
    )
    assert policy.region_set_evidence[("mouth",)].flip_rate == pytest.approx(
        0.70
    )
    assert policy.global_effect(("mouth", "lower_lip")) == pytest.approx(0.42)
    assert len(policy.graph_sha256) == 64
    with pytest.raises(TypeError):
        policy.region_set_effects[("mouth",)] = 1.0


def test_new_graph_preserves_candidate_family_and_all_audit_edges(
    tmp_path,
):
    payload = influence_payload()
    payload["verified_edges"].append(
        {
            "source": "Smiling",
            "target": "skin",
            "relation": "classifier_counterfactual_influence",
        }
    )
    payload["region_set_evidence"].append(
        {"regions": ["skin"], "mean_effect": 0.95}
    )
    path = tmp_path / "influence_graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    policy = load_frozen_influence_policy(path)

    assert policy.verified_regions == (
        "lower_lip",
        "mouth",
        "skin",
        "upper_lip",
    )
    assert policy.candidate_region_sets == (
        ("lower_lip", "mouth", "upper_lip"),
        ("mouth",),
    )
    assert ("skin",) not in policy.candidate_region_sets


def test_legacy_graph_migrates_generation_regions_to_single_candidate(
    tmp_path,
):
    payload = influence_payload()
    payload.pop("candidate_region_sets")
    payload.pop("fallback_regions")
    payload["generation_regions"] = ["mouth"]
    path = tmp_path / "legacy_graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    policy = load_frozen_influence_policy(path)

    assert policy.candidate_region_sets == (("mouth",),)
    assert policy.fallback_regions == ("mouth",)
    assert policy.verified_regions == ("lower_lip", "mouth", "upper_lip")


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"graph_type": "attribution_only"}, "graph type"),
        ({"verified_edges": []}, "verified"),
        ({"desired_value": 2}, "desired_value"),
        (
            {"fallback_regions": ["mouth", "nose"]},
            "fallback regions",
        ),
    ),
)
def test_load_frozen_influence_policy_rejects_invalid_graph(
    tmp_path,
    update,
    message,
):
    payload = influence_payload()
    payload.update(update)
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_frozen_influence_policy(path)


def test_selects_smallest_exact_union_reaching_coverage():
    saliency = np.zeros((3, 4), dtype=np.float32)
    saliency[0, 0] = 4
    saliency[1, 0] = 3
    saliency[2, 0] = 1
    mouth = np.zeros_like(saliency)
    mouth[0, 0:2] = 1
    lower = np.zeros_like(saliency)
    lower[1, 0] = 1
    upper = np.zeros_like(saliency)
    upper[2, 0] = 1

    selected = select_individual_region_set(
        saliency,
        {
            "mouth": mouth,
            "lower_lip": lower,
            "upper_lip": upper,
        },
        make_policy(),
        coverage_threshold=0.80,
    )

    assert selected.selected_regions == ("lower_lip", "mouth")
    assert selected.coverage == pytest.approx(7 / 8)
    assert selected.mask_fraction == pytest.approx(3 / 12)
    assert selected.candidate_count == 7
    assert not selected.fallback_used


def test_selection_is_not_limited_to_ranked_prefixes():
    saliency = np.array([[0.45, 0.0, 0.0, 0.0, 0.40, 0.15]])
    region_a = np.array([[1, 1, 1, 1, 0, 0]])
    region_b = np.array([[0, 0, 0, 0, 1, 0]])
    region_c = np.array([[0, 0, 0, 0, 0, 1]])
    policy = make_policy(
        verified_regions=("a", "b", "c"),
        fallback_regions=("a", "b", "c"),
        region_set_effects={},
    )

    selected = select_individual_region_set(
        saliency,
        {"a": region_a, "b": region_b, "c": region_c},
        policy,
        coverage_threshold=0.55,
    )

    assert selected.selected_regions == ("b", "c")
    assert selected.mask_fraction == pytest.approx(2 / 6)


def test_exact_union_does_not_double_count_overlapping_masks():
    saliency = np.array([[2.0, 1.0]])
    left = np.array([[1, 1]])
    right = np.array([[1, 0]])
    policy = make_policy(
        verified_regions=("left", "right"),
        fallback_regions=("left", "right"),
        region_set_effects={},
    )

    selected = select_individual_region_set(
        saliency,
        {"left": left, "right": right},
        policy,
        coverage_threshold=1.0,
    )

    assert selected.selected_regions == ("left",)
    assert selected.coverage == pytest.approx(1.0)


def test_global_effect_breaks_equal_area_and_region_count_ties():
    saliency = np.array([[1.0, 1.0]])
    policy = make_policy(
        verified_regions=("a", "b"),
        fallback_regions=("a", "b"),
        region_set_effects={("a",): 0.1, ("b",): 0.8},
    )

    selected = select_individual_region_set(
        saliency,
        {"a": np.array([[1, 0]]), "b": np.array([[0, 1]])},
        policy,
        coverage_threshold=0.5,
    )

    assert selected.selected_regions == ("b",)


def test_zero_saliency_uses_available_global_fallback_and_reports_missing():
    policy = make_policy()

    selected = select_individual_region_set(
        np.zeros((2, 2)),
        {"mouth": np.array([[1, 0], [0, 0]])},
        policy,
        coverage_threshold=0.8,
    )

    assert selected.selected_regions == ("mouth",)
    assert selected.available_regions == ("mouth",)
    assert selected.missing_regions == ("lower_lip", "upper_lip")
    assert selected.coverage == 0.0
    assert selected.fallback_used
    assert selected.fallback_reason == "zero_saliency_in_verified_union"


def test_zero_saliency_rejects_when_no_fallback_mask_is_available():
    policy = make_policy(fallback_regions=("lower_lip",))

    with pytest.raises(ValueError, match="fallback"):
        select_individual_region_set(
            np.zeros((2, 2)),
            {"mouth": np.array([[1, 0], [0, 0]])},
            policy,
            coverage_threshold=0.8,
        )
