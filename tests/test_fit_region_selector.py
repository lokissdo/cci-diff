import json
from pathlib import Path

import pytest

from cci_diff.risk_controlled_selection import FEATURE_NAMES
from scripts.fit_region_selector import (
    fit_region_selector,
    provenance_from_manifests,
)


MOUTH = ("mouth",)
PERIORAL = ("lower_lip", "mouth", "upper_lip")


def write_graph(path: Path) -> Path:
    payload = {
        "version": 1,
        "type": "classifier_counterfactual_influence",
        "graph_type": "classifier_counterfactual_influence",
        "target": "Smiling",
        "desired_value": 0,
        "candidate_region_sets": [list(MOUTH), list(PERIORAL)],
        "fallback_regions": list(PERIORAL),
        "selected_regions": list(PERIORAL),
        "generation_regions": list(PERIORAL),
        "verified_edges": [
            {
                "source": "Smiling",
                "target": region,
                "relation": "classifier_counterfactual_influence",
            }
            for region in PERIORAL
        ],
        "region_set_evidence": [
            {
                "regions": list(MOUTH),
                "mean_effect": 0.20,
                "flip_rate": 0.70,
                "effect_ci_low": 0.05,
                "mean_mask_fraction": 0.02,
            },
            {
                "regions": list(PERIORAL),
                "mean_effect": 0.35,
                "flip_rate": 0.97,
                "effect_ci_low": 0.15,
                "mean_mask_fraction": 0.05,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def provenance():
    return {
        "feature_signature": "b" * 64,
        "classifier_sha256": "c" * 64,
        "generation_policy_signature": "d" * 64,
        "discovery_manifest_sha256": "e" * 64,
        "fit_manifest_sha256": "f" * 64,
        "calibration_manifest_sha256": "1" * 64,
        "software_versions": {"numpy": "test"},
    }


def development_rows(*, fit_ids=range(100), calibration_ids=range(100, 200)):
    rows = []
    for cohort, sample_ids in (
        ("fit", fit_ids),
        ("calibration", calibration_ids),
    ):
        for sample_id in sample_ids:
            for regions in (MOUTH, PERIORAL):
                safe = regions == PERIORAL or sample_id % 10 < 7
                evidence = (
                    (0.20, 0.70, 0.05, 0.02, 1.0)
                    if regions == MOUTH
                    else (0.35, 0.97, 0.15, 0.05, 3.0)
                )
                values = {
                    "difficulty": 2.0,
                    "coverage": 0.95 if safe else 0.40,
                    "saliency_density": 1.0 if safe else 0.2,
                    "mask_fraction": evidence[3],
                    "component_count": evidence[4],
                    "global_mean_effect": evidence[0],
                    "global_flip_rate": evidence[1],
                    "global_effect_ci_low": evidence[2],
                }
                assert set(values) == set(FEATURE_NAMES)
                rows.append(
                    {
                        "cohort": cohort,
                        "sample_id": sample_id,
                        "regions": regions,
                        **values,
                        "desired_probability": 0.80 if safe else 0.40,
                        "identity_distance": 0.01,
                        "outside_locality": 0.01,
                    }
                )
    return rows


def test_fit_rejects_any_pairwise_cohort_overlap(tmp_path):
    graph = write_graph(tmp_path / "graph.json")
    rows = development_rows()

    with pytest.raises(ValueError, match="pairwise disjoint"):
        fit_region_selector(
            graph,
            rows,
            tmp_path / "out",
            provenance=provenance(),
            discovery_ids={0},
            evaluation_ids={200, 201},
        )


def test_provenance_can_be_built_from_source_and_split_manifests(tmp_path):
    source = tmp_path / "source_manifest.json"
    source.write_text(
        json.dumps(
            {
                "feature_signature": "b" * 64,
                "classifier_sha256": "c" * 64,
                "generation_policy_signature": "d" * 64,
            }
        ),
        encoding="utf-8",
    )
    split = tmp_path / "split_manifest.json"
    split.write_text(json.dumps({"cohorts": {"fit": [1]}}), encoding="utf-8")

    result = provenance_from_manifests(source, split)

    assert result["feature_signature"] == "b" * 64
    assert result["classifier_sha256"] == "c" * 64
    assert result["generation_policy_signature"] == "d" * 64
    assert len(result["source_feature_manifest_sha256"]) == 64
    assert len(result["split_manifest_sha256"]) == 64


def test_fit_rejects_incomplete_candidate_pairs(tmp_path):
    graph = write_graph(tmp_path / "graph.json")
    rows = development_rows()
    rows = [
        row
        for row in rows
        if not (
            row["cohort"] == "fit"
            and row["sample_id"] == 0
            and row["regions"] == MOUTH
        )
    ]

    with pytest.raises(ValueError, match="complete candidate family"):
        fit_region_selector(
            graph, rows, tmp_path / "out", provenance=provenance()
        )


def test_fit_rejects_candidate_family_not_in_graph(tmp_path):
    graph = write_graph(tmp_path / "graph.json")

    with pytest.raises(ValueError, match="graph candidate"):
        fit_region_selector(
            graph,
            development_rows(),
            tmp_path / "out",
            provenance=provenance(),
            candidate_family=(MOUTH, ("nose",), PERIORAL),
        )


def test_fit_rejects_oracle_or_final_metric_provenance(tmp_path):
    graph = write_graph(tmp_path / "graph.json")
    invalid = provenance()
    invalid["oracle_model"] = "evaluation-only.pth"

    with pytest.raises(ValueError, match="evaluation-only"):
        fit_region_selector(
            graph,
            development_rows(),
            tmp_path / "out",
            provenance=invalid,
        )


def test_fit_writes_bitwise_reproducible_frozen_artifacts(tmp_path):
    graph = write_graph(tmp_path / "graph.json")
    rows = development_rows()

    first = fit_region_selector(
        graph,
        rows,
        tmp_path / "first",
        provenance=provenance(),
        discovery_ids=range(300, 320),
        evaluation_ids=range(200, 300),
    )
    second = fit_region_selector(
        graph,
        rows,
        tmp_path / "second",
        provenance=provenance(),
        discovery_ids=range(300, 320),
        evaluation_ids=range(200, 300),
    )

    assert first == second
    assert (tmp_path / "first/selector_model.json").read_bytes() == (
        tmp_path / "second/selector_model.json"
    ).read_bytes()
    assert first.risk_calibration.accepted >= 60
    assert not first.risk_calibration.fallback_only
    assert first.fit_sample_ids == tuple(range(100))
    assert first.calibration_sample_ids == tuple(range(100, 200))
    for name in (
        "selector_data_manifest.json",
        "selector_fit_rows.csv",
        "selector_calibration_rows.csv",
        "selector_model.json",
        "selector_calibration_report.json",
        "selector_fit_report.md",
    ):
        assert (tmp_path / "first" / name).is_file()
