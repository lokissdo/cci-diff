import csv
import json

import pytest

from scripts.discover_counterfactual_graph import discover_graph, read_observations
from scripts.screen_counterfactual_regions import (
    aggregate_screening_rows,
    build_arg_parser as build_screening_arg_parser,
)


def write_results(path):
    rows = []
    for sample_id in range(4):
        for regions, output_probability in (
            (["mouth"], 0.3),
            (["lower_lip"], 0.7),
            (["lower_lip", "mouth"], 0.2),
        ):
            rows.append(
                {
                    "target": "Smiling",
                    "desired_value": 0,
                    "sample_id": sample_id,
                    "seed": 42,
                    "regions": json.dumps(regions),
                    "source_probability": 0.9,
                    "output_probability": output_probability,
                    "mask_fraction": 0.05 * len(regions),
                    "identity_cosine": 0.95,
                    "non_target_drift": 0.01,
                    "outside_l1": 0.02,
                    "changed_fraction": 0.03,
                    "output_path": f"{sample_id}.png",
                    "audit_path": f"{sample_id}.json",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_read_and_discover_graph_writes_evidence_artifacts(tmp_path):
    results = tmp_path / "intervention_results.csv"
    write_results(results)
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps({"region": {"components": [], "audit_role": "original"}}),
        encoding="utf-8",
    )

    observations = read_observations(results)
    result = discover_graph(
        results,
        tmp_path / "analysis",
        required_flip_rate=0.95,
        minimum_samples=4,
        bootstrap_samples=100,
        random_seed=7,
        template_graph_path=template,
    )

    assert len(observations) == 12
    assert result.selected_regions == ("mouth",)
    payload = json.loads(
        (tmp_path / "analysis" / "influence_graph.json").read_text()
    )
    assert payload["selected_regions"] == ["mouth"]
    assert payload["candidate_region_sets"] == [
        ["lower_lip", "mouth"],
        ["mouth"],
    ]
    assert payload["fallback_regions"] == ["mouth"]
    assert "mouth" in {
        edge["target"] for edge in payload["verified_edges"]
    }
    metrics_path = tmp_path / "analysis" / "region_set_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    assert {"pareto_optimal", "target_efficiency", "dominated_by"} <= set(
        metrics[0]
    )
    assert any(row["pareto_optimal"] == "True" for row in metrics)
    assert (tmp_path / "analysis" / "interactions.csv").is_file()
    execution = json.loads(
        (
            tmp_path / "analysis" / "selected_execution_graph.json"
        ).read_text()
    )
    assert execution["discovery"]["selection_rule"] == (
        "risk_controlled_candidate_pool_v1"
    )
    assert execution["discovery"]["required_flip_rate_role"] == (
        "fallback_reliability_threshold"
    )
    assert execution["discovery"]["candidate_region_sets"] == (
        payload["candidate_region_sets"]
    )
    report = (tmp_path / "analysis" / "discovery_report.md").read_text()
    assert "Adaptive candidate sets" in report
    assert "Reliable fallback" in report


def test_screening_summary_reports_robust_heatmap_statistics():
    rows = [
        {
            "sample_id": 0,
            "region": "mouth",
            "captured_mass": 0.10,
            "region_density": 0.8,
            "mask_fraction": 0.01,
            "proposal_score": 0.08,
        },
        {
            "sample_id": 1,
            "region": "mouth",
            "captured_mass": 0.20,
            "region_density": 0.4,
            "mask_fraction": 0.03,
            "proposal_score": 0.08,
        },
        {
            "sample_id": 0,
            "region": "upper_lip",
            "captured_mass": 0.07,
            "region_density": 0.5,
            "mask_fraction": 0.01,
            "proposal_score": 0.035,
        },
    ]

    summary = aggregate_screening_rows(rows, sample_count=2)
    by_region = {row["region"]: row for row in summary}

    assert [row["region"] for row in summary] == ["mouth", "upper_lip"]
    assert by_region["mouth"]["mean_proposal_score"] == pytest.approx(0.08)
    assert by_region["mouth"]["median_region_density"] == pytest.approx(0.6)
    assert by_region["mouth"]["median_captured_mass"] == pytest.approx(0.15)
    assert by_region["mouth"]["median_mask_fraction"] == pytest.approx(0.02)
    assert by_region["mouth"]["coverage_frequency"] == 1.0
    assert by_region["upper_lip"]["coverage_frequency"] == 0.5


def test_screening_cli_uses_dynamic_maximum_four_region_selection():
    args = build_screening_arg_parser().parse_args(
        [
            "--template_graph",
            "graph.json",
            "--classifier_path",
            "classifier.pth",
            "--sample_ids",
            "1",
            "--candidate_regions",
            "mouth",
            "--output_dir",
            "out",
        ]
    )

    assert args.max_selected_regions == 4
    assert args.saliency_coverage_threshold == pytest.approx(0.8)
    assert not hasattr(args, "select_top_k")
