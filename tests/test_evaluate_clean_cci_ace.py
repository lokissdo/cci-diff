from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_clean_cci_ace import (
    binary_cout_from_smile_curves,
    bootstrap_mean_interval,
    build_cout_transitions,
    collateral_flips,
    continuous_non_target_drift,
    correlation_difference,
    directional_target_metrics,
    group_variant_task_rows,
    paired_cosine_similarity,
    summarize_task_rows,
)


def test_binary_cout_uses_smile_probability_complement():
    curves = np.array([[0.9, 0.1, 0.1]])

    scores = binary_cout_from_smile_curves(curves)

    assert scores == pytest.approx([0.4])


def test_binary_cout_direction_reflects_early_desired_transition():
    early_desired = np.array([[0.9, 0.1, 0.1]])
    late_desired = np.array([[0.9, 0.9, 0.1]])
    symmetric = np.array([[0.9, 0.5, 0.1]])

    assert binary_cout_from_smile_curves(early_desired)[0] > 0
    assert binary_cout_from_smile_curves(late_desired)[0] < 0
    assert binary_cout_from_smile_curves(symmetric)[0] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "curves",
    [
        np.array([[0.9, np.nan]]),
        np.array([[0.9, 1.1]]),
        np.array([0.9, 0.1]),
        np.array([[0.9]]),
    ],
)
def test_binary_cout_rejects_invalid_probability_curves(curves):
    with pytest.raises(ValueError):
        binary_cout_from_smile_curves(curves)


def test_cout_transitions_insert_largest_changed_pixels_first():
    import torch

    source = torch.zeros((1, 1, 1, 3))
    output = torch.tensor([[[[0.1, 0.9, 0.5]]]])

    transitions = list(build_cout_transitions(source, output, steps=3))

    assert len(transitions) == 4
    assert torch.equal(transitions[0], source)
    torch.testing.assert_close(
        transitions[1],
        torch.tensor([[[[0.0, 0.9, 0.0]]]]),
    )
    torch.testing.assert_close(
        transitions[2],
        torch.tensor([[[[0.0, 0.9, 0.5]]]]),
    )
    assert torch.equal(transitions[3], output)


def test_classifier_metric_parser_accepts_local_attribute_model():
    from scripts.evaluate_clean_cci_ace import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--experiment_root",
            "outputs/test",
            "--ace_root",
            "thesis/evaluate/ACE",
            "--attribute_classifier_path",
            "models/classifier.pth",
        ]
    )

    assert args.attribute_classifier_path == "models/classifier.pth"


def test_continuous_non_target_drift_excludes_each_target():
    source = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    output = np.array([[0.2, 0.9, 0.5], [0.1, 0.7, 0.9]])

    values = continuous_non_target_drift(
        source,
        output,
        np.array([1, 0]),
    )

    assert values == pytest.approx([0.15, 0.25])


def test_directional_success_handles_desired_zero_and_one():
    source = np.array([[0.9, 0.2], [0.1, 0.8]])
    output = np.array([[0.4, 0.3], [0.2, 0.7]])

    metrics = directional_target_metrics(
        source,
        output,
        target_indices=np.array([0, 1]),
        desired_values=np.array([0, 1]),
    )

    assert metrics["desired_probability"].tolist() == pytest.approx([0.6, 0.7])
    assert metrics["target_success"].tolist() == [True, True]
    assert metrics["directional_flip"].tolist() == [True, False]


def test_mnac_excludes_each_rows_target_attribute():
    source = np.zeros((2, 4), dtype=bool)
    output = np.array(
        [
            [True, True, False, True],
            [False, True, True, True],
        ]
    )

    values = collateral_flips(source, output, np.array([0, 2]))

    assert values.tolist() == [2, 2]


def test_paired_cosine_compares_corresponding_source_and_output_vectors():
    source = np.array([[1.0, 0.0], [1.0, 0.0]])
    output = np.array([[2.0, 0.0], [0.0, 1.0]])

    similarities = paired_cosine_similarity(source, output)

    assert similarities.tolist() == pytest.approx([1.0, 0.0])


def test_cd_matches_sum_of_absolute_delta_correlations():
    source = np.zeros((4, 3), dtype=bool)
    output = np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    )

    assert correlation_difference(source, output, target_index=0) == pytest.approx(1.0)


def test_bootstrap_interval_is_fixed_seed_and_contains_mean():
    values = np.array([0.0, 1.0, 2.0, 3.0])

    first = bootstrap_mean_interval(values, seed=17, iterations=500)
    second = bootstrap_mean_interval(values, seed=17, iterations=500)

    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_task_summary_reports_unconditional_and_success_conditioned_metrics():
    rows = [
        {
            "target_success": True,
            "desired_probability": 0.9,
            "fva_cosine": 0.8,
            "fs_cosine": 0.7,
            "mnac": 1.0,
            "independent_non_target_drift": 0.1,
            "changed_fraction_5": 0.1,
        },
        {
            "target_success": False,
            "desired_probability": 0.4,
            "fva_cosine": 0.2,
            "fs_cosine": 0.3,
            "mnac": 3.0,
            "independent_non_target_drift": 0.3,
            "changed_fraction_5": 0.5,
        },
    ]

    summary = summarize_task_rows(rows, bootstrap_seed=3, bootstrap_iterations=100)

    assert summary["count"] == 2
    assert summary["target_success_count"] == 1
    assert summary["fr"] == pytest.approx(0.5)
    assert summary["unconditional"]["fva_cosine"]["mean"] == pytest.approx(0.5)
    assert summary["unconditional"]["independent_non_target_drift"][
        "mean"
    ] == pytest.approx(0.2)
    assert summary["target_success_conditioned"]["fva_cosine"]["mean"] == 0.8


def test_group_variant_task_rows_keeps_ablation_results_separate():
    rows = [
        {"variant": "A3", "feature": "smile", "sample_id": 0},
        {"variant": "A4", "feature": "smile", "sample_id": 0},
        {"variant": "A3", "feature": "hair", "sample_id": 0},
    ]

    grouped = group_variant_task_rows(rows)

    assert sorted(grouped) == ["A3", "A4"]
    assert grouped["A3"]["smile"] == [rows[0]]
    assert grouped["A3"]["hair"] == [rows[2]]
    assert grouped["A4"]["smile"] == [rows[1]]
