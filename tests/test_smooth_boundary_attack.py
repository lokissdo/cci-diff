from __future__ import annotations

import pytest


def test_comparison_parser_accepts_input_variant():
    from scripts.compare_smooth_boundary_attack import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            "outputs/test",
            "--sample_ids",
            "44",
            "--variant",
            "A3",
        ]
    )

    assert args.variant == "A3"


def test_soft_mask_has_no_support_outside_facepart():
    import torch

    from scripts.smooth_boundary_attack import soft_anatomical_mask

    facepart = torch.tensor([[[[1.0, 0.0]]]])
    saliency = torch.tensor([[[[0.4, 1.0]]]])

    result = soft_anatomical_mask(facepart, saliency)

    assert result.tolist() == [[[[pytest.approx(0.4), 0.0]]]]


def test_smooth_attack_respects_mask_and_epsilon():
    import torch

    from scripts.smooth_boundary_attack import targeted_smooth_boundary_attack

    class CenterPixelClassifier(torch.nn.Module):
        def forward(self, images):
            return torch.sigmoid(images[:, :, 1, 1])

    image = torch.full((1, 1, 3, 3), 0.8)
    image[:, :, 1, 1] = 0.02
    mask = torch.zeros_like(image)
    mask[:, :, 1, 1] = 1.0

    attacked, record = targeted_smooth_boundary_attack(
        CenterPixelClassifier(),
        image,
        mask,
        label_index=0,
        desired_value=0,
        epsilon=0.05,
        step_size=0.01,
        max_steps=10,
        boundary_margin=0.0,
    )

    assert torch.max(torch.abs(attacked - image)).item() <= 0.05 + 1e-6
    assert torch.equal(attacked[mask == 0], image[mask == 0])
    assert record["target_pass"] is True
    assert record["after_probability"] <= 0.5
    assert record["boundary_iterations"] > 0


def test_boundary_refinement_returns_closer_successful_segment_point():
    import torch

    from scripts.smooth_boundary_attack import refine_boundary

    class ScalarClassifier(torch.nn.Module):
        def forward(self, images):
            return torch.sigmoid(images.flatten(1)[:, :1])

    failed = torch.tensor([[[[1.0]]]])
    passed = torch.tensor([[[[-1.0]]]])

    refined, record = refine_boundary(
        ScalarClassifier(),
        failed,
        passed,
        label_index=0,
        desired_value=0,
        threshold=0.5,
        margin=0.01,
        max_steps=16,
    )

    assert record["target_pass"] is True
    assert record["after_probability"] <= 0.49
    assert torch.linalg.vector_norm(refined - failed) < torch.linalg.vector_norm(
        passed - failed
    )
    assert record["boundary_iterations"] == 16


def test_aggregate_comparison_reports_before_after_and_paired_deltas():
    from scripts.compare_smooth_boundary_attack import aggregate_comparison

    rows = [
        {
            "sample_id": 0,
            "method": "without_attack",
            "smile_probability": 0.8,
            "desired_probability": 0.2,
            "target_pass": False,
            "identity_cosine": 0.95,
            "mean_abs_change": 0.0,
            "linf": 0.0,
            "changed_fraction": 0.0,
            "residual_tv": 0.0,
            "outside_facepart_mae": 0.0,
            "iterations": 0,
            "boundary_iterations": 0,
        },
        {
            "sample_id": 0,
            "method": "with_attack",
            "smile_probability": 0.49,
            "desired_probability": 0.51,
            "target_pass": True,
            "identity_cosine": 0.93,
            "mean_abs_change": 0.001,
            "linf": 0.01,
            "changed_fraction": 0.02,
            "residual_tv": 0.004,
            "outside_facepart_mae": 0.0,
            "iterations": 4,
            "boundary_iterations": 16,
        },
    ]

    summary = aggregate_comparison(rows)

    assert summary["without_attack"]["target_pass_rate"] == 0.0
    assert summary["with_attack"]["target_pass_rate"] == 1.0
    assert summary["with_attack"]["mean_residual_tv"] == pytest.approx(0.004)
    assert summary["paired"]["target_pass_rate_delta"] == 1.0
    assert summary["paired"]["identity_delta"] == pytest.approx(-0.02)
    assert summary["paired"]["desired_probability_delta"] == pytest.approx(0.31)
    assert summary["paired"]["attempted_count"] == 1
    assert summary["paired"]["attempted_target_pass_rate"] == 1.0
    assert summary["paired"]["attempted_identity_delta"] == pytest.approx(-0.02)


def test_residual_tv_uses_only_facepart_interior_edges():
    import torch

    from scripts.compare_smooth_boundary_attack import residual_total_variation

    before = torch.zeros((1, 1, 2, 2))
    after = torch.tensor([[[[0.0, 0.2], [0.0, 0.0]]]])
    mask = torch.ones((1, 1, 2, 2))

    result = residual_total_variation(before, after, mask)

    assert result == pytest.approx(0.1)
