from __future__ import annotations

import pytest


def test_parse_epsilon_schedule_requires_increasing_positive_values():
    from cci_diff.post_attack import parse_epsilon_schedule

    assert parse_epsilon_schedule("0.05, 0.08,0.10") == (0.05, 0.08, 0.1)

    for invalid in ("", "0.05,0.05", "0.08,0.05", "0,-0.1", "nan,0.1"):
        with pytest.raises(ValueError, match="epsilon schedule"):
            parse_epsilon_schedule(invalid)


def test_adaptive_attack_restarts_and_stops_on_quantized_margin(monkeypatch):
    import torch

    from cci_diff import post_attack

    class MeanClassifier(torch.nn.Module):
        def forward(self, images):
            return torch.sigmoid(
                images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
            )

    source = post_attack.normalize_imagenet(
        torch.full((1, 3, 2, 2), 0.5)
    )
    starts = []
    attempted_epsilons = []

    def fake_attack(model, image, mask, **kwargs):
        starts.append(image.detach().clone())
        epsilon = kwargs["epsilon"]
        attempted_epsilons.append(epsilon)
        value = -0.1 if epsilon == 0.05 else 0.2
        candidate = torch.full_like(image, value)
        return candidate, {
            "after_probability": float(model(candidate)[:, 0].item()),
            "iterations": 3,
            "boundary_iterations": 4,
            "margin_pass": epsilon >= 0.08,
        }

    selected, record = post_attack.targeted_adaptive_smooth_boundary_attack(
        MeanClassifier(),
        source,
        torch.ones((1, 1, 2, 2)),
        epsilon_schedule=(0.05, 0.08, 0.1),
        label_index=0,
        desired_value=1,
        boundary_margin=0.01,
        attack_fn=fake_attack,
    )

    assert attempted_epsilons == [0.05, 0.08]
    assert all(torch.equal(start, source) for start in starts)
    assert record["selected_epsilon"] == 0.08
    assert record["escalated"] is True
    assert record["margin_pass"] is True
    assert record["total_iterations"] == 6
    assert len(record["attempts"]) == 2
    assert record["after_probability"] == pytest.approx(
        float(torch.sigmoid(torch.tensor(0.2)).item())
    )
    assert record["quantized_after_probability"] == pytest.approx(
        record["attempts"][1]["quantized_probability"]
    )
    assert record["attempts"][0]["quantized_margin_pass"] is False
    assert record["attempts"][1]["quantized_margin_pass"] is True
    assert torch.equal(selected, torch.full_like(source, 0.2))


def test_adaptive_attack_keeps_largest_budget_when_schedule_is_exhausted():
    import torch

    from cci_diff import post_attack

    class ConstantClassifier(torch.nn.Module):
        def forward(self, images):
            return torch.full(
                (images.shape[0], 1),
                0.7,
                device=images.device,
                dtype=images.dtype,
            )

    source = post_attack.normalize_imagenet(
        torch.full((1, 3, 2, 2), 0.5)
    )

    def fake_attack(model, image, mask, **kwargs):
        epsilon = kwargs["epsilon"]
        candidate = image - epsilon
        return candidate, {
            "after_probability": 0.7,
            "iterations": 2,
            "boundary_iterations": 0,
            "margin_pass": False,
        }

    selected, record = post_attack.targeted_adaptive_smooth_boundary_attack(
        ConstantClassifier(),
        source,
        torch.ones((1, 1, 2, 2)),
        epsilon_schedule=(0.05, 0.08),
        label_index=0,
        desired_value=0,
        attack_fn=fake_attack,
    )

    assert torch.allclose(selected, source - 0.08)
    assert record["selected_epsilon"] == 0.08
    assert record["margin_pass"] is False
    assert record["schedule_exhausted"] is True


def test_production_soft_mask_preserves_semantic_support():
    import torch

    from cci_diff.post_attack import soft_anatomical_mask

    semantic = torch.tensor([[[[1.0, 0.0]]]])
    saliency = torch.tensor([[[[0.25, 1.0]]]])

    result = soft_anatomical_mask(semantic, saliency)

    assert result.tolist() == [[[[0.25, 0.0]]]]


def test_horizontal_grid_round_trip_preserves_candidate_order():
    import torch

    from cci_diff.post_attack import join_horizontal_grid, split_horizontal_grid

    grid = torch.arange(12, dtype=torch.float32).view(1, 1, 2, 6)

    candidates = split_horizontal_grid(grid, count=3, crop_width=2)

    assert candidates.shape == (3, 1, 2, 2)
    assert candidates[0].flatten().tolist() == [0.0, 1.0, 6.0, 7.0]
    assert candidates[2].flatten().tolist() == [4.0, 5.0, 10.0, 11.0]
    assert torch.equal(join_horizontal_grid(candidates), grid)


def test_horizontal_grid_rejects_inconsistent_width():
    import torch

    from cci_diff.post_attack import split_horizontal_grid

    grid = torch.zeros((1, 3, 4, 5))

    with pytest.raises(ValueError, match="width"):
        split_horizontal_grid(grid, count=2, crop_width=3)
