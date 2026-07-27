from __future__ import annotations

import numpy as np
import pytest


def test_binary_mask_uses_inclusive_threshold():
    from scripts.compare_attack_masks import binary_mask_from_saliency

    saliency = np.array([[0.39, 0.40, 0.90]], dtype=np.float32)

    mask = binary_mask_from_saliency(saliency, threshold=0.4)

    assert mask.dtype == np.float32
    assert mask.tolist() == [[0.0, 1.0, 1.0]]


def test_targeted_masked_pgd_changes_only_selected_pixels():
    import torch

    from scripts.compare_attack_masks import targeted_masked_pgd

    class ToyClassifier(torch.nn.Module):
        def forward(self, images):
            score = images.flatten(1).sum(dim=1, keepdim=True)
            return torch.sigmoid(score)

    image = torch.tensor([[[[1.0, 1.0]]]])
    mask = torch.tensor([[[[1.0, 0.0]]]])

    attacked, record = targeted_masked_pgd(
        ToyClassifier(),
        image,
        mask,
        label_index=0,
        desired_value=0,
        epsilon=0.5,
        step_size=0.25,
        max_steps=4,
    )

    assert attacked[0, 0, 0, 1].item() == image[0, 0, 0, 1].item()
    assert 0 < image[0, 0, 0, 0].item() - attacked[0, 0, 0, 0].item() <= 0.5
    assert record["after_probability"] < record["before_probability"]


def test_perturbation_metrics_report_facepart_and_attack_locality():
    import torch

    from scripts.compare_attack_masks import perturbation_metrics

    before = torch.zeros((1, 3, 1, 2))
    after = before.clone()
    after[:, :, :, 0] = 0.2
    attack_mask = torch.tensor([[[[1.0, 0.0]]]])
    facepart_mask = torch.tensor([[[[0.0, 1.0]]]])

    metrics = perturbation_metrics(
        before,
        after,
        attack_mask=attack_mask,
        facepart_mask=facepart_mask,
        pixel_threshold=0.01,
    )

    assert metrics["changed_fraction"] == pytest.approx(0.5)
    assert metrics["outside_attack_mae"] == pytest.approx(0.0)
    assert metrics["outside_facepart_mae"] == pytest.approx(0.2)
    assert metrics["linf"] == pytest.approx(0.2)


def test_gradcam_pp_supports_frozen_classifier_parameters():
    import torch

    from scripts.compare_attack_masks import gradcam_pp_saliency

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = torch.nn.ModuleList([torch.nn.Conv2d(3, 2, 1)])

        def forward(self, images):
            return self.layer4[-1](images)

    class Classifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = Base()
            self.head = torch.nn.Linear(2, 1)

        def forward(self, images):
            features = self.base_model(images).mean(dim=(-2, -1))
            return torch.sigmoid(self.head(features))

    model = Classifier().eval().requires_grad_(False)
    image = torch.rand((1, 3, 8, 8))

    saliency = gradcam_pp_saliency(
        model,
        image,
        label_index=0,
        original_present=True,
    )

    assert saliency.shape == (8, 8)
    assert np.isfinite(saliency).all()


def test_aggregate_rows_reports_attack_efficiency_and_locality():
    from scripts.compare_attack_masks import aggregate_rows

    rows = []
    for mask_type, probability, iterations, changed, identity in (
        ("facepart", 0.7, 4, 0.1, 0.9),
        ("gradcam_pp", 0.8, 2, 0.2, 0.8),
    ):
        rows.append(
            {
                "mask_type": mask_type,
                "target_pass": True,
                "desired_probability": probability,
                "iterations": iterations,
                "mask_fraction": changed,
                "changed_fraction": changed,
                "mean_abs_change": changed / 10,
                "outside_facepart_mae": 0.0,
                "identity_cosine": identity,
            }
        )

    summary = aggregate_rows(rows)

    assert summary["facepart"]["mean_iterations"] == 4
    assert summary["gradcam_pp"]["mean_iterations"] == 2
    assert summary["gradcam_pp"]["mean_abs_change"] == pytest.approx(0.02)
