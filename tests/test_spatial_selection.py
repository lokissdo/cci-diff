from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cci_diff.spatial_selection import (
    measure_spatial_change,
    select_spatial_candidate,
)


def _save_rgb(path, values):
    Image.fromarray(np.asarray(values, dtype=np.uint8), mode="RGB").save(path)


def _save_mask(path, values):
    Image.fromarray(np.asarray(values, dtype=np.uint8), mode="L").save(path)


def test_measure_spatial_change_uses_fixed_thresholds_and_dual_masks(tmp_path):
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    output = source.copy()
    output[0, 0, 0] = 2
    output[0, 1] = 6
    output[1, 0] = 11
    output[1, 1, 0] = 255

    semantic = np.zeros((4, 4), dtype=np.uint8)
    semantic[0, :2] = 255
    generation = np.zeros((4, 4), dtype=np.uint8)
    generation[0, :2] = 255
    generation[1, :2] = 128

    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    semantic_path = tmp_path / "semantic.png"
    generation_path = tmp_path / "generation.png"
    _save_rgb(source_path, source)
    _save_rgb(output_path, output)
    _save_mask(semantic_path, semantic)
    _save_mask(generation_path, generation)

    metrics = measure_spatial_change(
        source_path,
        output_path,
        semantic_path,
        generation_path,
    )

    assert metrics["changed_fraction_1"] == pytest.approx(4 / 16)
    assert metrics["changed_fraction_5"] == pytest.approx(3 / 16)
    assert metrics["changed_fraction_10"] == pytest.approx(2 / 16)
    assert metrics["outside_semantic_fraction_1"] == pytest.approx(2 / 16)
    assert metrics["outside_semantic_fraction_5"] == pytest.approx(2 / 16)
    assert metrics["outside_semantic_fraction_10"] == pytest.approx(2 / 16)
    assert metrics["outside_generation_fraction_5"] == pytest.approx(
        2 * (127 / 255) / 16
    )
    assert metrics["semantic_mask_fraction"] == pytest.approx(2 / 16)
    assert metrics["generation_mask_fraction"] == pytest.approx(
        (2 + 2 * (128 / 255)) / 16
    )

    pixel_l1 = np.mean(np.abs(output.astype(float) - source), axis=2) / 255
    assert metrics["inside_semantic_l1"] == pytest.approx(
        pixel_l1[semantic >= 128].mean()
    )
    assert metrics["outside_semantic_l1"] == pytest.approx(
        pixel_l1[semantic < 128].mean()
    )


def test_measure_spatial_change_resizes_masks_to_image_size(tmp_path):
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    output = source.copy()
    output[:2, :2] = 255
    semantic = np.array([[255, 0], [0, 0]], dtype=np.uint8)

    source_path = tmp_path / "source.png"
    output_path = tmp_path / "output.png"
    semantic_path = tmp_path / "semantic.png"
    generation_path = tmp_path / "generation.png"
    _save_rgb(source_path, source)
    _save_rgb(output_path, output)
    _save_mask(semantic_path, semantic)
    _save_mask(generation_path, semantic)

    metrics = measure_spatial_change(
        source_path,
        output_path,
        semantic_path,
        generation_path,
    )

    assert metrics["semantic_mask_fraction"] == pytest.approx(4 / 16)
    assert metrics["outside_semantic_fraction_10"] == 0


def test_measure_spatial_change_resizes_source_like_sd2_backend(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    semantic = tmp_path / "semantic.png"
    generation = tmp_path / "generation.png"
    _save_rgb(source, np.zeros((8, 8, 3), dtype=np.uint8))
    _save_rgb(output, np.zeros((4, 4, 3), dtype=np.uint8))
    _save_mask(semantic, np.ones((8, 8), dtype=np.uint8) * 255)
    _save_mask(generation, np.ones((8, 8), dtype=np.uint8) * 255)

    metrics = measure_spatial_change(source, output, semantic, generation)

    assert metrics["changed_fraction_1"] == 0


def test_selects_smallest_passing_candidate_before_stronger_candidate():
    selected = select_spatial_candidate(
        [
            {
                "candidate": "d0",
                "desired_probability": 0.79,
                "changed_fraction_5": 0.01,
                "outside_semantic_fraction_5": 0.0,
            },
            {
                "candidate": "d4",
                "desired_probability": 0.82,
                "changed_fraction_5": 0.05,
                "outside_semantic_fraction_5": 0.02,
            },
            {
                "candidate": "d8",
                "desired_probability": 0.91,
                "changed_fraction_5": 0.08,
                "outside_semantic_fraction_5": 0.03,
            },
        ]
    )

    assert selected["candidate"] == "d4"


def test_passing_candidate_ties_use_outside_change_then_probability_identity():
    selected = select_spatial_candidate(
        [
            {
                "candidate": "d0",
                "desired_probability": 0.90,
                "changed_fraction_5": 0.05,
                "outside_semantic_fraction_5": 0.03,
                "identity_cosine": 0.99,
            },
            {
                "candidate": "d4",
                "desired_probability": 0.85,
                "changed_fraction_5": 0.05,
                "outside_semantic_fraction_5": 0.01,
                "identity_cosine": 0.80,
            },
        ]
    )

    assert selected["candidate"] == "d4"


def test_all_failed_candidates_choose_highest_probability_then_smallest_area():
    selected = select_spatial_candidate(
        [
            {
                "candidate": "d0",
                "desired_probability": 0.70,
                "changed_fraction_5": 0.02,
            },
            {
                "candidate": "d4",
                "desired_probability": 0.79,
                "changed_fraction_5": 0.06,
            },
            {
                "candidate": "d8",
                "desired_probability": 0.79,
                "changed_fraction_5": 0.08,
            },
        ]
    )

    assert selected["candidate"] == "d4"


def test_candidate_selection_requires_rows_and_required_metrics():
    with pytest.raises(ValueError, match="at least one"):
        select_spatial_candidate([])
    with pytest.raises(ValueError, match="desired_probability"):
        select_spatial_candidate([{"changed_fraction_5": 0.1}])
