from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cci_diff.region_screening import (
    build_union_mask,
    canonical_region_sets,
    celebamask_component_path,
    score_region_masks,
    select_saliency_covering_regions,
    select_screened_regions,
)


def test_score_region_masks_uses_saliency_mass_and_density():
    saliency = np.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    mouth = np.zeros((4, 4), dtype=np.uint8)
    mouth[0, 0] = 255
    lips = np.zeros((4, 4), dtype=np.uint8)
    lips[1, 1] = 255
    lips[2, 2] = 255

    scores = score_region_masks(
        saliency,
        {"mouth": mouth, "upper_lip": lips},
    )

    assert [item.region for item in scores] == ["mouth", "upper_lip"]
    assert scores[0].captured_mass == pytest.approx(0.5)
    assert scores[0].region_density == pytest.approx(4.0)
    assert scores[0].mask_fraction == pytest.approx(1 / 16)
    assert scores[0].proposal_score == pytest.approx(2.0)
    assert scores[1].captured_mass == pytest.approx(3 / 8)
    assert scores[1].region_density == pytest.approx(1.5)


def test_score_region_masks_rejects_invalid_inputs():
    valid = np.ones((4, 4), dtype=np.float32)
    mask = np.ones((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="identical shapes"):
        score_region_masks(valid, {"mouth": np.ones((3, 4))})
    with pytest.raises(ValueError, match="finite"):
        score_region_masks(
            np.full((4, 4), np.nan), {"mouth": mask}
        )
    with pytest.raises(ValueError, match="non-empty"):
        score_region_masks(valid, {"mouth": np.zeros((4, 4))})


def test_score_region_masks_accepts_full_semantic_ontology():
    saliency = np.ones((4, 4), dtype=np.float32)
    mask = np.ones((4, 4), dtype=np.uint8)

    scores = score_region_masks(
        saliency,
        {f"region_{index}": mask for index in range(19)},
    )

    assert len(scores) == 19


def test_select_screened_regions_prioritizes_concentrated_heatmap_evidence():
    summary = [
        {
            "region": "skin",
            "median_region_density": 0.12,
            "median_captured_mass": 0.76,
            "coverage_frequency": 1.0,
            "median_mask_fraction": 0.31,
        },
        {
            "region": "mouth",
            "median_region_density": 0.76,
            "median_captured_mass": 0.10,
            "coverage_frequency": 1.0,
            "median_mask_fraction": 0.01,
        },
        {
            "region": "lower_lip",
            "median_region_density": 0.69,
            "median_captured_mass": 0.12,
            "coverage_frequency": 1.0,
            "median_mask_fraction": 0.01,
        },
        {
            "region": "upper_lip",
            "median_region_density": 0.64,
            "median_captured_mass": 0.06,
            "coverage_frequency": 1.0,
            "median_mask_fraction": 0.01,
        },
        {
            "region": "noise",
            "median_region_density": 0.99,
            "median_captured_mass": 0.01,
            "coverage_frequency": 1.0,
            "median_mask_fraction": 0.001,
        },
    ]

    selected = select_screened_regions(
        summary,
        top_k=3,
        minimum_coverage_frequency=0.95,
        minimum_captured_saliency=0.02,
    )

    assert selected == ("mouth", "lower_lip", "upper_lip")


def test_saliency_covering_selection_uses_smallest_dynamic_subset():
    rows = []
    for sample_id in range(3):
        rows.extend(
            [
                {
                    "sample_id": sample_id,
                    "region": "mouth",
                    "captured_mass": 0.48,
                    "mask_fraction": 0.02,
                },
                {
                    "sample_id": sample_id,
                    "region": "lower_lip",
                    "captured_mass": 0.34,
                    "mask_fraction": 0.01,
                },
                {
                    "sample_id": sample_id,
                    "region": "skin",
                    "captured_mass": 0.90,
                    "mask_fraction": 0.35,
                },
            ]
        )

    selected, evidence, status = select_saliency_covering_regions(
        rows,
        saliency_coverage_threshold=0.80,
        cohort_frequency_threshold=1.0,
        max_regions=4,
    )

    assert selected == ("lower_lip", "mouth")
    assert status == "meets_coverage"
    assert evidence[0].regions
    assert all(len(item.regions) <= 4 for item in evidence)


def test_saliency_covering_selection_caps_maximum_regions_at_four():
    rows = [
        {
            "sample_id": 0,
            "region": f"r{index}",
            "captured_mass": 0.2,
            "mask_fraction": 0.01,
        }
        for index in range(5)
    ]

    with pytest.raises(ValueError, match="cannot exceed 4"):
        select_saliency_covering_regions(
            rows,
            saliency_coverage_threshold=0.8,
            cohort_frequency_threshold=1.0,
            max_regions=5,
        )


def test_canonical_region_sets_are_sorted_unique_and_bounded():
    combinations = canonical_region_sets(
        ["upper_lip", "mouth", "lower_lip", "mouth"],
        max_set_size=2,
    )

    assert combinations == (
        ("lower_lip",),
        ("mouth",),
        ("upper_lip",),
        ("lower_lip", "mouth"),
        ("lower_lip", "upper_lip"),
        ("mouth", "upper_lip"),
    )
    with pytest.raises(ValueError, match="four"):
        canonical_region_sets([f"r{index}" for index in range(5)])
    assert len(
        canonical_region_sets(
            [f"r{index}" for index in range(4)], max_set_size=4
        )
    ) == 15


def test_build_union_mask_uses_exact_binary_union_and_writes_png(tmp_path):
    mouth = np.zeros((4, 4), dtype=np.uint8)
    mouth[1, 1] = 255
    lip = np.zeros((4, 4), dtype=np.uint8)
    lip[2, 2] = 10
    output = tmp_path / "target_region.png"

    union = build_union_mask(
        {"mouth": mouth, "lower_lip": lip},
        ("lower_lip", "mouth"),
        output_path=output,
    )

    assert union.dtype == np.uint8
    assert np.count_nonzero(union) == 2
    assert union[1, 1] == 255
    assert union[2, 2] == 255
    assert np.array_equal(np.asarray(Image.open(output)), union)


def test_celebamask_component_path_resolves_canonical_aliases():
    root = Path("/masks")

    assert celebamask_component_path(root, 1, "upper_lip") == (
        root / "0" / "00001_u_lip.png"
    )
    assert celebamask_component_path(root, 2345, "left_eye") == (
        root / "1" / "02345_l_eye.png"
    )
    with pytest.raises(ValueError, match="Unknown CelebAMask"):
        celebamask_component_path(root, 1, "cheek")
