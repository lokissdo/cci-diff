from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from scripts.build_cci_teaser_assets import (
    build_teaser_assets,
    mask_overlay,
    square_crop_box,
)


def _make_layout(tmp_path: Path) -> tuple[Path, Path, Image.Image]:
    experiment_root = tmp_path / "experiment"
    image_root = tmp_path / "images"
    image_root.mkdir()

    source = Image.new("RGB", (64, 64), (30, 60, 90))
    source.save(image_root / "1.jpg", quality=100, subsampling=0)

    mask = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask).rectangle((26, 28, 37, 35), fill=255)
    for variant, color in (("A0", (90, 70, 50)), ("A11", (50, 70, 90))):
        variant_dir = experiment_root / "smile" / "00001" / variant
        candidate_dir = variant_dir / "candidates" / "d8"
        candidate_dir.mkdir(parents=True)
        mask.save(variant_dir / "semantic_mask.png")
        Image.new("RGB", (64, 64), color).save(
            candidate_dir / "sd2_bld_grid_corrected.png"
        )
    return experiment_root, image_root, mask


def test_square_crop_box_and_overlay_are_mask_derived():
    source = Image.new("RGB", (64, 64), (30, 60, 90))
    mask = Image.new("L", (64, 64), 0)
    ImageDraw.Draw(mask).rectangle((26, 28, 37, 35), fill=255)

    assert square_crop_box(mask, min_side=48, scale=4.0) == (8, 8, 56, 56)

    overlay = mask_overlay(source, mask)
    assert overlay.getpixel((0, 0)) == source.getpixel((0, 0))
    assert overlay.getpixel((30, 30)) != source.getpixel((30, 30))


def test_build_teaser_assets_writes_normalized_images_and_provenance(
    tmp_path: Path,
):
    experiment_root, image_root, _ = _make_layout(tmp_path)
    output_dir = tmp_path / "paper-assets"
    provenance_path = tmp_path / "provenance.json"

    payload = build_teaser_assets(
        experiment_root,
        image_root,
        [1],
        output_dir,
        provenance_path,
    )

    assert payload["samples"][0]["sample_id"] == 1
    assert payload["samples"][0]["crop_box"] == [172, 172, 340, 340]
    assert Image.open(output_dir / "00001_source.jpg").size == (512, 512)
    assert Image.open(output_dir / "00001_mask_overlay.png").size == (512, 512)
    assert Image.open(output_dir / "00001_bld_crop.png").size == (512, 512)
    assert Image.open(output_dir / "00001_ours_crop.png").size == (512, 512)
    assert provenance_path.is_file()
    assert len(payload["samples"][0]["inputs"]["source"]["sha256"]) == 64


def test_build_teaser_assets_rejects_empty_or_inconsistent_masks(tmp_path: Path):
    experiment_root, image_root, _ = _make_layout(tmp_path)
    mask_a11 = (
        experiment_root
        / "smile"
        / "00001"
        / "A11"
        / "semantic_mask.png"
    )
    Image.new("L", (64, 64), 0).save(mask_a11)

    with pytest.raises(ValueError, match="semantic masks differ"):
        build_teaser_assets(
            experiment_root,
            image_root,
            [1],
            tmp_path / "out",
            tmp_path / "provenance.json",
        )

    empty = Image.new("L", (64, 64), 0)
    for variant in ("A0", "A11"):
        empty.save(
            experiment_root
            / "smile"
            / "00001"
            / variant
            / "semantic_mask.png"
        )
    with pytest.raises(ValueError, match="semantic mask is empty"):
        build_teaser_assets(
            experiment_root,
            image_root,
            [1],
            tmp_path / "out-empty",
            tmp_path / "empty.json",
        )


def test_build_teaser_assets_rejects_missing_output_source_and_duplicate_ids(
    tmp_path: Path,
):
    experiment_root, image_root, _ = _make_layout(tmp_path)
    missing_output = (
        experiment_root
        / "smile"
        / "00001"
        / "A11"
        / "candidates"
        / "d8"
        / "sd2_bld_grid_corrected.png"
    )
    missing_output.unlink()
    with pytest.raises(FileNotFoundError, match="adaptive output"):
        build_teaser_assets(
            experiment_root,
            image_root,
            [1],
            tmp_path / "out-missing-output",
            tmp_path / "missing-output.json",
        )

    second_case = tmp_path / "missing-source-case"
    second_case.mkdir()
    experiment_root, image_root, _ = _make_layout(second_case)
    (image_root / "1.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="source image"):
        build_teaser_assets(
            experiment_root,
            image_root,
            [1],
            tmp_path / "out-missing-source",
            tmp_path / "missing-source.json",
        )

    with pytest.raises(ValueError, match="non-empty and unique"):
        build_teaser_assets(
            experiment_root,
            image_root,
            [1, 1],
            tmp_path / "out-duplicate",
            tmp_path / "duplicate.json",
        )
