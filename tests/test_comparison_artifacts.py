from __future__ import annotations

import json

import numpy as np
from PIL import Image

from cci_diff.comparison_artifacts import (
    create_paginated_pair_sheets,
    create_pair_image,
    materialize_selected_artifacts,
)


def _solid_image(path, size, color):
    Image.new("RGB", size, color).save(path)


def test_create_pair_image_preserves_both_aspect_ratios(tmp_path):
    source = tmp_path / "wide.png"
    output = tmp_path / "tall.png"
    destination = tmp_path / "pair.png"
    _solid_image(source, (200, 100), (255, 0, 0))
    _solid_image(output, (100, 200), (0, 0, 255))

    result = create_pair_image(source, output, destination, "A3 d4")

    assert result == destination
    pair = np.asarray(Image.open(destination).convert("RGB"))
    assert pair.shape == (548, 1024, 3)
    red = np.all(pair == (255, 0, 0), axis=2)
    blue = np.all(pair == (0, 0, 255), axis=2)
    red_y, red_x = np.where(red)
    blue_y, blue_x = np.where(blue)
    assert (red_x.max() - red_x.min() + 1, red_y.max() - red_y.min() + 1) == (
        512,
        256,
    )
    assert (blue_x.max() - blue_x.min() + 1, blue_y.max() - blue_y.min() + 1) == (
        256,
        512,
    )
    assert red_x.max() < 512
    assert blue_x.min() >= 512


def test_materialize_selected_artifacts_copies_portable_result(tmp_path):
    source = tmp_path / "source.jpg"
    candidate = tmp_path / "candidate"
    result_dir = tmp_path / "selected"
    candidate.mkdir()
    _solid_image(source, (200, 100), (255, 0, 0))
    _solid_image(candidate / "sd2_bld_grid.png", (100, 200), (0, 0, 255))
    _solid_image(candidate / "semantic_mask.png", (100, 200), (255, 255, 255))
    _solid_image(candidate / "generation_mask.png", (100, 200), (128, 128, 128))
    (candidate / "audit.json").write_text('{"ok": true}', encoding="utf-8")

    artifacts = materialize_selected_artifacts(
        source,
        candidate,
        result_dir,
        {"feature": "smile", "sample_id": "00001", "dilation": 4},
    )

    expected = {
        "input_path": result_dir / "input.jpg",
        "output_path": result_dir / "sd2_bld_grid.png",
        "audit_path": result_dir / "audit.json",
        "semantic_mask_path": result_dir / "semantic_mask.png",
        "generation_mask_path": result_dir / "generation_mask.png",
        "selection_path": result_dir / "selected.json",
        "comparison_path": result_dir / "input_output.jpg",
    }
    assert artifacts == {name: str(path) for name, path in expected.items()}
    assert all(path.is_file() for path in expected.values())
    assert json.loads(expected["selection_path"].read_text())["dilation"] == 4
    assert Image.open(expected["input_path"]).size == (200, 100)
    assert Image.open(expected["output_path"]).size == (100, 200)


def test_paginated_pair_sheets_respect_page_size(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    _solid_image(source, (32, 16), (255, 0, 0))
    _solid_image(output, (16, 32), (0, 0, 255))
    rows = [
        {
            "source_path": str(source),
            "output_path": str(output),
            "label": f"sample {index}",
        }
        for index in range(3)
    ]

    sheets = create_paginated_pair_sheets(rows, tmp_path / "sheets", page_size=2)

    assert [path.name for path in sheets] == ["pairs_001.jpg", "pairs_002.jpg"]
    assert all(path.is_file() for path in sheets)
