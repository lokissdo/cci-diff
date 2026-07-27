"""Portable selected-result artifacts and visual comparison sheets."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps


PANEL_SIZE = (512, 512)
LABEL_HEIGHT = 36
BACKGROUND = (24, 24, 24)
TEXT_COLOR = (240, 240, 240)


def _atomic_image_save(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    image_format = "JPEG" if destination.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
    save_options = {"quality": 94} if image_format == "JPEG" else {}
    image.save(temporary, format=image_format, **save_options)
    os.replace(temporary, destination)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _render_pair(
    source: str | Path,
    output: str | Path,
    label: str,
) -> Image.Image:
    canvas = Image.new("RGB", (PANEL_SIZE[0] * 2, PANEL_SIZE[1] + LABEL_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 11), "Input", fill=TEXT_COLOR)
    draw.text((PANEL_SIZE[0] + 12, 11), f"Output | {label}", fill=TEXT_COLOR)
    for index, path in enumerate((source, output)):
        with Image.open(path) as image:
            contained = ImageOps.contain(image.convert("RGB"), PANEL_SIZE)
        x = index * PANEL_SIZE[0] + (PANEL_SIZE[0] - contained.width) // 2
        y = LABEL_HEIGHT + (PANEL_SIZE[1] - contained.height) // 2
        canvas.paste(contained, (x, y))
    return canvas


def create_pair_image(
    source: str | Path,
    output: str | Path,
    destination: str | Path,
    label: str,
) -> Path:
    """Render source and output in fixed, aspect-preserving panels."""

    destination = Path(destination)
    _atomic_image_save(_render_pair(source, output, label), destination)
    return destination


def materialize_selected_artifacts(
    source: str | Path,
    candidate_dir: str | Path,
    result_dir: str | Path,
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    """Copy a selected candidate into a self-contained result directory."""

    source = Path(source)
    candidate_dir = Path(candidate_dir)
    result_dir = Path(result_dir)
    candidate_output = Path(
        str(metadata.get("output_path") or candidate_dir / "sd2_bld_grid.png")
    )
    required = ["audit.json", "semantic_mask.png", "generation_mask.png"]
    missing = [name for name in required if not (candidate_dir / name).is_file()]
    if not candidate_output.is_file():
        missing.append(str(candidate_output))
    if not source.is_file():
        raise FileNotFoundError(f"Source image not found: {source}")
    if missing:
        raise FileNotFoundError(f"Selected candidate artifacts not found: {missing}")

    paths = {
        "input_path": result_dir / "input.jpg",
        "output_path": result_dir / "sd2_bld_grid.png",
        "audit_path": result_dir / "audit.json",
        "semantic_mask_path": result_dir / "semantic_mask.png",
        "generation_mask_path": result_dir / "generation_mask.png",
        "selection_path": result_dir / "selected.json",
        "comparison_path": result_dir / "input_output.jpg",
    }
    _atomic_copy(source, paths["input_path"])
    _atomic_copy(candidate_output, paths["output_path"])
    for key, name in (
        ("audit_path", "audit.json"),
        ("semantic_mask_path", "semantic_mask.png"),
        ("generation_mask_path", "generation_mask.png"),
    ):
        _atomic_copy(candidate_dir / name, paths[key])

    selection = dict(metadata)
    selection["candidate_dir"] = str(candidate_dir)
    temporary = paths["selection_path"].with_name(".selected.json.tmp")
    temporary.write_text(
        json.dumps(selection, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, paths["selection_path"])
    label = str(metadata.get("label") or metadata.get("candidate") or "selected")
    create_pair_image(
        paths["input_path"],
        paths["output_path"],
        paths["comparison_path"],
        label,
    )
    return {name: str(path) for name, path in paths.items()}


def create_paginated_pair_sheets(
    rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    page_size: int = 20,
) -> list[Path]:
    """Create two-column contact sheets from selected source/output rows."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    tile_size = (512, 274)
    columns = 2
    for page_start in range(0, len(rows), page_size):
        page_rows = rows[page_start : page_start + page_size]
        row_count = (len(page_rows) + columns - 1) // columns
        sheet = Image.new(
            "RGB",
            (tile_size[0] * columns, tile_size[1] * row_count),
            BACKGROUND,
        )
        for index, row in enumerate(page_rows):
            pair = _render_pair(
                row["source_path"],
                row["output_path"],
                str(row.get("label", "selected")),
            )
            tile = pair.resize(tile_size, Image.Resampling.LANCZOS)
            sheet.paste(
                tile,
                ((index % columns) * tile_size[0], (index // columns) * tile_size[1]),
            )
        page_number = page_start // page_size + 1
        destination = output_dir / f"pairs_{page_number:03d}.jpg"
        _atomic_image_save(sheet, destination)
        sheets.append(destination)
    return sheets
