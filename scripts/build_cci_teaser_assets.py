#!/usr/bin/env python3
"""Build reproducible, paper-owned qualitative CCI figure assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps


IMAGE_SIZE = 512
OVERLAY_ALPHA = 0.45
CROP_SCALE = 1.75
MINIMUM_CROP_SIDE = 128
OVERLAY_COLOR = (220, 32, 96)


@dataclass(frozen=True)
class SamplePaths:
    sample_id: int
    source: Path
    mask_a0: Path
    mask_a11: Path
    bld: Path
    adaptive: Path


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def resolve_sample_paths(
    experiment_root: Path,
    image_root: Path,
    sample_id: int,
) -> SamplePaths:
    """Resolve one paired sample from the reviewed experiment layout."""

    sample_dir = Path(experiment_root) / "smile" / f"{sample_id:05d}"
    source = _require_file(
        Path(image_root) / f"{sample_id}.jpg", "source image"
    )
    mask_a0 = _require_file(
        sample_dir / "A0" / "semantic_mask.png", "A0 semantic mask"
    )
    mask_a11 = _require_file(
        sample_dir / "A11" / "semantic_mask.png", "A11 semantic mask"
    )
    bld = _require_file(
        sample_dir
        / "A0"
        / "candidates"
        / "d8"
        / "sd2_bld_grid_corrected.png",
        "BLD output",
    )
    adaptive = _require_file(
        sample_dir
        / "A11"
        / "candidates"
        / "d8"
        / "sd2_bld_grid_corrected.png",
        "adaptive output",
    )
    return SamplePaths(
        sample_id=sample_id,
        source=source,
        mask_a0=mask_a0,
        mask_a11=mask_a11,
        bld=bld,
        adaptive=adaptive,
    )


def _resampling(name: str) -> int:
    enum = getattr(Image, "Resampling", Image)
    return getattr(enum, name)


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        return normalized.resize(
            (IMAGE_SIZE, IMAGE_SIZE), _resampling("LANCZOS")
        )


def _load_mask(path: Path) -> Image.Image:
    with Image.open(path) as image:
        normalized = image.convert("L").resize(
            (IMAGE_SIZE, IMAGE_SIZE), _resampling("NEAREST")
        )
    return normalized.point(lambda value: 255 if value > 0 else 0)


def square_crop_box(
    mask: Image.Image,
    min_side: int,
    scale: float,
) -> tuple[int, int, int, int]:
    """Return a clipped square crop centered on a non-empty mask."""

    if min_side <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("crop settings must be positive")
    normalized = mask.convert("L")
    bounds = normalized.getbbox()
    if bounds is None:
        raise ValueError("semantic mask is empty")
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    canvas_width, canvas_height = normalized.size
    side = max(min_side, int(math.ceil(scale * max(width, height))))
    side = min(side, canvas_width, canvas_height)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_left = int(round(center_x - side / 2.0))
    crop_top = int(round(center_y - side / 2.0))
    crop_left = max(0, min(crop_left, canvas_width - side))
    crop_top = max(0, min(crop_top, canvas_height - side))
    return (
        crop_left,
        crop_top,
        crop_left + side,
        crop_top + side,
    )


def mask_overlay(
    source: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int] = OVERLAY_COLOR,
    alpha: float = OVERLAY_ALPHA,
) -> Image.Image:
    """Overlay a color only on non-zero mask pixels."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("overlay alpha must be in [0, 1]")
    source_rgb = source.convert("RGB")
    normalized_mask = mask.convert("L")
    if normalized_mask.size != source_rgb.size:
        normalized_mask = normalized_mask.resize(
            source_rgb.size, _resampling("NEAREST")
        )
    binary_mask = normalized_mask.point(
        lambda value: 255 if value > 0 else 0
    )
    tint = Image.new("RGB", source_rgb.size, color)
    blended = Image.blend(source_rgb, tint, alpha)
    return Image.composite(blended, source_rgb, binary_mask)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _save_jpeg(image: Image.Image, path: Path) -> None:
    image.save(
        path,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
        progressive=False,
    )


def build_sample_assets(
    paths: SamplePaths,
    output_dir: Path,
) -> dict[str, object]:
    """Build source, overlay, paired output, and mouth-crop assets."""

    if paths.mask_a0.read_bytes() != paths.mask_a11.read_bytes():
        raise ValueError(
            f"semantic masks differ for sample {paths.sample_id}"
        )
    source = _load_rgb(paths.source)
    mask = _load_mask(paths.mask_a11)
    crop_box = square_crop_box(
        mask,
        min_side=MINIMUM_CROP_SIDE,
        scale=CROP_SCALE,
    )
    bld = _load_rgb(paths.bld)
    adaptive = _load_rgb(paths.adaptive)

    stem = f"{paths.sample_id:05d}"
    outputs = {
        "source": output_dir / f"{stem}_source.jpg",
        "mask_overlay": output_dir / f"{stem}_mask_overlay.png",
        "bld": output_dir / f"{stem}_bld.jpg",
        "ours": output_dir / f"{stem}_ours.jpg",
        "bld_crop": output_dir / f"{stem}_bld_crop.png",
        "ours_crop": output_dir / f"{stem}_ours_crop.png",
    }
    _save_jpeg(source, outputs["source"])
    mask_overlay(source, mask).save(outputs["mask_overlay"], format="PNG")
    _save_jpeg(bld, outputs["bld"])
    _save_jpeg(adaptive, outputs["ours"])
    bld.crop(crop_box).resize(
        (IMAGE_SIZE, IMAGE_SIZE), _resampling("LANCZOS")
    ).save(outputs["bld_crop"], format="PNG")
    adaptive.crop(crop_box).resize(
        (IMAGE_SIZE, IMAGE_SIZE), _resampling("LANCZOS")
    ).save(outputs["ours_crop"], format="PNG")

    return {
        "sample_id": paths.sample_id,
        "crop_box": list(crop_box),
        "inputs": {
            "source": _input_record(paths.source),
            "semantic_mask": _input_record(paths.mask_a11),
            "bld": _input_record(paths.bld),
            "adaptive": _input_record(paths.adaptive),
        },
        "outputs": {
            label: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for label, path in outputs.items()
        },
    }


def build_teaser_assets(
    experiment_root: Path,
    image_root: Path,
    sample_ids: Sequence[int],
    output_dir: Path,
    provenance_path: Path,
) -> dict[str, object]:
    """Build all qualitative assets and a provenance manifest."""

    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample IDs must be non-empty and unique")
    output_dir = Path(output_dir)
    provenance_path = Path(provenance_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_id in sample_ids:
        paths = resolve_sample_paths(
            Path(experiment_root), Path(image_root), int(sample_id)
        )
        records.append(build_sample_assets(paths, output_dir))
    payload = {
        "settings": {
            "image_size": IMAGE_SIZE,
            "overlay_alpha": OVERLAY_ALPHA,
            "overlay_color": list(OVERLAY_COLOR),
            "crop_scale": CROP_SCALE,
            "minimum_crop_side": MINIMUM_CROP_SIDE,
        },
        "samples": records,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument(
        "--sample_ids", type=int, nargs="+", required=True
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--provenance_out", type=Path, required=True)
    args = parser.parse_args()
    build_teaser_assets(
        args.experiment_root,
        args.image_root,
        args.sample_ids,
        args.output_dir,
        args.provenance_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
