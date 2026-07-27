"""Semantic mask preparation helpers for localized CCI generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MaskArtifacts:
    """Saved hard semantic and feathered generation masks."""

    semantic_path: str
    generation_path: str
    semantic_fraction: float


def _validate_dilation_radius(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _maximum_filter_axis(values, radius: int, *, axis: int):
    if radius == 0:
        return values

    import numpy as np

    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="constant", constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        2 * radius + 1,
        axis=axis,
    )
    return windows.max(axis=-1)


def prepare_semantic_masks(
    component_paths: Iterable[str | Path],
    *,
    feather_radius: float,
    dilation_radius: int = 0,
    dilation_x: int | None = None,
    dilation_y: int | None = None,
    hard_output: str | Path,
    soft_output: str | Path,
) -> MaskArtifacts:
    """Union aligned masks and save a dilated, feathered generation copy."""

    if feather_radius < 0:
        raise ValueError("feather_radius must be non-negative")
    _validate_dilation_radius("dilation_radius", dilation_radius)
    effective_x = dilation_radius if dilation_x is None else dilation_x
    effective_y = dilation_radius if dilation_y is None else dilation_y
    _validate_dilation_radius("dilation_x", effective_x)
    _validate_dilation_radius("dilation_y", effective_y)

    import numpy as np
    from PIL import Image, ImageChops, ImageFilter

    paths = [Path(path) for path in component_paths]
    if not paths:
        raise ValueError("At least one generation mask component is required")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Generation mask components not found: {missing}")

    masks = [Image.open(path).convert("L") for path in paths]
    sizes = {mask.size for mask in masks}
    if len(sizes) != 1:
        raise ValueError("Generation mask components must have the same dimensions")

    semantic = masks[0].point(lambda value: 255 if value >= 128 else 0)
    for mask in masks[1:]:
        binary = mask.point(lambda value: 255 if value >= 128 else 0)
        semantic = ImageChops.lighter(semantic, binary)
    semantic_values = np.asarray(semantic, dtype=np.uint8)
    dilated_values = _maximum_filter_axis(
        semantic_values,
        effective_x,
        axis=1,
    )
    dilated_values = _maximum_filter_axis(
        dilated_values,
        effective_y,
        axis=0,
    )
    dilated = Image.fromarray(dilated_values, mode="L")
    generation = dilated.filter(ImageFilter.GaussianBlur(feather_radius))

    hard_output = Path(hard_output)
    soft_output = Path(soft_output)
    hard_output.parent.mkdir(parents=True, exist_ok=True)
    soft_output.parent.mkdir(parents=True, exist_ok=True)
    semantic.save(hard_output)
    generation.save(soft_output)

    semantic_fraction = float((np.array(semantic) >= 128).mean())
    return MaskArtifacts(
        semantic_path=str(hard_output),
        generation_path=str(soft_output),
        semantic_fraction=semantic_fraction,
    )
