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


def prepare_semantic_masks(
    component_paths: Iterable[str | Path],
    *,
    feather_radius: float,
    hard_output: str | Path,
    soft_output: str | Path,
) -> MaskArtifacts:
    """Union aligned binary component masks and save a feathered copy."""

    if feather_radius < 0:
        raise ValueError("feather_radius must be non-negative")

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
    generation = semantic.filter(ImageFilter.GaussianBlur(feather_radius))

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
