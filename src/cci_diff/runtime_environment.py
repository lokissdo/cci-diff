"""Portable, fail-closed runtime resolution for local inference jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def resolve_device(requested: str, torch_module: Any) -> str:
    """Resolve ``auto`` as CUDA, then MPS, then CPU."""

    device = str(requested).strip().lower()
    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be one of auto, cuda, mps, or cpu")
    cuda_available = bool(torch_module.cuda.is_available())
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(
        mps_backend is not None and mps_backend.is_available()
    )
    if device == "auto":
        return "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if device == "cuda" and not cuda_available:
        raise ValueError("CUDA is unavailable")
    if device == "mps" and not mps_available:
        raise ValueError("MPS is unavailable")
    return device


def validate_local_artifacts(
    paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    """Resolve required local files/directories and reject missing artifacts."""

    resolved = {}
    for raw_name, raw_path in paths.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("artifact names must be non-empty")
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{name} artifact not found: {path}")
        resolved[name] = path
    return resolved
