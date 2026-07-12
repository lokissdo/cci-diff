"""Serializable diffusion run state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiffusionState:
    """Lightweight record for one denoising step."""

    step_index: int
    timestep: int | float
    prompt: str
    latent_shape: tuple[int, ...]
    phase: str = "denoise"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["latent_shape"] = list(self.latent_shape)
        if "extra" in payload:
            payload["extra"] = _json_ready(payload["extra"])
        return payload


@dataclass(frozen=True)
class DiffusionRunResult:
    """Serializable result of a smoke diffusion run."""

    image_path: str
    prompt: str
    backend: str
    states: list[DiffusionState]

    def to_dict(self) -> dict[str, object]:
        return {
            "image_path": self.image_path,
            "prompt": self.prompt,
            "backend": self.backend,
            "states": [state.to_dict() for state in self.states],
        }


def _json_ready(value):
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value
