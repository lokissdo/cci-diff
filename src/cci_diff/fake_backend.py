"""Small local backend for testing the diffusion runner without ML dependencies."""

from __future__ import annotations

import random
from pathlib import Path

from cci_diff.diffusion_state import DiffusionState


class FakeDiffusionBackend:
    """Deterministic stand-in for a diffusion backend.

    It writes a tiny PPM image and records synthetic denoising states. This lets
    the runner, config, prompt, and audit path be tested on machines without GPU
    dependencies.
    """

    name = "fake"

    def generate(
        self,
        *,
        prompt: str,
        output_path: Path,
        num_inference_steps: int,
        seed: int,
    ) -> list[DiffusionState]:
        rng = random.Random(seed)
        width = 16
        height = 16
        color = (
            96 + rng.randrange(96),
            96 + rng.randrange(96),
            96 + rng.randrange(96),
        )
        _write_ppm(output_path, width, height, color)
        return [
            DiffusionState(
                step_index=i,
                timestep=num_inference_steps - i,
                prompt=prompt,
                latent_shape=(1, 4, height // 2, width // 2),
            )
            for i in range(num_inference_steps)
        ]


def _write_ppm(path: Path, width: int, height: int, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P3\n{width} {height}\n255\n"
    pixel = f"{color[0]} {color[1]} {color[2]}"
    body = "\n".join(" ".join([pixel] * width) for _ in range(height))
    path.write_text(header + body + "\n", encoding="ascii")
