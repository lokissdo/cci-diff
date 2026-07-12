"""Executable diffusion smoke runner."""

from __future__ import annotations

import json
from pathlib import Path

from cci_diff.config import load_cci_config
from cci_diff.diffusers_backend import DiffusersTextToImageBackend
from cci_diff.diffusion_state import DiffusionRunResult
from cci_diff.fake_backend import FakeDiffusionBackend
from cci_diff.prompts import build_concept_prompt


def run_diffusion_smoke(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    backend_name: str,
    num_inference_steps: int,
    seed: int,
    model_id: str = "hf-internal-testing/tiny-stable-diffusion-pipe",
    device: str = "cpu",
    torch_dtype: str = "auto",
    local_files_only: bool = False,
) -> DiffusionRunResult:
    """Run a tiny generation path and write an audit JSON file."""

    config = load_cci_config(config_path)
    prompt = build_concept_prompt(config.intervention)
    output_dir = Path(output_dir)
    backend = _build_backend(
        backend_name=backend_name,
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    suffix = ".ppm" if backend.name == "fake" else ".png"
    image_path = output_dir / f"sample{suffix}"
    states = backend.generate(
        prompt=prompt.positive,
        output_path=image_path,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )
    result = DiffusionRunResult(
        image_path=str(image_path),
        prompt=prompt.positive,
        backend=backend.name,
        states=states,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return result


def _build_backend(
    *,
    backend_name: str,
    model_id: str,
    device: str,
    torch_dtype: str,
    local_files_only: bool,
):
    if backend_name == "fake":
        return FakeDiffusionBackend()
    if backend_name == "diffusers":
        return DiffusersTextToImageBackend(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
    raise ValueError("backend_name must be fake or diffusers")
