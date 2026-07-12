"""Optional real diffusers backend."""

from __future__ import annotations

from pathlib import Path

from cci_diff.diffusion_state import DiffusionState


def require_diffusers():
    """Import optional ML dependencies with a useful installation message."""

    try:
        import torch
        from diffusers import DiffusionPipeline
    except ImportError as exc:
        raise ImportError(
            "Diffusers backend requires ML dependencies. "
            "Create a compatible Python 3.10+ venv, then run: "
            "pip install -e '.[ml]'"
        ) from exc
    return torch, DiffusionPipeline


class DiffusersTextToImageBackend:
    """Tiny text-to-image backend for smoke tests with real diffusers models."""

    name = "diffusers"

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cpu",
        torch_dtype: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        torch, pipeline_cls = require_diffusers()
        dtype = _resolve_dtype(torch, torch_dtype, device)
        self.device = device
        self.pipe = pipeline_cls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=local_files_only,
        )
        self.pipe = self.pipe.to(device)

    def generate(
        self,
        *,
        prompt: str,
        output_path: Path,
        num_inference_steps: int,
        seed: int,
    ) -> list[DiffusionState]:
        torch, _ = require_diffusers()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        states: list[DiffusionState] = []

        def capture_state(pipe, step_index, timestep, callback_kwargs):
            states.append(
                callback_state_from_kwargs(
                    step_index=step_index,
                    timestep=timestep,
                    prompt=prompt,
                    callback_kwargs=callback_kwargs,
                )
            )
            return callback_kwargs

        try:
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                callback_on_step_end=capture_state,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        except TypeError:
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output_path)
        if states:
            return states
        return [
            DiffusionState(
                step_index=i,
                timestep=num_inference_steps - i,
                prompt=prompt,
                latent_shape=(),
            )
            for i in range(num_inference_steps)
        ]


def _resolve_dtype(torch, torch_dtype: str, device: str):
    if torch_dtype == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "float32":
        return torch.float32
    raise ValueError("torch_dtype must be auto, float16, or float32")


def callback_state_from_kwargs(
    *,
    step_index: int,
    timestep,
    prompt: str,
    callback_kwargs: dict,
) -> DiffusionState:
    latents = callback_kwargs.get("latents")
    latent_shape = tuple(getattr(latents, "shape", ()))
    return DiffusionState(
        step_index=step_index,
        timestep=_to_int(timestep),
        prompt=prompt,
        latent_shape=latent_shape,
    )


def _to_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)
