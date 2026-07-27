"""Stable Diffusion 2 blended-latent backend with an in-loop CCI hook."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cci_diff.diffusion_state import DiffusionRunResult, DiffusionState


CCIGuidanceHook = Callable[["SD2DenoisingStep"], Any | None]
CCILatentGuidanceHook = Callable[["SD2DenoisingStep"], Any | None]


@dataclass(frozen=True)
class SD2DenoisingStep:
    """Objects available at the CCI hook point inside the SD2 denoising loop."""

    step_index: int
    timestep: Any
    prompt: str
    latents: Any
    noise_pred: Any
    source_latents: Any
    latent_mask: Any
    semantic_mask: Any | None = None
    total_steps: int = 1
    progress: float = 0.0


def require_sd2_dependencies():
    """Import optional SD2 GPU dependencies only when the backend is used."""

    try:
        import numpy as np
        import torch
        from diffusers import DDIMScheduler, DiffusionPipeline
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "SD2 BLD backend requires ML dependencies. "
            "Create a compatible Python 3.10+ venv, then run: "
            "pip install -e '.[ml]'"
        ) from exc
    return torch, np, Image, DDIMScheduler, DiffusionPipeline


@contextmanager
def suppress_transformers_progress():
    """Hide per-parameter model loading bars and restore the prior setting."""

    try:
        from transformers.utils import logging as transformers_logging
    except ImportError:
        yield
        return

    was_enabled = transformers_logging.is_progress_bar_enabled()
    if was_enabled:
        transformers_logging.disable_progress_bar()
    try:
        yield
    finally:
        if was_enabled:
            transformers_logging.enable_progress_bar()


def blending_start_index(num_timesteps: int, blending_percentage: float) -> int:
    """Return the first denoising index used by the BLD loop."""

    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    raw_index = int(num_timesteps * blending_percentage)
    return max(0, min(num_timesteps - 1, raw_index))


def denoising_progress(step_index: int, total_steps: int) -> float:
    """Normalize a step over the selected reverse-diffusion interval."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step_index < 0 or step_index >= total_steps:
        raise ValueError("step_index must be inside the selected reverse interval")
    return step_index / max(total_steps - 1, 1)


def apply_cci_guidance(
    noise_pred: Any,
    step: SD2DenoisingStep,
    hook: CCIGuidanceHook | None,
) -> Any:
    """Let a CCI hook replace CFG noise; keep CFG noise when hook returns None."""

    if hook is None:
        return noise_pred
    guided_noise = hook(step)
    return noise_pred if guided_noise is None else guided_noise


def apply_cci_latent_guidance_hook(
    latents: Any,
    step: SD2DenoisingStep,
    hook: CCILatentGuidanceHook | None,
) -> Any:
    """Let a CCI hook replace latents; keep scheduler latents when it returns None."""

    if hook is None:
        return latents
    guided_latents = hook(step)
    return latents if guided_latents is None else guided_latents


def diffusion_state_from_step(step: SD2DenoisingStep, *, phase: str) -> DiffusionState:
    """Build a serializable state record from tensors at a denoising step."""

    return DiffusionState(
        step_index=step.step_index,
        timestep=_to_number(step.timestep),
        prompt=step.prompt,
        latent_shape=_shape_tuple(step.latents),
        phase=phase,
        extra={
            "noise_pred_shape": _shape_tuple(step.noise_pred),
            "source_latent_shape": _shape_tuple(step.source_latents),
            "mask_shape": _shape_tuple(step.latent_mask),
            "semantic_mask_shape": _shape_tuple(step.semantic_mask),
            "total_steps": step.total_steps,
            "progress": step.progress,
        },
    )


def blend_latents(latents: Any, latent_mask: Any, noise_source_latents: Any) -> Any:
    """Select edited latents inside the mask and source latents outside it."""

    return noise_source_latents.where(~latent_mask.bool(), latents)


def blend_soft_latents(
    edited_latents: Any,
    generation_mask: Any,
    source_latents: Any,
) -> Any:
    """Interpolate edited and source latents with a fractional mask."""

    return generation_mask * edited_latents + (1.0 - generation_mask) * source_latents


def seeded_noise_like(reference: Any, generator: Any) -> Any:
    """Draw reference-shaped noise from an explicit reproducible generator."""

    import torch

    return torch.randn(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=reference.dtype,
        layout=reference.layout,
    )


def replace_nonfinite_latents(latents: Any, fallback_latents: Any) -> Any:
    """Replace NaN/Inf latent values with finite fallback values."""

    torch, _, _, _, _ = require_sd2_dependencies()
    return torch.where(torch.isfinite(latents), latents, fallback_latents)


class BlendedLatentDiffusionSD2Backend:
    """GPU Stable Diffusion 2 image-editing backend following ESWA BLD."""

    name = "sd2-bld"

    def __init__(
        self,
        *,
        model_path: str = "stabilityai/stable-diffusion-2-base",
        device: str = "cuda",
        torch_dtype: str = "float16",
        lora_path: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        torch, _, _, scheduler_cls, pipeline_cls = require_sd2_dependencies()
        dtype = _resolve_torch_dtype(torch, torch_dtype)
        try:
            with suppress_transformers_progress():
                pipe = pipeline_cls.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    safety_checker=None,
                    local_files_only=local_files_only,
                )
        except OSError as exc:
            raise OSError(
                f"Cannot load SD2 model from {model_path!r}. "
                "If this environment cannot reach Hugging Face, download it first: "
                "python scripts/download_hf_model.py --local_dir checkpoints/sd2-base "
                "then rerun with --model_path checkpoints/sd2-base --local_files_only."
            ) from exc
        if lora_path:
            pipe.load_lora_weights(lora_path)
        self.device = device
        self.torch_dtype = dtype
        self.vae = pipe.vae.to(device).eval()
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(device).eval()
        self.unet = pipe.unet.to(device).eval()
        for component in (self.vae, self.text_encoder, self.unet):
            component.requires_grad_(False)
        self.scheduler = scheduler_cls(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

    def edit_image(
        self,
        *,
        init_image: str | Path,
        mask: str | Path,
        generation_mask: str | Path | None = None,
        semantic_mask: str | Path | None = None,
        prompt: str,
        output_path: str | Path,
        batch_size: int = 4,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        blending_percentage: float = 0.25,
        seed: int = 42,
        cci_guidance_hook: CCIGuidanceHook | None = None,
        cci_latent_guidance_hook: CCILatentGuidanceHook | None = None,
        initial_latent_mode: str = "random",
    ) -> DiffusionRunResult:
        """Run SD2 blended-latent editing and save a horizontal result grid."""

        torch, np, Image, _, _ = require_sd2_dependencies()
        prompts = [prompt] * batch_size
        source_latents = self._image_to_latents(init_image, height=height, width=width)
        latent_mask = self._read_mask(
            generation_mask if generation_mask is not None else mask,
            dest_size=(height // 8, width // 8),
            binary=generation_mask is None,
        )
        semantic_latent_mask = (
            self._read_mask(
                semantic_mask,
                dest_size=(height // 8, width // 8),
                binary=True,
            )
            if semantic_mask is not None
            else (latent_mask >= 0.5).to(latent_mask.dtype)
        )
        source_latents = source_latents.repeat((batch_size, 1, 1, 1))
        latent_mask = latent_mask.repeat((batch_size, 1, 1, 1))
        semantic_latent_mask = semantic_latent_mask.repeat((batch_size, 1, 1, 1))

        text_embeddings = self._encode_prompts(prompts)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        start_index = blending_start_index(len(timesteps), blending_percentage)
        latents = self._initial_latents(
            source_latents=source_latents,
            latent_shape=(batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
            start_timestep=timesteps[start_index],
            mode=initial_latent_mode,
        )

        states: list[DiffusionState] = []
        selected_timesteps = timesteps[start_index:]
        total_steps = len(selected_timesteps)
        for step_index, timestep in enumerate(selected_timesteps):
            progress = denoising_progress(step_index, total_steps)
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input,
                timestep=timestep,
            )
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            step = SD2DenoisingStep(
                step_index=step_index,
                timestep=timestep,
                prompt=prompt,
                latents=latents,
                noise_pred=noise_pred,
                source_latents=source_latents,
                latent_mask=latent_mask,
                semantic_mask=semantic_latent_mask,
                total_steps=total_steps,
                progress=progress,
            )
            noise_pred = apply_cci_guidance(noise_pred, step, cci_guidance_hook)
            step = SD2DenoisingStep(
                step_index=step_index,
                timestep=timestep,
                prompt=prompt,
                latents=latents,
                noise_pred=noise_pred,
                source_latents=source_latents,
                latent_mask=latent_mask,
                semantic_mask=semantic_latent_mask,
                total_steps=total_steps,
                progress=progress,
            )
            states.append(diffusion_state_from_step(step, phase="cci_guidance"))

            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
            step = SD2DenoisingStep(
                step_index=step_index,
                timestep=timestep,
                prompt=prompt,
                latents=latents,
                noise_pred=noise_pred,
                source_latents=source_latents,
                latent_mask=latent_mask,
                semantic_mask=semantic_latent_mask,
                total_steps=total_steps,
                progress=progress,
            )
            states.append(diffusion_state_from_step(step, phase="scheduler_step"))

            if cci_latent_guidance_hook is not None and not getattr(
                cci_latent_guidance_hook,
                "apply_after_blend",
                False,
            ):
                latents = apply_cci_latent_guidance_hook(
                    latents,
                    step,
                    cci_latent_guidance_hook,
                )
                step = SD2DenoisingStep(
                    step_index=step_index,
                    timestep=timestep,
                    prompt=prompt,
                    latents=latents,
                    noise_pred=noise_pred,
                    source_latents=source_latents,
                    latent_mask=latent_mask,
                    semantic_mask=semantic_latent_mask,
                    total_steps=total_steps,
                    progress=progress,
                )
                states.append(
                    diffusion_state_from_step(step, phase="cci_latent_guidance")
                )

            noise_source_latents = self.scheduler.add_noise(
                source_latents,
                seeded_noise_like(latents, generator),
                timestep,
            )
            latents = replace_nonfinite_latents(latents, noise_source_latents)
            if generation_mask is None:
                latents = blend_latents(latents, latent_mask, noise_source_latents)
            else:
                latents = blend_soft_latents(
                    latents,
                    latent_mask,
                    noise_source_latents,
                )
            latents = replace_nonfinite_latents(latents, noise_source_latents)
            if cci_latent_guidance_hook is not None and getattr(
                cci_latent_guidance_hook,
                "apply_after_blend",
                False,
            ):
                post_blend_step = SD2DenoisingStep(
                    step_index=step_index,
                    timestep=timestep,
                    prompt=prompt,
                    latents=latents,
                    noise_pred=noise_pred,
                    source_latents=source_latents,
                    latent_mask=latent_mask,
                    semantic_mask=semantic_latent_mask,
                    total_steps=total_steps,
                    progress=progress,
                )
                latents = apply_cci_latent_guidance_hook(
                    latents,
                    post_blend_step,
                    cci_latent_guidance_hook,
                )
            step = SD2DenoisingStep(
                step_index=step_index,
                timestep=timestep,
                prompt=prompt,
                latents=latents,
                noise_pred=noise_pred,
                source_latents=source_latents,
                latent_mask=latent_mask,
                semantic_mask=semantic_latent_mask,
                total_steps=total_steps,
                progress=progress,
            )
            states.append(diffusion_state_from_step(step, phase="blend"))

        latents = replace_nonfinite_latents(latents, source_latents)
        images = self._decode_latents(latents, np=np)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid = np.concatenate(images, axis=1)
        Image.fromarray(grid).save(output_path)
        return DiffusionRunResult(
            image_path=str(output_path),
            prompt=prompt,
            backend=self.name,
            states=states,
        )

    def _encode_prompts(self, prompts: list[str]):
        torch, _, _, _, _ = require_sd2_dependencies()
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.tokenizer(
            [""] * len(prompts),
            padding="max_length",
            max_length=text_input.input_ids.shape[-1],
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        return torch.cat([uncond_embeddings, text_embeddings])

    def _image_to_latents(self, image_path: str | Path, *, height: int, width: int):
        torch, np, Image, _, _ = require_sd2_dependencies()
        image = Image.open(image_path).convert("RGB").resize((height, width), Image.BILINEAR)
        image = np.array(image)[:, :, :3]
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.torch_dtype == torch.float16:
            image = image.half()
        latents = self.vae.encode(image)["latent_dist"].mean
        return latents * 0.18215

    def _read_mask(
        self,
        mask_path: str | Path,
        *,
        dest_size: tuple[int, int],
        binary: bool = True,
    ):
        torch, np, Image, _, _ = require_sd2_dependencies()
        resample = Image.NEAREST if binary else Image.BILINEAR
        mask = Image.open(mask_path).convert("L").resize(dest_size, resample)
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        if binary:
            mask_array[mask_array < 0.5] = 0
            mask_array[mask_array >= 0.5] = 1
        mask_array = mask_array[np.newaxis, np.newaxis, ...]
        tensor = torch.from_numpy(mask_array).to(self.device)
        if self.torch_dtype == torch.float16:
            tensor = tensor.half()
        return tensor

    def _initial_latents(
        self,
        *,
        source_latents,
        latent_shape: tuple[int, int, int, int],
        generator,
        start_timestep,
        mode: str,
    ):
        torch, _, _, _, _ = require_sd2_dependencies()
        if mode == "random":
            return torch.randn(
                latent_shape,
                generator=generator,
                device=self.device,
                dtype=self.torch_dtype,
            )
        if mode == "source_noise":
            return self.scheduler.add_noise(
                source_latents,
                seeded_noise_like(source_latents, generator),
                start_timestep,
            )
        raise ValueError("initial_latent_mode must be random or source_noise")

    def _decode_latents(self, latents, *, np):
        latents = 1 / 0.18215 * latents
        decoded_images = self.vae.decode(latents).sample
        images = (decoded_images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        return (images * 255).round().astype(np.uint8)


def _resolve_torch_dtype(torch, torch_dtype: str):
    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "float32":
        return torch.float32
    raise ValueError("torch_dtype must be float16 or float32")


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def _to_number(value: Any) -> int | float:
    if hasattr(value, "item"):
        return value.item()
    return value
