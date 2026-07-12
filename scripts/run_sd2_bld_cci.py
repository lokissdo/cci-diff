#!/usr/bin/env python3
"""Run GPU Stable Diffusion 2 blended-latent editing with CCI audit state."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cci_diff.adapters.sd2_cci import apply_cci_latent_guidance, infer_target_rgb
from cci_diff.adapters.sd2_robust import (
    apply_robust_latent_guidance,
    boundary_loss,
    multi_scale_classifier_loss,
    residual_tv_loss,
    robust_step_size,
)
from cci_diff.classifiers.celeba_resnet50 import (
    CELEBA_ATTRIBUTES,
    classifier_logits,
    classifier_probabilities,
    load_celeba_resnet50,
    resolve_celeba_attribute_index,
)
from cci_diff.config import load_cci_config
from cci_diff.guidance import GuidanceTerms
from cci_diff.masking import MaskArtifacts, prepare_semantic_masks
from cci_diff.prompts import build_concept_prompt
from cci_diff.sd2_bld_backend import BlendedLatentDiffusionSD2Backend


@dataclass(frozen=True)
class ClassifierRuntime:
    """Loaded classifier state shared by guidance and post-run auditing."""

    model: Any
    path: str
    label_index: int
    attribute: str
    input_size: int
    device: str
    applied_steps: list[int] = field(default_factory=list)
    guidance_trace: list[dict[str, float | int]] = field(default_factory=list)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cci_config", required=True)
    parser.add_argument("--init_image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--model_path", default="stabilityai/stable-diffusion-2-base")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--blending_start_percentage", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load the model only from local cache/path; do not contact Hugging Face.",
    )
    parser.add_argument(
        "--initial_latent_mode",
        choices=["random", "source_noise"],
        default="random",
        help="Use random to match the ESWA SD2 script; source_noise matches the SDXL variant.",
    )
    parser.add_argument(
        "--cci_hook",
        choices=["none", "latent_color", "latent_classifier"],
        default="none",
        help="Enable optional masked color or classifier latent guidance.",
    )
    parser.add_argument(
        "--cci_step_size",
        type=float,
        default=0.03,
        help="Gradient step size for the latent CCI hook.",
    )
    parser.add_argument(
        "--cci_every_n_steps",
        type=int,
        default=4,
        help="Apply the latent CCI hook every N denoising steps.",
    )
    parser.add_argument(
        "--cci_normalize_grad",
        action="store_true",
        help="Normalize each latent gradient step before applying cci_step_size.",
    )
    parser.add_argument(
        "--cci_start_step",
        type=int,
        default=0,
        help="First denoising step index where the latent CCI hook may run.",
    )
    parser.add_argument(
        "--cci_end_step",
        type=int,
        default=None,
        help="Last denoising step index where the latent CCI hook may run.",
    )
    parser.add_argument(
        "--cci_target_rgb",
        default=None,
        help="Optional target RGB prior as R,G,B floats in [0, 1].",
    )
    parser.add_argument(
        "--classifier_path",
        default=None,
        help="CelebA ResNet50 state-dict path for latent_classifier guidance.",
    )
    parser.add_argument(
        "--classifier_label_index",
        type=int,
        default=None,
        help="Optional explicit CelebA output index; inferred from the concept by default.",
    )
    parser.add_argument(
        "--classifier_input_size",
        type=int,
        default=512,
        help="Square differentiable classifier input size.",
    )
    parser.add_argument(
        "--robust_classifier_guidance",
        action="store_true",
        help="Use multi-scale classifier and realism gradients with a semantic mask.",
    )
    parser.add_argument(
        "--generation_mask_component",
        action="append",
        default=[],
        help="Repeat for each aligned semantic component mask.",
    )
    parser.add_argument("--generation_mask_feather", type=float, default=3.0)
    parser.add_argument("--classifier_scales", default="256,384,512")
    parser.add_argument("--classifier_blur_sigma", type=float, default=1.0)
    parser.add_argument("--boundary_weight", type=float, default=0.3)
    parser.add_argument("--tv_weight", type=float, default=0.05)
    return parser


def resolve_prompt(*, config_path: str | Path, override: str | None) -> str:
    if override:
        return override
    config = load_cci_config(config_path)
    return build_concept_prompt(config.intervention).positive


def parse_target_rgb(value: str) -> tuple[float, float, float]:
    """Parse a user-supplied RGB prior in normalized [0, 1] space."""

    channels = tuple(float(channel.strip()) for channel in value.split(","))
    if len(channels) != 3:
        raise ValueError("--cci_target_rgb must contain exactly three channels")
    if any(channel < 0.0 or channel > 1.0 for channel in channels):
        raise ValueError("--cci_target_rgb channels must be in [0, 1]")
    return channels


def parse_classifier_scales(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive classifier view sizes."""

    scales = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("--classifier_scales must contain positive integers")
    return scales


def build_cci_latent_guidance_hook(backend, config, args):
    """Build the optional denoising-time CCI latent hook."""

    if args.cci_hook == "none":
        return None, None
    if args.cci_every_n_steps <= 0:
        raise ValueError("--cci_every_n_steps must be positive")
    if args.cci_step_size < 0:
        raise ValueError("--cci_step_size must be non-negative")

    target_rgb = None
    classifier = None
    label_index = None
    classifier_runtime = None
    classifier_scales: tuple[int, ...] = ()
    if args.robust_classifier_guidance and args.cci_hook != "latent_classifier":
        raise ValueError("Robust classifier guidance requires latent_classifier")
    if args.robust_classifier_guidance:
        classifier_scales = parse_classifier_scales(args.classifier_scales)
        if args.classifier_blur_sigma < 0:
            raise ValueError("--classifier_blur_sigma must be non-negative")
        if args.boundary_weight < 0 or args.tv_weight < 0:
            raise ValueError("Robust realism weights must be non-negative")
    if args.cci_hook == "latent_color":
        target_rgb = (
            parse_target_rgb(args.cci_target_rgb)
            if args.cci_target_rgb
            else infer_target_rgb(config.intervention.target_concept)
        )
    elif args.cci_hook == "latent_classifier":
        import torch

        if not args.classifier_path:
            raise ValueError("--classifier_path is required for latent_classifier")
        if args.classifier_input_size <= 0:
            raise ValueError("--classifier_input_size must be positive")
        label_index = (
            args.classifier_label_index
            if args.classifier_label_index is not None
            else resolve_celeba_attribute_index(config.intervention.target_concept)
        )
        if label_index < 0 or label_index >= 40:
            raise ValueError("--classifier_label_index must be in [0, 39]")
        classifier = load_celeba_resnet50(
            args.classifier_path,
            device=backend.device,
            dtype=torch.float32,
        )
        classifier_runtime = ClassifierRuntime(
            model=classifier,
            path=str(args.classifier_path),
            label_index=label_index,
            attribute=CELEBA_ATTRIBUTES[label_index],
            input_size=args.classifier_input_size,
            device=backend.device,
        )
    source_image_cache = None

    def decode_latents(latents):
        decoded = backend.vae.decode(latents / 0.18215).sample
        return decoded / 2 + 0.5

    def should_apply(step_index: int) -> bool:
        if step_index < args.cci_start_step:
            return False
        if args.cci_end_step is not None and step_index > args.cci_end_step:
            return False
        return (step_index - args.cci_start_step) % args.cci_every_n_steps == 0

    def hook(step):
        if not should_apply(step.step_index):
            return None
        if classifier_runtime is not None:
            classifier_runtime.applied_steps.append(step.step_index)

        import torch
        import torch.nn.functional as functional

        nonlocal source_image_cache
        target = None
        if target_rgb is not None:
            target = torch.tensor(
                target_rgb,
                device=step.latents.device,
                dtype=step.latents.dtype,
            ).view(1, 3, 1, 1)

        if args.robust_classifier_guidance:
            robust_end = args.cci_end_step if args.cci_end_step is not None else 16
            effective_step_size = robust_step_size(
                step.step_index,
                start=args.cci_start_step,
                end=robust_end,
                every=args.cci_every_n_steps,
                base=args.cci_step_size,
            )
            if effective_step_size is None:
                return None

            def robust_loss_fn(decoded):
                nonlocal source_image_cache
                generation_image_mask = functional.interpolate(
                    step.latent_mask.detach().float(),
                    size=decoded.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).to(device=decoded.device, dtype=decoded.dtype)
                boundary_image_mask = (
                    4.0 * generation_image_mask * (1.0 - generation_image_mask)
                ).clamp(0.0, 1.0)
                if (
                    source_image_cache is None
                    or source_image_cache.shape != decoded.shape
                    or source_image_cache.device != decoded.device
                ):
                    with torch.no_grad():
                        source_image_cache = decode_latents(
                            step.source_latents.detach()
                        ).detach()
                source_image = source_image_cache.to(
                    device=decoded.device,
                    dtype=decoded.dtype,
                )
                return {
                    "smile": multi_scale_classifier_loss(
                        classifier,
                        decoded,
                        label_index=label_index,
                        desired_value=config.intervention.desired_value,
                        scales=classifier_scales,
                        input_size=args.classifier_input_size,
                        blur_sigma=args.classifier_blur_sigma,
                    ),
                    "boundary": boundary_loss(
                        decoded,
                        source_image,
                        boundary_image_mask,
                    ),
                    "tv": residual_tv_loss(
                        decoded,
                        source_image,
                        generation_image_mask,
                    ),
                }

            guided, stats = apply_robust_latent_guidance(
                step.latents,
                decode_fn=decode_latents,
                loss_fn=robust_loss_fn,
                weights={
                    "smile": 1.0,
                    "boundary": args.boundary_weight,
                    "tv": args.tv_weight,
                },
                step_size=effective_step_size,
                generation_mask=step.latent_mask,
            )
            if classifier_runtime is not None:
                classifier_runtime.guidance_trace.append(
                    {
                        "step_index": step.step_index,
                        "step_size": effective_step_size,
                        **stats,
                    }
                )
            return guided

        def loss_fn(decoded):
            nonlocal source_image_cache
            image_mask = step.latent_mask.detach().float()
            if image_mask.shape[-2:] != decoded.shape[-2:]:
                image_mask = functional.interpolate(
                    image_mask,
                    size=decoded.shape[-2:],
                    mode="nearest",
                )
            image_mask = image_mask.to(device=decoded.device, dtype=decoded.dtype)
            outside_mask = 1.0 - image_mask

            if (
                source_image_cache is None
                or source_image_cache.shape != decoded.shape
                or source_image_cache.device != decoded.device
            ):
                with torch.no_grad():
                    source_image_cache = decode_latents(step.source_latents.detach())
                    source_image_cache = source_image_cache.detach()

            source_image = source_image_cache.to(
                device=decoded.device,
                dtype=decoded.dtype,
            )
            if classifier is not None and label_index is not None:
                return build_classifier_guidance_terms(
                    decoded,
                    classifier=classifier,
                    label_index=label_index,
                    desired_value=config.intervention.desired_value,
                    image_mask=image_mask,
                    source_image=source_image,
                    input_size=args.classifier_input_size,
                )

            target_loss = _masked_mean_rgb_mse(decoded, target, image_mask)
            outside_delta = _masked_mse(decoded, source_image, outside_mask)
            return GuidanceTerms(
                target=target_loss,
                preservation=outside_delta,
                leakage=outside_delta,
                classifier=target_loss,
                outside_mask=outside_delta,
            )

        return apply_cci_latent_guidance(
            step.latents,
            decode_fn=decode_latents,
            loss_fn=loss_fn,
            weights=config.weights,
            step_size=args.cci_step_size,
            latent_mask=step.latent_mask,
            normalize_gradient=args.cci_normalize_grad,
        )

    return hook, classifier_runtime


def build_classifier_guidance_terms(
    decoded,
    *,
    classifier,
    label_index: int,
    desired_value: int,
    image_mask,
    source_image,
    input_size: int,
):
    """Build independent classifier and outside-mask CCI loss terms."""

    import torch
    import torch.nn.functional as functional

    logits = classifier_logits(classifier, decoded, size=input_size)
    selected_logits = logits[:, label_index]
    target = torch.full_like(selected_logits, float(desired_value))
    classifier_loss = functional.binary_cross_entropy_with_logits(
        selected_logits,
        target,
    )
    outside_delta = _masked_mse(decoded, source_image, 1.0 - image_mask)
    zero = classifier_loss * 0.0
    return GuidanceTerms(
        target=zero,
        preservation=zero,
        leakage=zero,
        classifier=classifier_loss,
        outside_mask=outside_delta,
    )


def _masked_mse(value, target, mask):
    channels = value.shape[1]
    denominator = (mask.sum() * channels).clamp_min(1.0)
    return (((value - target) ** 2) * mask).sum() / denominator


def _masked_mean_rgb_mse(value, target_rgb, mask):
    denominator = mask.sum().clamp_min(1.0)
    mean_rgb = (value * mask).sum(dim=(0, 2, 3)) / denominator
    return ((mean_rgb - target_rgb.view(3)) ** 2).mean()


def score_classifier_image_grid(
    image_path: str | Path,
    *,
    classifier,
    label_index: int,
    input_size: int,
    device: str,
    batch_size: int,
    crop_width: int | None = None,
) -> list[float]:
    """Score one image or each horizontal image in a saved batch grid."""

    import numpy as np
    import torch
    from PIL import Image

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    image = Image.open(image_path).convert("RGB")
    if batch_size == 1:
        crops = [image]
    else:
        if crop_width is None or crop_width <= 0:
            raise ValueError("crop_width must be positive for batch grids")
        required_width = batch_size * crop_width
        if image.width < required_width:
            raise ValueError(
                f"Image grid width {image.width} is smaller than {required_width}"
            )
        crops = [
            image.crop((index * crop_width, 0, (index + 1) * crop_width, image.height))
            for index in range(batch_size)
        ]

    tensors = [
        torch.from_numpy(np.array(crop, dtype=np.float32))
        .permute(2, 0, 1)
        .div(255.0)
        for crop in crops
    ]
    images = torch.stack(tensors).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        probabilities = classifier_probabilities(
            classifier,
            images,
            size=input_size,
        )[:, label_index]
    return [float(value) for value in probabilities.detach().cpu().tolist()]


def masked_image_change_metrics(
    source_path: str | Path,
    output_path: str | Path,
    mask_path: str | Path,
) -> dict[str, float]:
    """Measure RGB MAE inside and outside a binary audit mask."""

    import numpy as np
    from PIL import Image

    output_image = Image.open(output_path).convert("RGB")
    source_image = Image.open(source_path).convert("RGB").resize(
        output_image.size,
        Image.BILINEAR,
    )
    mask_image = Image.open(mask_path).convert("L").resize(
        output_image.size,
        Image.NEAREST,
    )
    source = np.array(source_image, dtype=np.float32)
    output = np.array(output_image, dtype=np.float32)
    mask = np.array(mask_image, dtype=np.uint8) >= 128
    if not mask.any() or mask.all():
        raise ValueError("Audit mask must contain inside and outside pixels")
    delta = np.abs(output - source)
    return {
        "inside_mae": float(delta[mask].mean()),
        "outside_mae": float(delta[~mask].mean()),
        "mask_fraction": float(mask.mean()),
    }


def classifier_audit_metadata(
    args,
    config,
    runtime: ClassifierRuntime,
    *,
    source_probability: float,
    output_probabilities: list[float],
) -> dict[str, Any]:
    """Build serializable classifier provenance and before/after scores."""

    return {
        "path": str(args.classifier_path or runtime.path),
        "attribute": runtime.attribute,
        "label_index": runtime.label_index,
        "desired_value": config.intervention.desired_value,
        "input_size": runtime.input_size,
        "source_probability": source_probability,
        "output_probabilities": output_probabilities,
        "applied_steps": list(runtime.applied_steps),
        "guidance_trace": list(runtime.guidance_trace),
    }


def run(args: argparse.Namespace) -> str:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_cci_config(args.cci_config)
    mask_artifacts: MaskArtifacts | None = None
    generation_mask_path: str | None = None
    if args.robust_classifier_guidance:
        if len(args.generation_mask_component) != 3:
            raise ValueError(
                "Robust smile guidance requires exactly three mask components: "
                "mouth, upper lip, and lower lip"
            )
        mask_artifacts = prepare_semantic_masks(
            args.generation_mask_component,
            feather_radius=args.generation_mask_feather,
            hard_output=output_dir / "semantic_mask.png",
            soft_output=output_dir / "generation_mask.png",
        )
        generation_mask_path = mask_artifacts.generation_path
    prompt = (
        args.prompt
        if args.prompt
        else build_concept_prompt(config.intervention).positive
    )
    backend = BlendedLatentDiffusionSD2Backend(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        lora_path=args.lora_path,
        local_files_only=args.local_files_only,
    )
    cci_latent_guidance_hook, classifier_runtime = build_cci_latent_guidance_hook(
        backend,
        config,
        args,
    )
    result = backend.edit_image(
        init_image=args.init_image,
        mask=args.mask,
        generation_mask=generation_mask_path,
        prompt=prompt,
        output_path=output_dir / "sd2_bld_grid.png",
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        blending_percentage=args.blending_start_percentage,
        seed=args.seed,
        initial_latent_mode=args.initial_latent_mode,
        cci_latent_guidance_hook=cci_latent_guidance_hook,
    )
    audit = result.to_dict()
    mask_metrics = None
    if mask_artifacts is not None:
        mask_metrics = {
            "strict_mouth": masked_image_change_metrics(
                args.init_image,
                result.image_path,
                args.mask,
            ),
            "semantic_union": masked_image_change_metrics(
                args.init_image,
                result.image_path,
                mask_artifacts.semantic_path,
            ),
        }
    classifier_audit = None
    if classifier_runtime is not None:
        source_probability = score_classifier_image_grid(
            args.init_image,
            classifier=classifier_runtime.model,
            label_index=classifier_runtime.label_index,
            input_size=classifier_runtime.input_size,
            device=classifier_runtime.device,
            batch_size=1,
        )[0]
        output_probabilities = score_classifier_image_grid(
            result.image_path,
            classifier=classifier_runtime.model,
            label_index=classifier_runtime.label_index,
            input_size=classifier_runtime.input_size,
            device=classifier_runtime.device,
            batch_size=args.batch_size,
            crop_width=args.width,
        )
        classifier_audit = classifier_audit_metadata(
            args,
            config,
            classifier_runtime,
            source_probability=source_probability,
            output_probabilities=output_probabilities,
        )
    audit["cci"] = {
        "hook": args.cci_hook,
        "step_size": args.cci_step_size,
        "every_n_steps": args.cci_every_n_steps,
        "normalize_gradient": args.cci_normalize_grad,
        "start_step": args.cci_start_step,
        "end_step": args.cci_end_step,
        "target_rgb": (
            parse_target_rgb(args.cci_target_rgb)
            if args.cci_target_rgb
            else infer_target_rgb(config.intervention.target_concept)
            if args.cci_hook == "latent_color"
            else None
        ),
        "classifier": classifier_audit,
        "robust": (
            {
                "enabled": True,
                "semantic_mask": mask_artifacts.semantic_path,
                "generation_mask": mask_artifacts.generation_path,
                "semantic_fraction": mask_artifacts.semantic_fraction,
                "feather_radius": args.generation_mask_feather,
                "classifier_scales": list(
                    parse_classifier_scales(args.classifier_scales)
                ),
                "classifier_blur_sigma": args.classifier_blur_sigma,
                "boundary_weight": args.boundary_weight,
                "tv_weight": args.tv_weight,
                "change_metrics": mask_metrics,
            }
            if mask_artifacts is not None
            else None
        ),
        "weights": config.weights.__dict__,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    return result.image_path


def main() -> int:
    args = build_arg_parser().parse_args()
    print(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
