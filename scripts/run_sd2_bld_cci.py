#!/usr/bin/env python3
"""Run GPU Stable Diffusion 2 blended-latent editing with CCI audit state."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cci_diff.adapters.sd2_clean_cci import (
    CleanCCIGuidanceHook,
    FinalTargetLatentCorrectionHook,
)
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
from cci_diff.cci_trace import JSONLTraceWriter
from cci_diff.compilers.json_graph import JsonConceptGraphCompiler
from cci_diff.concept_graph import (
    load_concept_graph,
    load_sample_bindings,
    sha256_file,
)
from cci_diff.concept_registry import default_concept_registry
from cci_diff.config import load_cci_config
from cci_diff.constraint_controller import ConstraintFeedbackController
from cci_diff.constraints import (
    CelebAAttributeConstraint,
    CelebAAttributeTarget,
    MaskedResidualTVConstraint,
    OutsideL1Constraint,
)
from cci_diff.guidance import GuidanceTerms
from cci_diff.identity.facenet import (
    FaceNetIdentityConstraint,
    build_face_detector,
    load_facenet_identity,
    load_identity_export_manifest,
)
from cci_diff.masking import MaskArtifacts, prepare_semantic_masks
from cci_diff.post_attack import (
    gradcam_pp_saliency,
    join_horizontal_grid,
    normalize_imagenet,
    parse_epsilon_schedule,
    soft_anatomical_mask,
    split_horizontal_grid,
    targeted_adaptive_smooth_boundary_attack,
    targeted_smooth_boundary_attack,
    unnormalize_imagenet,
)
from cci_diff.prompts import build_concept_prompt
from cci_diff.sd2_bld_backend import BlendedLatentDiffusionSD2Backend
from cci_diff.spec import ConceptIntervention


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


@dataclass(frozen=True)
class ClipRuntime:
    """Loaded CLIP state for differentiable semantic guidance."""

    model: Any
    text_features: Any
    text: str
    model_name: str
    pretrained: str
    input_size: int
    device: str


@dataclass(frozen=True)
class CleanRunSetup:
    """Resolved runtime objects and provenance for one clean CCI run."""

    plan: Any
    mask_artifacts: MaskArtifacts
    guidance_hook: Any
    classifier_runtime: ClassifierRuntime
    identity_checkpoint_sha256: str
    trace_path: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cci_config", default=None)
    parser.add_argument("--init_image", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cci_graph", default=None)
    parser.add_argument("--cci_sample_bindings", default=None)
    parser.add_argument("--identity_model_path", default=None)
    parser.add_argument("--cci_trace", default=None)
    parser.add_argument(
        "--cci_frame_dir",
        default=None,
        help="Optional directory for predicted-clean before/after frame snapshots.",
    )
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
        choices=["none", "latent_color", "latent_classifier", "clean_constraint"],
        default="none",
        help="Select legacy latent guidance or predicted-clean constraint feedback.",
    )
    parser.add_argument(
        "--cci_step_size",
        type=float,
        default=None,
        help="Gradient step size for the latent CCI hook.",
    )
    parser.add_argument(
        "--cci_every_n_steps",
        type=int,
        default=None,
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
        default=None,
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
    parser.add_argument("--generation_mask_feather", type=float, default=None)
    parser.add_argument("--generation_mask_dilation", type=int, default=0)
    parser.add_argument("--generation_mask_dilation_x", type=int, default=None)
    parser.add_argument("--generation_mask_dilation_y", type=int, default=None)
    parser.add_argument("--classifier_scales", default="256,384,512")
    parser.add_argument("--classifier_blur_sigma", type=float, default=1.0)
    parser.add_argument("--boundary_weight", type=float, default=0.3)
    parser.add_argument("--tv_weight", type=float, default=0.05)
    parser.add_argument(
        "--clip_guidance_text",
        default=None,
        help="Optional CLIP text target for semantic latent guidance.",
    )
    parser.add_argument("--clip_model", default="ViT-B-32")
    parser.add_argument("--clip_pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--clip_input_size", type=int, default=224)
    parser.add_argument(
        "--cci_controller_mode",
        choices=["disabled", "fixed_equal", "feedback"],
        default="feedback",
        help="Ablation mode for predicted-clean CCI; feedback is the proposed method.",
    )
    parser.add_argument("--cci_disable_target_projection", action="store_true")
    parser.add_argument("--cci_disable_target_guidance", action="store_true")
    parser.add_argument("--cci_disable_gradient_normalization", action="store_true")
    parser.add_argument("--cci_disable_target_budget", action="store_true")
    parser.add_argument("--cci_disable_guidance_schedule", action="store_true")
    parser.add_argument("--cci_disable_final_correction", action="store_true")
    parser.add_argument(
        "--cci_final_correction_mask",
        choices=["generation", "semantic", "semantic_attribution"],
        default="semantic_attribution",
        help=(
            "Localize final target correction with the soft generation mask, "
            "hard semantic mask, or classifier attribution inside semantic support."
        ),
    )
    parser.add_argument(
        "--cci_post_attack",
        choices=["none", "smooth_boundary"],
        default="none",
    )
    parser.add_argument(
        "--cci_post_attack_epsilon",
        type=float,
        default=None,
        help="Use one fixed epsilon instead of the adaptive schedule.",
    )
    parser.add_argument(
        "--cci_post_attack_epsilon_schedule",
        default="0.05,0.08,0.10,0.30,0.50",
        help="Increasing epsilon budgets used until the saved-image margin passes.",
    )
    parser.add_argument("--cci_post_attack_step_size", type=float, default=0.005)
    parser.add_argument("--cci_post_attack_max_steps", type=int, default=500)
    parser.add_argument(
        "--cci_post_attack_boundary_margin",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--cci_post_attack_boundary_steps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--cci_post_attack_gaussian_kernel_size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--cci_post_attack_gaussian_sigma",
        type=float,
        default=1.0,
    )
    return parser


def validate_mode_args(args: argparse.Namespace) -> None:
    """Enforce clean/legacy option ownership and restore legacy defaults."""

    if args.generation_mask_dilation < 0:
        raise ValueError("generation_mask_dilation must be non-negative")
    for name in ("generation_mask_dilation_x", "generation_mask_dilation_y"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.cci_hook == "clean_constraint":
        required = {
            "--cci_graph": args.cci_graph,
            "--cci_sample_bindings": args.cci_sample_bindings,
            "--classifier_path": args.classifier_path,
            "--identity_model_path": args.identity_model_path,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"clean_constraint requires: {', '.join(missing)}")
        if args.batch_size != 1:
            raise ValueError("clean_constraint version 1 requires --batch_size 1")
        duplicates = {
            "--cci_config": args.cci_config,
            "--init_image": args.init_image,
            "--mask": args.mask,
            "--cci_step_size": args.cci_step_size,
            "--cci_every_n_steps": args.cci_every_n_steps,
            "--cci_start_step": args.cci_start_step,
            "--cci_end_step": args.cci_end_step,
            "--classifier_label_index": args.classifier_label_index,
        }
        supplied = [name for name, value in duplicates.items() if value is not None]
        if (
            args.cci_normalize_grad
            or args.robust_classifier_guidance
            or args.generation_mask_component
        ):
            supplied.append("legacy guidance/mask option")
        if supplied:
            raise ValueError(
                "Graph and sample bindings are the single source of truth; remove: "
                + ", ".join(supplied)
            )
        if args.torch_dtype != "float32" and args.device == "mps":
            raise ValueError(
                "clean_constraint on MPS requires --torch_dtype float32"
            )
        if args.cci_post_attack_epsilon is not None:
            args.cci_post_attack_epsilon_schedule = parse_epsilon_schedule(
                (args.cci_post_attack_epsilon,)
            )
        else:
            args.cci_post_attack_epsilon_schedule = parse_epsilon_schedule(
                args.cci_post_attack_epsilon_schedule
            )
        if args.cci_post_attack == "smooth_boundary":
            positive_values = {
                "cci_post_attack_step_size": args.cci_post_attack_step_size,
                "cci_post_attack_max_steps": args.cci_post_attack_max_steps,
                "cci_post_attack_boundary_steps": (
                    args.cci_post_attack_boundary_steps
                ),
                "cci_post_attack_gaussian_sigma": (
                    args.cci_post_attack_gaussian_sigma
                ),
            }
            invalid = [
                name for name, value in positive_values.items() if value <= 0
            ]
            if invalid:
                raise ValueError(
                    "Post-attack values must be positive: " + ", ".join(invalid)
                )
            if not 0 <= args.cci_post_attack_boundary_margin < 0.5:
                raise ValueError(
                    "cci_post_attack_boundary_margin must be in [0, 0.5)"
                )
            kernel_size = args.cci_post_attack_gaussian_kernel_size
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(
                    "cci_post_attack_gaussian_kernel_size must be a positive "
                    "odd integer"
                )
        return
    if not args.cci_config or not args.init_image or not args.mask:
        raise ValueError(
            "Legacy CCI modes require --cci_config, --init_image, and --mask"
        )
    if (
        args.cci_graph
        or args.cci_sample_bindings
        or args.identity_model_path
        or args.cci_trace
        or args.cci_frame_dir
        or args.cci_controller_mode != "feedback"
        or args.cci_disable_target_projection
        or args.cci_disable_target_guidance
        or args.cci_disable_gradient_normalization
        or args.cci_disable_target_budget
        or args.cci_disable_guidance_schedule
        or args.cci_disable_final_correction
        or args.cci_post_attack != "none"
    ):
        raise ValueError("Clean graph options require --cci_hook clean_constraint")
    if args.cci_step_size is None:
        args.cci_step_size = 0.03
    if args.cci_every_n_steps is None:
        args.cci_every_n_steps = 4
    if args.cci_start_step is None:
        args.cci_start_step = 0
    if args.generation_mask_feather is None:
        args.generation_mask_feather = 3.0


def resolve_generation_mask_geometry(
    args: argparse.Namespace,
    *,
    default_feather: float,
) -> tuple[int, int, float]:
    """Resolve scalar-compatible generation-mask geometry."""

    dilation_x = (
        args.generation_mask_dilation
        if args.generation_mask_dilation_x is None
        else args.generation_mask_dilation_x
    )
    dilation_y = (
        args.generation_mask_dilation
        if args.generation_mask_dilation_y is None
        else args.generation_mask_dilation_y
    )
    feather = (
        default_feather
        if args.generation_mask_feather is None
        else args.generation_mask_feather
    )
    return dilation_x, dilation_y, feather


def validate_robust_mask_components(args: argparse.Namespace) -> None:
    """Require at least one aligned semantic component for robust guidance."""

    if args.robust_classifier_guidance and not args.generation_mask_component:
        raise ValueError(
            "Robust classifier guidance requires at least one mask component"
        )


def prepare_clean_plan(args: argparse.Namespace, output_dir: Path):
    """Compile a graph and build its hard and feathered semantic masks."""

    graph = load_concept_graph(args.cci_graph)
    bindings = load_sample_bindings(args.cci_sample_bindings)
    plan = JsonConceptGraphCompiler(graph, args.cci_graph).compile(
        graph.intervention,
        bindings,
        default_concept_registry(),
    )
    dilation_x, dilation_y, feather = resolve_generation_mask_geometry(
        args,
        default_feather=plan.graph.region.feather_radius,
    )
    mask_artifacts = prepare_semantic_masks(
        [path for _, path in plan.component_paths],
        feather_radius=feather,
        dilation_radius=args.generation_mask_dilation,
        dilation_x=dilation_x,
        dilation_y=dilation_y,
        hard_output=output_dir / "semantic_mask.png",
        soft_output=output_dir / "generation_mask.png",
    )
    return plan, mask_artifacts


def build_clean_evaluators(
    plan,
    *,
    classifier,
    identity_model,
    face_detector,
    classifier_input_size: int,
):
    """Instantiate only reviewed target and constraint evaluator adapters."""

    if plan.target.attribute_index is None:
        raise ValueError("Clean CCI target requires a resolved attribute index")
    target = CelebAAttributeTarget(
        classifier,
        plan.target.attribute_index,
        classifier_input_size,
    )
    constraints = []
    for node in plan.constraints:
        if node.tolerance is None:
            raise ValueError(f"Constraint {node.id!r} requires a tolerance")
        if node.evaluator == "celeba_attribute":
            if node.attribute_index is None:
                raise ValueError(
                    f"CelebA constraint {node.id!r} requires an attribute index"
                )
            evaluator = CelebAAttributeConstraint(
                node.id,
                classifier,
                node.attribute_index,
                input_size=classifier_input_size,
                tolerance=node.tolerance,
            )
        elif node.evaluator == "facenet_identity":
            evaluator = FaceNetIdentityConstraint(
                node.id,
                identity_model,
                face_detector,
                tolerance=node.tolerance,
            )
        elif node.evaluator == "outside_l1":
            evaluator = OutsideL1Constraint(node.id, node.tolerance)
        elif node.evaluator == "masked_residual_tv":
            evaluator = MaskedResidualTVConstraint(node.id, node.tolerance)
        else:
            raise ValueError(
                f"No runtime constraint adapter for {node.evaluator!r}"
            )
        constraints.append(evaluator)
    return target, tuple(constraints)


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


def load_clip_runtime(args, device: str) -> ClipRuntime:
    """Load OpenCLIP and precompute the semantic text embedding."""

    if args.clip_input_size <= 0:
        raise ValueError("--clip_input_size must be positive")
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise ImportError(
            "CLIP guidance requires open-clip-torch and torch. Install the ML "
            "dependencies with pip install -e '.[ml]'."
        ) from exc

    model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        tokens = open_clip.tokenize([args.clip_guidance_text]).to(device)
        text_features = model.encode_text(tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return ClipRuntime(
        model=model,
        text_features=text_features,
        text=args.clip_guidance_text,
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        input_size=args.clip_input_size,
        device=device,
    )


def clip_semantic_loss(decoded, runtime: ClipRuntime):
    """Differentiable CLIP image-text loss: 1 - cosine(image, text)."""

    import torch
    import torch.nn.functional as functional

    images = functional.interpolate(
        decoded,
        size=(runtime.input_size, runtime.input_size),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    normalized = (images - mean) / std
    image_features = runtime.model.encode_image(normalized).float()
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    similarity = image_features @ runtime.text_features.to(image_features.device).T
    return (1.0 - similarity.squeeze(-1)).mean()


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
    clip_runtime = None
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
    if args.clip_guidance_text:
        clip_runtime = load_clip_runtime(args, backend.device)
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
            clip_loss = (
                clip_semantic_loss(decoded, clip_runtime)
                if clip_runtime is not None
                else None
            )
            smooth_loss = (
                residual_tv_loss(decoded, source_image, image_mask)
                if config.weights.smooth > 0
                else None
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
                    clip_loss=clip_loss,
                    smooth_loss=smooth_loss,
                )

            target_loss = _masked_mean_rgb_mse(decoded, target, image_mask)
            outside_delta = _masked_mse(decoded, source_image, outside_mask)
            zero = target_loss * 0.0
            return GuidanceTerms(
                target=target_loss,
                preservation=outside_delta,
                leakage=outside_delta,
                classifier=target_loss,
                outside_mask=outside_delta,
                clip=clip_loss if clip_loss is not None else zero,
                smooth=smooth_loss if smooth_loss is not None else zero,
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
    clip_loss=None,
    smooth_loss=None,
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
        clip=clip_loss if clip_loss is not None else zero,
        smooth=smooth_loss if smooth_loss is not None else zero,
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


def load_rgb_image_tensor(path: str | Path, *, device: str):
    """Load one RGB image as a float32 NCHW tensor in [0, 1]."""

    import numpy as np
    import torch
    from PIL import Image

    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )


def _post_attack_target_pass(
    probability: float,
    *,
    desired_value: int,
    margin: float = 0.0,
) -> bool:
    if desired_value == 1:
        return probability >= 0.5 + margin
    return probability <= 0.5 - margin


def _save_rgb_grid(path: str | Path, grid) -> None:
    import numpy as np
    from PIL import Image

    array = (
        grid.detach()[0]
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    Image.fromarray(np.asarray(array), mode="RGB").save(path)


class PredictedCleanFrameWriter:
    """Persist detached predicted-clean snapshots and scalar metadata."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        if self.output_dir.is_dir() and any(self.output_dir.iterdir()):
            raise FileExistsError(
                f"Predicted-clean frame directory is not empty: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, float | int | str]] = []

    def __call__(self, frame: dict[str, Any]) -> None:
        step = int(frame["step"])
        before_name = f"step_{step:02d}_before.png"
        after_name = f"step_{step:02d}_after.png"
        _save_rgb_grid(self.output_dir / before_name, frame["before_image"])
        _save_rgb_grid(self.output_dir / after_name, frame["after_image"])
        self.records.append(
            {
                "step": step,
                "timestep": int(frame["timestep"]),
                "progress": float(frame["progress"]),
                "before": before_name,
                "after": after_name,
            }
        )
        manifest_path = self.output_dir / "manifest.json"
        temporary_path = manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(self.records, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(manifest_path)


def _load_post_attack_mask(
    path: str | Path,
    *,
    height: int,
    width: int,
    device: str,
):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(path).convert("L").resize(
        (width, height),
        Image.NEAREST,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).view(1, 1, height, width).to(
        device=device,
        dtype=torch.float32,
    )


def _post_attack_identity_score(
    source,
    candidate,
    *,
    model,
    detector,
) -> float:
    import torch

    from cci_diff.identity.facenet import (
        detect_largest_face_box,
        fixed_face_crop,
        standardize_face,
    )

    box = detect_largest_face_box(detector, source)
    with torch.no_grad():
        source_embedding = torch.nn.functional.normalize(
            model(standardize_face(fixed_face_crop(source, box))),
            dim=1,
        )
        candidate_embedding = torch.nn.functional.normalize(
            model(standardize_face(fixed_face_crop(candidate, box))),
            dim=1,
        )
    return float(
        torch.nn.functional.cosine_similarity(
            source_embedding,
            candidate_embedding,
            dim=1,
        ).mean().item()
    )


def _post_attack_change_metrics(before, after, semantic_mask) -> dict[str, float]:
    import torch

    delta = torch.abs(after - before)
    pixel_delta = delta.max(dim=1, keepdim=True).values
    outside = 1.0 - semantic_mask
    denominator = (outside.sum() * delta.shape[1]).clamp_min(1.0)
    return {
        "mean_abs_change": float(delta.mean().item()),
        "l2": float(torch.linalg.vector_norm(delta.float()).item()),
        "linf": float(delta.max().item()),
        "changed_fraction": float(
            (pixel_delta > (1.0 / 255.0)).float().mean().item()
        ),
        "outside_semantic_mae": float(
            (delta * outside).sum().item() / denominator.item()
        ),
    }


def run_clean_post_attack(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    plan,
    mask_artifacts: MaskArtifacts,
    classifier_runtime: ClassifierRuntime,
    identity_model,
    face_detector,
    raw_output_path: str | Path,
    saliency_fn=gradcam_pp_saliency,
    attack_fn=targeted_smooth_boundary_attack,
    identity_score_fn=_post_attack_identity_score,
) -> dict[str, Any] | None:
    """Optionally correct each saved clean-CCI candidate near its target boundary."""

    if args.cci_post_attack == "none":
        return None
    if args.cci_post_attack != "smooth_boundary":
        raise ValueError(f"Unsupported post-attack mode: {args.cci_post_attack}")
    if (
        args.height != classifier_runtime.input_size
        or args.width != classifier_runtime.input_size
    ):
        raise ValueError(
            "smooth post-attack requires height and width to match "
            "classifier_input_size"
        )

    import numpy as np
    import torch
    import torch.nn.functional as functional

    raw_output_path = Path(raw_output_path)
    corrected_output_path = output_dir / "sd2_bld_grid_corrected.png"
    soft_mask_path = output_dir / "post_attack_soft_mask.png"
    raw_grid = load_rgb_image_tensor(
        raw_output_path,
        device=classifier_runtime.device,
    )
    if raw_grid.shape[-2] != args.height:
        raise ValueError(
            f"grid height {raw_grid.shape[-2]} does not equal {args.height}"
        )
    candidates = split_horizontal_grid(
        raw_grid,
        count=args.batch_size,
        crop_width=args.width,
    )
    source = load_rgb_image_tensor(
        plan.source_image,
        device=classifier_runtime.device,
    )
    source = functional.interpolate(
        source,
        size=(args.height, args.width),
        mode="bilinear",
        align_corners=False,
    )
    semantic_mask = _load_post_attack_mask(
        mask_artifacts.semantic_path,
        height=args.height,
        width=args.width,
        device=classifier_runtime.device,
    )
    with torch.no_grad():
        source_probability = float(
            classifier_probabilities(
                classifier_runtime.model,
                source,
                size=classifier_runtime.input_size,
            )[:, classifier_runtime.label_index].item()
        )
        before_probabilities = classifier_probabilities(
            classifier_runtime.model,
            candidates,
            size=classifier_runtime.input_size,
        )[:, classifier_runtime.label_index]
    saliency = saliency_fn(
        classifier_runtime.model,
        normalize_imagenet(source),
        label_index=classifier_runtime.label_index,
        original_present=source_probability >= 0.5,
    )
    saliency_tensor = torch.as_tensor(
        np.asarray(saliency),
        device=classifier_runtime.device,
        dtype=torch.float32,
    ).view(1, 1, *np.asarray(saliency).shape[-2:])
    if saliency_tensor.shape[-2:] != semantic_mask.shape[-2:]:
        saliency_tensor = functional.interpolate(
            saliency_tensor,
            size=semantic_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    soft_mask = soft_anatomical_mask(semantic_mask, saliency_tensor)
    _save_rgb_grid(
        soft_mask_path,
        soft_mask.expand(-1, 3, -1, -1),
    )

    desired_value = int(plan.graph.intervention.desired_value)

    def score_identity(candidate):
        try:
            return identity_score_fn(
                source,
                candidate,
                model=identity_model,
                detector=face_detector,
            )
        except ValueError:
            return None

    corrected_candidates = []
    internal_records = []
    for index, candidate in enumerate(candidates):
        candidate = candidate.unsqueeze(0)
        before_probability = float(before_probabilities[index].item())
        already_successful = _post_attack_target_pass(
            before_probability,
            desired_value=desired_value,
            margin=args.cci_post_attack_boundary_margin,
        )
        if already_successful:
            corrected = candidate
            internal_record = {
                "after_probability": before_probability,
                "quantized_after_probability": before_probability,
                "iterations": 0,
                "boundary_iterations": 0,
                "total_iterations": 0,
                "margin_pass": True,
                "selected_epsilon": None,
                "escalated": False,
                "schedule_exhausted": False,
                "attempts": [],
            }
        else:
            corrected_normalized, internal_record = (
                targeted_adaptive_smooth_boundary_attack(
                    classifier_runtime.model,
                    normalize_imagenet(candidate),
                    soft_mask,
                    epsilon_schedule=args.cci_post_attack_epsilon_schedule,
                    label_index=classifier_runtime.label_index,
                    desired_value=desired_value,
                    step_size=args.cci_post_attack_step_size,
                    max_steps=args.cci_post_attack_max_steps,
                    decision_threshold=0.5,
                    boundary_margin=args.cci_post_attack_boundary_margin,
                    boundary_steps=args.cci_post_attack_boundary_steps,
                    kernel_size=args.cci_post_attack_gaussian_kernel_size,
                    sigma=args.cci_post_attack_gaussian_sigma,
                    attack_fn=attack_fn,
                )
            )
            corrected = unnormalize_imagenet(corrected_normalized).clamp(0, 1)
        corrected_candidates.append(corrected)
        internal_records.append(
            {
                **internal_record,
                "already_successful": already_successful,
            }
        )

    corrected_batch = torch.cat(corrected_candidates, dim=0)
    _save_rgb_grid(
        corrected_output_path,
        join_horizontal_grid(corrected_batch),
    )
    saved_grid = load_rgb_image_tensor(
        corrected_output_path,
        device=classifier_runtime.device,
    )
    saved_candidates = split_horizontal_grid(
        saved_grid,
        count=args.batch_size,
        crop_width=args.width,
    )
    with torch.no_grad():
        after_probabilities = classifier_probabilities(
            classifier_runtime.model,
            saved_candidates,
            size=classifier_runtime.input_size,
        )[:, classifier_runtime.label_index]

    candidate_records = []
    for index, (before, after) in enumerate(zip(candidates, saved_candidates)):
        before = before.unsqueeze(0)
        after = after.unsqueeze(0)
        before_probability = float(before_probabilities[index].item())
        after_probability = float(after_probabilities[index].item())
        internal_record = internal_records[index]
        candidate_records.append(
            {
                "index": index,
                "before_probability": before_probability,
                "after_probability": after_probability,
                "desired_probability": (
                    after_probability
                    if desired_value == 1
                    else 1.0 - after_probability
                ),
                "already_successful": internal_record["already_successful"],
                "target_pass": _post_attack_target_pass(
                    after_probability,
                    desired_value=desired_value,
                ),
                "margin_pass": _post_attack_target_pass(
                    after_probability,
                    desired_value=desired_value,
                    margin=args.cci_post_attack_boundary_margin,
                ),
                "iterations": int(internal_record["iterations"]),
                "boundary_iterations": int(
                    internal_record["boundary_iterations"]
                ),
                "internal_after_probability": float(
                    internal_record["after_probability"]
                ),
                "selection_quantized_probability": float(
                    internal_record["quantized_after_probability"]
                ),
                "internal_margin_pass": bool(internal_record["margin_pass"]),
                "selected_epsilon": internal_record["selected_epsilon"],
                "escalated": bool(internal_record["escalated"]),
                "schedule_exhausted": bool(
                    internal_record["schedule_exhausted"]
                ),
                "total_iterations": int(internal_record["total_iterations"]),
                "attempts": list(internal_record["attempts"]),
                **_post_attack_change_metrics(before, after, semantic_mask),
                "identity_before": score_identity(before),
                "identity_after": score_identity(after),
            }
        )

    return {
        "mode": "smooth_boundary",
        "raw_output_path": str(raw_output_path),
        "corrected_output_path": str(corrected_output_path),
        "soft_mask_path": str(soft_mask_path),
        "configuration": {
            "decision_threshold": 0.5,
            "epsilon_schedule": list(args.cci_post_attack_epsilon_schedule),
            "fixed_epsilon_override": args.cci_post_attack_epsilon,
            "step_size": args.cci_post_attack_step_size,
            "max_steps": args.cci_post_attack_max_steps,
            "boundary_margin": args.cci_post_attack_boundary_margin,
            "boundary_steps": args.cci_post_attack_boundary_steps,
            "gaussian_kernel_size": args.cci_post_attack_gaussian_kernel_size,
            "gaussian_sigma": args.cci_post_attack_gaussian_sigma,
        },
        "candidates": candidate_records,
    }


def _runtime_package_versions() -> dict[str, str | None]:
    versions = {}
    for distribution in (
        "torch",
        "torchvision",
        "diffusers",
        "numpy",
        "Pillow",
        "opencv-python",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _residual_quality_metrics(
    source_path: str | Path,
    output_path: str | Path,
    semantic_mask_path: str | Path,
    generation_mask_path: str | Path,
) -> dict[str, float]:
    import numpy as np
    from PIL import Image

    output_image = Image.open(output_path).convert("RGB")
    source = np.asarray(
        Image.open(source_path).convert("RGB").resize(
            output_image.size,
            Image.BILINEAR,
        ),
        dtype=np.float32,
    ) / 255.0
    output = np.asarray(output_image, dtype=np.float32) / 255.0
    semantic = np.asarray(
        Image.open(semantic_mask_path).convert("L").resize(
            output_image.size,
            Image.NEAREST,
        ),
        dtype=np.float32,
    ) / 255.0
    generation = np.asarray(
        Image.open(generation_mask_path).convert("L").resize(
            output_image.size,
            Image.BILINEAR,
        ),
        dtype=np.float32,
    ) / 255.0
    residual = output - source
    dx_mask = semantic[:, 1:] * semantic[:, :-1]
    dy_mask = semantic[1:, :] * semantic[:-1, :]
    dx = np.abs(residual[:, 1:] - residual[:, :-1]) * dx_mask[..., None]
    dy = np.abs(residual[1:, :] - residual[:-1, :]) * dy_mask[..., None]
    denominator = max(float((dx_mask.sum() + dy_mask.sum()) * 3), 1.0)
    ring = 4.0 * generation * (1.0 - generation)
    ring_denominator = max(float(ring.sum() * 3), 1.0)
    return {
        "residual_tv": float((dx.sum() + dy.sum()) / denominator),
        "boundary_discontinuity": float(
            (np.abs(residual) * ring[..., None]).sum() / ring_denominator
        ),
    }


def build_clean_postrun_metrics(
    *,
    plan,
    mask_artifacts: MaskArtifacts,
    guidance_hook: CleanCCIGuidanceHook,
    classifier_runtime: ClassifierRuntime,
    result,
) -> dict[str, Any]:
    """Measure the complete paired clean-CCI metric surface."""

    import torch

    source_tensor = load_rgb_image_tensor(
        plan.source_image,
        device=classifier_runtime.device,
    )
    output_tensor = load_rgb_image_tensor(
        result.image_path,
        device=classifier_runtime.device,
    )
    with torch.no_grad():
        source_probabilities = classifier_probabilities(
            classifier_runtime.model,
            source_tensor,
            size=classifier_runtime.input_size,
        )[0]
        output_probabilities = classifier_probabilities(
            classifier_runtime.model,
            output_tensor,
            size=classifier_runtime.input_size,
        )[0]
    source_values = [float(value) for value in source_probabilities.cpu().tolist()]
    output_values = [float(value) for value in output_probabilities.cpu().tolist()]
    excluded = {plan.target.attribute_index}
    drift_indices = [index for index in range(len(CELEBA_ATTRIBUTES)) if index not in excluded]
    mean_non_target_drift = sum(
        abs(output_values[index] - source_values[index])
        for index in drift_indices
    ) / len(drift_indices)
    feasibility = guidance_hook.evaluate_image(output_tensor)
    identity_payload = feasibility["constraints"].get("identity")
    identity_cosine = (
        1.0 - identity_payload["value"]
        if identity_payload and identity_payload["value"] is not None
        else None
    )
    locality = {
        "strict_audit_mask": masked_image_change_metrics(
            plan.source_image,
            result.image_path,
            plan.audit_mask_path,
        ),
        "semantic_union": masked_image_change_metrics(
            plan.source_image,
            result.image_path,
            mask_artifacts.semantic_path,
        ),
    }
    return {
        "final_feasibility": feasibility,
        "locality": locality,
        "attributes": {
            "names": list(CELEBA_ATTRIBUTES),
            "source_probabilities": source_values,
            "output_probabilities": output_values,
            "excluded_indices": sorted(excluded),
            "mean_non_target_drift": mean_non_target_drift,
        },
        "identity_cosine": identity_cosine,
        "quality": _residual_quality_metrics(
            plan.source_image,
            result.image_path,
            mask_artifacts.semantic_path,
            mask_artifacts.generation_path,
        ),
        "target_signed_margin": feasibility["target"]["signed_margin"],
        "independent_semantic_agreement": {
            "value": None,
            "status": "not_configured",
        },
        "outside_perceptual_distance": {
            "value": None,
            "status": "not_configured",
        },
    }


def run_clean(args: argparse.Namespace, output_dir: Path) -> str:
    """Run graph-compiled predicted-clean constraint feedback."""

    import torch

    plan, mask_artifacts = prepare_clean_plan(args, output_dir)
    preserved = tuple(
        edge.target
        for edge in plan.graph.edges
        if edge.relation == "must_preserve"
    )
    prompt = args.prompt or build_concept_prompt(
        ConceptIntervention(
            target_concept=plan.graph.intervention.concept,
            desired_value=plan.graph.intervention.desired_value,
            preserved_concepts=preserved,
            candidate_concepts=tuple(node.id for node in plan.graph.nodes),
        )
    ).positive
    backend = BlendedLatentDiffusionSD2Backend(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        lora_path=args.lora_path,
        local_files_only=args.local_files_only,
    )
    classifier = load_celeba_resnet50(
        args.classifier_path,
        device=backend.device,
        dtype=torch.float32,
    )
    identity_manifest = load_identity_export_manifest(args.identity_model_path)
    identity_model = load_facenet_identity(
        args.identity_model_path,
        device=backend.device,
    )
    face_detector = build_face_detector()
    target, constraints = build_clean_evaluators(
        plan,
        classifier=classifier,
        identity_model=identity_model,
        face_detector=face_detector,
        classifier_input_size=args.classifier_input_size,
    )
    if plan.target.attribute_index is None or plan.target.attribute is None:
        raise ValueError("Clean CCI target classifier resolution is incomplete")
    classifier_runtime = ClassifierRuntime(
        model=classifier,
        path=str(args.classifier_path),
        label_index=plan.target.attribute_index,
        attribute=plan.target.attribute,
        input_size=args.classifier_input_size,
        device=backend.device,
    )
    trace_path = Path(args.cci_trace) if args.cci_trace else output_dir / "cci_trace.jsonl"
    frame_observer = (
        PredictedCleanFrameWriter(args.cci_frame_dir)
        if args.cci_frame_dir
        else None
    )
    clean_hook = CleanCCIGuidanceHook(
        scheduler=backend.scheduler,
        vae=backend.vae,
        target_evaluator=target,
        constraint_evaluators=constraints,
        controller=ConstraintFeedbackController(
            plan.controller,
            use_target_guidance=not args.cci_disable_target_guidance,
            normalize_gradients=not args.cci_disable_gradient_normalization,
            budget_constraints=not args.cci_disable_target_budget,
        ),
        desired_value=plan.graph.intervention.desired_value,
        target_probability=plan.graph.intervention.target_probability,
        trace_writer=JSONLTraceWriter(trace_path),
        frame_observer=frame_observer,
        controller_mode=args.cci_controller_mode,
        project_conflicts=not args.cci_disable_target_projection,
        scheduled_guidance=not args.cci_disable_guidance_schedule,
    )
    final_hook = (
        FinalTargetLatentCorrectionHook(
            vae=backend.vae,
            target_evaluator=target,
            desired_value=plan.graph.intervention.desired_value,
            target_probability=plan.graph.intervention.target_probability,
            max_steps=plan.controller.final_corrections,
            step_radius=plan.controller.trust_radius,
            mask_mode=args.cci_final_correction_mask,
        )
        if args.cci_controller_mode != "disabled"
        and plan.controller.final_corrections
        and not args.cci_disable_target_guidance
        and not args.cci_disable_final_correction
        else None
    )
    setup = CleanRunSetup(
        plan=plan,
        mask_artifacts=mask_artifacts,
        guidance_hook=clean_hook,
        classifier_runtime=classifier_runtime,
        identity_checkpoint_sha256=sha256_file(args.identity_model_path),
        trace_path=str(trace_path),
    )
    started = time.perf_counter()
    result = backend.edit_image(
        init_image=plan.source_image,
        mask=plan.audit_mask_path,
        semantic_mask=mask_artifacts.semantic_path,
        generation_mask=mask_artifacts.generation_path,
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
        cci_guidance_hook=clean_hook,
        cci_latent_guidance_hook=final_hook,
    )
    wall_seconds = time.perf_counter() - started
    metrics = build_clean_postrun_metrics(
        plan=plan,
        mask_artifacts=mask_artifacts,
        guidance_hook=clean_hook,
        classifier_runtime=classifier_runtime,
        result=result,
    )
    post_attack = run_clean_post_attack(
        args=args,
        output_dir=output_dir,
        plan=plan,
        mask_artifacts=mask_artifacts,
        classifier_runtime=classifier_runtime,
        identity_model=identity_model,
        face_detector=face_detector,
        raw_output_path=result.image_path,
    )
    resolved_nodes = [
        asdict(node)
        for group in (
            (plan.target,),
            plan.constraints,
            plan.audit_only,
        )
        for node in group
    ]
    audit = result.to_dict()
    dilation_x, dilation_y, feather = resolve_generation_mask_geometry(
        args,
        default_feather=plan.graph.region.feather_radius,
    )
    audit["cci"] = {
        "hook": "clean_constraint",
        "graph_path": plan.graph_path,
        "graph_sha256": plan.graph_sha256,
        "graph": plan.graph.to_dict(),
        "resolved_nodes": resolved_nodes,
        "sample_bindings_path": str(args.cci_sample_bindings),
        "source_image": plan.source_image,
        "audit_mask": plan.audit_mask_path,
        "component_paths": [list(value) for value in plan.component_paths],
        "mask_artifacts": asdict(mask_artifacts),
        "generation_mask_dilation": args.generation_mask_dilation,
        "generation_mask_dilation_x": dilation_x,
        "generation_mask_dilation_y": dilation_y,
        "generation_mask_feather": feather,
        "classifier_checkpoint_sha256": sha256_file(args.classifier_path),
        "identity_checkpoint_sha256": setup.identity_checkpoint_sha256,
        "identity_export_manifest": identity_manifest,
        "runtime_packages": _runtime_package_versions(),
        "trace_path": setup.trace_path,
        "controller_mode": args.cci_controller_mode,
        "target_projection": not args.cci_disable_target_projection,
        "target_guidance": not args.cci_disable_target_guidance,
        "gradient_normalization": not args.cci_disable_gradient_normalization,
        "target_budget": not args.cci_disable_target_budget,
        "guidance_schedule": not args.cci_disable_guidance_schedule,
        "final_correction_enabled": not args.cci_disable_final_correction,
        "final_correction": final_hook.record if final_hook is not None else None,
        "post_attack": post_attack,
        "wall_seconds": wall_seconds,
        "peak_mps_bytes": clean_hook.peak_mps_bytes,
        "final_feasibility": metrics.get("final_feasibility"),
        "metrics": metrics,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result.image_path


def run(args: argparse.Namespace) -> str:
    validate_mode_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.cci_hook == "clean_constraint":
        return run_clean(args, output_dir)
    config = load_cci_config(args.cci_config)
    mask_artifacts: MaskArtifacts | None = None
    generation_mask_path: str | None = None
    validate_robust_mask_components(args)
    if args.robust_classifier_guidance:
        dilation_x, dilation_y, feather = resolve_generation_mask_geometry(
            args,
            default_feather=3.0,
        )
        mask_artifacts = prepare_semantic_masks(
            args.generation_mask_component,
            feather_radius=feather,
            dilation_radius=args.generation_mask_dilation,
            dilation_x=dilation_x,
            dilation_y=dilation_y,
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
        "clip": (
            {
                "text": args.clip_guidance_text,
                "model": args.clip_model,
                "pretrained": args.clip_pretrained,
                "input_size": args.clip_input_size,
            }
            if args.clip_guidance_text
            else None
        ),
        "robust": (
            {
                "enabled": True,
                "semantic_mask": mask_artifacts.semantic_path,
                "generation_mask": mask_artifacts.generation_path,
                "semantic_fraction": mask_artifacts.semantic_fraction,
                "feather_radius": args.generation_mask_feather,
                "dilation_radius": args.generation_mask_dilation,
                "dilation_x": (
                    args.generation_mask_dilation
                    if args.generation_mask_dilation_x is None
                    else args.generation_mask_dilation_x
                ),
                "dilation_y": (
                    args.generation_mask_dilation
                    if args.generation_mask_dilation_y is None
                    else args.generation_mask_dilation_y
                ),
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
