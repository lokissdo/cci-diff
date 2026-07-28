from __future__ import annotations

import math
from typing import Any


def parse_epsilon_schedule(value: str | Any) -> tuple[float, ...]:
    """Parse a finite, positive, strictly increasing epsilon schedule."""

    try:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError
            schedule = tuple(float(item.strip()) for item in value.split(","))
        else:
            schedule = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "epsilon schedule must contain increasing positive numbers"
        ) from error
    if (
        not schedule
        or any(not math.isfinite(item) or item <= 0 for item in schedule)
        or any(right <= left for left, right in zip(schedule, schedule[1:]))
    ):
        raise ValueError(
            "epsilon schedule must contain increasing positive numbers"
        )
    return schedule


def normalize_imagenet(images: Any):
    """ImageNet-normalize NCHW RGB images without resizing."""

    import torch

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    return (images - mean) / std


def unnormalize_imagenet(images: Any):
    """Undo ImageNet normalization for NCHW RGB images."""

    import torch

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    return images * std + mean


def soft_anatomical_mask(facepart_mask: Any, saliency: Any):
    """Weight classifier saliency while preserving anatomical support."""

    import torch

    facepart = torch.as_tensor(facepart_mask)
    attention = torch.as_tensor(
        saliency,
        device=facepart.device,
        dtype=facepart.dtype,
    )
    while attention.ndim < facepart.ndim:
        attention = attention.unsqueeze(0)
    try:
        attention = torch.broadcast_to(attention, facepart.shape)
    except RuntimeError as error:
        raise ValueError("saliency must be broadcastable to facepart_mask") from error
    return facepart.clamp(0, 1) * attention.clamp(0, 1)


def _expand_mask(mask: Any, image: Any):
    attack_mask = mask.to(device=image.device, dtype=image.dtype)
    if attack_mask.ndim == image.ndim - 1:
        attack_mask = attack_mask.unsqueeze(1)
    if attack_mask.shape[0] == 1 and image.shape[0] != 1:
        attack_mask = attack_mask.expand(image.shape[0], -1, -1, -1)
    if attack_mask.shape[1] == 1 and image.shape[1] != 1:
        attack_mask = attack_mask.expand(-1, image.shape[1], -1, -1)
    if attack_mask.shape != image.shape:
        raise ValueError("mask must be broadcastable across image channels")
    return attack_mask


def _gaussian_kernel(
    kernel_size: int,
    sigma: float,
    *,
    device: Any,
    dtype: Any,
):
    import torch

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coordinates = torch.arange(kernel_size, device=device, dtype=dtype)
    coordinates = coordinates - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d


def smooth_masked_gradient(
    gradient: Any,
    mask: Any,
    *,
    kernel_size: int = 5,
    sigma: float = 1.0,
    eps: float = 1e-8,
):
    """Return a Gaussian-smoothed, soft-masked, per-image RMS direction."""

    import torch
    import torch.nn.functional as functional

    if eps <= 0:
        raise ValueError("eps must be positive")
    attack_mask = _expand_mask(mask, gradient)
    kernel = _gaussian_kernel(
        kernel_size,
        sigma,
        device=gradient.device,
        dtype=gradient.dtype,
    )
    kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(
        gradient.shape[1],
        1,
        -1,
        -1,
    )
    padding = kernel_size // 2
    padded = functional.pad(
        gradient,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    smoothed = functional.conv2d(
        padded,
        kernel,
        groups=gradient.shape[1],
    )
    weighted = smoothed * attack_mask
    active = (attack_mask > 0).to(dtype=gradient.dtype)
    active_count = active.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1)
    rms = torch.sqrt(
        weighted.square().sum(dim=(1, 2, 3), keepdim=True) / active_count
    )
    normalized = weighted / (rms + eps)
    has_direction = rms > eps
    return torch.where(has_direction, normalized, torch.zeros_like(normalized))


def _probability(model: Any, image: Any, label_index: int) -> float:
    import torch

    with torch.no_grad():
        return float(model(image)[:, label_index].item())


def _target_pass(
    probability: float,
    *,
    desired_value: int,
    threshold: float,
    margin: float = 0.0,
) -> bool:
    if desired_value == 1:
        return probability >= threshold + margin
    return probability <= threshold - margin


def refine_boundary(
    model: Any,
    failed: Any,
    passed: Any,
    *,
    label_index: int,
    desired_value: int,
    threshold: float = 0.5,
    margin: float = 0.01,
    max_steps: int = 16,
):
    """Bisect a failed/passed segment and retain its closest passing point."""

    if desired_value not in (0, 1):
        raise ValueError("desired_value must be 0 or 1")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if margin < 0 or margin >= 0.5:
        raise ValueError("margin must be in [0, 0.5)")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    failed_probability = _probability(model, failed, label_index)
    passed_probability = _probability(model, passed, label_index)
    if not _target_pass(
        passed_probability,
        desired_value=desired_value,
        threshold=threshold,
        margin=margin,
    ):
        raise ValueError("passed endpoint does not satisfy the target margin")
    if _target_pass(
        failed_probability,
        desired_value=desired_value,
        threshold=threshold,
        margin=margin,
    ):
        return failed.detach(), {
            "before_probability": failed_probability,
            "after_probability": failed_probability,
            "desired_probability": (
                failed_probability
                if desired_value == 1
                else 1.0 - failed_probability
            ),
            "target_pass": _target_pass(
                failed_probability,
                desired_value=desired_value,
                threshold=threshold,
            ),
            "margin_pass": True,
            "boundary_iterations": 0,
            "interpolation_fraction": 0.0,
        }

    low = 0.0
    high = 1.0
    best = passed.detach()
    best_probability = passed_probability
    for _ in range(max_steps):
        fraction = (low + high) / 2
        candidate = failed + fraction * (passed - failed)
        candidate_probability = _probability(model, candidate, label_index)
        if _target_pass(
            candidate_probability,
            desired_value=desired_value,
            threshold=threshold,
            margin=margin,
        ):
            high = fraction
            best = candidate.detach()
            best_probability = candidate_probability
        else:
            low = fraction

    return best, {
        "before_probability": failed_probability,
        "after_probability": best_probability,
        "desired_probability": (
            best_probability if desired_value == 1 else 1.0 - best_probability
        ),
        "target_pass": _target_pass(
            best_probability,
            desired_value=desired_value,
            threshold=threshold,
        ),
        "margin_pass": _target_pass(
            best_probability,
            desired_value=desired_value,
            threshold=threshold,
            margin=margin,
        ),
        "boundary_iterations": max_steps,
        "interpolation_fraction": high,
    }


def targeted_smooth_boundary_attack(
    model: Any,
    image: Any,
    mask: Any,
    *,
    label_index: int,
    desired_value: int,
    epsilon: float = 0.05,
    step_size: float = 0.005,
    max_steps: int = 500,
    decision_threshold: float = 0.5,
    boundary_margin: float = 0.01,
    boundary_steps: int = 16,
    kernel_size: int = 5,
    sigma: float = 1.0,
):
    """Run a low-budget smooth attack and refine its first boundary crossing."""

    import torch
    import torch.nn.functional as functional

    if desired_value not in (0, 1):
        raise ValueError("desired_value must be 0 or 1")
    if epsilon <= 0 or step_size <= 0 or max_steps <= 0:
        raise ValueError("epsilon, step_size, and max_steps must be positive")
    reference = image.detach()
    attack_mask = _expand_mask(mask, reference)
    before_probability = _probability(model, reference, label_index)
    if _target_pass(
        before_probability,
        desired_value=desired_value,
        threshold=decision_threshold,
        margin=boundary_margin,
    ):
        return reference.clone(), {
            "before_probability": before_probability,
            "after_probability": before_probability,
            "desired_probability": (
                before_probability
                if desired_value == 1
                else 1.0 - before_probability
            ),
            "iterations": 0,
            "boundary_iterations": 0,
            "interpolation_fraction": 0.0,
            "target_pass": True,
            "margin_pass": True,
        }

    target = torch.full(
        (reference.shape[0],),
        float(desired_value),
        device=reference.device,
        dtype=reference.dtype,
    )
    attacked = reference.clone()
    iterations = 0
    boundary_record = {
        "boundary_iterations": 0,
        "interpolation_fraction": 1.0,
    }
    for iteration in range(max_steps):
        failed = attacked.detach()
        attacked = failed.requires_grad_(True)
        probability = model(attacked)[:, label_index]
        loss = functional.binary_cross_entropy(probability, target)
        gradient = torch.autograd.grad(loss, attacked)[0]
        direction = smooth_masked_gradient(
            gradient,
            attack_mask,
            kernel_size=kernel_size,
            sigma=sigma,
        )
        if not bool(torch.any(direction != 0).item()):
            attacked = failed
            break
        candidate = attacked - step_size * direction
        candidate = torch.maximum(
            torch.minimum(candidate, reference + epsilon),
            reference - epsilon,
        ).detach()
        candidate_probability = _probability(model, candidate, label_index)
        attacked = candidate
        iterations = iteration + 1
        if _target_pass(
            candidate_probability,
            desired_value=desired_value,
            threshold=decision_threshold,
            margin=boundary_margin,
        ):
            attacked, boundary_record = refine_boundary(
                model,
                failed,
                candidate,
                label_index=label_index,
                desired_value=desired_value,
                threshold=decision_threshold,
                margin=boundary_margin,
                max_steps=boundary_steps,
            )
            break

    after_probability = _probability(model, attacked, label_index)
    return attacked.detach(), {
        "before_probability": before_probability,
        "after_probability": after_probability,
        "desired_probability": (
            after_probability if desired_value == 1 else 1.0 - after_probability
        ),
        "iterations": iterations,
        "boundary_iterations": boundary_record["boundary_iterations"],
        "interpolation_fraction": boundary_record["interpolation_fraction"],
        "target_pass": _target_pass(
            after_probability,
            desired_value=desired_value,
            threshold=decision_threshold,
        ),
        "margin_pass": _target_pass(
            after_probability,
            desired_value=desired_value,
            threshold=decision_threshold,
            margin=boundary_margin,
        ),
    }


def targeted_adaptive_smooth_boundary_attack(
    model: Any,
    image: Any,
    mask: Any,
    *,
    epsilon_schedule: str | Any,
    label_index: int,
    desired_value: int,
    step_size: float = 0.005,
    max_steps: int = 500,
    decision_threshold: float = 0.5,
    boundary_margin: float = 0.01,
    boundary_steps: int = 16,
    kernel_size: int = 5,
    sigma: float = 1.0,
    attack_fn=None,
):
    """Escalate epsilon from the same source until the quantized image passes."""

    schedule = parse_epsilon_schedule(epsilon_schedule)
    single_attack = attack_fn or targeted_smooth_boundary_attack
    reference = image.detach()
    attempts = []
    selected = reference.clone()
    selected_record: dict[str, Any] = {
        "after_probability": _probability(model, reference, label_index),
        "iterations": 0,
        "boundary_iterations": 0,
        "margin_pass": False,
    }

    for epsilon in schedule:
        candidate, attack_record = single_attack(
            model,
            reference,
            mask,
            label_index=label_index,
            desired_value=desired_value,
            epsilon=epsilon,
            step_size=step_size,
            max_steps=max_steps,
            decision_threshold=decision_threshold,
            boundary_margin=boundary_margin,
            boundary_steps=boundary_steps,
            kernel_size=kernel_size,
            sigma=sigma,
        )
        quantized_rgb = (
            unnormalize_imagenet(candidate)
            .clamp(0, 1)
            .mul(255)
            .round()
            .div(255)
        )
        quantized_probability = _probability(
            model,
            normalize_imagenet(quantized_rgb),
            label_index,
        )
        quantized_target_pass = _target_pass(
            quantized_probability,
            desired_value=desired_value,
            threshold=decision_threshold,
        )
        quantized_margin_pass = _target_pass(
            quantized_probability,
            desired_value=desired_value,
            threshold=decision_threshold,
            margin=boundary_margin,
        )
        attempt_record = {
            "epsilon": epsilon,
            "internal_after_probability": float(
                attack_record["after_probability"]
            ),
            "quantized_probability": quantized_probability,
            "quantized_target_pass": quantized_target_pass,
            "quantized_margin_pass": quantized_margin_pass,
            "iterations": int(attack_record["iterations"]),
            "boundary_iterations": int(
                attack_record["boundary_iterations"]
            ),
        }
        attempts.append(attempt_record)
        selected = candidate.detach()
        selected_record = attack_record
        if quantized_margin_pass:
            break

    final_attempt = attempts[-1]
    total_iterations = sum(item["iterations"] for item in attempts)
    return selected, {
        **selected_record,
        "after_probability": float(selected_record["after_probability"]),
        "quantized_after_probability": final_attempt[
            "quantized_probability"
        ],
        "target_pass": final_attempt["quantized_target_pass"],
        "margin_pass": final_attempt["quantized_margin_pass"],
        "selected_epsilon": final_attempt["epsilon"],
        "escalated": len(attempts) > 1,
        "schedule_exhausted": (
            not final_attempt["quantized_margin_pass"]
            and len(attempts) == len(schedule)
        ),
        "total_iterations": total_iterations,
        "attempts": attempts,
    }


class AttributeOutputTarget:
    """pytorch-grad-cam target for one binary multi-label output."""

    def __init__(self, label_index: int, present: bool) -> None:
        self.label_index = label_index
        self.present = present

    def __call__(self, model_output):
        value = model_output[self.label_index]
        return value if self.present else 1.0 - value


def gradcam_pp_saliency(
    model: Any,
    original_normalized: Any,
    *,
    label_index: int,
    original_present: bool,
):
    """Return normalized Grad-CAM++ saliency without an optional dependency.

    Kaggle's offline image does not consistently ship ``pytorch-grad-cam``.
    This is the Grad-CAM++ weighting rule directly in PyTorch, which keeps
    region discovery self-contained and differentiates the requested output
    score with respect to the final ResNet block activations.
    """

    import numpy as np
    import torch
    import torch.nn.functional as functional

    target_layer = model.base_model.layer4[-1]
    captured: list[Any] = []
    handle = target_layer.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        cam_input = original_normalized.detach().requires_grad_(True)
        model_output = model(cam_input)
        score = model_output[0, label_index]
        if not original_present:
            score = 1.0 - score
        if len(captured) != 1:
            raise RuntimeError("Grad-CAM++ did not capture target activations")
        activations = captured[0]
        gradients = torch.autograd.grad(
            score,
            activations,
            retain_graph=False,
            create_graph=False,
        )[0]
    finally:
        handle.remove()

    # Grad-CAM++ alpha coefficients and channel weights. The small epsilon
    # keeps regions with near-zero derivatives numerically well-defined.
    gradients_2 = gradients.square()
    gradients_3 = gradients_2 * gradients
    denominator = 2.0 * gradients_2 + (
        activations * gradients_3
    ).sum(dim=(-2, -1), keepdim=True)
    alphas = gradients_2 / (denominator + 1e-8)
    channel_weights = (alphas * gradients.relu()).sum(dim=(-2, -1), keepdim=True)
    saliency = (channel_weights * activations).sum(dim=1, keepdim=True).relu()
    saliency = functional.interpolate(
        saliency,
        size=cam_input.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    saliency = saliency / saliency.amax().clamp_min(1e-8)
    return np.asarray(saliency.detach().cpu(), dtype=np.float32)


def split_horizontal_grid(grid: Any, *, count: int, crop_width: int):
    """Split a one-row image grid into an NCHW candidate batch."""

    import torch

    if grid.ndim != 4 or grid.shape[0] != 1:
        raise ValueError("grid must have shape [1, C, H, W]")
    if count <= 0 or crop_width <= 0:
        raise ValueError("count and crop_width must be positive")
    required_width = count * crop_width
    if grid.shape[-1] != required_width:
        raise ValueError(
            f"grid width {grid.shape[-1]} does not equal "
            f"count * crop_width ({required_width})"
        )
    return torch.cat(
        [
            grid[:, :, :, index * crop_width : (index + 1) * crop_width]
            for index in range(count)
        ],
        dim=0,
    )


def join_horizontal_grid(candidates: Any):
    """Join an NCHW candidate batch into a one-row image grid."""

    import torch

    if candidates.ndim != 4 or candidates.shape[0] <= 0:
        raise ValueError("candidates must have shape [N, C, H, W]")
    return torch.cat(
        [candidates[index : index + 1] for index in range(candidates.shape[0])],
        dim=-1,
    )
