"""Robust multi-term latent guidance for semantic SD2 interventions."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from cci_diff.classifiers.celeba_resnet50 import classifier_logits


def robust_step_size(
    step: int,
    *,
    start: int,
    end: int,
    every: int,
    base: float,
) -> float | None:
    """Return a linearly decayed step size inside the active schedule."""

    if start > end:
        raise ValueError("start must be less than or equal to end")
    if every <= 0:
        raise ValueError("every must be positive")
    if base < 0:
        raise ValueError("base must be non-negative")
    if step < start or step > end or (step - start) % every != 0:
        return None
    return base * (end - step + 1) / (end - start + 1)


def boundary_loss(decoded: Any, source: Any, boundary_mask: Any) -> Any:
    """Preserve source appearance in a soft semantic boundary ring."""

    channels = decoded.shape[1]
    denominator = (boundary_mask.sum() * channels).clamp_min(1.0)
    return ((decoded - source).abs() * boundary_mask).sum() / denominator


def residual_tv_loss(decoded: Any, source: Any, generation_mask: Any) -> Any:
    """Penalize high-frequency edit residuals inside the generation region."""

    residual = generation_mask * (decoded - source)
    vertical = (residual[:, :, 1:, :] - residual[:, :, :-1, :]).abs().mean()
    horizontal = (residual[:, :, :, 1:] - residual[:, :, :, :-1]).abs().mean()
    return vertical + horizontal


def multi_scale_classifier_loss(
    classifier: Any,
    decoded: Any,
    *,
    label_index: int,
    desired_value: int,
    scales: tuple[int, ...],
    input_size: int,
    blur_sigma: float,
) -> Any:
    """Average classifier BCE over deterministic low-pass image scales."""

    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("classifier scales must contain positive values")
    if blur_sigma < 0:
        raise ValueError("blur_sigma must be non-negative")

    import torch
    import torch.nn.functional as functional
    from torchvision.transforms.functional import gaussian_blur

    filtered = (
        gaussian_blur(decoded.float(), [5, 5], [blur_sigma, blur_sigma])
        if blur_sigma > 0
        else decoded.float()
    )
    losses = []
    for scale in scales:
        view = functional.interpolate(
            filtered,
            size=(scale, scale),
            mode="bilinear",
            align_corners=False,
        )
        logits = classifier_logits(classifier, view, size=input_size)
        selected = logits[:, label_index]
        target = torch.full_like(selected, float(desired_value))
        losses.append(functional.binary_cross_entropy_with_logits(selected, target))
    return torch.stack(losses).mean()


def apply_robust_latent_guidance(
    latents: Any,
    *,
    decode_fn: Callable[[Any], Any],
    loss_fn: Callable[[Any], Mapping[str, Any]],
    weights: Mapping[str, float],
    step_size: float,
    generation_mask: Any,
    gradient_eps: float = 1e-8,
) -> tuple[Any, dict[str, float]]:
    """Apply separately normalized semantic and realism latent gradients."""

    if step_size < 0:
        raise ValueError("step_size must be non-negative")
    if gradient_eps <= 0:
        raise ValueError("gradient_eps must be positive")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("robust guidance weights must be non-negative")

    import torch

    with torch.enable_grad():
        guided_latents = latents.detach().clone().float().requires_grad_(True)
        decoded = decode_fn(guided_latents)
        terms = dict(loss_fn(decoded))
        update = torch.zeros_like(guided_latents)
        stats: dict[str, float] = {}
        term_items = list(terms.items())
        for index, (name, loss) in enumerate(term_items):
            gradient = torch.autograd.grad(
                loss,
                guided_latents,
                retain_graph=index < len(term_items) - 1,
                create_graph=False,
                allow_unused=True,
            )[0]
            if gradient is None:
                gradient = torch.zeros_like(guided_latents)
            gradient = gradient * generation_mask.to(
                device=gradient.device,
                dtype=gradient.dtype,
            )
            reduce_dims = tuple(range(1, gradient.ndim))
            norm = gradient.pow(2).sum(dim=reduce_dims, keepdim=True).sqrt()
            normalized = torch.where(
                norm > gradient_eps,
                gradient / norm.clamp_min(gradient_eps),
                torch.zeros_like(gradient),
            )
            update = update + float(weights.get(name, 0.0)) * normalized
            stats[f"{name}_loss"] = float(loss.detach().mean().cpu())
            stats[f"{name}_gradient_norm"] = float(norm.detach().mean().cpu())

    return (latents - step_size * update.to(latents.dtype)).detach(), stats
