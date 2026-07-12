"""Stable Diffusion 2 CCI latent guidance adapter."""

from __future__ import annotations

from typing import Any, Callable

from cci_diff.guidance import GuidanceTerms, compose_guidance_loss
from cci_diff.spec import GuidanceWeights


DecodeFn = Callable[[Any], Any]
LossFn = Callable[[Any], GuidanceTerms]


def apply_cci_latent_guidance(
    latents: Any,
    *,
    decode_fn: DecodeFn,
    loss_fn: LossFn,
    weights: GuidanceWeights,
    step_size: float,
    latent_mask: Any | None = None,
    normalize_gradient: bool = False,
    gradient_eps: float = 1e-8,
) -> Any:
    """Apply one differentiable CCI loss step to SD2 latents.

    ``decode_fn`` maps latents to a differentiable image/concept space.
    ``loss_fn`` returns framework-neutral ``GuidanceTerms`` from that decoded
    value. Torch is imported here, rather than at module import time, so the
    core CCI package remains usable without ML dependencies installed.
    """

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "CCI SD2 adapter requires torch. Install the ML dependencies with "
            "pip install -e '.[ml]' before enabling the SD2 CCI hook."
        ) from exc

    if step_size < 0:
        raise ValueError("step_size must be non-negative")

    with torch.enable_grad():
        guided_latents = latents.detach().clone().requires_grad_(True)
        decoded = decode_fn(guided_latents)
        loss = compose_guidance_loss(loss_fn(decoded), weights)
        gradient = torch.autograd.grad(
            loss,
            guided_latents,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

    if latent_mask is not None:
        gradient = gradient * latent_mask.to(
            device=gradient.device,
            dtype=gradient.dtype,
        )
    if normalize_gradient:
        gradient = _normalize_gradient(gradient, gradient_eps)

    return (latents - step_size * gradient).detach()


def infer_target_rgb(target_concept: str) -> tuple[float, float, float]:
    """Return a simple RGB prior for color-like face interventions."""

    normalized = target_concept.casefold()
    if "blond" in normalized or "blonde" in normalized:
        return (0.95, 0.78, 0.38)
    if "black hair" in normalized:
        return (0.05, 0.04, 0.035)
    if "brown hair" in normalized:
        return (0.32, 0.18, 0.08)
    if "gray hair" in normalized or "grey hair" in normalized:
        return (0.62, 0.60, 0.56)
    if "red hair" in normalized:
        return (0.72, 0.20, 0.08)
    raise ValueError(
        "No built-in CCI color prior for "
        f"{target_concept!r}. Pass --cci_target_rgb R,G,B to use latent_color."
    )


def _normalize_gradient(gradient, eps: float):
    if eps <= 0:
        raise ValueError("gradient_eps must be positive")
    if gradient.ndim <= 1:
        norm = gradient.pow(2).sum().sqrt().clamp_min(eps)
        return gradient / norm
    reduce_dims = tuple(range(1, gradient.ndim))
    norm = gradient.pow(2).sum(dim=reduce_dims, keepdim=True).sqrt().clamp_min(eps)
    return gradient / norm
