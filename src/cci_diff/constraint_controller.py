"""Automatic primal-dual gradient composition for clean CCI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from cci_diff.concept_graph import ControllerSpec
from cci_diff.constraints import ConstraintObservation


@dataclass(frozen=True)
class TargetMargin:
    loss: Any
    safeguard_loss: Any
    raw_logit: Any
    signed_logit: Any
    required_logit: float
    desired_probability: float
    residual: float
    activation: float


@dataclass(frozen=True)
class ControllerResult:
    delta: Any
    record: dict[str, Any]


def target_margin(
    logit: Any,
    desired_value: int,
    target_probability: float,
) -> TargetMargin:
    import torch

    if desired_value not in (0, 1) or isinstance(desired_value, bool):
        raise ValueError("desired_value must be 0 or 1")
    if not 0.5 < target_probability < 1.0:
        raise ValueError(
            "target_probability must be strictly between 0.5 and 1.0"
        )
    sign = 2 * desired_value - 1
    signed = sign * logit
    required = math.log(target_probability / (1.0 - target_probability))
    threshold = torch.as_tensor(
        required,
        device=logit.device,
        dtype=logit.dtype,
    )
    loss = torch.relu(threshold - signed)
    residual = float(required - signed.detach().item())
    activation = (
        min(max(residual / max(abs(required), 1.0), 0.0), 1.0)
        if math.isfinite(residual)
        else 0.0
    )
    probability = float(torch.sigmoid(signed.detach()).item())
    return TargetMargin(
        loss,
        -signed,
        logit,
        signed,
        required,
        probability,
        residual,
        activation,
    )


def update_dual_multiplier(
    current: float,
    *,
    value: float,
    tolerance: float,
    spec: ControllerSpec,
) -> tuple[float, float]:
    residual = value / tolerance - 1.0
    if not math.isfinite(residual):
        return current, residual
    updated = min(
        max(current + spec.dual_rate * residual, 0.0),
        spec.lambda_max,
    )
    return updated, residual


def normalize_with_ema(
    gradient: Any,
    *,
    previous_ema: float,
    beta: float,
    floor: float,
) -> tuple[Any, float, float]:
    import torch

    raw_norm = float(torch.linalg.vector_norm(gradient.detach().float()).item())
    if not math.isfinite(raw_norm):
        return torch.zeros_like(gradient), previous_ema, raw_norm
    ema = beta * previous_ema + (1.0 - beta) * raw_norm
    if raw_norm < floor or ema < floor:
        return torch.zeros_like(gradient), ema, raw_norm
    return gradient / max(ema, floor), ema, raw_norm


def project_target_conflict(
    target_gradient: Any,
    constraint_gradient: Any,
    *,
    gradient_floor: float,
) -> tuple[Any, bool, float | None]:
    import torch

    target_norm = torch.linalg.vector_norm(target_gradient.detach().float())
    constraint_norm = torch.linalg.vector_norm(
        constraint_gradient.detach().float()
    )
    if (
        target_norm.item() < gradient_floor
        or constraint_norm.item() < gradient_floor
    ):
        return constraint_gradient, False, None
    dot = torch.sum(target_gradient * constraint_gradient)
    cosine = float((dot / (target_norm * constraint_norm)).detach().item())
    if dot.detach().item() >= 0:
        return constraint_gradient, False, cosine
    denominator = torch.sum(target_gradient * target_gradient).clamp_min(
        gradient_floor**2
    )
    projected = constraint_gradient - dot / denominator * target_gradient
    return projected, True, cosine


def budget_constraint_for_target(
    target_gradient: Any,
    constraint_gradient: Any,
    *,
    target_activation: float,
    gradient_floor: float,
) -> tuple[Any, float, float | None]:
    """Cap preservation pressure until the target approaches feasibility."""

    import torch

    if target_activation <= 0:
        return constraint_gradient, 1.0, None
    target_norm = float(
        torch.linalg.vector_norm(target_gradient.detach().float()).item()
    )
    constraint_norm = float(
        torch.linalg.vector_norm(constraint_gradient.detach().float()).item()
    )
    budget = target_norm * max(0.0, 1.0 - target_activation)
    scale = min(1.0, budget / max(constraint_norm, gradient_floor))
    return constraint_gradient * scale, scale, budget


def clip_update_norm(
    update: Any,
    *,
    trust_radius: float,
    gradient_floor: float,
) -> tuple[Any, float, float]:
    import torch

    before = float(torch.linalg.vector_norm(update.detach().float()).item())
    if not math.isfinite(before):
        return torch.zeros_like(update), before, 0.0
    scale = min(1.0, trust_radius / max(before, gradient_floor))
    clipped = update * scale
    after = float(torch.linalg.vector_norm(clipped.detach().float()).item())
    return clipped, before, after


class ConstraintFeedbackController:
    def __init__(
        self,
        spec: ControllerSpec,
        *,
        use_target_guidance: bool = True,
        normalize_gradients: bool = True,
        budget_constraints: bool = True,
    ) -> None:
        self.spec = spec
        self.use_target_guidance = use_target_guidance
        self.normalize_gradients = normalize_gradients
        self.budget_constraints = budget_constraints
        self.multipliers: dict[str, float] = {}
        self.norm_ema: dict[str, float] = {}
        self.consecutive_nonfinite = 0
        self.consecutive_unreliable_target = 0

    def compute_update(
        self,
        *,
        latents: Any,
        target: TargetMargin,
        constraints: Sequence[ConstraintObservation],
        latent_mask: Any,
        eta: float,
        project_conflicts: bool = True,
        mode: str = "feedback",
    ) -> ControllerResult:
        import torch

        if mode not in {"disabled", "feedback", "fixed_equal"}:
            raise ValueError(f"Unknown controller mode: {mode}")
        if mode == "disabled":
            return self._disabled_result(latents, target, constraints, eta)

        multiplier_snapshot = dict(self.multipliers)
        norm_snapshot = dict(self.norm_ema)
        target_gradient = torch.zeros_like(latents)
        if self.use_target_guidance:
            target_value = torch.autograd.grad(
                target.safeguard_loss,
                latents,
                retain_graph=bool(constraints),
                allow_unused=True,
            )[0]
            if target_value is not None:
                target_gradient = target_value
        target_normalized, target_ema, target_raw_norm = self._prepare_gradient(
            target_gradient,
            name="target",
        )
        target_normalized_norm = float(
            torch.linalg.vector_norm(target_normalized.detach().float()).item()
        )

        constraint_gradient = torch.zeros_like(latents)
        records: dict[str, Any] = {}
        finite = math.isfinite(target_raw_norm) and bool(
            torch.isfinite(target.loss).all().item()
        )
        for index, observation in enumerate(constraints):
            value_float = float(observation.value.detach().item())
            before = self.multipliers.get(observation.name, 0.0)
            after, residual = update_dual_multiplier(
                before,
                value=value_float,
                tolerance=observation.tolerance,
                spec=self.spec,
            )
            violation = torch.relu(
                observation.value / observation.tolerance - 1.0
            )
            active = (
                math.isfinite(residual)
                and float(violation.detach().item()) > 0.0
            )
            raw = torch.zeros_like(latents)
            normalized = raw
            raw_norm = 0.0
            normalized_norm = 0.0
            if active:
                raw_value = torch.autograd.grad(
                    violation,
                    latents,
                    retain_graph=index < len(constraints) - 1,
                    allow_unused=True,
                )[0]
                if raw_value is not None:
                    raw = raw_value
                normalized, _, raw_norm = self._prepare_gradient(
                    raw,
                    name=observation.name,
                )
                normalized_norm = float(
                    torch.linalg.vector_norm(
                        normalized.detach().float()
                    ).item()
                )
            coefficient = (
                1.0
                if mode == "fixed_equal" and active
                else before + self.spec.penalty * max(residual, 0.0)
                if math.isfinite(residual)
                else 0.0
            )
            constraint_gradient = (
                constraint_gradient + coefficient * normalized
            )
            self.multipliers[observation.name] = after
            finite = (
                finite
                and math.isfinite(value_float)
                and math.isfinite(raw_norm)
            )
            records[observation.name] = {
                "value": _finite_or_none(value_float),
                "tolerance": observation.tolerance,
                "residual": _finite_or_none(residual),
                "violation": (
                    max(residual, 0.0) if math.isfinite(residual) else None
                ),
                "lambda_before": before,
                "lambda_after": after,
                "coefficient": coefficient,
                "gradient_norm": _finite_or_none(raw_norm),
                "normalized_gradient_norm": _finite_or_none(normalized_norm),
            }

        projected = False
        cosine = None
        if project_conflicts and self.use_target_guidance:
            constraint_gradient, projected, cosine = project_target_conflict(
                target_normalized,
                constraint_gradient,
                gradient_floor=self.spec.gradient_floor,
            )
        constraint_scale = 1.0
        constraint_budget = None
        effective_activation = target.activation if self.use_target_guidance else 0.0
        if mode == "feedback" and self.budget_constraints and self.use_target_guidance:
            (
                constraint_gradient,
                constraint_scale,
                constraint_budget,
            ) = budget_constraint_for_target(
                target_normalized,
                constraint_gradient,
                target_activation=effective_activation,
                gradient_floor=self.spec.gradient_floor,
            )
        combined = effective_activation * target_normalized + constraint_gradient
        mask = latent_mask.to(device=latents.device, dtype=latents.dtype)
        masked = eta * mask * combined
        delta, pre_clip_norm, final_norm = clip_update_norm(
            masked,
            trust_radius=self.spec.trust_radius,
            gradient_floor=self.spec.gradient_floor,
        )

        if not finite or not bool(torch.isfinite(delta).all().item()):
            self.multipliers = multiplier_snapshot
            self.norm_ema = norm_snapshot
            self.consecutive_nonfinite += 1
            if self.consecutive_nonfinite >= 2:
                raise FloatingPointError(
                    "Two consecutive non-finite clean CCI steps"
                )
            delta = torch.zeros_like(latents)
            skip_reason = "nonfinite"
        else:
            self.consecutive_nonfinite = 0
            skip_reason = None
        if (
            self.use_target_guidance
            and target.activation > 0
            and target_raw_norm < self.spec.gradient_floor
        ):
            self.consecutive_unreliable_target += 1
        else:
            self.consecutive_unreliable_target = 0

        record = {
            "target": self._target_record(
                target,
                target_raw_norm,
                target_ema,
                target_normalized_norm,
            ),
            "constraints": records,
            "update": {
                "eta": eta,
                "controller_mode": mode,
                "projected": projected,
                "target_constraint_cosine": cosine,
                "constraint_scale": constraint_scale,
                "constraint_budget": constraint_budget,
                "gradient_normalization": self.normalize_gradients,
                "target_budget": self.budget_constraints,
                "pre_clip_norm": pre_clip_norm,
                "norm": final_norm,
                "skip_reason": skip_reason,
            },
        }
        return ControllerResult(delta.detach(), record)

    def _prepare_gradient(
        self,
        gradient: Any,
        *,
        name: str,
    ) -> tuple[Any, float, float]:
        import torch

        if self.normalize_gradients:
            normalized, ema, raw_norm = normalize_with_ema(
                gradient,
                previous_ema=self.norm_ema.get(name, 0.0),
                beta=self.spec.norm_ema_beta,
                floor=self.spec.gradient_floor,
            )
        else:
            raw_norm = float(
                torch.linalg.vector_norm(gradient.detach().float()).item()
            )
            normalized = gradient if math.isfinite(raw_norm) else torch.zeros_like(gradient)
            ema = raw_norm
        self.norm_ema[name] = ema
        return normalized, ema, raw_norm

    def _disabled_result(
        self,
        latents: Any,
        target: TargetMargin,
        constraints: Sequence[ConstraintObservation],
        eta: float,
    ) -> ControllerResult:
        import torch

        finite = all(
            math.isfinite(float(observation.value.detach().item()))
            for observation in constraints
        ) and math.isfinite(target.desired_probability)
        self.consecutive_nonfinite = (
            0 if finite else self.consecutive_nonfinite + 1
        )
        if self.consecutive_nonfinite >= 2:
            raise FloatingPointError("Two consecutive non-finite clean CCI steps")
        records = {}
        for observation in constraints:
            value = float(observation.value.detach().item())
            residual = value / observation.tolerance - 1.0
            records[observation.name] = {
                "value": value if math.isfinite(value) else None,
                "tolerance": observation.tolerance,
                "residual": residual if math.isfinite(residual) else None,
                "violation": (
                    max(residual, 0.0) if math.isfinite(residual) else None
                ),
                "lambda_before": 0.0,
                "lambda_after": 0.0,
                "coefficient": 0.0,
                "gradient_norm": 0.0,
                "normalized_gradient_norm": 0.0,
            }
        return ControllerResult(
            torch.zeros_like(latents),
            {
                "target": self._target_record(target, 0.0, 0.0, 0.0),
                "constraints": records,
                "update": {
                    "eta": eta,
                    "controller_mode": "disabled",
                    "projected": False,
                    "target_constraint_cosine": None,
                    "gradient_normalization": self.normalize_gradients,
                    "target_budget": self.budget_constraints,
                    "pre_clip_norm": 0.0,
                    "norm": 0.0,
                    "skip_reason": (
                        None if finite else "nonfinite_measurement"
                    ),
                },
            },
        )

    def _target_record(
        self,
        target: TargetMargin,
        gradient_norm: float,
        norm_ema: float,
        normalized_gradient_norm: float,
    ) -> dict[str, Any]:
        return {
            "logit": _finite_or_none(target.raw_logit.detach().item()),
            "signed_logit": _finite_or_none(
                target.signed_logit.detach().item()
            ),
            "target_probability": _finite_or_none(
                target.desired_probability
            ),
            "required_probability": self._required_probability(target),
            "required_logit": target.required_logit,
            "margin_residual": _finite_or_none(target.residual),
            "activation": target.activation,
            "gradient_norm": _finite_or_none(gradient_norm),
            "normalized_gradient_norm": _finite_or_none(normalized_gradient_norm),
            "norm_ema": _finite_or_none(norm_ema),
            "guidance_enabled": self.use_target_guidance,
            "unreliable_target_gradient": (
                self.consecutive_unreliable_target >= 2
            ),
        }

    @staticmethod
    def _required_probability(target: TargetMargin) -> float:
        return 1.0 / (1.0 + math.exp(-target.required_logit))


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None
