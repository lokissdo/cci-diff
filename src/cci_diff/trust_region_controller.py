"""Lexicographic trust-region composition in masked clean-latent geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from cci_diff.concept_graph import TrustRegionSpec
from cci_diff.constraint_controller import TargetMargin
from cci_diff.constraints import ConstraintObservation
from cci_diff.trust_region_solver import (
    ProjectionResult,
    project_to_linear_constraints,
    solve_lexicographic_envelope,
)


@dataclass(frozen=True)
class TrustRegionResult:
    clean_delta: Any
    record: dict[str, Any]


class LexicographicTrustRegionController:
    """Solve target, safety, and drift priorities in their gradient span."""

    _SAFETY_NAMES = ("identity", "outside_locality")

    def __init__(self, spec: TrustRegionSpec) -> None:
        self.spec = spec
        self.radius = spec.initial_radius

    def compute_update(
        self,
        *,
        latents: Any,
        target: TargetMargin,
        drift_loss: Any,
        safety_constraints: Sequence[ConstraintObservation],
        effective_support: Any,
        mode: str = "trust_region",
    ) -> TrustRegionResult:
        import torch

        if mode not in {"trust_region", "fixed_trust_matched"}:
            raise ValueError(f"Unknown trust-region mode: {mode}")
        safety = self._ordered_safety(safety_constraints)
        target_loss = (
            target.loss if target.residual > 0 else target.safeguard_loss
        )
        losses = (
            target_loss,
            safety[0].value,
            safety[1].value,
            drift_loss,
        )
        names = ("target", *self._SAFETY_NAMES, "non_target_drift")
        raw_gradients = []
        gradients = []
        gradient_records: dict[str, dict[str, float]] = {}
        for name, loss in zip(names, losses):
            raw, masked = _masked_gradient(
                loss,
                latents,
                effective_support,
                retain_graph=True,
            )
            raw_gradients.append(raw)
            gradients.append(masked)
            gradient_records[name] = {
                "raw_norm": _tensor_norm(raw),
                "masked_norm": _tensor_norm(masked),
            }

        scalar_values = (
            target.residual,
            float(drift_loss.detach().item()),
            *(
                float(observation.value.detach().item())
                for observation in safety
            ),
            *(record["raw_norm"] for record in gradient_records.values()),
            *(record["masked_norm"] for record in gradient_records.values()),
        )
        if not all(math.isfinite(value) for value in scalar_values):
            return self._skip_result(
                latents,
                target,
                gradient_records,
                mode,
                "nonfinite",
            )

        target_norm = gradient_records["target"]["masked_norm"]
        if target_norm <= 1e-12:
            return self._skip_result(
                latents,
                target,
                gradient_records,
                mode,
                "unreliable_target_gradient",
            )

        gram = _gram_matrix(gradients)
        cosines = _cosine_matrix(gram)
        residuals = tuple(
            float(observation.value.detach().item()) - observation.tolerance
            for observation in safety
        )
        guard_mode = "progress" if target.residual > 0 else "maintain"
        requested_progress = (
            min(
                target.residual,
                self.spec.target_progress_fraction
                * self.radius
                * target_norm,
            )
            if target.residual > 0
            else 0.0
        )
        target_bound = (
            -requested_progress
            if guard_mode == "progress"
            else -target.residual
        )

        envelope = self._solve_with_progress_backoff(
            gram,
            target_bound=target_bound,
            requested_progress=requested_progress,
            safety_residuals=residuals,
        )
        if envelope is None:
            return self._skip_result(
                latents,
                target,
                gradient_records,
                mode,
                "solver_infeasible",
                gram=gram,
                cosines=cosines,
            )
        accepted_progress, tau, envelope_result = envelope
        if guard_mode == "progress":
            target_bound = -accepted_progress

        if mode == "trust_region":
            drift_norm = gradient_records["non_target_drift"]["masked_norm"]
            eta = self.radius / drift_norm if drift_norm > 1e-12 else 0.0
            nominal = (0.0, 0.0, 0.0, -eta)
        else:
            eta = None
            nominal = (-target.activation, -1.0, -1.0, -1.0)

        result = project_to_linear_constraints(
            nominal,
            gram,
            (
                target_bound,
                tau + self.spec.feasibility_tolerance - residuals[0],
                tau + self.spec.feasibility_tolerance - residuals[1],
            ),
            radius=self.radius,
            tolerance=self.spec.feasibility_tolerance,
        )
        if result is None:
            return self._skip_result(
                latents,
                target,
                gradient_records,
                mode,
                "solver_infeasible",
                gram=gram,
                cosines=cosines,
            )

        clean_delta = torch.zeros_like(latents)
        for coefficient, gradient in zip(result.step, gradients):
            clean_delta = clean_delta + coefficient * gradient
        if not bool(torch.isfinite(clean_delta).all().item()):
            return self._skip_result(
                latents,
                target,
                gradient_records,
                mode,
                "nonfinite",
                gram=gram,
                cosines=cosines,
            )

        products = tuple(
            sum(gram[row][column] * result.step[column] for column in range(4))
            for row in range(4)
        )
        record = {
            "target": {
                "residual": target.residual,
                "desired_probability": target.desired_probability,
                "guard_mode": guard_mode,
            },
            "safety": {
                observation.name: {
                    "value": float(observation.value.detach().item()),
                    "tolerance": observation.tolerance,
                    "residual_before": residual,
                    "linearized_residual_after": residual + products[index + 1],
                }
                for index, (observation, residual) in enumerate(
                    zip(safety, residuals)
                )
            },
            "gradients": gradient_records,
            "geometry": {
                "order": list(names),
                "gram": gram,
                "cosines": cosines,
            },
            "solver": {
                "mode": mode,
                "requested_target_progress": requested_progress,
                "accepted_target_progress": accepted_progress,
                "achieved_target_progress": -products[0],
                "envelope_tau": tau,
                "envelope_active_constraints": list(
                    envelope_result.active_indices
                ),
                "active_constraints": list(result.active_indices),
                "multipliers": list(result.multipliers),
                "coefficients": list(result.step),
                "nominal_coefficients": list(nominal),
                "eta": eta,
                "radius": self.radius,
                "step_norm": result.norm,
                "primal_violation": result.primal_violation,
                "dual_violation": result.dual_violation,
            },
            "update": {
                "skip_reason": None,
                "clean_norm": _tensor_norm(clean_delta),
            },
        }
        return TrustRegionResult(clean_delta, record)

    def observe_outcome(
        self,
        *,
        requested_progress: float,
        actual_progress: float,
        step_norm: float,
    ) -> None:
        if requested_progress <= 0 or not all(
            math.isfinite(value)
            for value in (requested_progress, actual_progress, step_norm)
        ):
            return
        previous = self.radius
        ratio = actual_progress / requested_progress
        if ratio < 0.25:
            updated = previous * 0.5
        elif ratio > 0.75 and step_norm >= 0.9 * previous:
            updated = previous * 1.5
        else:
            updated = previous
        self.radius = min(
            max(updated, self.spec.minimum_radius),
            self.spec.maximum_radius,
        )

    def _ordered_safety(
        self,
        observations: Sequence[ConstraintObservation],
    ) -> tuple[ConstraintObservation, ConstraintObservation]:
        by_name = {observation.name: observation for observation in observations}
        if (
            len(observations) != 2
            or len(by_name) != 2
            or set(by_name) != set(self._SAFETY_NAMES)
        ):
            raise ValueError(
                "safety constraints must contain exactly identity and "
                "outside_locality"
            )
        return by_name["identity"], by_name["outside_locality"]

    def _solve_with_progress_backoff(
        self,
        gram: list[list[float]],
        *,
        target_bound: float,
        requested_progress: float,
        safety_residuals: tuple[float, float],
    ) -> tuple[float, float, ProjectionResult] | None:
        def solve(bound: float):
            try:
                return solve_lexicographic_envelope(
                    gram,
                    bound,
                    safety_residuals,
                    radius=self.radius,
                    tolerance=self.spec.feasibility_tolerance,
                )
            except ValueError:
                return None

        direct = solve(target_bound)
        if direct is not None:
            return requested_progress, direct[0], direct[1]
        if requested_progress <= 0:
            return None

        low = 0.0
        high = requested_progress
        best = solve(0.0)
        if best is None:
            return None
        for _ in range(40):
            middle = (low + high) / 2.0
            candidate = solve(-middle)
            if candidate is None:
                high = middle
            else:
                low = middle
                best = candidate
        return low, best[0], best[1]

    def _skip_result(
        self,
        latents: Any,
        target: TargetMargin,
        gradient_records: dict[str, dict[str, float]],
        mode: str,
        reason: str,
        *,
        gram: list[list[float]] | None = None,
        cosines: list[list[float]] | None = None,
    ) -> TrustRegionResult:
        return TrustRegionResult(
            latents.new_zeros(latents.shape),
            {
                "target": {
                    "residual": target.residual,
                    "desired_probability": target.desired_probability,
                    "guard_mode": (
                        "progress" if target.residual > 0 else "maintain"
                    ),
                },
                "gradients": gradient_records,
                "geometry": {
                    "order": [
                        "target",
                        "identity",
                        "outside_locality",
                        "non_target_drift",
                    ],
                    "gram": gram,
                    "cosines": cosines,
                },
                "solver": {
                    "mode": mode,
                    "radius": self.radius,
                },
                "update": {
                    "skip_reason": reason,
                    "clean_norm": 0.0,
                },
            },
        )


def _masked_gradient(
    loss: Any,
    latents: Any,
    support: Any,
    *,
    retain_graph: bool,
) -> tuple[Any, Any]:
    import torch

    raw = torch.autograd.grad(
        loss,
        latents,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]
    if raw is None:
        raw = torch.zeros_like(latents)
    mask = support.to(device=latents.device, dtype=latents.dtype)
    return raw, raw * mask


def _gram_matrix(gradients: Sequence[Any]) -> list[list[float]]:
    import torch

    return [
        [
            float(
                torch.sum(
                    left.detach().float() * right.detach().float()
                ).item()
            )
            for right in gradients
        ]
        for left in gradients
    ]


def _cosine_matrix(gram: Sequence[Sequence[float]]) -> list[list[float]]:
    norms = [math.sqrt(max(row[index], 0.0)) for index, row in enumerate(gram)]
    return [
        [
            (
                gram[row][column] / (norms[row] * norms[column])
                if norms[row] > 0 and norms[column] > 0
                else 0.0
            )
            for column in range(len(gram))
        ]
        for row in range(len(gram))
    ]


def _tensor_norm(value: Any) -> float:
    import torch

    return float(torch.linalg.vector_norm(value.detach().float()).item())
