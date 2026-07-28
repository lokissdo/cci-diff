"""Predicted-clean pre-scheduler guidance for SD2 DDIM."""

from __future__ import annotations

import math
from typing import Any

from cci_diff.concept_graph import ControllerSpec
from cci_diff.constraint_controller import target_margin
from cci_diff.constraints import ConstraintContext, ConstraintObservation


LATENT_SCALE = 0.18215


def alpha_prod_for_step(scheduler: Any, timestep: Any, sample: Any) -> Any:
    import torch

    index = int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
    alpha = scheduler.alphas_cumprod[index]
    return torch.as_tensor(
        alpha,
        device=sample.device,
        dtype=sample.dtype,
    )


def predict_clean_latents(
    sample: Any,
    model_output: Any,
    alpha_prod_t: Any,
    prediction_type: str,
) -> Any:
    import torch

    alpha = torch.as_tensor(
        alpha_prod_t,
        device=sample.device,
        dtype=sample.dtype,
    )
    beta = (1.0 - alpha).clamp(min=0.0)
    detached_output = model_output.detach().to(
        device=sample.device,
        dtype=sample.dtype,
    )
    if prediction_type == "epsilon":
        return (sample - beta.sqrt() * detached_output) / alpha.sqrt()
    if prediction_type == "v_prediction":
        return alpha.sqrt() * sample - beta.sqrt() * detached_output
    if prediction_type == "sample":
        raise ValueError(
            "sample prediction is unsupported without U-Net backpropagation"
        )
    raise ValueError(f"Unsupported scheduler prediction type: {prediction_type}")


def clean_delta_to_epsilon_delta(
    clean_delta: Any,
    alpha_prod_t: Any,
) -> Any:
    """Map a predicted-clean displacement to epsilon-prediction coordinates."""

    import torch

    alpha = torch.as_tensor(
        alpha_prod_t,
        device=clean_delta.device,
        dtype=clean_delta.dtype,
    )
    beta = (1.0 - alpha).clamp_min(torch.finfo(clean_delta.dtype).eps)
    return -torch.sqrt(alpha / beta) * clean_delta


def decode_clean_latents(
    vae: Any,
    clean_latents: Any,
    latent_scale: float = LATENT_SCALE,
) -> Any:
    if latent_scale <= 0:
        raise ValueError("latent_scale must be positive")
    # The SD2 denoiser may run in FP16 while clean-CCI intentionally keeps
    # the VAE in FP32 for stable differentiable decoding. Match the VAE input
    # to its parameters before decoding so CUDA convolution receives one dtype.
    vae_dtype = getattr(vae, "dtype", clean_latents.dtype)
    vae_latents = clean_latents.to(dtype=vae_dtype)
    decoded = vae.decode(vae_latents / latent_scale).sample
    return (decoded / 2.0 + 0.5).clamp(0.0, 1.0)


def guidance_eta(
    step_index: int,
    progress: float,
    spec: ControllerSpec,
    *,
    scheduled: bool = True,
) -> float | None:
    if step_index % spec.every_n_steps != 0:
        return None
    if not scheduled:
        return spec.step_scale
    start, end = spec.active_progress
    if progress < start or progress > end:
        return None
    u = (progress - start) / (end - start)
    return spec.step_scale * math.sin(math.pi * u) ** 2


def classifier_attribution_mask(
    target_loss: Any,
    decoded_image: Any,
    semantic_latent_mask: Any,
    *,
    latent_size: tuple[int, int],
    eps: float = 1e-8,
) -> Any:
    """Build a soft classifier-saliency mask bounded by semantic support."""

    import torch
    import torch.nn.functional as functional

    image_gradient = torch.autograd.grad(
        target_loss,
        decoded_image,
        retain_graph=True,
    )[0]
    saliency = image_gradient.detach().abs().mean(dim=1, keepdim=True)
    semantic_image_mask = functional.interpolate(
        semantic_latent_mask.detach().float(),
        size=decoded_image.shape[-2:],
        mode="nearest",
    ).to(device=saliency.device, dtype=saliency.dtype)
    saliency = saliency * semantic_image_mask
    peak = saliency.amax(dim=(-2, -1), keepdim=True)
    normalized = saliency / peak.clamp_min(eps)
    normalized = torch.where(
        peak > eps,
        normalized,
        semantic_image_mask,
    )
    latent_support = functional.interpolate(
        normalized,
        size=latent_size,
        mode="bilinear",
        align_corners=False,
    )
    semantic_latent_mask = semantic_latent_mask.to(
        device=latent_support.device,
        dtype=latent_support.dtype,
    )
    latent_support = latent_support * semantic_latent_mask
    latent_peak = latent_support.amax(dim=(-2, -1), keepdim=True)
    return torch.where(
        latent_peak > eps,
        latent_support / latent_peak.clamp_min(eps),
        semantic_latent_mask,
    )


class FinalTargetLatentCorrectionHook:
    """Restore target feasibility on the final clean latent with line search."""

    apply_after_blend = True

    def __init__(
        self,
        *,
        vae: Any,
        target_evaluator: Any,
        desired_value: int,
        target_probability: float,
        max_steps: int,
        step_radius: float,
        mask_mode: str = "semantic_attribution",
    ) -> None:
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if step_radius <= 0:
            raise ValueError("step_radius must be positive")
        if mask_mode not in {
            "generation",
            "semantic",
            "semantic_attribution",
        }:
            raise ValueError(f"Unsupported final correction mask mode: {mask_mode}")
        self.vae = vae
        self.target_evaluator = target_evaluator
        self.desired_value = desired_value
        self.target_probability = target_probability
        self.max_steps = max_steps
        self.step_radius = step_radius
        self.mask_mode = mask_mode
        self.record: dict[str, Any] | None = None

    def _probability(self, latents: Any) -> float:
        import torch

        with torch.no_grad():
            image = decode_clean_latents(self.vae, latents)
            margin = target_margin(
                self.target_evaluator.logit(image),
                self.desired_value,
                self.target_probability,
            )
        return margin.desired_probability

    def __call__(self, step: Any) -> Any | None:
        import torch

        if step.progress < 1.0 - 1e-9 or self.max_steps == 0:
            return None
        current = step.latents.detach()
        initial_probability = self._probability(current)
        current_probability = initial_probability
        attempts = []
        accepted_steps = 0
        mask_record: dict[str, Any] = {"mode": self.mask_mode}
        for iteration in range(self.max_steps):
            if current_probability >= self.target_probability:
                break
            guided = current.detach().clone().requires_grad_(True)
            image = decode_clean_latents(self.vae, guided)
            margin = target_margin(
                self.target_evaluator.logit(image),
                self.desired_value,
                self.target_probability,
            )
            semantic_mask = (
                step.semantic_mask
                if step.semantic_mask is not None
                else (step.latent_mask >= 0.5).to(step.latent_mask.dtype)
            )
            if self.mask_mode == "generation":
                mask = step.latent_mask
            elif self.mask_mode == "semantic":
                mask = semantic_mask
            else:
                mask = classifier_attribution_mask(
                    margin.loss,
                    image,
                    semantic_mask,
                    latent_size=tuple(guided.shape[-2:]),
                )
            gradient = torch.autograd.grad(margin.loss, guided)[0]
            mask = mask.to(
                device=gradient.device,
                dtype=gradient.dtype,
            )
            gradient = gradient * mask
            mask_record = {
                "mode": self.mask_mode,
                "active_fraction": float((mask > 0).float().mean().item()),
                "mean_weight": float(mask.float().mean().item()),
            }
            norm = float(torch.linalg.vector_norm(gradient.float()).item())
            if not math.isfinite(norm) or norm <= 1e-8:
                attempts.append(
                    {"iteration": iteration, "accepted": False, "reason": "gradient"}
                )
                break
            direction = gradient / norm
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125):
                candidate = (guided - self.step_radius * fraction * direction).detach()
                probability = self._probability(candidate)
                if probability > current_probability + 1e-7:
                    attempts.append(
                        {
                            "iteration": iteration,
                            "accepted": True,
                            "step_fraction": fraction,
                            "probability_before": current_probability,
                            "probability_after": probability,
                        }
                    )
                    current = candidate
                    current_probability = probability
                    accepted_steps += 1
                    accepted = True
                    break
            if not accepted:
                attempts.append(
                    {"iteration": iteration, "accepted": False, "reason": "line_search"}
                )
                break
        self.record = {
            "initial_probability": initial_probability,
            "final_probability": current_probability,
            "required_probability": self.target_probability,
            "accepted_steps": accepted_steps,
            "attempts": attempts,
            "mask": mask_record,
        }
        return current if accepted_steps else None


class TrustRegionCleanCCIGuidanceHook:
    """Predicted-clean trust-region guidance with post-BLD retention tracing."""

    def __init__(
        self,
        *,
        scheduler: Any,
        vae: Any,
        target_evaluator: Any,
        drift_evaluator: Any,
        constraint_evaluators: tuple[Any, ...],
        controller: Any,
        desired_value: int,
        target_probability: float,
        trace_writer: Any,
        frame_observer: Any | None = None,
        controller_mode: str = "trust_region",
    ) -> None:
        self.scheduler = scheduler
        self.vae = vae
        self.target_evaluator = target_evaluator
        self.drift_evaluator = drift_evaluator
        self.constraint_evaluators = constraint_evaluators
        self.controller = controller
        self.desired_value = desired_value
        self.target_probability = target_probability
        self.trace_writer = trace_writer
        self.frame_observer = frame_observer
        self.controller_mode = controller_mode
        self._evaluators_bound = False
        self._pending_record: dict[str, Any] | None = None
        self._retention_reference = None
        self.current_step_active = False
        self.peak_mps_bytes: int | None = None

    def __call__(self, step: Any) -> Any | None:
        import torch

        self.current_step_active = False
        self._pending_record = None
        alpha = alpha_prod_for_step(self.scheduler, step.timestep, step.latents)
        alpha_number = float(alpha.detach().item())
        if alpha_number < self.controller.spec.reliability_alpha_min:
            return None
        prediction_type = self.scheduler.config.prediction_type
        if prediction_type != "epsilon":
            raise ValueError(
                "trust-region clean guidance currently requires epsilon prediction"
            )

        self._sample_mps_memory(torch)
        predicted_clean = predict_clean_latents(
            step.latents.detach().float(),
            step.noise_pred.detach().float(),
            alpha,
            prediction_type,
        )
        clean_latents = predicted_clean.detach().clone().requires_grad_(True)
        clean_image = decode_clean_latents(self.vae, clean_latents)
        self._bind_evaluators(step, clean_image)

        margin = target_margin(
            self.target_evaluator.logit(clean_image),
            self.desired_value,
            self.target_probability,
        )
        drift_loss = self.drift_evaluator.measure(clean_image)
        observations = tuple(
            ConstraintObservation(
                evaluator.name,
                evaluator.measure(clean_image),
                evaluator.tolerance,
            )
            for evaluator in self.constraint_evaluators
        )
        safety = tuple(
            observation
            for observation in observations
            if observation.name in {"identity", "outside_locality"}
        )
        support = _effective_blend_support(
            step.latent_mask,
            floor=self.controller.spec.support_floor,
            maximum_compensation=(
                self.controller.spec.maximum_blend_compensation
            ),
        )
        result = self.controller.compute_update(
            latents=clean_latents,
            target=margin,
            drift_loss=drift_loss,
            safety_constraints=safety,
            effective_support=support,
            mode=self.controller_mode,
        )
        skip_reason = result.record.get("update", {}).get("skip_reason")
        clean_norm = _norm(result.clean_delta)
        if skip_reason is not None or clean_norm <= 0:
            return None

        epsilon_delta = clean_delta_to_epsilon_delta(
            result.clean_delta,
            alpha,
        )
        guided_noise = step.noise_pred.detach() + epsilon_delta.to(
            device=step.noise_pred.device,
            dtype=step.noise_pred.dtype,
        )
        with torch.no_grad():
            post_clean_latents = predict_clean_latents(
                step.latents.detach().float(),
                guided_noise.detach().float(),
                alpha,
                prediction_type,
            )
            post_clean_image = decode_clean_latents(
                self.vae,
                post_clean_latents,
            )
            post_margin = target_margin(
                self.target_evaluator.logit(post_clean_image),
                self.desired_value,
                self.target_probability,
            )
            post_drift = self.drift_evaluator.audit(post_clean_image)

        requested_progress = float(
            result.record.get("solver", {}).get(
                "accepted_target_progress",
                result.record.get("solver", {}).get(
                    "requested_target_progress",
                    0.0,
                ),
            )
        )
        actual_progress = margin.residual - post_margin.residual
        self.controller.observe_outcome(
            requested_progress=requested_progress,
            actual_progress=actual_progress,
            step_norm=clean_norm,
        )
        result.record.setdefault("target", {}).update(
            {
                "target_probability": margin.desired_probability,
                "post_update_probability": post_margin.desired_probability,
                "residual_after": post_margin.residual,
                "actual_progress": actual_progress,
            }
        )
        result.record.setdefault("update", {}).update(
            {
                "clean_delta_norm": clean_norm,
                "epsilon_delta_norm": _norm(epsilon_delta),
            }
        )
        constraint_payload = {
            observation.name: {
                "value": float(observation.value.detach().item()),
                "tolerance": observation.tolerance,
                "residual": (
                    float(observation.value.detach().item())
                    - observation.tolerance
                ),
            }
            for observation in observations
        }
        record = {
            "step": step.step_index,
            "timestep": (
                int(step.timestep.item())
                if hasattr(step.timestep, "item")
                else int(step.timestep)
            ),
            "progress": step.progress,
            "prediction_type": prediction_type,
            "alpha_prod_t": alpha_number,
            "constraints": constraint_payload,
            "non_target_drift": {
                "before": self.drift_evaluator.audit(clean_image.detach()),
                "after_immediate": post_drift,
            },
            **result.record,
        }
        self._pending_record = record
        self._retention_reference = step.latents.detach().float().clone()
        self.current_step_active = True
        if self.frame_observer is not None:
            self.frame_observer(
                {
                    "step": step.step_index,
                    "timestep": record["timestep"],
                    "progress": step.progress,
                    "before_image": clean_image.detach(),
                    "after_image": post_clean_image.detach(),
                }
            )
        return guided_noise.detach()

    def observe_retention(self, phase: str, latents: Any) -> None:
        if (
            not self.current_step_active
            or self._pending_record is None
            or self._retention_reference is None
        ):
            return
        if phase not in {"scheduler_step", "blend"}:
            raise ValueError(f"Unknown retention phase: {phase}")
        displacement = (
            latents.detach().float()
            - self._retention_reference.to(
                device=latents.device,
                dtype=latents.dtype,
            )
        )
        self._pending_record["update"][f"{phase}_displacement_norm"] = _norm(
            displacement
        )
        if phase == "blend":
            self.trace_writer.write(self._pending_record)
            self._pending_record = None
            self._retention_reference = None
            self.current_step_active = False

    def _bind_evaluators(self, step: Any, clean_image: Any) -> None:
        import torch

        if self._evaluators_bound:
            return
        with torch.no_grad():
            source_image = decode_clean_latents(
                self.vae,
                step.source_latents.detach(),
            ).detach()
        generation_mask = _mask_for_image(
            step.latent_mask,
            clean_image,
            mode="bilinear",
        )
        semantic_latent = (
            step.semantic_mask
            if step.semantic_mask is not None
            else (step.latent_mask >= 0.5).to(step.latent_mask.dtype)
        )
        semantic_mask = _mask_for_image(
            semantic_latent,
            clean_image,
            mode="nearest",
        )
        context = ConstraintContext(
            source_image,
            generation_mask,
            semantic_mask,
        )
        self.drift_evaluator.bind(context)
        for evaluator in self.constraint_evaluators:
            evaluator.bind(context)
        self._evaluators_bound = True

    def _sample_mps_memory(self, torch: Any) -> None:
        if not hasattr(torch, "mps") or not torch.backends.mps.is_available():
            return
        current = int(torch.mps.current_allocated_memory())
        self.peak_mps_bytes = max(self.peak_mps_bytes or 0, current)


class FinalPreservationTrustRegionHook:
    """Backtracked final restoration that keeps target and safety priorities."""

    apply_after_blend = True

    def __init__(
        self,
        *,
        vae: Any,
        target_evaluator: Any,
        drift_evaluator: Any,
        constraint_evaluators: tuple[Any, ...],
        controller: Any,
        desired_value: int,
        target_probability: float,
        controller_mode: str = "trust_region",
    ) -> None:
        self.vae = vae
        self.target_evaluator = target_evaluator
        self.drift_evaluator = drift_evaluator
        self.constraint_evaluators = constraint_evaluators
        self.controller = controller
        self.desired_value = desired_value
        self.target_probability = target_probability
        self.controller_mode = controller_mode
        self.record: dict[str, Any] | None = None

    def __call__(self, step: Any) -> Any | None:
        import torch

        if (
            step.progress < 1.0 - 1e-9
            or self.controller.spec.final_iterations == 0
        ):
            return None
        safety_evaluators = self._safety_evaluators()
        initial = step.latents.detach().float()
        current = initial
        current_metrics = self._measure(current, safety_evaluators)
        initial_metrics = dict(current_metrics)
        attempts: list[dict[str, Any]] = []
        accepted_steps = 0

        for iteration in range(self.controller.spec.final_iterations):
            guided = current.detach().clone().requires_grad_(True)
            image = decode_clean_latents(self.vae, guided)
            margin = target_margin(
                self.target_evaluator.logit(image),
                self.desired_value,
                self.target_probability,
            )
            drift_loss = self.drift_evaluator.measure(image)
            observations = tuple(
                ConstraintObservation(
                    evaluator.name,
                    evaluator.measure(image),
                    evaluator.tolerance,
                )
                for evaluator in safety_evaluators
            )
            support = _effective_blend_support(
                step.latent_mask,
                floor=self.controller.spec.support_floor,
                maximum_compensation=(
                    self.controller.spec.maximum_blend_compensation
                ),
            )
            result = self.controller.compute_update(
                latents=guided,
                target=margin,
                drift_loss=drift_loss,
                safety_constraints=observations,
                effective_support=support,
                mode=self.controller_mode,
            )
            if result.record.get("update", {}).get("skip_reason") is not None:
                attempts.append(
                    {
                        "iteration": iteration,
                        "accepted": False,
                        "reason": result.record["update"]["skip_reason"],
                    }
                )
                break
            if _norm(result.clean_delta) <= 0:
                attempts.append(
                    {
                        "iteration": iteration,
                        "accepted": False,
                        "reason": "zero_step",
                    }
                )
                break

            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.125):
                candidate = (
                    guided + fraction * result.clean_delta
                ).detach()
                metrics = self._measure(candidate, safety_evaluators)
                cumulative_norm = _norm(candidate - initial)
                target_ok = (
                    metrics["probability"]
                    > current_metrics["probability"] + 1e-7
                    if current_metrics["probability"]
                    < self.target_probability
                    else metrics["probability"] >= self.target_probability
                )
                safety_ok = (
                    metrics["safety_envelope"]
                    <= current_metrics["safety_envelope"]
                    + self.controller.spec.feasibility_tolerance
                )
                drift_ok = (
                    True
                    if (
                        current_metrics["probability"]
                        < self.target_probability
                        or current_metrics["safety_envelope"] > 0
                    )
                    else (
                        metrics["drift"]
                        < current_metrics["drift"] - 1e-8
                    )
                )
                radius_ok = (
                    cumulative_norm
                    <= self.controller.spec.final_cumulative_radius
                )
                reason = (
                    "target"
                    if not target_ok
                    else (
                        "safety_envelope"
                        if not safety_ok
                        else (
                            "non_target_drift"
                            if not drift_ok
                            else (
                                "cumulative_radius"
                                if not radius_ok
                                else "accepted"
                            )
                        )
                    )
                )
                attempt = {
                    "iteration": iteration,
                    "step_fraction": fraction,
                    "accepted": reason == "accepted",
                    "reason": reason,
                    "probability": metrics["probability"],
                    "drift": metrics["drift"],
                    "safety_envelope": metrics["safety_envelope"],
                    "step_norm": _norm(fraction * result.clean_delta),
                    "cumulative_norm": cumulative_norm,
                }
                attempts.append(attempt)
                if reason == "accepted":
                    current = candidate
                    current_metrics = metrics
                    accepted_steps += 1
                    accepted = True
                    break
            if not accepted:
                break

        self.record = {
            "initial_probability": initial_metrics["probability"],
            "final_probability": current_metrics["probability"],
            "required_probability": self.target_probability,
            "initial_drift": initial_metrics["drift"],
            "final_drift": current_metrics["drift"],
            "initial_safety_envelope": initial_metrics["safety_envelope"],
            "final_safety_envelope": current_metrics["safety_envelope"],
            "cumulative_norm": _norm(current - initial),
            "accepted_steps": accepted_steps,
            "attempts": attempts,
        }
        return current if accepted_steps else None

    def _safety_evaluators(self) -> tuple[Any, Any]:
        by_name = {
            evaluator.name: evaluator
            for evaluator in self.constraint_evaluators
            if evaluator.name in {"identity", "outside_locality"}
        }
        if set(by_name) != {"identity", "outside_locality"}:
            raise ValueError(
                "final trust hook requires identity and outside_locality"
            )
        return by_name["identity"], by_name["outside_locality"]

    def _measure(
        self,
        latents: Any,
        safety_evaluators: tuple[Any, Any],
    ) -> dict[str, float]:
        import torch

        with torch.no_grad():
            image = decode_clean_latents(self.vae, latents)
            margin = target_margin(
                self.target_evaluator.logit(image),
                self.desired_value,
                self.target_probability,
            )
            drift = float(self.drift_evaluator.measure(image).item())
            residuals = [
                float(evaluator.measure(image).item()) - evaluator.tolerance
                for evaluator in safety_evaluators
            ]
        return {
            "probability": margin.desired_probability,
            "drift": drift,
            "safety_envelope": max(residuals),
        }


def _effective_blend_support(
    latent_mask: Any,
    *,
    floor: float,
    maximum_compensation: float,
) -> Any:
    import torch

    mask = latent_mask.detach().float()
    return torch.where(
        mask >= floor,
        torch.clamp(
            1.0 / mask.clamp_min(floor),
            max=maximum_compensation,
        )
        * mask,
        torch.zeros_like(mask),
    )


def _mask_for_image(mask: Any, image: Any, *, mode: str) -> Any:
    import torch.nn.functional as functional

    resized = mask.detach().float()
    if resized.ndim >= 3 and image.ndim >= 3:
        kwargs = {"align_corners": False} if mode == "bilinear" else {}
        resized = functional.interpolate(
            resized,
            size=image.shape[-2:],
            mode=mode,
            **kwargs,
        )
    return resized.to(device=image.device, dtype=image.dtype)


def _norm(value: Any) -> float:
    import torch

    return float(torch.linalg.vector_norm(value.detach().float()).item())


class CleanCCIGuidanceHook:
    def __init__(
        self,
        *,
        scheduler: Any,
        vae: Any,
        target_evaluator: Any,
        constraint_evaluators: tuple[Any, ...],
        controller: Any,
        desired_value: int,
        target_probability: float,
        trace_writer: Any,
        frame_observer: Any | None = None,
        controller_mode: str = "feedback",
        project_conflicts: bool = True,
        scheduled_guidance: bool = True,
    ) -> None:
        self.scheduler = scheduler
        self.vae = vae
        self.target_evaluator = target_evaluator
        self.constraint_evaluators = constraint_evaluators
        self.controller = controller
        self.desired_value = desired_value
        self.target_probability = target_probability
        self.trace_writer = trace_writer
        self.frame_observer = frame_observer
        self.controller_mode = controller_mode
        self.project_conflicts = project_conflicts
        self.scheduled_guidance = scheduled_guidance
        self._constraints_bound = False
        self.peak_mps_bytes: int | None = None

    def __call__(self, step: Any) -> Any | None:
        import torch
        import torch.nn.functional as functional

        eta = guidance_eta(
            step.step_index,
            step.progress,
            self.controller.spec,
            scheduled=self.scheduled_guidance,
        )
        if eta is None:
            return None
        self._sample_mps_memory(torch)
        guided_latents = (
            step.latents.detach().float().clone().requires_grad_(True)
        )
        alpha = alpha_prod_for_step(
            self.scheduler,
            step.timestep,
            guided_latents,
        )
        prediction_type = self.scheduler.config.prediction_type
        clean_latents = predict_clean_latents(
            guided_latents,
            step.noise_pred.detach(),
            alpha,
            prediction_type,
        )
        clean_image = decode_clean_latents(self.vae, clean_latents)

        if not self._constraints_bound:
            with torch.no_grad():
                source_image = decode_clean_latents(
                    self.vae,
                    step.source_latents.detach(),
                ).detach()
            generation_mask = functional.interpolate(
                step.latent_mask.detach().float(),
                size=clean_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(device=clean_image.device, dtype=clean_image.dtype)
            semantic_latent = (
                step.semantic_mask
                if step.semantic_mask is not None
                else (step.latent_mask >= 0.5).to(step.latent_mask.dtype)
            )
            semantic_mask = functional.interpolate(
                semantic_latent.detach().float(),
                size=clean_image.shape[-2:],
                mode="nearest",
            ).to(device=clean_image.device, dtype=clean_image.dtype)
            context = ConstraintContext(
                source_image,
                generation_mask,
                semantic_mask,
            )
            for evaluator in self.constraint_evaluators:
                evaluator.bind(context)
            self._constraints_bound = True

        margin = target_margin(
            self.target_evaluator.logit(clean_image),
            self.desired_value,
            self.target_probability,
        )
        observations = tuple(
            ConstraintObservation(
                evaluator.name,
                evaluator.measure(clean_image),
                evaluator.tolerance,
            )
            for evaluator in self.constraint_evaluators
        )
        result = self.controller.compute_update(
            latents=guided_latents,
            target=margin,
            constraints=observations,
            latent_mask=step.latent_mask,
            eta=eta,
            project_conflicts=self.project_conflicts,
            mode=self.controller_mode,
        )
        beta_sqrt = (1.0 - alpha).clamp_min(0.0).sqrt()
        guided_noise = step.noise_pred.detach() + beta_sqrt * result.delta.to(
            device=step.noise_pred.device,
            dtype=step.noise_pred.dtype,
        )
        with torch.no_grad():
            post_clean_latents = predict_clean_latents(
                guided_latents.detach(),
                guided_noise,
                alpha,
                prediction_type,
            )
            post_clean_image = decode_clean_latents(self.vae, post_clean_latents)
            post_margin = target_margin(
                self.target_evaluator.logit(post_clean_image),
                self.desired_value,
                self.target_probability,
            )
        post_probability = post_margin.desired_probability
        result.record["target"]["post_update_probability"] = (
            post_probability if math.isfinite(post_probability) else None
        )
        probability_delta = post_probability - margin.desired_probability
        result.record["target"]["probability_delta"] = (
            probability_delta if math.isfinite(probability_delta) else None
        )
        record = {
            "step": step.step_index,
            "timestep": (
                int(step.timestep.item())
                if hasattr(step.timestep, "item")
                else int(step.timestep)
            ),
            "progress": step.progress,
            "prediction_type": prediction_type,
            "alpha_prod_t": float(alpha.detach().item()),
            **result.record,
        }
        self.trace_writer.write(record)
        if self.frame_observer is not None:
            self.frame_observer(
                {
                    "step": step.step_index,
                    "timestep": record["timestep"],
                    "progress": step.progress,
                    "before_image": clean_image.detach(),
                    "after_image": post_clean_image.detach(),
                }
            )
        return guided_noise.detach()

    def evaluate_image(self, image: Any) -> dict[str, Any]:
        """Measure final feasibility without updating controller state."""

        import torch

        if not self._constraints_bound:
            raise RuntimeError("Clean CCI evaluators are not bound to a source image")
        with torch.no_grad():
            logit = self.target_evaluator.logit(image)
            margin = target_margin(
                logit,
                self.desired_value,
                self.target_probability,
            )
            measured = [
                (evaluator, evaluator.measure(image))
                for evaluator in self.constraint_evaluators
            ]
        target_passed = math.isfinite(margin.residual) and margin.residual <= 0
        failed_names = []
        constraint_payload = {}
        for evaluator, value in measured:
            number = float(value.item())
            passed = math.isfinite(number) and number <= evaluator.tolerance
            if not passed:
                failed_names.append(evaluator.name)
            constraint_payload[evaluator.name] = {
                "value": number if math.isfinite(number) else None,
                "tolerance": evaluator.tolerance,
                "passed": passed,
            }
        logit_number = float(logit.item())
        signed_margin = float(
            margin.signed_logit.item() - margin.required_logit
        )
        return {
            "target": {
                "logit": logit_number if math.isfinite(logit_number) else None,
                "desired_probability": (
                    margin.desired_probability
                    if math.isfinite(margin.desired_probability)
                    else None
                ),
                "required_probability": self.target_probability,
                "signed_margin": (
                    signed_margin if math.isfinite(signed_margin) else None
                ),
                "passed": target_passed,
            },
            "constraints": constraint_payload,
            "feasible": target_passed and not failed_names,
            "failed_constraints": failed_names,
        }

    def _sample_mps_memory(self, torch: Any) -> None:
        if not hasattr(torch, "mps") or not torch.backends.mps.is_available():
            return
        current = int(torch.mps.current_allocated_memory())
        self.peak_mps_bytes = max(self.peak_mps_bytes or 0, current)
