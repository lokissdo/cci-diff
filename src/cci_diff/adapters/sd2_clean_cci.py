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


def decode_clean_latents(
    vae: Any,
    clean_latents: Any,
    latent_scale: float = LATENT_SCALE,
) -> Any:
    if latent_scale <= 0:
        raise ValueError("latent_scale must be positive")
    decoded = vae.decode(clean_latents / latent_scale).sample
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
        guided_latents = step.latents.detach().clone().requires_grad_(True)
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
