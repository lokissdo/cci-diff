"""Differentiable target and preservation measurements for clean CCI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from cci_diff.classifiers.celeba_resnet50 import classifier_logits


@dataclass(frozen=True)
class ConstraintContext:
    source_image: Any
    generation_mask: Any
    semantic_mask: Any


@dataclass(frozen=True)
class ConstraintObservation:
    name: str
    value: Any
    tolerance: float


class TargetEvaluator(Protocol):
    def logit(self, image: Any) -> Any:
        ...


class ConstraintEvaluator(Protocol):
    name: str
    tolerance: float

    def bind(self, context: ConstraintContext) -> None:
        ...

    def measure(self, image: Any) -> Any:
        ...


class CelebAAttributeTarget:
    def __init__(self, model: Any, attribute_index: int, input_size: int) -> None:
        self.model = model
        self.attribute_index = attribute_index
        self.input_size = input_size

    def logit(self, image: Any) -> Any:
        return classifier_logits(self.model, image, size=self.input_size)[
            :, self.attribute_index
        ].mean()


class CelebAAttributeConstraint:
    def __init__(
        self,
        name: str,
        model: Any,
        attribute_index: int,
        *,
        input_size: int,
        tolerance: float,
    ) -> None:
        self.name = name
        self.model = model
        self.attribute_index = attribute_index
        self.input_size = input_size
        self.tolerance = tolerance
        self._source_probability = None

    def bind(self, context: ConstraintContext) -> None:
        import torch

        with torch.no_grad():
            source_logit = classifier_logits(
                self.model,
                context.source_image,
                size=self.input_size,
            )[:, self.attribute_index]
            self._source_probability = torch.sigmoid(source_logit).mean().detach()

    def measure(self, image: Any) -> Any:
        import torch

        if self._source_probability is None:
            raise RuntimeError(f"Constraint {self.name!r} is not bound to a source")
        logit = classifier_logits(self.model, image, size=self.input_size)[
            :, self.attribute_index
        ]
        return (torch.sigmoid(logit).mean() - self._source_probability).abs()


class OutsideL1Constraint:
    def __init__(self, name: str, tolerance: float) -> None:
        self.name = name
        self.tolerance = tolerance
        self._source = None
        self._outside = None

    def bind(self, context: ConstraintContext) -> None:
        self._source = context.source_image.detach()
        self._outside = 1.0 - _resize_mask(context.generation_mask, self._source)
        if float(self._outside.sum().item()) <= 0:
            raise ValueError("outside_l1 requires at least one outside-mask pixel")

    def measure(self, image: Any) -> Any:
        if self._source is None or self._outside is None:
            raise RuntimeError(f"Constraint {self.name!r} is not bound to a source")
        outside = self._outside.to(device=image.device, dtype=image.dtype)
        source = self._source.to(device=image.device, dtype=image.dtype)
        denominator = outside.sum() * image.shape[1]
        return ((image - source).abs() * outside).sum() / denominator


class MaskedResidualTVConstraint:
    def __init__(self, name: str, tolerance: float) -> None:
        self.name = name
        self.tolerance = tolerance
        self._source = None
        self._semantic = None

    def bind(self, context: ConstraintContext) -> None:
        self._source = context.source_image.detach()
        self._semantic = _resize_mask(context.semantic_mask, self._source)
        if float(self._semantic.sum().item()) <= 0:
            raise ValueError("masked_residual_tv requires a non-empty semantic mask")

    def measure(self, image: Any) -> Any:
        if self._source is None or self._semantic is None:
            raise RuntimeError(f"Constraint {self.name!r} is not bound to a source")
        source = self._source.to(device=image.device, dtype=image.dtype)
        mask = self._semantic.to(device=image.device, dtype=image.dtype)
        residual = (image - source) * mask
        dx_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        dy_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        dx = (residual[:, :, :, 1:] - residual[:, :, :, :-1]).abs() * dx_mask
        dy = (residual[:, :, 1:, :] - residual[:, :, :-1, :]).abs() * dy_mask
        channels = image.shape[1]
        denominator = (dx_mask.sum() + dy_mask.sum()) * channels
        return (dx.sum() + dy.sum()) / denominator.clamp_min(1.0)


def _resize_mask(mask: Any, reference: Any) -> Any:
    import torch.nn.functional as functional

    resized = mask.detach().float()
    if resized.shape[-2:] != reference.shape[-2:]:
        resized = functional.interpolate(
            resized,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return resized.to(
        device=reference.device,
        dtype=reference.dtype,
    ).clamp(0.0, 1.0)
