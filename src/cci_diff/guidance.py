"""Framework-neutral CCI guidance objective helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from cci_diff.spec import GuidanceWeights

Scalar = TypeVar("Scalar")


@dataclass(frozen=True)
class GuidanceTerms:
    """Loss terms used to guide a diffusion denoising step."""

    target: Scalar
    preservation: Scalar
    leakage: Scalar
    classifier: Scalar
    outside_mask: Scalar


def compose_guidance_loss(terms: GuidanceTerms, weights: GuidanceWeights):
    """Return weighted sum of CCI loss terms.

    The function deliberately avoids a concrete tensor type. If terms are torch
    scalar tensors, the returned value remains differentiable.
    """

    return (
        weights.target * terms.target
        + weights.preservation * terms.preservation
        + weights.leakage * terms.leakage
        + weights.classifier * terms.classifier
        + weights.outside_mask * terms.outside_mask
    )
