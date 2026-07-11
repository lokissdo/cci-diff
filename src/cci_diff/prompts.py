"""Prompt helpers for concept-level interventions."""

from __future__ import annotations

from dataclasses import dataclass

from cci_diff.spec import ConceptIntervention


@dataclass(frozen=True)
class ConceptPrompt:
    """Positive and negative prompt pair for a concept intervention."""

    positive: str
    negative: str


def build_concept_prompt(intervention: ConceptIntervention) -> ConceptPrompt:
    """Build a Stable-Diffusion-friendly prompt pair from an intervention spec."""

    action = "add" if intervention.desired_value == 1 else "remove"
    preserved = ", ".join(
        f"preserve {concept}" for concept in intervention.preserved_concepts
    )
    negative_preserved = ", ".join(
        f"do not change {concept}" for concept in intervention.preserved_concepts
    )

    positive_parts = [
        "(photo-realistic:1.2)",
        "ultra-high-resolution portrait of the same person",
        f"{action} {intervention.target_concept}",
    ]
    if preserved:
        positive_parts.append(preserved)
    positive_parts.extend(
        [
            "natural expression",
            "detailed skin texture",
            "soft cinematic lighting",
        ]
    )

    negative_parts = [
        "identity drift",
        "distorted face",
        "unrealistic artifacts",
    ]
    if negative_preserved:
        negative_parts.append(negative_preserved)

    return ConceptPrompt(
        positive=", ".join(positive_parts),
        negative=", ".join(negative_parts),
    )
