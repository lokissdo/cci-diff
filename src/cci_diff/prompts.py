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
    target = intervention.target_concept.replace("_", " ").strip().lower()
    visual_descriptions = {
        ("smile", 0): "relaxed neutral facial expression, closed relaxed lips, no smile",
        ("smiling", 0): "relaxed neutral facial expression, closed relaxed lips, no smile",
        ("smile", 1): "natural genuine smile",
        ("smiling", 1): "natural genuine smile",
        ("blond hair", 1): "natural warm blond hair with realistic detailed strands",
    }
    target_instruction = f"{action} {target}"
    description = visual_descriptions.get((target, intervention.desired_value))
    if description:
        target_instruction = f"{target_instruction}: {description}"
    preserved = ", ".join(
        f"preserve {concept}" for concept in intervention.preserved_concepts
    )
    negative_preserved = ", ".join(
        f"do not change {concept}" for concept in intervention.preserved_concepts
    )

    positive_parts = [
        "(photo-realistic:1.2)",
        "ultra-high-resolution portrait of the same person",
        target_instruction,
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
