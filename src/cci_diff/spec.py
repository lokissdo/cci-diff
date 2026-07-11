"""Typed configuration objects for CCI-Diff interventions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConceptIntervention:
    """A single concept-level intervention request."""

    target_concept: str
    desired_value: int
    preserved_concepts: tuple[str, ...] = ()
    candidate_concepts: tuple[str, ...] = ()
    target_mask: str | None = None
    preserve_mask: str | None = None
    audit_concepts: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.desired_value not in (0, 1):
            raise ValueError("desired_value must be 0 or 1")
        if self.target_concept in self.preserved_concepts:
            raise ValueError("target_concept cannot also be preserved")

        audit_concepts = tuple(
            concept
            for concept in self.candidate_concepts
            if concept != self.target_concept and concept not in self.preserved_concepts
        )
        object.__setattr__(self, "audit_concepts", audit_concepts)


@dataclass(frozen=True)
class GuidanceWeights:
    """Relative weights for CCI-Diff guidance losses."""

    target: float = 1.0
    preservation: float = 1.0
    leakage: float = 0.5
    classifier: float = 1.0
    outside_mask: float = 1.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
