"""Reviewed evaluator and semantic-mask registrations for concept graphs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from cci_diff.classifiers.celeba_resnet50 import resolve_celeba_attribute_index
from cci_diff.concept_graph import ConceptNode
from cci_diff.region_screening import CELEBAMASK_COMPONENT_SUFFIXES


@dataclass(frozen=True)
class EvaluatorRegistration:
    name: str
    version: str
    roles: frozenset[str]
    differentiable: bool


class ConceptRegistry:
    def __init__(
        self,
        evaluators: Mapping[str, EvaluatorRegistration],
        mask_roles: frozenset[str],
    ) -> None:
        self.evaluators = MappingProxyType(dict(evaluators))
        self.mask_roles = mask_roles

    def resolve_node(
        self,
        node: ConceptNode,
    ) -> tuple[EvaluatorRegistration, int | None]:
        try:
            registration = self.evaluators[node.evaluator]
        except KeyError as exc:
            raise ValueError(f"Unknown evaluator: {node.evaluator}") from exc
        if node.role not in registration.roles:
            raise ValueError(
                f"Evaluator {node.evaluator!r} cannot serve role {node.role!r}"
            )
        if node.evaluator == "celeba_attribute":
            if not node.attribute:
                raise ValueError(f"CelebA node {node.id!r} requires attribute")
            return registration, resolve_celeba_attribute_index(node.attribute)
        reviewed_ids = {
            "facenet_identity": {"identity"},
            "outside_l1": {"outside_locality"},
            "masked_residual_tv": {"residual_tv"},
            "clip_image_audit": {"clip_image_similarity"},
        }[node.evaluator]
        if node.id not in reviewed_ids:
            raise ValueError(
                f"Unknown concept {node.id!r} for evaluator {node.evaluator!r}"
            )
        return registration, None

    def validate_mask_role(self, role: str) -> None:
        if role not in self.mask_roles:
            raise ValueError(f"Unknown semantic mask role: {role}")


def default_concept_registry() -> ConceptRegistry:
    registrations = {
        "celeba_attribute": EvaluatorRegistration(
            "celeba_attribute",
            "celeba-resnet50-v1",
            frozenset({"target", "constraint", "audit_only"}),
            True,
        ),
        "facenet_identity": EvaluatorRegistration(
            "facenet_identity",
            "facenet-vggface2-v1",
            frozenset({"constraint"}),
            True,
        ),
        "outside_l1": EvaluatorRegistration(
            "outside_l1",
            "outside-l1-v1",
            frozenset({"constraint"}),
            True,
        ),
        "masked_residual_tv": EvaluatorRegistration(
            "masked_residual_tv",
            "masked-residual-tv-v1",
            frozenset({"constraint"}),
            True,
        ),
        "clip_image_audit": EvaluatorRegistration(
            "clip_image_audit",
            "open-clip-image-v1",
            frozenset({"audit_only"}),
            False,
        ),
    }
    return ConceptRegistry(
        registrations,
        frozenset(CELEBAMASK_COMPONENT_SUFFIXES) | {"target_region"},
    )
