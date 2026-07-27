"""Compile a validated JSON graph into an image-bound clean CCI plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageChops

from cci_diff.classifiers.celeba_resnet50 import resolve_celeba_attribute_index
from cci_diff.concept_graph import (
    ConceptGraph,
    ControllerSpec,
    InterventionRequest,
    SampleBindings,
    sha256_file,
)
from cci_diff.concept_registry import ConceptRegistry


@dataclass(frozen=True)
class ResolvedConceptNode:
    id: str
    role: str
    evaluator: str
    evaluator_version: str
    attribute: str | None
    attribute_index: int | None
    tolerance: float | None


@dataclass(frozen=True)
class CompiledCCIPlan:
    graph: ConceptGraph
    graph_path: str
    graph_sha256: str
    source_image: str
    audit_mask_path: str
    component_paths: tuple[tuple[str, str], ...]
    target: ResolvedConceptNode
    constraints: tuple[ResolvedConceptNode, ...]
    audit_only: tuple[ResolvedConceptNode, ...]
    controller: ControllerSpec


class ConceptGraphCompiler(Protocol):
    def compile(
        self,
        request: InterventionRequest,
        bindings: SampleBindings,
        registry: ConceptRegistry,
    ) -> CompiledCCIPlan:
        ...


class JsonConceptGraphCompiler:
    def __init__(self, graph: ConceptGraph, graph_path: str | Path) -> None:
        self.graph = graph
        self.graph_path = Path(graph_path)

    def compile(
        self,
        request: InterventionRequest,
        bindings: SampleBindings,
        registry: ConceptRegistry,
    ) -> CompiledCCIPlan:
        if request != self.graph.intervention:
            raise ValueError("JSON graph intervention is the single source of truth")
        if not self.graph_path.is_file():
            raise FileNotFoundError(f"Concept graph not found: {self.graph_path}")
        source = Path(bindings.source_image)
        if not source.is_file():
            raise FileNotFoundError(f"Source image not found: {source}")

        expected_roles = set(self.graph.region.components) | {
            self.graph.region.audit_role
        }
        supplied_roles = set(bindings.masks)
        missing = sorted(expected_roles - supplied_roles)
        unused = sorted(supplied_roles - expected_roles)
        if missing or unused:
            raise ValueError(
                f"Mask binding mismatch; missing={missing}, unused={unused}"
            )
        for role in expected_roles:
            registry.validate_mask_role(role)
        _validate_mask_files(bindings, expected_roles)

        target_id = next(
            node.id for node in self.graph.nodes if node.role == "target"
        )
        relations = {
            (edge.target, edge.relation)
            for edge in self.graph.edges
            if edge.source == target_id
        }
        for node in self.graph.nodes:
            required_relation = {"constraint": "must_preserve"}.get(node.role)
            if required_relation and (node.id, required_relation) not in relations:
                raise ValueError(
                    f"Node {node.id!r} requires {required_relation!r} from the target"
                )

        resolved = []
        for node in self.graph.nodes:
            registration, attribute_index = registry.resolve_node(node)
            resolved.append(
                ResolvedConceptNode(
                    id=node.id,
                    role=node.role,
                    evaluator=node.evaluator,
                    evaluator_version=registration.version,
                    attribute=node.attribute,
                    attribute_index=attribute_index,
                    tolerance=node.tolerance,
                )
            )
        by_role = {
            role: tuple(node for node in resolved if node.role == role)
            for role in ("target", "constraint", "audit_only")
        }
        target = by_role["target"][0]
        if target.evaluator != "celeba_attribute":
            raise ValueError("Version 1 requires a CelebA attribute target")
        if target.attribute_index != resolve_celeba_attribute_index(request.concept):
            raise ValueError(
                "Intervention concept and target evaluator attribute disagree"
            )
        return CompiledCCIPlan(
            graph=self.graph,
            graph_path=str(self.graph_path),
            graph_sha256=sha256_file(self.graph_path),
            source_image=str(source),
            audit_mask_path=bindings.masks[self.graph.region.audit_role],
            component_paths=tuple(
                (role, bindings.masks[role])
                for role in self.graph.region.components
            ),
            target=target,
            constraints=by_role["constraint"],
            audit_only=by_role["audit_only"],
            controller=self.graph.controller,
        )


def _validate_mask_files(bindings: SampleBindings, roles: set[str]) -> None:
    images = []
    for role in sorted(roles):
        path = Path(bindings.masks[role])
        if not path.is_file():
            raise FileNotFoundError(f"Mask for role {role!r} not found: {path}")
        with Image.open(path) as image:
            images.append(image.convert("L"))
    sizes = {image.size for image in images}
    if len(sizes) != 1:
        raise ValueError("All bound masks must have identical dimensions")
    union = images[0].point(lambda value: 255 if value >= 128 else 0)
    for image in images[1:]:
        union = ImageChops.lighter(
            union,
            image.point(lambda value: 255 if value >= 128 else 0),
        )
    if union.getbbox() is None:
        raise ValueError("Bound mask union must be non-empty")
