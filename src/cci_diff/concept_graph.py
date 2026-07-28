"""Versioned concept-graph and image-binding schema for clean CCI."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


NODE_ROLES = frozenset({"target", "constraint", "audit_only"})
EDGE_RELATIONS = frozenset({"must_preserve", "measured_by"})


@dataclass(frozen=True)
class InterventionRequest:
    concept: str
    desired_value: int
    target_probability: float


@dataclass(frozen=True)
class RegionSpec:
    audit_role: str
    components: tuple[str, ...]
    feather_radius: float


@dataclass(frozen=True)
class ConceptNode:
    id: str
    role: str
    evaluator: str
    attribute: str | None = None
    tolerance: float | None = None


@dataclass(frozen=True)
class ConceptEdge:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class TrustRegionSpec:
    initial_radius: float = 0.15
    minimum_radius: float = 0.01
    maximum_radius: float = 0.30
    target_progress_fraction: float = 0.5
    feasibility_tolerance: float = 0.0001
    reliability_alpha_min: float = 0.10
    huber_delta: float = 0.02
    support_floor: float = 0.05
    maximum_blend_compensation: float = 4.0
    final_cumulative_radius: float = 0.60
    final_iterations: int = 12


DEFAULT_TRUST_REGION_SPEC = TrustRegionSpec()


@dataclass(frozen=True)
class ControllerSpec:
    dual_rate: float
    penalty: float
    lambda_max: float
    step_scale: float
    trust_radius: float
    norm_ema_beta: float
    gradient_floor: float
    active_progress: tuple[float, float]
    every_n_steps: int
    final_corrections: int = 0
    trust_region: TrustRegionSpec = DEFAULT_TRUST_REGION_SPEC
    trust_region_explicit: bool = False


@dataclass(frozen=True)
class ConceptGraph:
    version: int
    intervention: InterventionRequest
    region: RegionSpec
    nodes: tuple[ConceptNode, ...]
    edges: tuple[ConceptEdge, ...]
    controller: ControllerSpec

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical version-one JSON representation."""

        return {
            "version": self.version,
            "intervention": {
                "concept": self.intervention.concept,
                "desired_value": self.intervention.desired_value,
                "target_probability": self.intervention.target_probability,
            },
            "region": {
                "audit_role": self.region.audit_role,
                "components": list(self.region.components),
                "feather_radius": self.region.feather_radius,
            },
            "nodes": [_node_to_dict(node) for node in self.nodes],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                }
                for edge in self.edges
            ],
            "controller": {
                "dual_rate": self.controller.dual_rate,
                "penalty": self.controller.penalty,
                "lambda_max": self.controller.lambda_max,
                "step_scale": self.controller.step_scale,
                "trust_radius": self.controller.trust_radius,
                "norm_ema_beta": self.controller.norm_ema_beta,
                "gradient_floor": self.controller.gradient_floor,
                "active_progress": list(self.controller.active_progress),
                "every_n_steps": self.controller.every_n_steps,
                **(
                    {"final_corrections": self.controller.final_corrections}
                    if self.controller.final_corrections
                    else {}
                ),
                **(
                    {
                        "trust_region": _trust_region_to_dict(
                            self.controller.trust_region
                        )
                    }
                    if (
                        self.controller.trust_region_explicit
                        or self.controller.trust_region
                        != DEFAULT_TRUST_REGION_SPEC
                    )
                    else {}
                ),
            },
        }


@dataclass(frozen=True)
class SampleBindings:
    source_image: str
    masks: Mapping[str, str]


def concept_graph_from_dict(payload: Mapping[str, Any]) -> ConceptGraph:
    """Parse and validate one versioned concept graph payload."""

    intervention_payload = payload["intervention"]
    intervention = InterventionRequest(
        concept=str(intervention_payload["concept"]),
        desired_value=intervention_payload["desired_value"],
        target_probability=float(intervention_payload["target_probability"]),
    )
    region_payload = payload["region"]
    region = RegionSpec(
        audit_role=str(region_payload["audit_role"]),
        components=tuple(str(value) for value in region_payload["components"]),
        feather_radius=float(region_payload["feather_radius"]),
    )
    controller_payload = payload["controller"]
    trust_region_payload = controller_payload.get("trust_region", {})
    controller = ControllerSpec(
        dual_rate=float(controller_payload["dual_rate"]),
        penalty=float(controller_payload["penalty"]),
        lambda_max=float(controller_payload["lambda_max"]),
        step_scale=float(controller_payload["step_scale"]),
        trust_radius=float(controller_payload["trust_radius"]),
        norm_ema_beta=float(controller_payload["norm_ema_beta"]),
        gradient_floor=float(controller_payload["gradient_floor"]),
        active_progress=tuple(
            float(value) for value in controller_payload["active_progress"]
        ),
        every_n_steps=int(controller_payload["every_n_steps"]),
        final_corrections=int(controller_payload.get("final_corrections", 0)),
        trust_region=TrustRegionSpec(
            initial_radius=float(
                trust_region_payload.get(
                    "initial_radius",
                    DEFAULT_TRUST_REGION_SPEC.initial_radius,
                )
            ),
            minimum_radius=float(
                trust_region_payload.get(
                    "minimum_radius",
                    DEFAULT_TRUST_REGION_SPEC.minimum_radius,
                )
            ),
            maximum_radius=float(
                trust_region_payload.get(
                    "maximum_radius",
                    DEFAULT_TRUST_REGION_SPEC.maximum_radius,
                )
            ),
            target_progress_fraction=float(
                trust_region_payload.get(
                    "target_progress_fraction",
                    DEFAULT_TRUST_REGION_SPEC.target_progress_fraction,
                )
            ),
            feasibility_tolerance=float(
                trust_region_payload.get(
                    "feasibility_tolerance",
                    DEFAULT_TRUST_REGION_SPEC.feasibility_tolerance,
                )
            ),
            reliability_alpha_min=float(
                trust_region_payload.get(
                    "reliability_alpha_min",
                    DEFAULT_TRUST_REGION_SPEC.reliability_alpha_min,
                )
            ),
            huber_delta=float(
                trust_region_payload.get(
                    "huber_delta",
                    DEFAULT_TRUST_REGION_SPEC.huber_delta,
                )
            ),
            support_floor=float(
                trust_region_payload.get(
                    "support_floor",
                    DEFAULT_TRUST_REGION_SPEC.support_floor,
                )
            ),
            maximum_blend_compensation=float(
                trust_region_payload.get(
                    "maximum_blend_compensation",
                    DEFAULT_TRUST_REGION_SPEC.maximum_blend_compensation,
                )
            ),
            final_cumulative_radius=float(
                trust_region_payload.get(
                    "final_cumulative_radius",
                    DEFAULT_TRUST_REGION_SPEC.final_cumulative_radius,
                )
            ),
            final_iterations=int(
                trust_region_payload.get(
                    "final_iterations",
                    DEFAULT_TRUST_REGION_SPEC.final_iterations,
                )
            ),
        ),
        trust_region_explicit="trust_region" in controller_payload,
    )
    nodes = tuple(
        ConceptNode(
            id=str(node["id"]),
            role=str(node["role"]),
            evaluator=str(node["evaluator"]),
            attribute=(
                str(node["attribute"])
                if node.get("attribute") is not None
                else None
            ),
            tolerance=(
                float(node["tolerance"])
                if node.get("tolerance") is not None
                else None
            ),
        )
        for node in payload["nodes"]
    )
    edges = tuple(
        ConceptEdge(
            source=str(edge["source"]),
            target=str(edge["target"]),
            relation=str(edge["relation"]),
        )
        for edge in payload["edges"]
    )
    graph = ConceptGraph(
        version=int(payload["version"]),
        intervention=intervention,
        region=region,
        nodes=nodes,
        edges=edges,
        controller=controller,
    )
    validate_concept_graph(graph)
    return graph


def sample_bindings_from_dict(payload: Mapping[str, Any]) -> SampleBindings:
    """Parse image-specific source and semantic-mask paths."""

    source_image = str(payload["source_image"])
    masks = {
        str(role): str(path)
        for role, path in dict(payload["masks"]).items()
    }
    if not source_image:
        raise ValueError("source_image must be non-empty")
    if not masks or any(not role or not path for role, path in masks.items()):
        raise ValueError("masks must contain non-empty role-to-path entries")
    return SampleBindings(source_image, MappingProxyType(masks))


def load_concept_graph(path: str | Path) -> ConceptGraph:
    """Load and validate a concept graph JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return concept_graph_from_dict(payload)


def load_sample_bindings(path: str | Path) -> SampleBindings:
    """Load image-specific bindings from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return sample_bindings_from_dict(payload)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_concept_graph(graph: ConceptGraph) -> None:
    """Reject structurally invalid or semantically ambiguous graphs."""

    if graph.version != 1:
        raise ValueError(f"Unsupported concept graph version: {graph.version}")
    request = graph.intervention
    if not request.concept:
        raise ValueError("intervention concept must be non-empty")
    if isinstance(request.desired_value, bool) or request.desired_value not in (0, 1):
        raise ValueError("desired_value must be 0 or 1")
    if (
        not math.isfinite(request.target_probability)
        or not 0.5 < request.target_probability < 1.0
    ):
        raise ValueError(
            "target_probability must be strictly between 0.5 and 1.0"
        )

    region = graph.region
    if not region.audit_role or not region.components:
        raise ValueError(
            "region requires an audit_role and at least one component"
        )
    if any(not component for component in region.components):
        raise ValueError("region components must be non-empty")
    if len(set(region.components)) != len(region.components):
        raise ValueError("region components must be unique")
    if not math.isfinite(region.feather_radius) or region.feather_radius < 0:
        raise ValueError("feather_radius must be finite and non-negative")

    node_ids = [node.id for node in graph.nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Duplicate node id")
    for node in graph.nodes:
        if node.role not in NODE_ROLES:
            raise ValueError(f"Unknown node role: {node.role}")
        if not node.id or not node.evaluator:
            raise ValueError("Every node needs a valid id, role, and evaluator")
    targets = [node for node in graph.nodes if node.role == "target"]
    if len(targets) != 1:
        raise ValueError("Graph must contain exactly one target node")
    for node in graph.nodes:
        if node.role == "constraint" and (
            node.tolerance is None
            or not math.isfinite(node.tolerance)
            or node.tolerance <= 0
        ):
            raise ValueError(
                f"Constraint {node.id!r} requires a positive tolerance"
            )
        if node.role != "constraint" and node.tolerance is not None:
            raise ValueError(
                f"Only constraint nodes may define tolerance: {node.id!r}"
            )

    known = set(node_ids)
    adjacency = {node_id: [] for node_id in node_ids}
    seen_edges: set[ConceptEdge] = set()
    for edge in graph.edges:
        if edge in seen_edges:
            raise ValueError(f"Duplicate concept edge: {edge}")
        seen_edges.add(edge)
        if edge.relation not in EDGE_RELATIONS:
            raise ValueError(f"Unsupported edge relation: {edge.relation}")
        if edge.source not in known or edge.target not in known:
            raise ValueError("Every edge endpoint must name an existing node")
        adjacency[edge.source].append(edge.target)
    _validate_acyclic(adjacency)
    _validate_controller(graph.controller)


def _node_to_dict(node: ConceptNode) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": node.id,
        "role": node.role,
        "evaluator": node.evaluator,
    }
    if node.attribute is not None:
        payload["attribute"] = node.attribute
    if node.tolerance is not None:
        payload["tolerance"] = node.tolerance
    return payload


def _trust_region_to_dict(spec: TrustRegionSpec) -> dict[str, Any]:
    return {
        "initial_radius": spec.initial_radius,
        "minimum_radius": spec.minimum_radius,
        "maximum_radius": spec.maximum_radius,
        "target_progress_fraction": spec.target_progress_fraction,
        "feasibility_tolerance": spec.feasibility_tolerance,
        "reliability_alpha_min": spec.reliability_alpha_min,
        "huber_delta": spec.huber_delta,
        "support_floor": spec.support_floor,
        "maximum_blend_compensation": spec.maximum_blend_compensation,
        "final_cumulative_radius": spec.final_cumulative_radius,
        "final_iterations": spec.final_iterations,
    }


def _validate_acyclic(adjacency: Mapping[str, list[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("Concept graph must be acyclic")
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency[node]:
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in adjacency:
        visit(node)


def _validate_controller(spec: ControllerSpec) -> None:
    numeric_values = (
        spec.dual_rate,
        spec.penalty,
        spec.lambda_max,
        spec.step_scale,
        spec.trust_radius,
        spec.norm_ema_beta,
        spec.gradient_floor,
        *spec.active_progress,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("controller values must be finite")
    if spec.dual_rate <= 0 or spec.lambda_max <= 0:
        raise ValueError("dual_rate and lambda_max must be positive")
    if spec.penalty < 0 or spec.step_scale <= 0 or spec.trust_radius <= 0:
        raise ValueError(
            "penalty must be non-negative; step_scale and trust_radius "
            "must be positive"
        )
    if not 0 <= spec.norm_ema_beta < 1 or spec.gradient_floor <= 0:
        raise ValueError(
            "norm_ema_beta must be in [0, 1) and gradient_floor must be positive"
        )
    if len(spec.active_progress) != 2:
        raise ValueError("active_progress must contain [start, end]")
    start, end = spec.active_progress
    if not 0 <= start < end <= 1:
        raise ValueError(
            "active_progress must satisfy 0 <= start < end <= 1"
        )
    if spec.every_n_steps <= 0:
        raise ValueError("every_n_steps must be positive")
    if spec.final_corrections < 0:
        raise ValueError("final_corrections must be non-negative")
    _validate_trust_region(spec.trust_region)


def _validate_trust_region(spec: TrustRegionSpec) -> None:
    numeric_values = (
        spec.initial_radius,
        spec.minimum_radius,
        spec.maximum_radius,
        spec.target_progress_fraction,
        spec.feasibility_tolerance,
        spec.reliability_alpha_min,
        spec.huber_delta,
        spec.support_floor,
        spec.maximum_blend_compensation,
        spec.final_cumulative_radius,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("trust_region values must be finite")
    if not (
        0
        < spec.minimum_radius
        <= spec.initial_radius
        <= spec.maximum_radius
    ):
        raise ValueError(
            "trust_region radii must satisfy "
            "0 < minimum_radius <= initial_radius <= maximum_radius"
        )
    if not 0 < spec.target_progress_fraction < 1:
        raise ValueError(
            "target_progress_fraction must be strictly between 0 and 1"
        )
    if spec.feasibility_tolerance <= 0:
        raise ValueError("feasibility_tolerance must be positive")
    if not 0 < spec.reliability_alpha_min < 1:
        raise ValueError(
            "reliability_alpha_min must be strictly between 0 and 1"
        )
    if spec.huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    if not 0 < spec.support_floor <= 1:
        raise ValueError("support_floor must be in (0, 1]")
    if spec.maximum_blend_compensation < 1:
        raise ValueError("maximum_blend_compensation must be at least 1")
    if spec.final_cumulative_radius <= 0:
        raise ValueError("final_cumulative_radius must be positive")
    if isinstance(spec.final_iterations, bool) or spec.final_iterations < 0:
        raise ValueError("final_iterations must be non-negative")
