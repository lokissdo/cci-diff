# Constraint-Feedback Clean CCI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in SD2 CCI mode that evaluates the requested counterfactual on the predicted clean image, generates preservation weights from measured constraint violations, and emits an inspectable concept graph and per-step weight trace.

**Architecture:** A validated JSON concept graph and image-specific bindings compile into a reusable `CompiledCCIPlan`. During selected DDIM steps, a pre-scheduler hook reconstructs `x0_hat`, evaluates a target margin plus hard constraints, composes their gradients with a primal-dual controller and target-priority projection, and modifies the CFG noise prediction before the existing BLD soft blend. Legacy `none`, `latent_color`, and `latent_classifier` modes remain unchanged.

**Tech Stack:** Python 3.10+, PyTorch, torchvision, diffusers DDIM, Pillow, NumPy, OpenCV, a locally exported VGGFace2 FaceNet TorchScript model, optional facenet-pytorch only in an isolated export environment, optional matplotlib, `unittest`, Apple MPS in float32.

## Global Constraints

- Make changes only under `/Users/hung.domodec.com/Documents/my-docs/cci-diff`.
- Do not create commits, stage files, or modify the parent repository history.
- Preserve all existing user changes in the dirty worktree.
- Keep `none`, `latent_color`, `latent_classifier`, and robust-classifier behavior backward compatible.
- Add `clean_constraint` as an opt-in pre-scheduler hook; do not replace a legacy mode.
- Use float32 for clean guidance on MPS; never create float64 MPS tensors.
- Do not backpropagate through the U-Net. Detach the CFG noise prediction before reconstructing `z0_hat`.
- Support scheduler prediction types `epsilon` and `v_prediction`; reject `sample` and unknown values clearly.
- The target classifier margin is mandatory. CLIP remains optional and audit-only in this implementation.
- Graph JSON contains no manual per-loss weights. Runtime constraint coefficients come from normalized violations and dual multipliers.
- Constraint tolerances and global controller settings come only from the graph. Source images and mask paths come only from sample bindings.
- Generation never downloads model weights implicitly. Unit tests use fakes and require neither network access nor large checkpoints.
- Do not install facenet-pytorch's dependency set into `.venv-ml`: release 2.6.0 pins torch/torchvision/NumPy/Pillow versions incompatible with the working torch 2.13 environment. Export FaceNet in an isolated venv and load TorchScript at runtime.
- Keep the authoritative trace as JSONL; CSV and PNG are derived artifacts and must not be required for generation.
- Run MPS experiments sequentially with `batch_size=1`; the initial pilot uses one seed per image.
- Stop before the 15-image pilot if focused tests, the full suite, trace validation, or either one-image mechanism check fails.

---

## File Map

**Create:**

- `src/cci_diff/concept_graph.py`: immutable graph/binding schema, JSON conversion, structural validation, and SHA-256 helpers.
- `src/cci_diff/concept_registry.py`: reviewed evaluator and semantic-mask registrations.
- `src/cci_diff/compilers/__init__.py`: compiler public exports.
- `src/cci_diff/compilers/json_graph.py`: graph-to-runtime-plan compilation and binding/file validation.
- `src/cci_diff/constraints.py`: target and constraint evaluator protocols plus CelebA, locality, and TV evaluators.
- `src/cci_diff/identity/__init__.py`: identity adapter public exports.
- `src/cci_diff/identity/facenet.py`: local TorchScript FaceNet loading, OpenCV one-time face-box detection, differentiable fixed crop, and identity distance.
- `src/cci_diff/constraint_controller.py`: target margin, dual state, norm EMA, conflict projection, trust clipping, and trace-ready step records.
- `src/cci_diff/adapters/sd2_clean_cci.py`: scheduler-aware predicted-clean reconstruction and callable pre-scheduler hook.
- `src/cci_diff/cci_trace.py`: deterministic JSONL writer and trace summary validation.
- `scripts/download_identity_model.py`: explicit VGGFace2 FaceNet acquisition and checksum output.
- `scripts/plot_cci_trace.py`: JSONL-to-CSV/PNG reporting.
- `scripts/run_clean_cci_pilot.py`: eligibility selection and sequential A0-A4 experiment orchestration.
- `examples/graphs/remove_smile_clean_cci.json`: reusable remove-smile graph.
- `examples/graphs/blond_hair_clean_cci.json`: reusable blond-hair graph.
- `examples/bindings/sample_0_mouth.json`: image 0 mouth/lips binding.
- `examples/bindings/sample_0_hair.json`: image 0 hair binding.
- `tests/test_concept_graph.py`, `tests/test_json_graph_compiler.py`, `tests/test_constraints.py`, `tests/test_facenet_identity.py`, `tests/test_constraint_controller.py`, `tests/test_sd2_clean_cci.py`, `tests/test_cci_trace.py`, `tests/test_plot_cci_trace.py`, `tests/test_clean_cci_cli.py`, and `tests/test_clean_cci_pilot.py`.

**Modify:**

- `src/cci_diff/sd2_bld_backend.py`: expose progress and semantic mask in `SD2DenoisingStep`, while retaining both existing hook points.
- `scripts/run_sd2_bld_cci.py`: validate mode-specific CLI sources, compile clean plans, load evaluators, pass the pre-scheduler hook, and extend audit output.
- `pyproject.toml`: add isolated `identity-export` and lazy `plot` optional dependency groups.
- `.gitignore`: ignore the isolated `.venv-facenet-export/` environment.
- `docs/superpowers/specs/2026-07-14-constraint-feedback-clean-cci-design.md`: record the FaceNet export compatibility erratum without changing the accepted algorithm.
- `tests/test_sd2_bld_backend.py`: cover progress/semantic context and legacy-hook compatibility.
- `tests/test_sd2_bld_cli.py`: preserve legacy parser coverage after mode-specific argument validation.

### Task 1: Versioned Concept Graph And Sample Bindings

**Files:**
- Create: `src/cci_diff/concept_graph.py`
- Create: `tests/test_concept_graph.py`

**Interfaces:**
- Consumes: standard-library `dataclasses`, `json`, `hashlib`, `pathlib`, and typing only.
- Produces: `InterventionRequest`, `RegionSpec`, `ConceptNode`, `ConceptEdge`, `ControllerSpec`, `ConceptGraph`, `SampleBindings`, `concept_graph_from_dict(payload)`, `sample_bindings_from_dict(payload)`, `load_concept_graph(path)`, `load_sample_bindings(path)`, and `sha256_file(path)`.

- [x] **Step 1: Write graph parsing and round-trip tests**

Create `tests/test_concept_graph.py` with a complete valid fixture and assertions for tuples, immutable values, and canonical serialization:

```python
import json
import tempfile
import unittest
from pathlib import Path


def valid_graph_payload():
    return {
        "version": 1,
        "intervention": {
            "concept": "Smiling",
            "desired_value": 0,
            "target_probability": 0.8,
        },
        "region": {
            "audit_role": "mouth",
            "components": ["mouth", "upper_lip", "lower_lip"],
            "feather_radius": 3.0,
        },
        "nodes": [
            {
                "id": "smiling",
                "role": "target",
                "evaluator": "celeba_attribute",
                "attribute": "Smiling",
            },
            {
                "id": "identity",
                "role": "constraint",
                "evaluator": "facenet_identity",
                "tolerance": 0.08,
            },
        ],
        "edges": [
            {
                "source": "smiling",
                "target": "identity",
                "relation": "must_preserve",
            }
        ],
        "controller": {
            "dual_rate": 0.2,
            "penalty": 0.5,
            "lambda_max": 4.0,
            "step_scale": 0.2,
            "trust_radius": 0.15,
            "norm_ema_beta": 0.9,
            "gradient_floor": 0.00001,
            "active_progress": [0.15, 0.65],
            "every_n_steps": 2,
        },
    }


class TestConceptGraph(unittest.TestCase):
    def test_parse_and_round_trip_version_one_graph(self):
        from cci_diff.concept_graph import concept_graph_from_dict

        payload = valid_graph_payload()
        graph = concept_graph_from_dict(payload)

        self.assertEqual(graph.version, 1)
        self.assertEqual(graph.intervention.desired_value, 0)
        self.assertEqual(graph.region.components, ("mouth", "upper_lip", "lower_lip"))
        self.assertEqual(graph.controller.active_progress, (0.15, 0.65))
        self.assertEqual(graph.to_dict(), payload)

    def test_load_sample_bindings(self):
        from cci_diff.concept_graph import load_sample_bindings

        payload = {
            "source_image": "data/0.jpg",
            "masks": {"mouth": "data/00000_mouth.png"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "binding.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            bindings = load_sample_bindings(path)

        self.assertEqual(bindings.source_image, "data/0.jpg")
        self.assertEqual(dict(bindings.masks), payload["masks"])

    def test_sha256_file_is_stable(self):
        from cci_diff.concept_graph import sha256_file

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "value.txt"
            path.write_bytes(b"cci")
            first = sha256_file(path)
            second = sha256_file(path)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the parsing tests and verify the expected import failure**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_concept_graph.py' -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'cci_diff.concept_graph'`.

- [x] **Step 3: Implement immutable schema and canonical conversion**

Create `src/cci_diff/concept_graph.py` with these exact public types and converters:

```python
"""Versioned concept-graph and image-binding schema for clean CCI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


NODE_ROLES = frozenset({"target", "allowed_change", "constraint", "audit_only"})
EDGE_RELATIONS = frozenset({"may_affect", "must_preserve", "measured_by"})


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


@dataclass(frozen=True)
class ConceptGraph:
    version: int
    intervention: InterventionRequest
    region: RegionSpec
    nodes: tuple[ConceptNode, ...]
    edges: tuple[ConceptEdge, ...]
    controller: ControllerSpec

    def to_dict(self) -> dict[str, Any]:
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
            "nodes": [
                {
                    key: value
                    for key, value in {
                        "id": node.id,
                        "role": node.role,
                        "evaluator": node.evaluator,
                        "attribute": node.attribute,
                        "tolerance": node.tolerance,
                    }.items()
                    if value is not None
                }
                for node in self.nodes
            ],
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
            },
        }


@dataclass(frozen=True)
class SampleBindings:
    source_image: str
    masks: Mapping[str, str]


def concept_graph_from_dict(payload: Mapping[str, Any]) -> ConceptGraph:
    intervention = InterventionRequest(**payload["intervention"])
    region_payload = payload["region"]
    region = RegionSpec(
        audit_role=region_payload["audit_role"],
        components=tuple(region_payload["components"]),
        feather_radius=float(region_payload["feather_radius"]),
    )
    controller_payload = payload["controller"]
    controller = ControllerSpec(
        dual_rate=float(controller_payload["dual_rate"]),
        penalty=float(controller_payload["penalty"]),
        lambda_max=float(controller_payload["lambda_max"]),
        step_scale=float(controller_payload["step_scale"]),
        trust_radius=float(controller_payload["trust_radius"]),
        norm_ema_beta=float(controller_payload["norm_ema_beta"]),
        gradient_floor=float(controller_payload["gradient_floor"]),
        active_progress=tuple(controller_payload["active_progress"]),
        every_n_steps=int(controller_payload["every_n_steps"]),
    )
    graph = ConceptGraph(
        version=int(payload["version"]),
        intervention=intervention,
        region=region,
        nodes=tuple(ConceptNode(**node) for node in payload["nodes"]),
        edges=tuple(ConceptEdge(**edge) for edge in payload["edges"]),
        controller=controller,
    )
    validate_concept_graph(graph)
    return graph


def sample_bindings_from_dict(payload: Mapping[str, Any]) -> SampleBindings:
    source_image = str(payload["source_image"])
    masks = {str(role): str(path) for role, path in payload["masks"].items()}
    if not source_image:
        raise ValueError("source_image must be non-empty")
    if not masks or any(not role or not path for role, path in masks.items()):
        raise ValueError("masks must contain non-empty role-to-path entries")
    return SampleBindings(source_image, MappingProxyType(masks))


def load_concept_graph(path: str | Path) -> ConceptGraph:
    return concept_graph_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_sample_bindings(path: str | Path) -> SampleBindings:
    return sample_bindings_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

- [x] **Step 4: Add exact graph validation tests**

Append tests that mutate the valid payload and assert exact failures:

```python
    def test_rejects_invalid_version_target_and_controller_bounds(self):
        from cci_diff.concept_graph import concept_graph_from_dict

        cases = (
            (("version",), 2, "Unsupported concept graph version"),
            (("intervention", "desired_value"), 2, "desired_value must be 0 or 1"),
            (("intervention", "target_probability"), 0.5, "target_probability"),
            (("controller", "every_n_steps"), 0, "every_n_steps must be positive"),
            (("controller", "active_progress"), [0.7, 0.2], "active_progress"),
        )
        for keys, value, message in cases:
            payload = valid_graph_payload()
            cursor = payload
            for key in keys[:-1]:
                cursor = cursor[key]
            cursor[keys[-1]] = value
            with self.subTest(keys=keys):
                with self.assertRaisesRegex(ValueError, message):
                    concept_graph_from_dict(payload)

    def test_rejects_cycle_duplicate_ids_and_role_conflict(self):
        from cci_diff.concept_graph import concept_graph_from_dict

        duplicate = valid_graph_payload()
        duplicate["nodes"].append(dict(duplicate["nodes"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate node id"):
            concept_graph_from_dict(duplicate)

        cycle = valid_graph_payload()
        cycle["edges"].append(
            {"source": "identity", "target": "smiling", "relation": "measured_by"}
        )
        with self.assertRaisesRegex(ValueError, "acyclic"):
            concept_graph_from_dict(cycle)

        conflict = valid_graph_payload()
        conflict["edges"].append(
            {"source": "smiling", "target": "identity", "relation": "may_affect"}
        )
        with self.assertRaisesRegex(ValueError, "both may_affect and must_preserve"):
            concept_graph_from_dict(conflict)
```

- [x] **Step 5: Implement structural and semantic validation**

Add these helpers to `concept_graph.py` and call `validate_concept_graph` as shown above:

```python
def validate_concept_graph(graph: ConceptGraph) -> None:
    if graph.version != 1:
        raise ValueError(f"Unsupported concept graph version: {graph.version}")
    request = graph.intervention
    if not request.concept:
        raise ValueError("intervention concept must be non-empty")
    if isinstance(request.desired_value, bool) or request.desired_value not in (0, 1):
        raise ValueError("desired_value must be 0 or 1")
    if not 0.5 < request.target_probability < 1.0:
        raise ValueError("target_probability must be strictly between 0.5 and 1.0")
    if not graph.region.audit_role or not graph.region.components:
        raise ValueError("region requires an audit_role and at least one component")
    if len(set(graph.region.components)) != len(graph.region.components):
        raise ValueError("region components must be unique")
    if graph.region.feather_radius < 0:
        raise ValueError("feather_radius must be non-negative")

    node_ids = [node.id for node in graph.nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Duplicate node id")
    if any(not node.id or node.role not in NODE_ROLES or not node.evaluator for node in graph.nodes):
        raise ValueError("Every node needs a valid id, role, and evaluator")
    targets = [node for node in graph.nodes if node.role == "target"]
    if len(targets) != 1:
        raise ValueError("Graph must contain exactly one target node")
    for node in graph.nodes:
        if node.role == "constraint" and (node.tolerance is None or node.tolerance <= 0):
            raise ValueError(f"Constraint {node.id!r} requires a positive tolerance")
        if node.role != "constraint" and node.tolerance is not None:
            raise ValueError(f"Only constraint nodes may define tolerance: {node.id!r}")

    known = set(node_ids)
    relations_by_pair: dict[tuple[str, str], set[str]] = {}
    adjacency = {node_id: [] for node_id in node_ids}
    for edge in graph.edges:
        if edge.relation not in EDGE_RELATIONS:
            raise ValueError(f"Unsupported edge relation: {edge.relation}")
        if edge.source not in known or edge.target not in known:
            raise ValueError("Every edge endpoint must name an existing node")
        relations_by_pair.setdefault((edge.source, edge.target), set()).add(edge.relation)
        adjacency[edge.source].append(edge.target)
    for pair, relations in relations_by_pair.items():
        if {"may_affect", "must_preserve"}.issubset(relations):
            raise ValueError(f"Edge {pair!r} is both may_affect and must_preserve")
    _validate_acyclic(adjacency)
    _validate_controller(graph.controller)


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
    if spec.dual_rate <= 0 or spec.lambda_max <= 0:
        raise ValueError("dual_rate and lambda_max must be positive")
    if spec.penalty < 0 or spec.step_scale <= 0 or spec.trust_radius <= 0:
        raise ValueError("penalty must be non-negative; step_scale and trust_radius must be positive")
    if not 0 <= spec.norm_ema_beta < 1 or spec.gradient_floor <= 0:
        raise ValueError("norm_ema_beta must be in [0, 1) and gradient_floor must be positive")
    if len(spec.active_progress) != 2:
        raise ValueError("active_progress must contain [start, end]")
    start, end = spec.active_progress
    if not 0 <= start < end <= 1:
        raise ValueError("active_progress must satisfy 0 <= start < end <= 1")
    if spec.every_n_steps <= 0:
        raise ValueError("every_n_steps must be positive")
```

- [x] **Step 6: Run the complete schema tests**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_concept_graph.py' -v
```

Expected: all graph tests PASS.

- [x] **Step 7: Checkpoint without committing**

Run:

```bash
git diff --check
rg -n '[[:blank:]]+$' src/cci_diff/concept_graph.py tests/test_concept_graph.py
```

Expected: both commands print no diagnostics. Do not stage or commit.

### Task 2: Reviewed Registry And JSON Graph Compiler

**Files:**
- Create: `src/cci_diff/concept_registry.py`
- Create: `src/cci_diff/compilers/__init__.py`
- Create: `src/cci_diff/compilers/json_graph.py`
- Create: `tests/test_json_graph_compiler.py`

**Interfaces:**
- Consumes: Task 1 graph objects and `resolve_celeba_attribute_index(concept: str) -> int`.
- Produces: `EvaluatorRegistration`, `ConceptRegistry`, `default_concept_registry()`, `ResolvedConceptNode`, `CompiledCCIPlan`, `ConceptGraphCompiler`, and `JsonConceptGraphCompiler(graph, graph_path).compile(request, bindings, registry) -> CompiledCCIPlan`.

- [x] **Step 1: Write compiler success and registry-resolution tests**

Create `tests/test_json_graph_compiler.py` with temporary real image/mask files so file validation is exercised without model loading:

```python
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from test_concept_graph import valid_graph_payload


class TestJsonGraphCompiler(unittest.TestCase):
    def make_files(self, root: Path):
        source = root / "source.png"
        Image.new("RGB", (8, 8), "gray").save(source)
        masks = {}
        for role in ("mouth", "upper_lip", "lower_lip"):
            path = root / f"{role}.png"
            image = Image.new("L", (8, 8), 0)
            image.putpixel((3, 3), 255)
            image.save(path)
            masks[role] = str(path)
        return source, masks

    def test_compiles_resolved_target_constraints_and_masks(self):
        from cci_diff.compilers.json_graph import JsonConceptGraphCompiler
        from cci_diff.concept_graph import concept_graph_from_dict, sample_bindings_from_dict
        from cci_diff.concept_registry import default_concept_registry

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            graph_path = root / "graph.json"
            payload = valid_graph_payload()
            graph_path.write_text(json.dumps(payload), encoding="utf-8")
            graph = concept_graph_from_dict(payload)
            bindings = sample_bindings_from_dict(
                {"source_image": str(source), "masks": masks}
            )
            plan = JsonConceptGraphCompiler(graph, graph_path).compile(
                graph.intervention,
                bindings,
                default_concept_registry(),
            )

        self.assertEqual(plan.target.id, "smiling")
        self.assertEqual(plan.target.attribute_index, 31)
        self.assertEqual(plan.constraints[0].id, "identity")
        self.assertEqual(plan.component_paths[0][0], "mouth")
        self.assertEqual(len(plan.graph_sha256), 64)
```

- [x] **Step 2: Run compiler test and verify the expected import failure**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_json_graph_compiler.py' -v
```

Expected: FAIL because `cci_diff.compilers.json_graph` does not exist.

- [x] **Step 3: Implement the reviewed evaluator and mask registry**

Create `src/cci_diff/concept_registry.py`:

```python
"""Reviewed evaluator and semantic-mask registrations for concept graphs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from cci_diff.classifiers.celeba_resnet50 import resolve_celeba_attribute_index
from cci_diff.concept_graph import ConceptNode


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

    def resolve_node(self, node: ConceptNode) -> tuple[EvaluatorRegistration, int | None]:
        try:
            registration = self.evaluators[node.evaluator]
        except KeyError as exc:
            raise ValueError(f"Unknown evaluator: {node.evaluator}") from exc
        if node.role not in registration.roles:
            raise ValueError(f"Evaluator {node.evaluator!r} cannot serve role {node.role!r}")
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
            raise ValueError(f"Unknown concept {node.id!r} for evaluator {node.evaluator!r}")
        return registration, None

    def validate_mask_role(self, role: str) -> None:
        if role not in self.mask_roles:
            raise ValueError(f"Unknown semantic mask role: {role}")


def default_concept_registry() -> ConceptRegistry:
    registrations = {
        "celeba_attribute": EvaluatorRegistration(
            "celeba_attribute",
            "celeba-resnet50-v1",
            frozenset({"target", "allowed_change", "constraint", "audit_only"}),
            True,
        ),
        "facenet_identity": EvaluatorRegistration(
            "facenet_identity", "facenet-vggface2-v1", frozenset({"constraint"}), True
        ),
        "outside_l1": EvaluatorRegistration(
            "outside_l1", "outside-l1-v1", frozenset({"constraint"}), True
        ),
        "masked_residual_tv": EvaluatorRegistration(
            "masked_residual_tv", "masked-residual-tv-v1", frozenset({"constraint"}), True
        ),
        "clip_image_audit": EvaluatorRegistration(
            "clip_image_audit", "open-clip-image-v1", frozenset({"audit_only"}), False
        ),
    }
    return ConceptRegistry(
        registrations,
        frozenset({"mouth", "upper_lip", "lower_lip", "hair"}),
    )
```

- [x] **Step 4: Implement the compiler protocol, compiled types, and file checks**

Create `src/cci_diff/compilers/json_graph.py` with these public signatures and validations:

```python
"""Compile a validated JSON graph into an image-bound clean CCI plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageChops

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
    allowed_changes: tuple[ResolvedConceptNode, ...]
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

        expected_roles = set(self.graph.region.components) | {self.graph.region.audit_role}
        supplied_roles = set(bindings.masks)
        missing = sorted(expected_roles - supplied_roles)
        unused = sorted(supplied_roles - expected_roles)
        if missing or unused:
            raise ValueError(f"Mask binding mismatch; missing={missing}, unused={unused}")
        for role in expected_roles:
            registry.validate_mask_role(role)
        _validate_mask_files(bindings, expected_roles)

        target_id = next(node.id for node in self.graph.nodes if node.role == "target")
        relations = {
            (edge.target, edge.relation)
            for edge in self.graph.edges
            if edge.source == target_id
        }
        for node in self.graph.nodes:
            required_relation = {
                "allowed_change": "may_affect",
                "constraint": "must_preserve",
            }.get(node.role)
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
            for role in ("target", "allowed_change", "constraint", "audit_only")
        }
        target = by_role["target"][0]
        if target.evaluator != "celeba_attribute":
            raise ValueError("Version 1 requires a CelebA attribute target")
        from cci_diff.classifiers.celeba_resnet50 import resolve_celeba_attribute_index
        if target.attribute_index != resolve_celeba_attribute_index(request.concept):
            raise ValueError("Intervention concept and target evaluator attribute disagree")
        return CompiledCCIPlan(
            graph=self.graph,
            graph_path=str(self.graph_path),
            graph_sha256=sha256_file(self.graph_path),
            source_image=str(source),
            audit_mask_path=bindings.masks[self.graph.region.audit_role],
            component_paths=tuple(
                (role, bindings.masks[role]) for role in self.graph.region.components
            ),
            target=target,
            allowed_changes=by_role["allowed_change"],
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
        images.append(Image.open(path).convert("L"))
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
```

Create `src/cci_diff/compilers/__init__.py`:

```python
"""Concept-graph compiler exports."""

from cci_diff.compilers.json_graph import (
    CompiledCCIPlan,
    ConceptGraphCompiler,
    JsonConceptGraphCompiler,
    ResolvedConceptNode,
)

__all__ = [
    "CompiledCCIPlan",
    "ConceptGraphCompiler",
    "JsonConceptGraphCompiler",
    "ResolvedConceptNode",
]
```

- [x] **Step 5: Add rejection tests for unknown concepts and binding failures**

Add tests for unknown evaluator, unknown mask role, missing/unused role, missing file, mismatched dimensions, and empty union. Use the `make_files` helper and assert the exact leading message from the implementation, for example:

```python
    def test_rejects_missing_and_unused_bindings(self):
        from cci_diff.compilers.json_graph import JsonConceptGraphCompiler
        from cci_diff.concept_graph import concept_graph_from_dict, sample_bindings_from_dict
        from cci_diff.concept_registry import default_concept_registry

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            graph_path = root / "graph.json"
            payload = valid_graph_payload()
            graph_path.write_text(json.dumps(payload), encoding="utf-8")
            graph = concept_graph_from_dict(payload)
            masks.pop("upper_lip")
            masks["hair"] = masks["mouth"]
            bindings = sample_bindings_from_dict(
                {"source_image": str(source), "masks": masks}
            )
            with self.assertRaisesRegex(ValueError, "missing=.*upper_lip.*unused=.*hair"):
                JsonConceptGraphCompiler(graph, graph_path).compile(
                    graph.intervention,
                    bindings,
                    default_concept_registry(),
                )
```

Add one determinism test that writes the same canonical payload to two graph paths and compares plans after removing path provenance:

```python
    def test_equivalent_json_compiles_to_equivalent_runtime_plan(self):
        from dataclasses import replace
        from cci_diff.compilers.json_graph import JsonConceptGraphCompiler
        from cci_diff.concept_graph import concept_graph_from_dict, sample_bindings_from_dict
        from cci_diff.concept_registry import default_concept_registry

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            payload = valid_graph_payload()
            plans = []
            for name in ("first.json", "second.json"):
                path = root / name
                path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                graph = concept_graph_from_dict(payload)
                plan = JsonConceptGraphCompiler(graph, path).compile(
                    graph.intervention,
                    sample_bindings_from_dict({"source_image": str(source), "masks": masks}),
                    default_concept_registry(),
                )
                plans.append(replace(plan, graph_path="", graph_sha256=""))

        self.assertEqual(plans[0], plans[1])
```

- [x] **Step 6: Run compiler and schema tests**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_*graph*.py' -v
```

Expected: all graph and compiler tests PASS.

- [x] **Step 7: Checkpoint without committing**

Run:

```bash
git diff --check
rg -n '[[:blank:]]+$' src/cci_diff/concept_registry.py src/cci_diff/compilers tests/test_json_graph_compiler.py
```

Expected: no diagnostics. Do not stage or commit.

### Task 3: Differentiable Target, Locality, Attribute, And TV Evaluators

**Files:**
- Create: `src/cci_diff/constraints.py`
- Create: `tests/test_constraints.py`

**Interfaces:**
- Consumes: `classifier_logits(model, images, size)` and Task 2 `ResolvedConceptNode` descriptions.
- Produces: `ConstraintContext`, `TargetEvaluator`, `ConstraintEvaluator`, `CelebAAttributeTarget`, `CelebAAttributeConstraint`, `OutsideL1Constraint`, `MaskedResidualTVConstraint`, and `ConstraintObservation`.

- [x] **Step 1: Write evaluator tests with a differentiable fake classifier**

Create `tests/test_constraints.py`:

```python
import unittest


class MeanClassifier:
    def forward_logits(self, images):
        import torch

        logits = torch.zeros((images.shape[0], 40), device=images.device, dtype=images.dtype)
        logits[:, 31] = images.mean(dim=(1, 2, 3))
        logits[:, 21] = images[:, :, :1, :].mean(dim=(1, 2, 3))
        return logits


class TestConstraintEvaluators(unittest.TestCase):
    def test_target_returns_differentiable_mean_logit(self):
        import torch
        from cci_diff.constraints import CelebAAttributeTarget

        image = torch.full((1, 3, 4, 4), 0.5, requires_grad=True)
        target = CelebAAttributeTarget(MeanClassifier(), attribute_index=31, input_size=4)
        logit = target.logit(image)
        logit.backward()

        self.assertAlmostEqual(logit.item(), 0.5)
        self.assertIsNotNone(image.grad)

    def test_outside_l1_ignores_changes_inside_generation_mask(self):
        import torch
        from cci_diff.constraints import ConstraintContext, OutsideL1Constraint

        source = torch.zeros((1, 3, 2, 2))
        generation_mask = torch.zeros((1, 1, 2, 2))
        generation_mask[:, :, 0, 0] = 1.0
        context = ConstraintContext(source, generation_mask, generation_mask)
        evaluator = OutsideL1Constraint("outside_locality", tolerance=0.02)
        evaluator.bind(context)
        inside_change = source.clone()
        inside_change[:, :, 0, 0] = 1.0
        outside_change = source.clone()
        outside_change[:, :, 1, 1] = 1.0

        self.assertEqual(evaluator.measure(inside_change).item(), 0.0)
        self.assertGreater(evaluator.measure(outside_change).item(), 0.0)

    def test_attribute_constraint_measures_source_probability_drift(self):
        import torch
        from cci_diff.constraints import (
            CelebAAttributeConstraint,
            ConstraintContext,
        )

        source = torch.zeros((1, 3, 4, 4))
        mask = torch.ones((1, 1, 4, 4))
        evaluator = CelebAAttributeConstraint(
            "mouth_open", MeanClassifier(), 21, input_size=4, tolerance=0.1
        )
        evaluator.bind(ConstraintContext(source, mask, mask))

        self.assertEqual(evaluator.measure(source).item(), 0.0)
        self.assertGreater(evaluator.measure(torch.ones_like(source)).item(), 0.0)

    def test_masked_residual_tv_penalizes_non_smooth_edit_only_inside_semantic_mask(self):
        import torch
        from cci_diff.constraints import ConstraintContext, MaskedResidualTVConstraint

        source = torch.zeros((1, 3, 3, 3))
        semantic = torch.ones((1, 1, 3, 3))
        evaluator = MaskedResidualTVConstraint("residual_tv", tolerance=0.015)
        evaluator.bind(ConstraintContext(source, semantic, semantic))
        checker = source.clone()
        checker[:, :, 1, 1] = 1.0

        self.assertEqual(evaluator.measure(source).item(), 0.0)
        self.assertGreater(evaluator.measure(checker).item(), 0.0)
```

- [x] **Step 2: Run evaluator tests and verify the expected import failure**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_constraints.py' -v
```

Expected: FAIL because `cci_diff.constraints` does not exist.

- [x] **Step 3: Implement evaluator protocols and concrete measurements**

Create `src/cci_diff/constraints.py`:

```python
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
                self.model, context.source_image, size=self.input_size
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
    return resized.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)
```

- [x] **Step 4: Add a no-parameter-gradient contract test**

Add a tiny `torch.nn.Module` fake with a frozen parameter, backpropagate the target logit, and assert `image.grad is not None` while every model parameter has `grad is None`. This guards the intended gradient boundary for later MPS integration.

```python
    def test_frozen_evaluator_passes_gradient_only_to_image(self):
        import torch
        from cci_diff.constraints import CelebAAttributeTarget

        class FrozenClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

            def forward_logits(self, images):
                logits = torch.zeros((images.shape[0], 40), device=images.device)
                return logits + images.mean() * self.scale

        model = FrozenClassifier()
        image = torch.ones((1, 3, 2, 2), requires_grad=True)
        CelebAAttributeTarget(model, 31, 2).logit(image).backward()

        self.assertIsNotNone(image.grad)
        self.assertIsNone(model.scale.grad)
```

- [x] **Step 5: Run evaluator tests**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_constraints.py' -v
```

Expected: all evaluator tests PASS.

- [x] **Step 6: Checkpoint without committing**

Run:

```bash
git diff --check
rg -n '[[:blank:]]+$' src/cci_diff/constraints.py tests/test_constraints.py
```

Expected: no diagnostics. Do not stage or commit.

### Task 4: Explicit FaceNet Identity Adapter And Acquisition Command

**Files:**
- Create: `src/cci_diff/identity/__init__.py`
- Create: `src/cci_diff/identity/facenet.py`
- Create: `scripts/download_identity_model.py`
- Create: `tests/test_facenet_identity.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Modify: `docs/superpowers/specs/2026-07-14-constraint-feedback-clean-cci-design.md`

**Interfaces:**
- Consumes: Task 3 `ConstraintContext`; a local VGGFace2 FaceNet TorchScript export.
- Produces: `load_facenet_identity(checkpoint_path, device)`, `load_identity_export_manifest(checkpoint_path)`, `build_face_detector()`, `detect_largest_face_box(detector, source_image)`, `fixed_face_crop(images, box, size=160)`, and `FaceNetIdentityConstraint`.

- [x] **Step 1: Add an isolated export dependency group and lazy plotting group**

Add these groups under `[project.optional-dependencies]` in `pyproject.toml`; `identity-export` is installed only in `.venv-facenet-export`, never in `.venv-ml`:

```toml
identity-export = [
  "facenet-pytorch==2.6.0",
]
plot = [
  "matplotlib>=3.8",
]
```

Add `.venv-facenet-export/` to `.gitignore` without changing any existing user entries.

Append an implementation compatibility note to the accepted design: facenet-pytorch 2.6.0 pins torch `<2.3`, torchvision `<0.18`, NumPy `<2`, and Pillow `<10.3`, while `.venv-ml` uses torch 2.13, torchvision 0.28, NumPy 2.2, and Pillow 12.3. State that the algorithm is unchanged, but acquisition occurs in `.venv-facenet-export`; runtime loads the exported VGGFace2 TorchScript model and uses the already-installed OpenCV detector.

- [x] **Step 2: Write identity tests using fake detector and embedder**

Create `tests/test_facenet_identity.py`:

```python
import tempfile
import unittest
from pathlib import Path


class FakeDetector:
    def detect(self, image):
        import numpy as np

        return np.array([[1.0, 1.0, 7.0, 7.0]]), np.array([0.99])


class MeanEmbedder:
    def __call__(self, images):
        import torch

        mean = images.mean(dim=(2, 3))
        return torch.nn.functional.normalize(mean + 0.01, dim=1)


class TestFaceNetIdentity(unittest.TestCase):
    def test_fixed_crop_is_differentiable_and_has_stable_size(self):
        import torch
        from cci_diff.identity.facenet import fixed_face_crop

        image = torch.rand((1, 3, 8, 8), requires_grad=True)
        crop = fixed_face_crop(image, (1, 1, 7, 7), size=4)
        crop.sum().backward()

        self.assertEqual(tuple(crop.shape), (1, 3, 4, 4))
        self.assertIsNotNone(image.grad)

    def test_identity_distance_is_zero_for_source_and_positive_after_change(self):
        import torch
        from cci_diff.constraints import ConstraintContext
        from cci_diff.identity.facenet import FaceNetIdentityConstraint

        source = torch.zeros((1, 3, 8, 8))
        mask = torch.ones((1, 1, 8, 8))
        evaluator = FaceNetIdentityConstraint(
            "identity",
            MeanEmbedder(),
            FakeDetector(),
            tolerance=0.08,
            crop_size=4,
        )
        evaluator.bind(ConstraintContext(source, mask, mask))

        self.assertAlmostEqual(evaluator.measure(source).item(), 0.0, places=6)
        changed = source.clone()
        changed[:, 0] = 1.0
        self.assertGreater(evaluator.measure(changed).item(), 0.0)

    def test_missing_checkpoint_fails_without_downloading(self):
        from cci_diff.identity.facenet import load_facenet_identity

        with self.assertRaisesRegex(FileNotFoundError, "Identity checkpoint not found"):
            load_facenet_identity("missing-facenet.ts", device="cpu")

    def test_export_manifest_verifies_checkpoint_digest(self):
        import hashlib
        import json
        from cci_diff.identity.facenet import load_identity_export_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "facenet.ts"
            checkpoint.write_bytes(b"model")
            Path(str(checkpoint) + ".json").write_text(
                json.dumps({
                    "facenet_pytorch_version": "2.6.0",
                    "export_torch_version": "2.2.0",
                    "sha256": hashlib.sha256(b"model").hexdigest(),
                }),
                encoding="utf-8",
            )
            manifest = load_identity_export_manifest(checkpoint)
            checkpoint.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_identity_export_manifest(checkpoint)

        self.assertEqual(manifest["facenet_pytorch_version"], "2.6.0")
```

- [x] **Step 3: Run identity tests and verify the expected import failure**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_facenet_identity.py' -v
```

Expected: FAIL because `cci_diff.identity.facenet` does not exist.

- [x] **Step 4: Implement local loading, one-time detection, and differentiable cropping**

Create `src/cci_diff/identity/facenet.py`:

```python
"""FaceNet identity constraint with a local TorchScript model and fixed crop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cci_diff.concept_graph import sha256_file
from cci_diff.constraints import ConstraintContext


def load_facenet_identity(checkpoint_path: str | Path, *, device: str) -> Any:
    import torch

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Identity checkpoint not found: {path}")
    model = torch.jit.load(str(path), map_location="cpu")
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return model


def load_identity_export_manifest(checkpoint_path: str | Path) -> dict[str, Any]:
    import json

    checkpoint = Path(checkpoint_path)
    manifest_path = Path(str(checkpoint) + ".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Identity export manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"facenet_pytorch_version", "export_torch_version", "sha256"}
    if not required.issubset(payload):
        raise ValueError("Identity export manifest is missing required provenance")
    if payload["sha256"] != sha256_file(checkpoint):
        raise ValueError("Identity TorchScript digest does not match its export manifest")
    return payload


class OpenCVHaarDetector:
    def __init__(self) -> None:
        import cv2

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError(f"Cannot load OpenCV face cascade: {cascade_path}")

    def detect(self, image: Any):
        import cv2
        import numpy as np

        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return None, None
        boxes = np.asarray(
            [[x, y, x + width, y + height] for x, y, width, height in faces],
            dtype=np.float32,
        )
        return boxes, np.ones((len(boxes),), dtype=np.float32)


def build_face_detector() -> Any:
    return OpenCVHaarDetector()


def detect_largest_face_box(detector: Any, source_image: Any) -> tuple[int, int, int, int]:
    from PIL import Image

    image = source_image.detach()[0].clamp(0, 1).mul(255).byte().cpu()
    pil = Image.fromarray(image.permute(1, 2, 0).numpy())
    boxes, probabilities = detector.detect(pil)
    if boxes is None or len(boxes) == 0:
        raise ValueError("FaceNet identity could not detect a source face")
    height, width = source_image.shape[-2:]
    candidates = []
    for box, probability in zip(boxes, probabilities):
        x1, y1, x2, y2 = [float(value) for value in box]
        area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
        candidates.append((float(probability), area, x1, y1, x2, y2))
    _, _, x1, y1, x2, y2 = max(candidates)
    margin = 0.15 * max(x2 - x1, y2 - y1)
    return (
        max(0, int(round(x1 - margin))),
        max(0, int(round(y1 - margin))),
        min(width, int(round(x2 + margin))),
        min(height, int(round(y2 + margin))),
    )


def fixed_face_crop(images: Any, box: tuple[int, int, int, int], *, size: int = 160) -> Any:
    import torch.nn.functional as functional

    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid fixed face box: {box}")
    crop = images[:, :, y1:y2, x1:x2]
    return functional.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)


def standardize_face(crop: Any) -> Any:
    return (crop * 255.0 - 127.5) / 128.0


class FaceNetIdentityConstraint:
    def __init__(
        self,
        name: str,
        model: Any,
        detector: Any,
        *,
        tolerance: float,
        crop_size: int = 160,
    ) -> None:
        self.name = name
        self.model = model
        self.detector = detector
        self.tolerance = tolerance
        self.crop_size = crop_size
        self.face_box = None
        self._source_embedding = None

    def bind(self, context: ConstraintContext) -> None:
        import torch

        self.face_box = detect_largest_face_box(self.detector, context.source_image)
        crop = fixed_face_crop(context.source_image, self.face_box, size=self.crop_size)
        normalized = standardize_face(crop)
        with torch.no_grad():
            self._source_embedding = torch.nn.functional.normalize(
                self.model(normalized), dim=1
            ).detach()

    def measure(self, image: Any) -> Any:
        import torch

        if self.face_box is None or self._source_embedding is None:
            raise RuntimeError(f"Constraint {self.name!r} is not bound to a source")
        crop = fixed_face_crop(image, self.face_box, size=self.crop_size)
        embedding = torch.nn.functional.normalize(self.model(standardize_face(crop)), dim=1)
        source = self._source_embedding.to(device=image.device, dtype=embedding.dtype)
        return (1.0 - torch.nn.functional.cosine_similarity(embedding, source, dim=1)).mean()
```

Create `src/cci_diff/identity/__init__.py` exporting `FaceNetIdentityConstraint`, `build_face_detector`, `detect_largest_face_box`, `fixed_face_crop`, `load_facenet_identity`, and `load_identity_export_manifest`.

- [x] **Step 5: Add the explicit acquisition script**

Create `scripts/download_identity_model.py`; this is the only code path allowed to request pretrained FaceNet weights:

```python
#!/usr/bin/env python3
"""Explicitly acquire and export facenet-pytorch VGGFace2 as TorchScript."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output: str | Path) -> tuple[str, str, str]:
    import torch
    from facenet_pytorch import InceptionResnetV1

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = InceptionResnetV1(pretrained="vggface2", classify=False).eval()
    example = torch.zeros((1, 3, 160, 160), dtype=torch.float32)
    exported = torch.jit.freeze(torch.jit.trace(model, example, strict=False))
    exported.save(str(destination))
    digest = sha256_file(destination)
    manifest_path = Path(str(destination) + ".json")
    manifest_path.write_text(
        json.dumps(
            {
                "facenet_pytorch_version": importlib.metadata.version("facenet-pytorch"),
                "export_torch_version": torch.__version__,
                "sha256": digest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(destination), digest, str(manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/facenet_vggface2.ts")
    args = parser.parse_args()
    path, digest, manifest = download(args.output)
    print(f"saved={path}")
    print(f"sha256={digest}")
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 6: Run identity tests without installing or downloading FaceNet**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_facenet_identity.py' -v
```

Expected: all tests PASS. Runtime identity code imports neither facenet-pytorch nor the network; the missing-checkpoint check occurs before `torch.jit.load`.

- [x] **Step 7: Checkpoint without committing**

Run:

```bash
git diff --check
rg -n '[[:blank:]]+$' .gitignore pyproject.toml src/cci_diff/identity scripts/download_identity_model.py tests/test_facenet_identity.py docs/superpowers/specs/2026-07-14-constraint-feedback-clean-cci-design.md
```

Expected: no diagnostics. Do not stage or commit.

### Task 5: Constraint-Feedback Controller And Target-Priority Gradient Composition

**Files:**
- Create: `src/cci_diff/constraint_controller.py`
- Create: `tests/test_constraint_controller.py`

**Interfaces:**
- Consumes: Task 1 `ControllerSpec` and Task 3 `ConstraintObservation`.
- Produces: `TargetMargin`, `ControllerResult`, `target_margin(logit, desired_value, target_probability)`, `update_dual_multiplier`, `project_target_conflict`, `clip_update_norm`, and stateful `ConstraintFeedbackController.compute_update(...)`.

- [x] **Step 1: Write target-margin and dual-update tests**

Create `tests/test_constraint_controller.py`:

```python
import math
import unittest


class TestConstraintController(unittest.TestCase):
    def controller_spec(self):
        from cci_diff.concept_graph import concept_graph_from_dict
        from test_concept_graph import valid_graph_payload

        return concept_graph_from_dict(valid_graph_payload()).controller

    def test_margin_has_correct_direction_for_both_binary_targets(self):
        import torch
        from cci_diff.constraint_controller import target_margin

        positive = target_margin(torch.tensor(2.0), 1, 0.8)
        negative = target_margin(torch.tensor(2.0), 0, 0.8)

        self.assertEqual(positive.loss.item(), 0.0)
        self.assertEqual(positive.activation, 0.0)
        self.assertAlmostEqual(negative.signed_logit.item(), -2.0)
        self.assertAlmostEqual(negative.loss.item(), math.log(4.0) + 2.0)
        self.assertGreater(negative.activation, 0.0)

    def test_dual_multiplier_rises_on_violation_and_falls_on_satisfaction(self):
        from cci_diff.constraint_controller import update_dual_multiplier

        spec = self.controller_spec()
        raised, residual = update_dual_multiplier(0.1, value=0.03, tolerance=0.02, spec=spec)
        lowered, _ = update_dual_multiplier(raised, value=0.005, tolerance=0.02, spec=spec)

        self.assertAlmostEqual(residual, 0.5)
        self.assertGreater(raised, 0.1)
        self.assertLess(lowered, raised)
```

- [x] **Step 2: Write projection, trust-radius, and tiny-gradient tests**

Append:

```python
    def test_projection_removes_only_target_opposing_component(self):
        import torch
        from cci_diff.constraint_controller import project_target_conflict

        target = torch.tensor([1.0, 0.0])
        constraint = torch.tensor([-1.0, 2.0])
        projected, applied, cosine = project_target_conflict(
            target, constraint, gradient_floor=1e-5
        )

        self.assertTrue(applied)
        self.assertLess(cosine, 0.0)
        self.assertTrue(torch.allclose(projected, torch.tensor([0.0, 2.0])))
        self.assertGreaterEqual(torch.dot(target, projected).item(), 0.0)

    def test_trust_clip_bounds_full_masked_update(self):
        import torch
        from cci_diff.constraint_controller import clip_update_norm

        update, before, after = clip_update_norm(
            torch.tensor([3.0, 4.0]), trust_radius=0.2, gradient_floor=1e-5
        )
        self.assertAlmostEqual(before, 5.0)
        self.assertAlmostEqual(after, 0.2, places=6)
        self.assertAlmostEqual(torch.linalg.vector_norm(update).item(), 0.2, places=6)

    def test_zero_gradient_is_not_inflated_or_made_non_finite(self):
        import torch
        from cci_diff.constraint_controller import normalize_with_ema

        normalized, ema, raw_norm = normalize_with_ema(
            torch.zeros(3), previous_ema=0.0, beta=0.9, floor=1e-5
        )
        self.assertEqual(raw_norm, 0.0)
        self.assertEqual(ema, 0.0)
        self.assertTrue(torch.equal(normalized, torch.zeros(3)))
```

- [x] **Step 3: Run pure controller tests and verify missing module**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_constraint_controller.py' -v
```

Expected: FAIL because `cci_diff.constraint_controller` does not exist.

- [x] **Step 4: Implement target margin and pure controller helpers**

Create the first part of `src/cci_diff/constraint_controller.py`:

```python
"""Automatic primal-dual gradient composition for clean CCI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from cci_diff.concept_graph import ControllerSpec
from cci_diff.constraints import ConstraintObservation


@dataclass(frozen=True)
class TargetMargin:
    loss: Any
    raw_logit: Any
    signed_logit: Any
    required_logit: float
    desired_probability: float
    residual: float
    activation: float


@dataclass(frozen=True)
class ControllerResult:
    delta: Any
    record: dict[str, Any]


def target_margin(logit: Any, desired_value: int, target_probability: float) -> TargetMargin:
    import torch

    if desired_value not in (0, 1):
        raise ValueError("desired_value must be 0 or 1")
    if not 0.5 < target_probability < 1.0:
        raise ValueError("target_probability must be strictly between 0.5 and 1.0")
    sign = 2 * desired_value - 1
    signed = sign * logit
    required = math.log(target_probability / (1.0 - target_probability))
    loss = torch.relu(torch.as_tensor(required, device=logit.device, dtype=logit.dtype) - signed)
    residual = float((required - signed.detach().item()))
    activation = (
        min(max(residual / max(abs(required), 1.0), 0.0), 1.0)
        if math.isfinite(residual)
        else 0.0
    )
    probability = float(torch.sigmoid(signed.detach()).item())
    return TargetMargin(loss, logit, signed, required, probability, residual, activation)


def update_dual_multiplier(
    current: float,
    *,
    value: float,
    tolerance: float,
    spec: ControllerSpec,
) -> tuple[float, float]:
    residual = value / tolerance - 1.0
    if not math.isfinite(residual):
        return current, residual
    updated = min(max(current + spec.dual_rate * residual, 0.0), spec.lambda_max)
    return updated, residual


def normalize_with_ema(
    gradient: Any,
    *,
    previous_ema: float,
    beta: float,
    floor: float,
) -> tuple[Any, float, float]:
    import torch

    raw_norm = float(torch.linalg.vector_norm(gradient.detach().float()).item())
    if not math.isfinite(raw_norm):
        return torch.zeros_like(gradient), previous_ema, raw_norm
    ema = beta * previous_ema + (1.0 - beta) * raw_norm
    if raw_norm < floor or ema < floor:
        return torch.zeros_like(gradient), ema, raw_norm
    return gradient / max(ema, floor), ema, raw_norm


def project_target_conflict(
    target_gradient: Any,
    constraint_gradient: Any,
    *,
    gradient_floor: float,
) -> tuple[Any, bool, float | None]:
    import torch

    target_norm = torch.linalg.vector_norm(target_gradient.detach().float())
    constraint_norm = torch.linalg.vector_norm(constraint_gradient.detach().float())
    if target_norm.item() < gradient_floor or constraint_norm.item() < gradient_floor:
        return constraint_gradient, False, None
    dot = torch.sum(target_gradient * constraint_gradient)
    cosine = float((dot / (target_norm * constraint_norm)).detach().item())
    if dot.detach().item() >= 0:
        return constraint_gradient, False, cosine
    denominator = torch.sum(target_gradient * target_gradient).clamp_min(gradient_floor**2)
    projected = constraint_gradient - dot / denominator * target_gradient
    return projected, True, cosine


def clip_update_norm(
    update: Any,
    *,
    trust_radius: float,
    gradient_floor: float,
) -> tuple[Any, float, float]:
    import torch

    before = float(torch.linalg.vector_norm(update.detach().float()).item())
    if not math.isfinite(before):
        return torch.zeros_like(update), before, 0.0
    scale = min(1.0, trust_radius / max(before, gradient_floor))
    clipped = update * scale
    after = float(torch.linalg.vector_norm(clipped.detach().float()).item())
    return clipped, before, after
```

- [x] **Step 5: Write an end-to-end controller-step test**

Add a test where the target is unmet and one constraint is violated. It must prove immediate augmented action, automatic multiplier growth, masked update, and complete trace fields:

```python
    def test_compute_update_uses_target_and_first_step_constraint_feedback(self):
        import torch
        from cci_diff.constraint_controller import ConstraintFeedbackController, target_margin
        from cci_diff.constraints import ConstraintObservation

        latents = torch.tensor([1.0, 1.0], requires_grad=True)
        margin = target_margin(latents[0], desired_value=0, target_probability=0.8)
        observations = (
            ConstraintObservation("locality", latents[1].square(), tolerance=0.5),
        )
        controller = ConstraintFeedbackController(self.controller_spec())
        result = controller.compute_update(
            latents=latents,
            target=margin,
            constraints=observations,
            latent_mask=torch.tensor([1.0, 0.0]),
            eta=0.2,
            project_conflicts=True,
            mode="feedback",
        )

        self.assertGreater(result.record["constraints"]["locality"]["coefficient"], 0.0)
        self.assertGreater(result.record["constraints"]["locality"]["lambda_after"], 0.0)
        self.assertEqual(result.delta[1].item(), 0.0)
        self.assertLessEqual(result.record["update"]["norm"], self.controller_spec().trust_radius)
```

- [x] **Step 6: Implement stateful gradient composition and reliability handling**

Add `ConstraintFeedbackController` to the same module. Use one autograd call per objective, current-step augmented coefficients, post-step dual updates, and two-strike non-finite/unreliable counters:

```python
class ConstraintFeedbackController:
    def __init__(self, spec: ControllerSpec) -> None:
        self.spec = spec
        self.multipliers: dict[str, float] = {}
        self.norm_ema: dict[str, float] = {}
        self.consecutive_nonfinite = 0
        self.consecutive_unreliable_target = 0

    def compute_update(
        self,
        *,
        latents: Any,
        target: TargetMargin,
        constraints: Sequence[ConstraintObservation],
        latent_mask: Any,
        eta: float,
        project_conflicts: bool = True,
        mode: str = "feedback",
    ) -> ControllerResult:
        import torch

        if mode not in {"disabled", "feedback", "fixed_equal"}:
            raise ValueError(f"Unknown controller mode: {mode}")
        if mode == "disabled":
            finite = all(
                math.isfinite(float(observation.value.detach().item()))
                for observation in constraints
            ) and math.isfinite(target.desired_probability)
            self.consecutive_nonfinite = 0 if finite else self.consecutive_nonfinite + 1
            if self.consecutive_nonfinite >= 2:
                raise FloatingPointError("Two consecutive non-finite clean CCI steps")
            disabled_constraints = {}
            for observation in constraints:
                value = float(observation.value.detach().item())
                residual = value / observation.tolerance - 1.0
                disabled_constraints[observation.name] = {
                    "value": value if math.isfinite(value) else None,
                    "tolerance": observation.tolerance,
                    "residual": residual if math.isfinite(residual) else None,
                    "violation": max(residual, 0.0) if math.isfinite(residual) else None,
                    "lambda_before": 0.0,
                    "lambda_after": 0.0,
                    "coefficient": 0.0,
                    "gradient_norm": 0.0,
                    "normalized_gradient_norm": 0.0,
                }
            return ControllerResult(
                torch.zeros_like(latents),
                {
                    "target": {
                        "logit": _finite_or_none(target.raw_logit.detach().item()),
                        "signed_logit": _finite_or_none(target.signed_logit.detach().item()),
                        "target_probability": _finite_or_none(target.desired_probability),
                        "required_probability": self._required_probability(target),
                        "required_logit": target.required_logit,
                        "margin_residual": _finite_or_none(target.residual),
                        "activation": target.activation,
                        "gradient_norm": 0.0,
                        "norm_ema": 0.0,
                        "unreliable_target_gradient": False,
                    },
                    "constraints": disabled_constraints,
                    "update": {
                        "eta": eta,
                        "controller_mode": mode,
                        "projected": False,
                        "target_constraint_cosine": None,
                        "pre_clip_norm": 0.0,
                        "norm": 0.0,
                        "skip_reason": None if finite else "nonfinite_measurement",
                    },
                },
            )
        multiplier_snapshot = dict(self.multipliers)
        norm_snapshot = dict(self.norm_ema)
        target_gradient = torch.autograd.grad(
            target.loss,
            latents,
            retain_graph=bool(constraints),
            allow_unused=True,
        )[0]
        if target_gradient is None:
            target_gradient = torch.zeros_like(latents)
        target_normalized, target_ema, target_raw_norm = normalize_with_ema(
            target_gradient,
            previous_ema=self.norm_ema.get("target", 0.0),
            beta=self.spec.norm_ema_beta,
            floor=self.spec.gradient_floor,
        )
        self.norm_ema["target"] = target_ema

        constraint_gradient = torch.zeros_like(latents)
        records: dict[str, Any] = {}
        finite = math.isfinite(target_raw_norm) and torch.isfinite(target.loss).all().item()
        for index, observation in enumerate(constraints):
            value_float = float(observation.value.detach().item())
            before = self.multipliers.get(observation.name, 0.0)
            after, residual = update_dual_multiplier(
                before,
                value=value_float,
                tolerance=observation.tolerance,
                spec=self.spec,
            )
            violation = torch.relu(
                observation.value / observation.tolerance - 1.0
            )
            active = math.isfinite(residual) and float(violation.detach().item()) > 0.0
            raw = torch.zeros_like(latents)
            normalized = raw
            raw_norm = 0.0
            normalized_norm = 0.0
            if active:
                raw_value = torch.autograd.grad(
                    violation,
                    latents,
                    retain_graph=index < len(constraints) - 1,
                    allow_unused=True,
                )[0]
                if raw_value is not None:
                    raw = raw_value
                normalized, ema, raw_norm = normalize_with_ema(
                    raw,
                    previous_ema=self.norm_ema.get(observation.name, 0.0),
                    beta=self.spec.norm_ema_beta,
                    floor=self.spec.gradient_floor,
                )
                self.norm_ema[observation.name] = ema
                normalized_norm = float(
                    torch.linalg.vector_norm(normalized.detach().float()).item()
                )
            coefficient = (
                1.0
                if mode == "fixed_equal" and active
                else before + self.spec.penalty * max(residual, 0.0)
                if math.isfinite(residual)
                else 0.0
            )
            constraint_gradient = constraint_gradient + coefficient * normalized
            self.multipliers[observation.name] = after
            finite = finite and math.isfinite(value_float) and math.isfinite(raw_norm)
            records[observation.name] = {
                "value": _finite_or_none(value_float),
                "tolerance": observation.tolerance,
                "residual": _finite_or_none(residual),
                "violation": max(residual, 0.0) if math.isfinite(residual) else None,
                "lambda_before": before,
                "lambda_after": after,
                "coefficient": coefficient,
                "gradient_norm": raw_norm,
                "normalized_gradient_norm": normalized_norm,
            }

        projected = False
        cosine = None
        if project_conflicts and target.activation > 0:
            constraint_gradient, projected, cosine = project_target_conflict(
                target_normalized,
                constraint_gradient,
                gradient_floor=self.spec.gradient_floor,
            )
        combined = target.activation * target_normalized + constraint_gradient
        masked = eta * latent_mask.to(device=latents.device, dtype=latents.dtype) * combined
        delta, pre_clip_norm, final_norm = clip_update_norm(
            masked,
            trust_radius=self.spec.trust_radius,
            gradient_floor=self.spec.gradient_floor,
        )

        if not finite or not torch.isfinite(delta).all().item():
            self.multipliers = multiplier_snapshot
            self.norm_ema = norm_snapshot
            self.consecutive_nonfinite += 1
            if self.consecutive_nonfinite >= 2:
                raise FloatingPointError("Two consecutive non-finite clean CCI steps")
            delta = torch.zeros_like(latents)
            skip_reason = "nonfinite"
        else:
            self.consecutive_nonfinite = 0
            skip_reason = None
        if target.activation > 0 and target_raw_norm < self.spec.gradient_floor:
            self.consecutive_unreliable_target += 1
        else:
            self.consecutive_unreliable_target = 0

        record = {
            "target": {
                "logit": _finite_or_none(target.raw_logit.detach().item()),
                "signed_logit": _finite_or_none(target.signed_logit.detach().item()),
                "target_probability": _finite_or_none(target.desired_probability),
                "required_probability": self._required_probability(target),
                "required_logit": target.required_logit,
                "margin_residual": _finite_or_none(target.residual),
                "activation": target.activation,
                "gradient_norm": target_raw_norm,
                "norm_ema": target_ema,
                "unreliable_target_gradient": self.consecutive_unreliable_target >= 2,
            },
            "constraints": records,
            "update": {
                "eta": eta,
                "controller_mode": mode,
                "projected": projected,
                "target_constraint_cosine": cosine,
                "pre_clip_norm": pre_clip_norm,
                "norm": final_norm,
                "skip_reason": skip_reason,
            },
        }
        return ControllerResult(delta.detach(), record)

    @staticmethod
    def _required_probability(target: TargetMargin) -> float:
        return 1.0 / (1.0 + math.exp(-target.required_logit))


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None
```

Add this explicit baseline test after the implementation:

```python
    def test_disabled_mode_measures_but_never_updates_noise_direction_or_duals(self):
        import torch
        from cci_diff.constraint_controller import ConstraintFeedbackController, target_margin
        from cci_diff.constraints import ConstraintObservation

        latents = torch.tensor([1.0], requires_grad=True)
        controller = ConstraintFeedbackController(self.controller_spec())
        result = controller.compute_update(
            latents=latents,
            target=target_margin(latents[0], 0, 0.8),
            constraints=(ConstraintObservation("locality", latents.square().mean(), 0.5),),
            latent_mask=torch.ones_like(latents),
            eta=0.2,
            mode="disabled",
        )

        self.assertTrue(torch.equal(result.delta, torch.zeros_like(latents)))
        self.assertEqual(controller.multipliers, {})
        self.assertEqual(result.record["constraints"]["locality"]["coefficient"], 0.0)
```

- [x] **Step 7: Run all controller tests**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_constraint_controller.py' -v
```

Expected: all tests PASS, including subtests proving `disabled` returns an all-zero delta without changing multipliers, `fixed_equal` uses coefficient `1.0` only for violated constraints, and projection-disabled feedback leaves the original constraint direction unchanged.

- [x] **Step 8: Checkpoint without committing**

Run:

```bash
git diff --check
rg -n '[[:blank:]]+$' src/cci_diff/constraint_controller.py tests/test_constraint_controller.py
```

Expected: no diagnostics. Do not stage or commit.

### Task 6: Scheduler-Aware Clean Prediction And Guidance Schedule

**Files:**
- Create: `src/cci_diff/adapters/sd2_clean_cci.py`
- Create: `tests/test_sd2_clean_cci.py`

**Interfaces:**
- Consumes: Task 1 `ControllerSpec`; torch tensors and a diffusers-style scheduler with `alphas_cumprod` and `config.prediction_type`.
- Produces: `alpha_prod_for_step(scheduler, timestep, sample)`, `predict_clean_latents(sample, model_output, alpha_prod_t, prediction_type)`, `decode_clean_latents(vae, clean_latents, latent_scale=0.18215)`, and `guidance_eta(step_index, progress, spec)`.

- [x] **Step 1: Write exact epsilon, velocity, rejection, dtype, and schedule tests**

Create `tests/test_sd2_clean_cci.py`:

```python
import math
import unittest


class TestSD2CleanPrediction(unittest.TestCase):
    def test_epsilon_prediction_matches_ddim_equation(self):
        import torch
        from cci_diff.adapters.sd2_clean_cci import predict_clean_latents

        sample = torch.tensor([2.0], dtype=torch.float32)
        epsilon = torch.tensor([1.0], dtype=torch.float32)
        result = predict_clean_latents(sample, epsilon, torch.tensor(0.64), "epsilon")

        self.assertTrue(torch.allclose(result, torch.tensor([1.75])))

    def test_velocity_prediction_matches_diffusers_equation(self):
        import torch
        from cci_diff.adapters.sd2_clean_cci import predict_clean_latents

        sample = torch.tensor([2.0], dtype=torch.float32)
        velocity = torch.tensor([1.0], dtype=torch.float32)
        result = predict_clean_latents(sample, velocity, torch.tensor(0.64), "v_prediction")

        self.assertTrue(torch.allclose(result, torch.tensor([1.0])))

    def test_sample_and_unknown_prediction_types_are_rejected(self):
        import torch
        from cci_diff.adapters.sd2_clean_cci import predict_clean_latents

        for prediction_type in ("sample", "mystery"):
            with self.subTest(prediction_type=prediction_type):
                with self.assertRaisesRegex(ValueError, prediction_type):
                    predict_clean_latents(
                        torch.ones(1), torch.ones(1), torch.tensor(0.5), prediction_type
                    )

    def test_guidance_eta_uses_normalized_bell_gate(self):
        from cci_diff.adapters.sd2_clean_cci import guidance_eta
        from test_concept_graph import valid_graph_payload
        from cci_diff.concept_graph import concept_graph_from_dict

        spec = concept_graph_from_dict(valid_graph_payload()).controller
        self.assertIsNone(guidance_eta(0, 0.10, spec))
        self.assertAlmostEqual(guidance_eta(2, 0.40, spec), 0.2)
        self.assertIsNone(guidance_eta(3, 0.40, spec))
        self.assertAlmostEqual(guidance_eta(4, 0.15, spec), 0.0)
        self.assertAlmostEqual(guidance_eta(4, 0.65, spec), 0.0)
```

- [x] **Step 2: Run the focused tests and verify missing symbols**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_sd2_clean_cci.py' -v
```

Expected: FAIL because the clean-prediction functions are not defined.

- [x] **Step 3: Implement clean-prediction math without a U-Net gradient path**

Create `src/cci_diff/adapters/sd2_clean_cci.py` with the first pure functions:

```python
"""Predicted-clean pre-scheduler guidance for SD2 DDIM."""

from __future__ import annotations

import math
from typing import Any

from cci_diff.concept_graph import ControllerSpec


LATENT_SCALE = 0.18215


def alpha_prod_for_step(scheduler: Any, timestep: Any, sample: Any) -> Any:
    import torch

    index = int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
    alpha = scheduler.alphas_cumprod[index]
    return torch.as_tensor(alpha, device=sample.device, dtype=sample.dtype)


def predict_clean_latents(
    sample: Any,
    model_output: Any,
    alpha_prod_t: Any,
    prediction_type: str,
) -> Any:
    import torch

    alpha = torch.as_tensor(alpha_prod_t, device=sample.device, dtype=sample.dtype)
    beta = (1.0 - alpha).clamp(min=0.0)
    detached_output = model_output.detach().to(device=sample.device, dtype=sample.dtype)
    if prediction_type == "epsilon":
        return (sample - beta.sqrt() * detached_output) / alpha.sqrt()
    if prediction_type == "v_prediction":
        return alpha.sqrt() * sample - beta.sqrt() * detached_output
    if prediction_type == "sample":
        raise ValueError("sample prediction is unsupported without U-Net backpropagation")
    raise ValueError(f"Unsupported scheduler prediction type: {prediction_type}")


def decode_clean_latents(vae: Any, clean_latents: Any, latent_scale: float = LATENT_SCALE) -> Any:
    decoded = vae.decode(clean_latents / latent_scale).sample
    return (decoded / 2.0 + 0.5).clamp(0.0, 1.0)


def guidance_eta(
    step_index: int,
    progress: float,
    spec: ControllerSpec,
) -> float | None:
    if step_index % spec.every_n_steps != 0:
        return None
    start, end = spec.active_progress
    if progress < start or progress > end:
        return None
    u = (progress - start) / (end - start)
    return spec.step_scale * math.sin(math.pi * u) ** 2
```

- [x] **Step 4: Add alpha lookup and gradient-path tests**

Add a fake scheduler and fake VAE test proving gradients reach `sample` but not the detached model output:

```python
    def test_clean_prediction_backpropagates_only_to_sample(self):
        import torch
        from cci_diff.adapters.sd2_clean_cci import predict_clean_latents

        sample = torch.tensor([1.0], requires_grad=True)
        model_output = torch.tensor([0.25], requires_grad=True)
        clean = predict_clean_latents(sample, model_output, torch.tensor(0.81), "epsilon")
        clean.sum().backward()

        self.assertIsNotNone(sample.grad)
        self.assertIsNone(model_output.grad)
```

- [x] **Step 5: Run clean-prediction tests**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_sd2_clean_cci.py' -v
```

Expected: all tests PASS on CPU and no float64 tensor is created.

- [x] **Step 6: Checkpoint without committing**

Run:

```bash
git diff --check
rg -n '[[:blank:]]+$' src/cci_diff/adapters/sd2_clean_cci.py tests/test_sd2_clean_cci.py
```

Expected: no diagnostics. Do not stage or commit.

### Task 7: Denoising Progress And Semantic Context In The Existing Backend

**Files:**
- Modify: `src/cci_diff/sd2_bld_backend.py`
- Modify: `tests/test_sd2_bld_backend.py`

**Interfaces:**
- Consumes: the existing `CCIGuidanceHook`, `CCILatentGuidanceHook`, `edit_image`, and BLD blend order.
- Produces: `denoising_progress(step_index, total_steps) -> float`; new optional `SD2DenoisingStep.semantic_mask`, `.total_steps`, and `.progress`; new optional `edit_image(semantic_mask=...)` argument.

- [x] **Step 1: Write progress and backward-compatibility tests**

Append to `tests/test_sd2_bld_backend.py`:

```python
    def test_denoising_progress_covers_selected_reverse_interval(self):
        from cci_diff.sd2_bld_backend import denoising_progress

        self.assertEqual(denoising_progress(0, 1), 0.0)
        self.assertEqual(denoising_progress(0, 5), 0.0)
        self.assertEqual(denoising_progress(2, 5), 0.5)
        self.assertEqual(denoising_progress(4, 5), 1.0)

    def test_old_step_constructor_gets_safe_context_defaults(self):
        from cci_diff.sd2_bld_backend import SD2DenoisingStep

        step = SD2DenoisingStep(
            step_index=0,
            timestep=17,
            prompt="legacy",
            latents=object(),
            noise_pred=object(),
            source_latents=object(),
            latent_mask=object(),
        )

        self.assertIsNone(step.semantic_mask)
        self.assertEqual(step.total_steps, 1)
        self.assertEqual(step.progress, 0.0)
```

- [x] **Step 2: Run backend tests and verify the new helper is missing**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_sd2_bld_backend.py' -v
```

Expected: FAIL because `denoising_progress` and the context defaults do not exist.

- [x] **Step 3: Extend the immutable step context without changing legacy call sites**

Add fields at the end of `SD2DenoisingStep` so all current tests and external hooks remain valid:

```python
    semantic_mask: Any | None = None
    total_steps: int = 1
    progress: float = 0.0
```

Add:

```python
def denoising_progress(step_index: int, total_steps: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step_index < 0 or step_index >= total_steps:
        raise ValueError("step_index must be inside the selected reverse interval")
    return step_index / max(total_steps - 1, 1)
```

Include `semantic_mask_shape`, `total_steps`, and `progress` in `diffusion_state_from_step(...).extra`, while preserving every existing key.

- [x] **Step 4: Load and propagate the semantic mask through every hook phase**

Add `semantic_mask: str | Path | None = None` after `generation_mask` in `edit_image`. Read it at latent resolution with `binary=True`; when absent, use `(latent_mask >= 0.5).to(latent_mask.dtype)`. Repeat it to `batch_size` exactly as for `latent_mask`.

Replace the loop setup and each `SD2DenoisingStep(...)` construction with the following values in addition to existing fields:

```python
        selected_timesteps = timesteps[start_index:]
        total_steps = len(selected_timesteps)
        for step_index, timestep in enumerate(selected_timesteps):
            progress = denoising_progress(step_index, total_steps)
            # Existing CFG calculation remains here.
            step = SD2DenoisingStep(
                step_index=step_index,
                timestep=timestep,
                prompt=prompt,
                latents=latents,
                noise_pred=noise_pred,
                source_latents=source_latents,
                latent_mask=latent_mask,
                semantic_mask=semantic_latent_mask,
                total_steps=total_steps,
                progress=progress,
            )
```

Apply the same three context fields to the reconstructed step after CCI noise guidance, scheduler step, latent guidance, and blend. Do not move these operations: `cci_guidance_hook -> scheduler.step -> cci_latent_guidance_hook -> source blend` remains exact.

- [x] **Step 5: Add a source-order regression assertion**

In `tests/test_sd2_bld_backend.py`, inspect `BlendedLatentDiffusionSD2Backend.edit_image` with `inspect.getsource` and assert the call positions remain ordered. This is intentionally a narrow guard against moving clean CCI after `scheduler.step`:

```python
    def test_hook_and_blend_order_remains_pre_scheduler_then_post_scheduler(self):
        import inspect
        from cci_diff.sd2_bld_backend import BlendedLatentDiffusionSD2Backend

        source = inspect.getsource(BlendedLatentDiffusionSD2Backend.edit_image)
        self.assertLess(source.index("apply_cci_guidance("), source.index("self.scheduler.step("))
        self.assertLess(source.index("self.scheduler.step("), source.index("apply_cci_latent_guidance_hook("))
        self.assertLess(source.index("apply_cci_latent_guidance_hook("), source.index("blend_soft_latents("))
```

- [x] **Step 6: Run backend and legacy adapter tests**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_sd2_bld_backend.py' -v
.venv-ml/bin/python -m unittest discover -s tests -p 'test_sd2_adapter_contract.py' -v
.venv-ml/bin/python -m unittest discover -s tests -p 'test_sd2_robust_adapter.py' -v
```

Expected: all three commands PASS.

- [x] **Step 7: Checkpoint without committing**

Run:

```bash
git diff --check
rg -n '[[:blank:]]+$' src/cci_diff/sd2_bld_backend.py tests/test_sd2_bld_backend.py
```

Expected: no diagnostics. Do not stage or commit.

### Task 8: Predicted-Clean Noise Hook And Authoritative JSONL Trace

**Files:**
- Create: `src/cci_diff/cci_trace.py`
- Modify: `src/cci_diff/adapters/sd2_clean_cci.py`
- Create: `tests/test_cci_trace.py`
- Extend: `tests/test_sd2_clean_cci.py`

**Interfaces:**
- Consumes: Tasks 3, 5, 6, and 7 evaluator/controller/clean-step interfaces.
- Produces: `JSONLTraceWriter`, `load_cci_trace(path)`, `validate_cci_trace(records)`, and callable `CleanCCIGuidanceHook` returning replacement noise or `None`.

- [x] **Step 1: Write deterministic trace writer tests**

Create `tests/test_cci_trace.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path


class TestCCITrace(unittest.TestCase):
    def test_writer_truncates_then_appends_deterministic_json_lines(self):
        from cci_diff.cci_trace import JSONLTraceWriter, load_cci_trace

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            path.write_text("stale\n", encoding="utf-8")
            writer = JSONLTraceWriter(path)
            writer.write({"step": 2, "target": {"activation": 1.0}, "constraints": {}, "update": {}})
            writer.write({"step": 4, "target": {"activation": 0.5}, "constraints": {}, "update": {}})
            records = load_cci_trace(path)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual([record["step"] for record in records], [2, 4])
        self.assertEqual(lines[0], json.dumps(records[0], sort_keys=True, separators=(",", ":")))

    def test_validation_rejects_missing_fields_and_non_increasing_steps(self):
        from cci_diff.cci_trace import validate_cci_trace

        with self.assertRaisesRegex(ValueError, "required fields"):
            validate_cci_trace([{"step": 0}])
        record = {"step": 2, "target": {}, "constraints": {}, "update": {}}
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_cci_trace([record, dict(record)])
```

- [x] **Step 2: Implement trace I/O with no plotting dependency**

Create `src/cci_diff/cci_trace.py`:

```python
"""Authoritative JSONL trace I/O for clean CCI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_TRACE_FIELDS = frozenset({"step", "timestep", "progress", "target", "constraints", "update"})


class JSONLTraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_cci_trace(path: str | Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_cci_trace(records)
    return records


def validate_cci_trace(records: Iterable[dict[str, Any]]) -> None:
    previous = -1
    for record in records:
        if not REQUIRED_TRACE_FIELDS.issubset(record):
            raise ValueError("CCI trace record is missing required fields")
        step = int(record["step"])
        if step <= previous:
            raise ValueError("CCI trace steps must be strictly increasing")
        previous = step
```

- [x] **Step 3: Write a complete fake-hook integration test**

Extend `tests/test_sd2_clean_cci.py` with local fakes and a test that executes one selected guidance step:

```python
    def test_clean_hook_changes_noise_traces_clean_target_and_detaches_unet_output(self):
        import tempfile
        import torch
        from pathlib import Path
        from types import SimpleNamespace

        from cci_diff.adapters.sd2_clean_cci import CleanCCIGuidanceHook
        from cci_diff.cci_trace import JSONLTraceWriter, load_cci_trace
        from cci_diff.constraint_controller import ConstraintFeedbackController
        from cci_diff.constraints import ConstraintContext
        from cci_diff.sd2_bld_backend import SD2DenoisingStep

        class Scheduler:
            alphas_cumprod = torch.linspace(0.01, 0.99, 1000)
            config = SimpleNamespace(prediction_type="epsilon")

        class VAE:
            def decode(self, latent):
                return SimpleNamespace(sample=latent[:, :3])

        class Target:
            def logit(self, image):
                return image.mean()

        class Locality:
            name = "locality"
            tolerance = 0.02

            def bind(self, context: ConstraintContext):
                self.source = context.source_image

            def measure(self, image):
                return (image - self.source).abs().mean()

        latents = torch.ones((1, 4, 2, 2))
        noise = torch.zeros_like(latents, requires_grad=True)
        mask = torch.ones((1, 1, 2, 2))
        step = SD2DenoisingStep(
            4, torch.tensor(500), "neutral expression", latents, noise,
            torch.zeros_like(latents), mask, mask, 11, 0.4
        )
        spec = self.controller_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            trace = Path(tmpdir) / "trace.jsonl"
            hook = CleanCCIGuidanceHook(
                scheduler=Scheduler(),
                vae=VAE(),
                target_evaluator=Target(),
                constraint_evaluators=(Locality(),),
                controller=ConstraintFeedbackController(spec),
                desired_value=0,
                target_probability=0.8,
                trace_writer=JSONLTraceWriter(trace),
            )
            guided = hook(step)
            records = load_cci_trace(trace)

        self.assertFalse(torch.equal(guided, noise))
        self.assertIsNone(noise.grad)
        self.assertEqual(records[0]["prediction_type"], "epsilon")
        self.assertIn("locality", records[0]["constraints"])
```

Add this helper method to the test class so the fixture does not depend on another test module:

```python
    def controller_spec(self):
        from cci_diff.concept_graph import ControllerSpec

        return ControllerSpec(0.2, 0.5, 4.0, 0.2, 0.15, 0.9, 1e-5, (0.15, 0.65), 2)
```

- [x] **Step 4: Implement the callable pre-scheduler clean hook**

Append to `src/cci_diff/adapters/sd2_clean_cci.py`:

```python
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
        controller_mode: str = "feedback",
        project_conflicts: bool = True,
    ) -> None:
        self.scheduler = scheduler
        self.vae = vae
        self.target_evaluator = target_evaluator
        self.constraint_evaluators = constraint_evaluators
        self.controller = controller
        self.desired_value = desired_value
        self.target_probability = target_probability
        self.trace_writer = trace_writer
        self.controller_mode = controller_mode
        self.project_conflicts = project_conflicts
        self._constraints_bound = False

    def __call__(self, step: Any) -> Any | None:
        import torch
        import torch.nn.functional as functional

        eta = guidance_eta(step.step_index, step.progress, self.controller.spec)
        if eta is None:
            return None
        guided_latents = step.latents.detach().clone().requires_grad_(True)
        alpha = alpha_prod_for_step(self.scheduler, step.timestep, guided_latents)
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
                    self.vae, step.source_latents.detach()
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
            context = ConstraintContext(source_image, generation_mask, semantic_mask)
            for evaluator in self.constraint_evaluators:
                evaluator.bind(context)
            self._constraints_bound = True

        margin = target_margin(
            self.target_evaluator.logit(clean_image),
            self.desired_value,
            self.target_probability,
        )
        observations = tuple(
            ConstraintObservation(evaluator.name, evaluator.measure(clean_image), evaluator.tolerance)
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
        record = {
            "step": step.step_index,
            "timestep": int(step.timestep.item()) if hasattr(step.timestep, "item") else int(step.timestep),
            "progress": step.progress,
            "prediction_type": prediction_type,
            "alpha_prod_t": float(alpha.detach().item()),
            **result.record,
        }
        self.trace_writer.write(record)
        return guided_noise.detach()
```

Add imports for `ConstraintFeedbackController`, `target_margin`, `ConstraintContext`, `ConstraintObservation`, and `JSONLTraceWriter` only for type checking or at module level where they do not trigger optional model imports.

- [x] **Step 5: Add direct-sample hook rejection and non-finite two-strike tests**

Reuse the fakes from Step 3, set `Scheduler.config.prediction_type="sample"`, and assert a `ValueError` before any trace is written. Separately, use a target returning NaN twice and assert the first call returns unchanged finite noise while the second raises `FloatingPointError`.

- [x] **Step 6: Run clean-hook and trace tests**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_sd2_clean_cci.py' -v
.venv-ml/bin/python -m unittest discover -s tests -p 'test_cci_trace.py' -v
```

Expected: both commands PASS; trace JSON uses finite values and strictly increasing selected step indices.

- [x] **Step 7: Checkpoint without committing**

Run:

```bash
git diff --check
rg -n '[[:blank:]]+$' src/cci_diff/adapters/sd2_clean_cci.py src/cci_diff/cci_trace.py tests/test_sd2_clean_cci.py tests/test_cci_trace.py
```

Expected: no diagnostics. Do not stage or commit.

### Task 9: Clean-Mode CLI Wiring, Evaluator Factory, Examples, And Audit Provenance

**Files:**
- Modify: `scripts/run_sd2_bld_cci.py`
- Modify: `tests/test_sd2_bld_cli.py`
- Create: `tests/test_clean_cci_cli.py`
- Create: `examples/graphs/remove_smile_clean_cci.json`
- Create: `examples/graphs/blond_hair_clean_cci.json`
- Create: `examples/bindings/sample_0_mouth.json`
- Create: `examples/bindings/sample_0_hair.json`

**Interfaces:**
- Consumes: Tasks 2-8 compiled plan, evaluators, hook, trace writer, mask preparation, and existing classifier loader/audit helpers.
- Produces: `validate_mode_args(args)`, `prepare_clean_plan(args, output_dir)`, `build_clean_evaluators(plan, classifier, identity_model, face_detector, input_size)`, `CleanRunSetup`, and a fully wired `--cci_hook clean_constraint` execution path.

- [x] **Step 1: Write clean parser and ownership tests**

Create `tests/test_clean_cci_cli.py`:

```python
import unittest

from scripts.run_sd2_bld_cci import build_arg_parser, validate_mode_args


class TestCleanCCICLI(unittest.TestCase):
    def parse(self, *extra):
        return build_arg_parser().parse_args(
            [
                "--output_dir", "outputs/clean",
                "--cci_hook", "clean_constraint",
                "--cci_graph", "examples/graphs/remove_smile_clean_cci.json",
                "--cci_sample_bindings", "examples/bindings/sample_0_mouth.json",
                "--classifier_path", "models/resnet50_multilabel_model.pth",
                "--identity_model_path", "models/facenet_vggface2.ts",
                "--batch_size", "1",
                *extra,
            ]
        )

    def test_clean_mode_accepts_graph_and_binding_without_legacy_config_paths(self):
        args = self.parse()
        validate_mode_args(args)

        self.assertEqual(args.cci_hook, "clean_constraint")
        self.assertIsNone(args.cci_config)
        self.assertIsNone(args.init_image)
        self.assertIsNone(args.mask)

    def test_clean_mode_rejects_duplicate_graph_owned_cli_values(self):
        duplicate_cases = (
            ("--cci_config", "examples/remove_smile_intervention.json"),
            ("--init_image", "data/0.jpg"),
            ("--mask", "data/00000_mouth.png"),
            ("--cci_step_size", "0.2"),
            ("--cci_every_n_steps", "2"),
            ("--generation_mask_feather", "3"),
            ("--classifier_label_index", "31"),
        )
        for option, value in duplicate_cases:
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "single source of truth"):
                    validate_mode_args(self.parse(option, value))

    def test_legacy_mode_still_requires_config_image_and_mask(self):
        args = build_arg_parser().parse_args(["--output_dir", "outputs/legacy"])
        with self.assertRaisesRegex(ValueError, "Legacy CCI modes require"):
            validate_mode_args(args)
```

- [x] **Step 2: Change parser defaults so explicit duplicates are detectable**

In `build_arg_parser()`:

- Make `--cci_config`, `--init_image`, and `--mask` optional with `default=None`; keep `--output_dir` required.
- Add `clean_constraint` to `--cci_hook` choices.
- Change defaults for `--cci_step_size`, `--cci_every_n_steps`, `--cci_start_step`, and `--generation_mask_feather` to `None`.
- Add the following options:

```python
    parser.add_argument("--cci_graph", default=None)
    parser.add_argument("--cci_sample_bindings", default=None)
    parser.add_argument("--identity_model_path", default=None)
    parser.add_argument("--cci_trace", default=None)
    parser.add_argument(
        "--cci_controller_mode",
        choices=["disabled", "fixed_equal", "feedback"],
        default="feedback",
        help="Ablation mode for predicted-clean CCI; feedback is the proposed method.",
    )
    parser.add_argument("--cci_disable_target_projection", action="store_true")
```

Add `validate_mode_args` and call it as the first line of `run(args)`:

```python
def validate_mode_args(args: argparse.Namespace) -> None:
    if args.cci_hook == "clean_constraint":
        required = {
            "--cci_graph": args.cci_graph,
            "--cci_sample_bindings": args.cci_sample_bindings,
            "--classifier_path": args.classifier_path,
            "--identity_model_path": args.identity_model_path,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"clean_constraint requires: {', '.join(missing)}")
        if args.batch_size != 1:
            raise ValueError("clean_constraint version 1 requires --batch_size 1")
        duplicates = {
            "--cci_config": args.cci_config,
            "--init_image": args.init_image,
            "--mask": args.mask,
            "--cci_step_size": args.cci_step_size,
            "--cci_every_n_steps": args.cci_every_n_steps,
            "--cci_start_step": args.cci_start_step,
            "--cci_end_step": args.cci_end_step,
            "--generation_mask_feather": args.generation_mask_feather,
            "--classifier_label_index": args.classifier_label_index,
        }
        supplied = [name for name, value in duplicates.items() if value is not None]
        if args.cci_normalize_grad or args.robust_classifier_guidance or args.generation_mask_component:
            supplied.append("legacy guidance/mask option")
        if supplied:
            raise ValueError(
                "Graph and sample bindings are the single source of truth; remove: "
                + ", ".join(supplied)
            )
        if args.torch_dtype != "float32" and args.device == "mps":
            raise ValueError("clean_constraint on MPS requires --torch_dtype float32")
        return
    if not args.cci_config or not args.init_image or not args.mask:
        raise ValueError("Legacy CCI modes require --cci_config, --init_image, and --mask")
    if (
        args.cci_graph
        or args.cci_sample_bindings
        or args.identity_model_path
        or args.cci_trace
        or args.cci_controller_mode != "feedback"
        or args.cci_disable_target_projection
    ):
        raise ValueError("Clean graph options require --cci_hook clean_constraint")
    if args.cci_step_size is None:
        args.cci_step_size = 0.03
    if args.cci_every_n_steps is None:
        args.cci_every_n_steps = 4
    if args.cci_start_step is None:
        args.cci_start_step = 0
    if args.generation_mask_feather is None:
        args.generation_mask_feather = 3.0
```

- [x] **Step 3: Run parser tests before runtime wiring**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_clean_cci_cli.py' -v
.venv-ml/bin/python -m unittest discover -s tests -p 'test_sd2_bld_cli.py' -v
```

Expected: clean tests PASS after Step 2; legacy parser tests PASS after updating only assertions that previously depended on parser-level required arguments.

- [x] **Step 4: Add reusable graph and binding examples**

Create `examples/graphs/remove_smile_clean_cci.json`:

```json
{
  "version": 1,
  "intervention": {
    "concept": "Smiling",
    "desired_value": 0,
    "target_probability": 0.8
  },
  "region": {
    "audit_role": "mouth",
    "components": ["mouth", "upper_lip", "lower_lip"],
    "feather_radius": 3.0
  },
  "nodes": [
    {"id": "smiling", "role": "target", "evaluator": "celeba_attribute", "attribute": "Smiling"},
    {"id": "mouth_open", "role": "allowed_change", "evaluator": "celeba_attribute", "attribute": "Mouth_Slightly_Open"},
    {"id": "identity", "role": "constraint", "evaluator": "facenet_identity", "tolerance": 0.08},
    {"id": "outside_locality", "role": "constraint", "evaluator": "outside_l1", "tolerance": 0.02},
    {"id": "residual_tv", "role": "constraint", "evaluator": "masked_residual_tv", "tolerance": 0.015}
  ],
  "edges": [
    {"source": "smiling", "target": "mouth_open", "relation": "may_affect"},
    {"source": "smiling", "target": "identity", "relation": "must_preserve"},
    {"source": "smiling", "target": "outside_locality", "relation": "must_preserve"},
    {"source": "smiling", "target": "residual_tv", "relation": "must_preserve"}
  ],
  "controller": {
    "dual_rate": 0.2,
    "penalty": 0.5,
    "lambda_max": 4.0,
    "step_scale": 0.2,
    "trust_radius": 0.15,
    "norm_ema_beta": 0.9,
    "gradient_floor": 0.00001,
    "active_progress": [0.15, 0.65],
    "every_n_steps": 2
  }
}
```

Create `examples/graphs/blond_hair_clean_cci.json`:

```json
{
  "version": 1,
  "intervention": {
    "concept": "Blond_Hair",
    "desired_value": 1,
    "target_probability": 0.8
  },
  "region": {
    "audit_role": "hair",
    "components": ["hair"],
    "feather_radius": 3.0
  },
  "nodes": [
    {"id": "blond_hair", "role": "target", "evaluator": "celeba_attribute", "attribute": "Blond_Hair"},
    {"id": "black_hair", "role": "allowed_change", "evaluator": "celeba_attribute", "attribute": "Black_Hair"},
    {"id": "brown_hair", "role": "allowed_change", "evaluator": "celeba_attribute", "attribute": "Brown_Hair"},
    {"id": "gray_hair", "role": "allowed_change", "evaluator": "celeba_attribute", "attribute": "Gray_Hair"},
    {"id": "identity", "role": "constraint", "evaluator": "facenet_identity", "tolerance": 0.08},
    {"id": "outside_locality", "role": "constraint", "evaluator": "outside_l1", "tolerance": 0.02},
    {"id": "residual_tv", "role": "constraint", "evaluator": "masked_residual_tv", "tolerance": 0.015}
  ],
  "edges": [
    {"source": "blond_hair", "target": "black_hair", "relation": "may_affect"},
    {"source": "blond_hair", "target": "brown_hair", "relation": "may_affect"},
    {"source": "blond_hair", "target": "gray_hair", "relation": "may_affect"},
    {"source": "blond_hair", "target": "identity", "relation": "must_preserve"},
    {"source": "blond_hair", "target": "outside_locality", "relation": "must_preserve"},
    {"source": "blond_hair", "target": "residual_tv", "relation": "must_preserve"}
  ],
  "controller": {
    "dual_rate": 0.2,
    "penalty": 0.5,
    "lambda_max": 4.0,
    "step_scale": 0.2,
    "trust_radius": 0.15,
    "norm_ema_beta": 0.9,
    "gradient_floor": 0.00001,
    "active_progress": [0.15, 0.65],
    "every_n_steps": 2
  }
}
```

Create bindings with repository-relative real paths:

```json
{
  "source_image": "data/CelebAMask-HQ/CelebA-HQ-img/0.jpg",
  "masks": {
    "mouth": "data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_mouth.png",
    "upper_lip": "data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_u_lip.png",
    "lower_lip": "data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_l_lip.png"
  }
}
```

and:

```json
{
  "source_image": "data/CelebAMask-HQ/CelebA-HQ-img/0.jpg",
  "masks": {
    "hair": "data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_hair.png"
  }
}
```

- [x] **Step 5: Add the clean setup and evaluator factory**

Add imports and this runtime container to `scripts/run_sd2_bld_cci.py`:

```python
@dataclass(frozen=True)
class CleanRunSetup:
    plan: Any
    mask_artifacts: MaskArtifacts
    guidance_hook: Any
    classifier_runtime: ClassifierRuntime
    identity_checkpoint_sha256: str
    trace_path: str
```

Implement plan preparation exactly as follows:

```python
def prepare_clean_plan(args, output_dir: Path):
    graph = load_concept_graph(args.cci_graph)
    bindings = load_sample_bindings(args.cci_sample_bindings)
    plan = JsonConceptGraphCompiler(graph, args.cci_graph).compile(
        graph.intervention,
        bindings,
        default_concept_registry(),
    )
    mask_artifacts = prepare_semantic_masks(
        [path for _, path in plan.component_paths],
        feather_radius=plan.graph.region.feather_radius,
        hard_output=output_dir / "semantic_mask.png",
        soft_output=output_dir / "generation_mask.png",
    )
    return plan, mask_artifacts
```

Implement the exact evaluator dispatch:

```python
def build_clean_evaluators(
    plan,
    *,
    classifier,
    identity_model,
    face_detector,
    classifier_input_size: int,
):
    target = CelebAAttributeTarget(
        classifier,
        plan.target.attribute_index,
        classifier_input_size,
    )
    constraints = []
    for node in plan.constraints:
        if node.evaluator == "celeba_attribute":
            evaluator = CelebAAttributeConstraint(
                node.id,
                classifier,
                node.attribute_index,
                input_size=classifier_input_size,
                tolerance=node.tolerance,
            )
        elif node.evaluator == "facenet_identity":
            evaluator = FaceNetIdentityConstraint(
                node.id,
                identity_model,
                face_detector,
                tolerance=node.tolerance,
            )
        elif node.evaluator == "outside_l1":
            evaluator = OutsideL1Constraint(node.id, node.tolerance)
        elif node.evaluator == "masked_residual_tv":
            evaluator = MaskedResidualTVConstraint(node.id, node.tolerance)
        else:
            raise ValueError(f"No runtime constraint adapter for {node.evaluator!r}")
        constraints.append(evaluator)
    return target, tuple(constraints)
```

Type narrowing in this factory must explicitly check that target `attribute_index` and every constraint `tolerance` are not `None` before constructing an evaluator.

- [x] **Step 6: Wire `clean_constraint` into `run` through `cci_guidance_hook`**

Branch before legacy config loading:

1. Compile the clean plan and masks.
2. Build the prompt from a `ConceptIntervention` whose preserved concepts are the `must_preserve` destination IDs.
3. Create the SD2 backend.
4. Load the CelebA model and local FaceNet model in float32, verify its adjacent export manifest with `load_identity_export_manifest`, and build the CPU face detector.
5. Set `trace_path = Path(args.cci_trace) if args.cci_trace else output_dir / "cci_trace.jsonl"`, then create `ConstraintFeedbackController(plan.controller)`, the evaluator tuple, `JSONLTraceWriter(trace_path)`, and `CleanCCIGuidanceHook`.
6. Call `backend.edit_image` with `init_image=plan.source_image`, `mask=plan.audit_mask_path`, `semantic_mask=mask_artifacts.semantic_path`, `generation_mask=mask_artifacts.generation_path`, `cci_guidance_hook=clean_hook`, and `cci_latent_guidance_hook=None`.

For `--cci_controller_mode disabled`, use the same `CleanCCIGuidanceHook` with `controller_mode="disabled"`. It binds and measures the same evaluators, traces zero coefficients, and returns noise that is numerically identical to CFG; this is the A0 baseline. For `fixed_equal`, pass `controller_mode="fixed_equal"`. For `feedback`, pass `controller_mode="feedback"`. Set `project_conflicts=not args.cci_disable_target_projection`.

Keep the existing legacy branch structurally unchanged after `validate_mode_args` assigns its old defaults.

One backward-compatible generalization is required for the hair A1 ablation: replace the runner's smile-specific `len(args.generation_mask_component) != 3` check with `not args.generation_mask_component`. Robust guidance must accept one or more aligned semantic components; the existing three-component mouth/lip command and its loss calculation remain unchanged. Add a CLI test with one hair component and a rejection test with zero components.

- [x] **Step 7: Add final feasibility evaluation to the hook**

Add this method to `CleanCCIGuidanceHook`; it reuses the source-bound evaluator instances and never changes controller state:

```python
    def evaluate_image(self, image: Any) -> dict[str, Any]:
        import torch

        if not self._constraints_bound:
            raise RuntimeError("Clean CCI evaluators are not bound to a source image")
        with torch.no_grad():
            logit = self.target_evaluator.logit(image)
            margin = target_margin(logit, self.desired_value, self.target_probability)
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
        return {
            "target": {
                "logit": (
                    float(logit.item()) if math.isfinite(float(logit.item())) else None
                ),
                "desired_probability": (
                    margin.desired_probability
                    if math.isfinite(margin.desired_probability)
                    else None
                ),
                "required_probability": self.target_probability,
                "signed_margin": (
                    float(margin.signed_logit.item() - margin.required_logit)
                    if math.isfinite(
                        float(margin.signed_logit.item() - margin.required_logit)
                    )
                    else None
                ),
                "passed": target_passed,
            },
            "constraints": constraint_payload,
            "feasible": target_passed and not failed_names,
            "failed_constraints": failed_names,
        }
```

Load the batch-size-1 output image with this helper, call the method, and add the result to `audit["cci"]["final_feasibility"]`:

```python
def load_rgb_image_tensor(path: str | Path, *, device: str):
    import numpy as np
    import torch
    from PIL import Image

    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
```

- [x] **Step 8: Record complete clean provenance and runtime**

Under `audit["cci"]`, record `graph_path`, graph SHA-256, `plan.graph.to_dict()`, resolved nodes and evaluator versions, source/binding paths, mask artifacts, classifier and identity checkpoint SHA-256 values, the verified FaceNet export manifest, runtime package versions, trace path, controller mode, projection flag, wall-clock seconds, and the maximum `torch.mps.current_allocated_memory()` sampled at selected hook steps when that API is available. Never call a download function from `run`.

Also record the full paired metric surface required by the design:

- strict audit-mask and semantic-union inside/outside MAE via `masked_image_change_metrics`;
- all 40 source/output CelebA probabilities and mean non-target drift after excluding the target plus every `may_affect` destination;
- FaceNet cosine as `1 - identity distance`;
- residual TV and a boundary discontinuity defined as mean absolute residual on the soft ring `4 * generation_mask * (1 - generation_mask)`;
- target signed margin, feasibility, runtime, and peak MPS bytes;
- `independent_semantic_agreement` and `outside_perceptual_distance` as either measured values from an explicitly configured reviewed audit adapter or JSON `null` with `status="not_configured"`; never substitute the target classifier or FaceNet for these fields.

Write `audit.json` with `json.dumps(audit, indent=2, allow_nan=False)`. A non-finite final metric is `null`, fails its corresponding validity check, and is never serialized as JavaScript-style `NaN`.

- [x] **Step 9: Add a mocked clean-run wiring test**

In `tests/test_clean_cci_cli.py`, patch the backend, model loaders, detector, and hook; execute `run(args)` against temporary graph/image/mask files. Assert `edit_image` receives `cci_guidance_hook`, receives no latent hook, uses the binding source path, and gets distinct audit/semantic/generation mask paths. Assert `audit.json` contains graph digest and `controller_mode`.

- [x] **Step 10: Run CLI, graph, hook, and legacy tests**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_clean_cci_cli.py' -v
.venv-ml/bin/python -m unittest discover -s tests -p 'test_sd2_bld_cli.py' -v
.venv-ml/bin/python -m unittest discover -s tests -p 'test_smile_classifier_hook.py' -v
.venv-ml/bin/python -m unittest discover -s tests -p 'test_*graph*.py' -v
```

Expected: all commands PASS; mocked generation makes no network request.

- [x] **Step 11: Checkpoint without committing**

Run:

```bash
git diff --check
rg -n '[[:blank:]]+$' scripts/run_sd2_bld_cci.py tests/test_clean_cci_cli.py tests/test_sd2_bld_cli.py examples/graphs examples/bindings
```

Expected: no diagnostics. Do not stage or commit.

### Task 10: Trace CSV/PNG Reporting And Machine-Readable Pilot Metrics

**Files:**
- Create: `scripts/plot_cci_trace.py`
- Create: `tests/test_plot_cci_trace.py`
- Modify: `src/cci_diff/cci_trace.py`

**Interfaces:**
- Consumes: Task 8 JSONL records.
- Produces: `trace_rows(records)`, `write_trace_csv(records, path)`, and optional `plot_trace(records, path)`; generation imports none of these plotting functions.

- [x] **Step 1: Write flattening and CSV tests without matplotlib**

Create `tests/test_plot_cci_trace.py` with two records containing `identity` and `outside_locality`. Assert CSV headers include `target_probability`, `update_norm`, `identity.lambda`, `identity.residual`, `outside_locality.lambda`, and `outside_locality.residual`; assert both rows have equal columns.

Use this concrete test shape:

```python
    records = [{
        "step": 2,
        "timestep": 800,
        "progress": 0.2,
        "target": {"target_probability": 0.3, "required_probability": 0.8, "activation": 1.0, "gradient_norm": 0.1},
        "constraints": {
            "identity": {"lambda_after": 0.1, "residual": 0.2, "gradient_norm": 0.03},
            "outside_locality": {"lambda_after": 0.0, "residual": -0.4, "gradient_norm": 0.0}
        },
        "update": {"eta": 0.1, "norm": 0.12, "target_constraint_cosine": -0.2}
    }]
```

- [x] **Step 2: Implement stable wide-row flattening and CSV output**

Create `scripts/plot_cci_trace.py` with:

```python
def trace_rows(records):
    names = sorted({name for record in records for name in record["constraints"]})
    rows = []
    for record in records:
        row = {
            "step": record["step"],
            "timestep": record["timestep"],
            "progress": record["progress"],
            "target_probability": record["target"]["target_probability"],
            "required_probability": record["target"]["required_probability"],
            "target_activation": record["target"]["activation"],
            "target_gradient_norm": record["target"]["gradient_norm"],
            "eta": record["update"]["eta"],
            "update_norm": record["update"]["norm"],
            "target_constraint_cosine": record["update"]["target_constraint_cosine"],
        }
        for name in names:
            values = record["constraints"].get(name, {})
            row[f"{name}.lambda"] = values.get("lambda_after")
            row[f"{name}.residual"] = values.get("residual")
            row[f"{name}.gradient_norm"] = values.get("gradient_norm")
        rows.append(row)
    return rows


def write_trace_csv(records, path):
    import csv
    from pathlib import Path

    rows = trace_rows(records)
    if not rows:
        raise ValueError("Cannot export an empty CCI trace")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
```

- [x] **Step 3: Implement optional four-panel plotting behind a lazy import**

Implement plotting with a lazy import so generation never imports matplotlib:

```python
def plot_trace(records, path):
    from pathlib import Path

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Trace PNG output requires: pip install -e '.[plot]'") from exc
    rows = trace_rows(records)
    if not rows:
        raise ValueError("Cannot plot an empty CCI trace")
    steps = [row["step"] for row in rows]
    constraint_names = sorted(
        key.removesuffix(".lambda")
        for key in rows[0]
        if key.endswith(".lambda")
    )
    figure, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(steps, [row["target_probability"] for row in rows], label="desired probability")
    axes[0].plot(steps, [row["required_probability"] for row in rows], "--", label="required")
    axes[0].set_ylabel("probability")
    axes[0].legend()
    for name in constraint_names:
        axes[1].plot(steps, [row[f"{name}.lambda"] for row in rows], label=name)
        axes[2].plot(steps, [row[f"{name}.residual"] for row in rows], label=name)
    axes[1].set_ylabel("dual lambda")
    axes[1].legend()
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("normalized residual")
    axes[2].legend()
    axes[3].plot(steps, [row["target_gradient_norm"] for row in rows], label="target grad")
    for name in constraint_names:
        axes[3].plot(
            steps,
            [row[f"{name}.gradient_norm"] for row in rows],
            label=f"{name} grad",
        )
    axes[3].plot(steps, [row["update_norm"] for row in rows], label="update norm")
    axes[3].plot(steps, [row["eta"] for row in rows], label="eta")
    axes[3].plot(steps, [row["target_constraint_cosine"] for row in rows], label="cosine")
    axes[3].set_xlabel("selected denoising step")
    axes[3].legend()
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
```

Add CLI options `--trace`, `--csv`, and optional `--png`. Always validate via `load_cci_trace`, always write CSV, and only call `plot_trace` when `--png` is supplied.

- [x] **Step 4: Run reporting tests**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_plot_cci_trace.py' -v
.venv-ml/bin/python -m unittest discover -s tests -p 'test_cci_trace.py' -v
```

Expected: PASS without importing matplotlib during the CSV test.

- [x] **Step 5: Checkpoint without committing**

Run:

```bash
git diff --check
rg -n '[[:blank:]]+$' scripts/plot_cci_trace.py tests/test_plot_cci_trace.py src/cci_diff/cci_trace.py
```

Expected: no diagnostics. Do not stage or commit.

### Task 11: Sequential A0-A4 Pilot Runner, Verification Gates, And Actual MPS Runs

**Files:**
- Create: `scripts/run_clean_cci_pilot.py`
- Create: `tests/test_clean_cci_pilot.py`

**Interfaces:**
- Consumes: the clean and legacy CLI modes, CelebAMask-HQ image/mask naming, and audit JSON from Tasks 9-10.
- Produces: deterministic per-sample bindings, sequential A0-A4 runs, `pilot_manifest.json`, `pilot_results.csv`, and `pilot_summary.json` with the approved success decision.

- [ ] **Step 1: Write pilot selection and summary tests with fake audits**

Create `tests/test_clean_cci_pilot.py`. Use a temporary CelebAMask-shaped directory to prove that selection skips samples missing any required component, computes annotation subdirectory as `image_id // 2000`, and stops at exactly `limit`. Feed fake A2/A3 paired audits to `summarize_results` and assert `adaptive_supported` is true when A3 has higher flip rate, or equal flips with lower identity/locality violation.

- [ ] **Step 2: Implement deterministic sample discovery and eligibility**

In `scripts/run_clean_cci_pilot.py`, implement:

```python
def annotation_paths(mask_root: Path, image_id: int, components: tuple[str, ...]):
    stem = f"{image_id:05d}"
    directory = mask_root / str(image_id // 2000)
    return {component: directory / f"{stem}_{component}.png" for component in components}


def discover_samples(image_root, mask_root, components, *, limit, max_image_id=30000):
    selected = []
    for image_id in range(max_image_id):
        source = Path(image_root) / f"{image_id}.jpg"
        masks = annotation_paths(Path(mask_root), image_id, components)
        if source.is_file() and all(path.is_file() for path in masks.values()):
            selected.append((image_id, source, masks))
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise ValueError(f"Found only {len(selected)} complete samples; required {limit}")
    return selected
```

Before generation, score sources with the frozen CelebA classifier and retain remove-smile samples with `P(Smiling) >= 0.5` and add-blond samples with `P(Blond_Hair) < 0.5`. Run the same OpenCV source-face detection preflight and exclude a sample if no fixed identity crop can be established. Continue scanning until 15 eligible samples are found. Record all scanned IDs, source probabilities, face-detection result/box, eligibility decisions, model digest, and threshold in `pilot_manifest.json`.

- [ ] **Step 3: Implement exact ablation variants and sequential subprocess calls**

Define variants as data, not shell-string concatenation:

```python
VARIANTS = {
    "A0": {"hook": "clean_constraint", "controller_mode": "disabled", "projection": True},
    "A1": {"hook": "latent_classifier", "robust": True},
    "A2": {"hook": "clean_constraint", "controller_mode": "fixed_equal", "projection": True},
    "A3": {"hook": "clean_constraint", "controller_mode": "feedback", "projection": True},
    "A4": {"hook": "clean_constraint", "controller_mode": "feedback", "projection": False},
}
```

Build subprocess argument lists beginning with `.venv-ml/bin/python scripts/run_sd2_bld_cci.py`. All variants use the same source, strict audit mask, semantic components, prompt, seed `42`, model path `checkpoints/sd2-1-base`, 35 inference steps, guidance scale `5.0`, blending start `0.25`, MPS float32, and batch size 1. Clean variants consume a generated per-sample binding and the feature graph. A1 consumes the matching legacy intervention config and all semantic components through robust classifier guidance; record that its fixed boundary/TV coefficients are the historical A1 baseline.

Run one subprocess at a time with `check=True`. Write a `FAILED` record containing variant, sample ID, exit code, and audit path, then stop the pilot on the first failed command. Never launch parallel MPS processes.

- [ ] **Step 4: Implement paired metric extraction and support decision**

Flatten each audit into one CSV row containing feature, sample ID, variant, source/desired target probabilities, signed margin, target pass, feasibility, FaceNet cosine (`1 - identity distance`), strict and semantic inside MAE, outside-mask MAE, non-target CelebA drift, residual TV, boundary discontinuity, configured independent-audit values, identity/locality/TV pass flags, runtime, peak MPS bytes, graph digest, and trace path.

`summarize_results(rows)` must compute per-variant flip rate, feasibility rate, mean identity cosine, mean outside MAE, mean TV, median runtime, and mean delta from paired A0. Set:

```python
adaptive_supported = (
    a3_flip_rate > a2_flip_rate
    or (
        a3_flip_rate == a2_flip_rate
        and (
            a3_mean_identity_cosine > a2_mean_identity_cosine
            or a3_mean_outside_mae < a2_mean_outside_mae
        )
    )
)
```

Also report each approved threshold separately: smile A3 at least 8/15 flips; hair A3 at least 11/15; outside MAE no more than 10 percent above A0; mean identity cosine at least 0.90 and no more than 0.02 below A0; artifact review pending until manually entered; median runtime no more than 3x A0. Never convert a failed target or failed hard constraint into success via an aggregate score.

- [ ] **Step 5: Run the full CPU unit suite before any model acquisition or MPS generation**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -v
```

Expected: all existing and new tests PASS. Record the count and elapsed time in the implementation log. If any test fails, stop and fix it before continuing.

- [ ] **Step 6: Export FaceNet in an isolated compatibility venv**

These commands require user approval if network access is needed:

```bash
/opt/homebrew/bin/python3.10 -m venv .venv-facenet-export
.venv-facenet-export/bin/pip install -e '.[identity-export]'
.venv-facenet-export/bin/python scripts/download_identity_model.py --output models/facenet_vggface2.ts
.venv-ml/bin/python -c 'import torch; model=torch.jit.load("models/facenet_vggface2.ts", map_location="cpu"); x=torch.zeros((1,3,160,160), requires_grad=True); model(x).sum().backward(); print(tuple(model(x).shape), x.grad is not None)'
```

Expected: the export script prints `saved=models/facenet_vggface2.ts`, a 64-character SHA-256 digest, and `manifest=models/facenet_vggface2.ts.json`; the `.venv-ml` compatibility check prints `(1, 512) True`. Verify `.venv-ml` still reports its original torch, torchvision, NumPy, and Pillow versions and runtime loading makes no network request.

- [ ] **Step 7: Run one 35-step remove-smile mechanism check on MPS**

Run:

```bash
.venv-ml/bin/python scripts/run_sd2_bld_cci.py --output_dir outputs/clean_cci_mechanism_smile --cci_hook clean_constraint --cci_graph examples/graphs/remove_smile_clean_cci.json --cci_sample_bindings examples/bindings/sample_0_mouth.json --classifier_path models/resnet50_multilabel_model.pth --identity_model_path models/facenet_vggface2.ts --model_path checkpoints/sd2-1-base --local_files_only --device mps --torch_dtype float32 --batch_size 1 --num_inference_steps 35 --seed 42
```

Expected: process exits 0; `sd2_bld_grid.png`, `audit.json`, semantic/generation masks, and `cci_trace.jsonl` exist; every trace line parses; in-loop desired probability is no longer falsely near success solely because the latent is noisy; no MPS float64 error occurs.

- [ ] **Step 8: Run one blond-hair mechanism check and derive trace plots**

Run the same command with `examples/graphs/blond_hair_clean_cci.json`, `examples/bindings/sample_0_hair.json`, and output `outputs/clean_cci_mechanism_hair`. Installing the plot extra requires approval if it is not cached. Then run:

```bash
.venv-ml/bin/pip install -e '.[plot]'
.venv-ml/bin/python scripts/plot_cci_trace.py --trace outputs/clean_cci_mechanism_smile/cci_trace.jsonl --csv outputs/clean_cci_mechanism_smile/cci_trace.csv --png outputs/clean_cci_mechanism_smile/cci_trace.png
.venv-ml/bin/python scripts/plot_cci_trace.py --trace outputs/clean_cci_mechanism_hair/cci_trace.jsonl --csv outputs/clean_cci_mechanism_hair/cci_trace.csv --png outputs/clean_cci_mechanism_hair/cci_trace.png
```

Expected: both mechanism runs and both reporting commands exit 0; series lengths match trace record counts; each non-zero lambda has a same-step or prior positive residual explaining it.

- [ ] **Step 9: Visually inspect both mechanism outputs before the expensive pilot**

Use `view_image` on each grid, semantic mask, and generation mask. Reject continuation if masks are misaligned, the face is structurally corrupted, the edit occurs outside the semantic region, or output is blank. Record observations in each `audit.json` under a separate manual-review key; do not change computed feasibility.

- [ ] **Step 10: Run the 15-image, two-feature A0-A4 pilot sequentially**

Run:

```bash
.venv-ml/bin/python scripts/run_clean_cci_pilot.py --features smile hair --limit 15 --seed 42 --num_inference_steps 35 --device mps --model_path checkpoints/sd2-1-base --classifier_path models/resnet50_multilabel_model.pth --identity_model_path models/facenet_vggface2.ts --output_dir outputs/clean_cci_pilot_15
```

Expected: 150 sequential runs at most, no concurrent MPS process, one row per completed feature/sample/variant, and machine-readable summary files. If runtime is excessive, stop only between subprocesses, retain partial rows, and resume from existing valid audits rather than rerunning completed work.

- [ ] **Step 11: Rank candidates and state the result without cherry-picking**

Rank feasible outputs first by signed target margin, then identity cosine, then lower outside MAE; place infeasible outputs after every feasible output and include their failed constraints. Generate contact sheets by variant and a paired metric table. State one of exactly three conclusions per feature: `adaptive CCI supported`, `adaptive CCI not supported`, or `pilot incomplete`. A visually preferred but classifier-invalid result remains invalid.

- [ ] **Step 12: Final verification and no-commit checkpoint**

Run:

```bash
.venv-ml/bin/python -m unittest discover -s tests -v
.venv-ml/bin/python -m compileall -q src scripts
git diff --check
git status --short
```

Expected: unit suite PASS, compileall exits 0, diff check prints no diagnostics, and status shows only intended `cci-diff` changes plus pre-existing user changes. Do not stage or commit anything.

## Execution Notes

- Tasks are dependency ordered by task number. Complete Tasks 1 through 10 before starting the model-backed steps in Task 11.
- `A2` is an explicit fixed-equal ablation, not a recommended runtime configuration and not a return to manual semantic weights.
- `A0` uses the compiled graph only to guarantee identical source/masks and returns unmodified CFG noise.
- `A4` changes only target-priority projection; all graph settings, masks, seed, prompt, and controller feedback remain identical to A3.
- The first publishable claim is conditional on the paired pilot. If A3 does not beat or constrain A2 under the accepted rule, document the negative result rather than retuning on the test images.

### 2026-07-14 Execution Outcome

- The implementation, FaceNet export, focused tests, mechanism runs, trace reports, and manual mask/output reviews completed.
- A sequential one-sample smoke evaluated all ten combinations for smile and hair across A0-A4. Outputs and summaries are under `outputs/clean_cci_pilot_smoke/`.
- A3 achieved zero valid target flips for both features. Visual review agreed: smile remained present, while the hair result did not improve meaningfully over the prompt-driven baseline.
- The original summary could label equal zero-flip results as supported when identity or locality improved slightly. The decision gate now requires A3 target validity as well as relative improvement: 1/1 for a one-sample smoke, 8/15 for smile, and 11/15 for hair.
- Corrected smoke conclusions are `adaptive CCI not supported` for smile and hair. The expensive 15-image A0-A4 pilot was therefore not started, in accordance with the pre-pilot stop rule.
- During workspace consolidation, the authoritative repository moved to `/Users/hung.domodec.com/my-docs/cci-diff`; `/Users/hung.domodec.com/Documents/my-docs` was removed after a checksum-verified merge.
