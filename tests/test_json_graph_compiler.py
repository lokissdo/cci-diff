import json
import tempfile
import unittest
from dataclasses import replace
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

    def compile(self, root: Path, payload, source: Path, masks):
        from cci_diff.compilers.json_graph import JsonConceptGraphCompiler
        from cci_diff.concept_graph import (
            concept_graph_from_dict,
            sample_bindings_from_dict,
        )
        from cci_diff.concept_registry import default_concept_registry

        graph_path = root / "graph.json"
        graph_path.write_text(json.dumps(payload), encoding="utf-8")
        graph = concept_graph_from_dict(payload)
        bindings = sample_bindings_from_dict(
            {"source_image": str(source), "masks": masks}
        )
        return JsonConceptGraphCompiler(graph, graph_path).compile(
            graph.intervention,
            bindings,
            default_concept_registry(),
        )

    def test_compiles_resolved_target_constraints_and_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            plan = self.compile(root, valid_graph_payload(), source, masks)

        self.assertEqual(plan.target.id, "smiling")
        self.assertEqual(plan.target.attribute_index, 31)
        self.assertEqual(plan.constraints[0].id, "identity")
        self.assertFalse(hasattr(plan, "allowed_changes"))
        self.assertEqual(plan.component_paths[0][0], "mouth")
        self.assertEqual(len(plan.graph_sha256), 64)

    def test_registry_resolves_reviewed_nodes_and_rejects_unknowns(self):
        from cci_diff.concept_graph import ConceptNode
        from cci_diff.concept_registry import default_concept_registry

        registry = default_concept_registry()
        registration, index = registry.resolve_node(
            ConceptNode("smiling", "target", "celeba_attribute", "Smiling", None)
        )
        self.assertEqual(registration.version, "celeba-resnet50-v1")
        self.assertEqual(index, 31)
        registration, index = registry.resolve_node(
            ConceptNode(
                "residual_tv",
                "audit_only",
                "masked_residual_tv",
                None,
                None,
            )
        )
        self.assertEqual(registration.version, "masked-residual-tv-v1")
        self.assertIsNone(index)

        with self.assertRaisesRegex(ValueError, "Unknown evaluator"):
            registry.resolve_node(
                ConceptNode("smiling", "target", "unknown", None, None)
            )
        with self.assertRaisesRegex(ValueError, "Unknown concept"):
            registry.resolve_node(
                ConceptNode("pose", "constraint", "facenet_identity", None, 0.1)
            )
        registry.validate_mask_role("nose")
        registry.validate_mask_role("target_region")
        with self.assertRaisesRegex(ValueError, "Unknown semantic mask role"):
            registry.validate_mask_role("forehead_zone")

    def test_rejects_missing_and_unused_bindings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            masks.pop("upper_lip")
            masks["hair"] = masks["mouth"]
            with self.assertRaisesRegex(
                ValueError,
                "missing=.*upper_lip.*unused=.*hair",
            ):
                self.compile(root, valid_graph_payload(), source, masks)

    def test_rejects_missing_mismatched_and_empty_mask_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            masks["mouth"] = str(root / "absent.png")
            with self.assertRaisesRegex(FileNotFoundError, "Mask for role 'mouth'"):
                self.compile(root, valid_graph_payload(), source, masks)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            Image.new("L", (7, 8), 255).save(masks["mouth"])
            with self.assertRaisesRegex(ValueError, "identical dimensions"):
                self.compile(root, valid_graph_payload(), source, masks)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            for path in masks.values():
                Image.new("L", (8, 8), 0).save(path)
            with self.assertRaisesRegex(ValueError, "union must be non-empty"):
                self.compile(root, valid_graph_payload(), source, masks)

    def test_rejects_unknown_mask_role_and_missing_graph_or_source(self):
        from cci_diff.compilers.json_graph import JsonConceptGraphCompiler
        from cci_diff.concept_graph import (
            concept_graph_from_dict,
            sample_bindings_from_dict,
        )
        from cci_diff.concept_registry import default_concept_registry

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            payload = valid_graph_payload()
            payload["region"]["audit_role"] = "forehead_zone"
            payload["region"]["components"][0] = "forehead_zone"
            nose_masks = dict(masks)
            nose_masks["forehead_zone"] = nose_masks.pop("mouth")
            with self.assertRaisesRegex(
                ValueError,
                "Unknown semantic mask role: forehead_zone",
            ):
                self.compile(root, payload, source, nose_masks)

            graph = concept_graph_from_dict(valid_graph_payload())
            bindings = sample_bindings_from_dict(
                {"source_image": str(source), "masks": masks}
            )
            with self.assertRaisesRegex(FileNotFoundError, "Concept graph not found"):
                JsonConceptGraphCompiler(graph, root / "absent.json").compile(
                    graph.intervention, bindings, default_concept_registry()
                )

            graph_path = root / "graph.json"
            graph_path.write_text(json.dumps(valid_graph_payload()), encoding="utf-8")
            missing_source = sample_bindings_from_dict(
                {"source_image": str(root / "absent.png"), "masks": masks}
            )
            with self.assertRaisesRegex(FileNotFoundError, "Source image not found"):
                JsonConceptGraphCompiler(graph, graph_path).compile(
                    graph.intervention,
                    missing_source,
                    default_concept_registry(),
                )

    def test_compiles_discovered_target_region_with_component_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            target_region = root / "target_region.png"
            Image.new("L", (8, 8), 0).save(target_region)
            image = Image.open(target_region)
            image.putpixel((3, 3), 255)
            image.save(target_region)
            masks["target_region"] = str(target_region)
            payload = valid_graph_payload()
            payload["region"]["audit_role"] = "target_region"

            plan = self.compile(root, payload, source, masks)

        self.assertEqual(plan.audit_mask_path, str(target_region))
        self.assertEqual(
            tuple(role for role, _ in plan.component_paths),
            ("mouth", "upper_lip", "lower_lip"),
        )

    def test_requires_reviewed_edge_relation_and_matching_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            payload = valid_graph_payload()
            payload["edges"] = []
            with self.assertRaisesRegex(ValueError, "requires 'must_preserve'"):
                self.compile(root, payload, source, masks)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            payload = valid_graph_payload()
            payload["intervention"]["concept"] = "Young"
            with self.assertRaisesRegex(ValueError, "target evaluator attribute disagree"):
                self.compile(root, payload, source, masks)

    def test_request_must_match_graph_intervention(self):
        from cci_diff.compilers.json_graph import JsonConceptGraphCompiler
        from cci_diff.concept_graph import (
            InterventionRequest,
            concept_graph_from_dict,
            sample_bindings_from_dict,
        )
        from cci_diff.concept_registry import default_concept_registry

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self.make_files(root)
            payload = valid_graph_payload()
            graph_path = root / "graph.json"
            graph_path.write_text(json.dumps(payload), encoding="utf-8")
            graph = concept_graph_from_dict(payload)
            bindings = sample_bindings_from_dict(
                {"source_image": str(source), "masks": masks}
            )
            different = InterventionRequest("Smiling", 1, 0.8)
            with self.assertRaisesRegex(ValueError, "single source of truth"):
                JsonConceptGraphCompiler(graph, graph_path).compile(
                    different,
                    bindings,
                    default_concept_registry(),
                )

    def test_equivalent_json_compiles_to_equivalent_runtime_plan(self):
        from cci_diff.compilers.json_graph import JsonConceptGraphCompiler
        from cci_diff.concept_graph import (
            concept_graph_from_dict,
            sample_bindings_from_dict,
        )
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
                    sample_bindings_from_dict(
                        {"source_image": str(source), "masks": masks}
                    ),
                    default_concept_registry(),
                )
                plans.append(replace(plan, graph_path="", graph_sha256=""))

        self.assertEqual(plans[0], plans[1])


if __name__ == "__main__":
    unittest.main()
