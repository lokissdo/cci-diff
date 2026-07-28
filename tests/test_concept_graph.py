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
    def test_trust_region_defaults_preserve_version_one_round_trip(self):
        from cci_diff.concept_graph import (
            DEFAULT_TRUST_REGION_SPEC,
            concept_graph_from_dict,
        )

        payload = valid_graph_payload()
        graph = concept_graph_from_dict(payload)

        self.assertEqual(graph.controller.trust_region, DEFAULT_TRUST_REGION_SPEC)
        self.assertEqual(graph.to_dict(), payload)

    def test_explicit_trust_region_settings_round_trip_and_validate(self):
        from cci_diff.concept_graph import concept_graph_from_dict

        payload = valid_graph_payload()
        settings = {
            "initial_radius": 0.15,
            "minimum_radius": 0.01,
            "maximum_radius": 0.30,
            "target_progress_fraction": 0.5,
            "feasibility_tolerance": 0.0001,
            "reliability_alpha_min": 0.10,
            "huber_delta": 0.02,
            "support_floor": 0.05,
            "maximum_blend_compensation": 4.0,
            "final_cumulative_radius": 0.60,
            "final_iterations": 12,
        }
        payload["controller"]["trust_region"] = settings

        graph = concept_graph_from_dict(payload)

        self.assertEqual(graph.controller.trust_region.initial_radius, 0.15)
        self.assertEqual(
            graph.to_dict()["controller"]["trust_region"],
            settings,
        )

        payload["controller"]["trust_region"]["minimum_radius"] = 0.31
        with self.assertRaisesRegex(ValueError, "minimum_radius"):
            concept_graph_from_dict(payload)

    def test_parse_and_round_trip_version_one_graph(self):
        from cci_diff.concept_graph import concept_graph_from_dict

        payload = valid_graph_payload()
        graph = concept_graph_from_dict(payload)

        self.assertEqual(graph.version, 1)
        self.assertEqual(graph.intervention.desired_value, 0)
        self.assertEqual(
            graph.region.components,
            ("mouth", "upper_lip", "lower_lip"),
        )
        self.assertEqual(graph.controller.active_progress, (0.15, 0.65))
        self.assertEqual(graph.to_dict(), payload)

    def test_load_sample_bindings_returns_immutable_mapping(self):
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
        with self.assertRaises(TypeError):
            bindings.masks["hair"] = "hair.png"

    def test_sha256_file_is_stable(self):
        from cci_diff.concept_graph import sha256_file

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "value.txt"
            path.write_bytes(b"cci")
            first = sha256_file(path)
            second = sha256_file(path)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_rejects_invalid_version_target_and_controller_bounds(self):
        from cci_diff.concept_graph import concept_graph_from_dict

        cases = (
            (("version",), 2, "Unsupported concept graph version"),
            (("intervention", "desired_value"), 2, "desired_value must be 0 or 1"),
            (("intervention", "desired_value"), True, "desired_value must be 0 or 1"),
            (
                ("intervention", "target_probability"),
                0.5,
                "target_probability",
            ),
            (("controller", "every_n_steps"), 0, "every_n_steps must be positive"),
            (("controller", "active_progress"), [0.7, 0.2], "active_progress"),
        )
        for keys, value, message in cases:
            payload = valid_graph_payload()
            cursor = payload
            for key in keys[:-1]:
                cursor = cursor[key]
            cursor[keys[-1]] = value
            with self.subTest(keys=keys, value=value):
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
            {
                "source": "identity",
                "target": "smiling",
                "relation": "measured_by",
            }
        )
        with self.assertRaisesRegex(ValueError, "acyclic"):
            concept_graph_from_dict(cycle)

        unsupported = valid_graph_payload()
        unsupported["edges"].append(
            {
                "source": "smiling",
                "target": "identity",
                "relation": "may_affect",
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported edge relation",
        ):
            concept_graph_from_dict(unsupported)

    def test_rejects_invalid_node_and_region_semantics(self):
        from cci_diff.concept_graph import concept_graph_from_dict

        payload = valid_graph_payload()
        payload["nodes"][1]["tolerance"] = 0
        with self.assertRaisesRegex(ValueError, "positive tolerance"):
            concept_graph_from_dict(payload)

        payload = valid_graph_payload()
        payload["region"]["components"] = ["mouth", "mouth"]
        with self.assertRaisesRegex(ValueError, "components must be unique"):
            concept_graph_from_dict(payload)

        payload = valid_graph_payload()
        payload["nodes"][0]["role"] = "constraint"
        payload["nodes"][0]["tolerance"] = 0.1
        with self.assertRaisesRegex(ValueError, "exactly one target"):
            concept_graph_from_dict(payload)

    def test_rejects_removed_allowed_change_role(self):
        from cci_diff.concept_graph import concept_graph_from_dict

        payload = valid_graph_payload()
        payload["nodes"].append(
            {
                "id": "mouth_open",
                "role": "allowed_change",
                "evaluator": "celeba_attribute",
                "attribute": "Mouth_Slightly_Open",
            }
        )
        payload["edges"].append(
            {
                "source": "smiling",
                "target": "mouth_open",
                "relation": "may_affect",
            }
        )

        with self.assertRaisesRegex(ValueError, "Unknown node role"):
            concept_graph_from_dict(payload)


if __name__ == "__main__":
    unittest.main()
