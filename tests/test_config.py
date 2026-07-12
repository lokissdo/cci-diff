import json
import tempfile
import unittest
from pathlib import Path

from cci_diff.config import load_cci_config
from cci_diff.prompts import build_concept_prompt


class TestConfigLoading(unittest.TestCase):
    def test_remove_smile_config_activates_classifier_and_outside_losses(self):
        config = load_cci_config("examples/remove_smile_intervention.json")

        self.assertEqual(config.intervention.target_concept, "smile")
        self.assertEqual(config.intervention.desired_value, 0)
        self.assertEqual(config.weights.target, 0.0)
        self.assertEqual(config.weights.preservation, 0.0)
        self.assertEqual(config.weights.leakage, 0.0)
        self.assertEqual(config.weights.classifier, 1.0)
        self.assertEqual(config.weights.outside_mask, 1.0)

    def test_load_cci_config_builds_intervention_and_weights(self):
        config = {
            "target_concept": "smile",
            "desired_value": 1,
            "preserved_concepts": ["identity", "hair"],
            "candidate_concepts": ["smile", "identity", "hair", "makeup"],
            "weights": {
                "target": 2.0,
                "preservation": 1.5,
                "leakage": 0.75,
                "classifier": 1.25,
                "outside_mask": 1.0,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cci.json"
            path.write_text(json.dumps(config), encoding="utf-8")

            loaded = load_cci_config(path)

        self.assertEqual(loaded.intervention.target_concept, "smile")
        self.assertEqual(loaded.intervention.audit_concepts, ("makeup",))
        self.assertEqual(loaded.weights.target, 2.0)
        self.assertEqual(loaded.weights.preservation, 1.5)

    def test_hair_example_targets_hair_without_preserving_hair(self):
        loaded = load_cci_config("examples/hair_intervention.json")

        prompt = build_concept_prompt(loaded.intervention)

        self.assertEqual(loaded.intervention.target_concept, "blond hair")
        self.assertNotIn("hair", loaded.intervention.preserved_concepts)
        self.assertIn("add blond hair", prompt.positive)
        self.assertNotIn("preserve hair", prompt.positive)


if __name__ == "__main__":
    unittest.main()
