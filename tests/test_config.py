import json
import tempfile
import unittest
from pathlib import Path

from cci_diff.config import load_cci_config


class TestConfigLoading(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
