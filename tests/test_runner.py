import json
import tempfile
import unittest
from pathlib import Path

from cci_diff.runner import run_diffusion_smoke


class TestDiffusionRunner(unittest.TestCase):
    def test_fake_backend_writes_image_and_audit_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "cci.json"
            output_dir = tmp_path / "out"
            config_path.write_text(
                json.dumps(
                    {
                        "target_concept": "smile",
                        "desired_value": 1,
                        "preserved_concepts": ["identity", "hair"],
                        "candidate_concepts": ["smile", "identity", "hair", "makeup"],
                    }
                ),
                encoding="utf-8",
            )

            result = run_diffusion_smoke(
                config_path=config_path,
                output_dir=output_dir,
                backend_name="fake",
                num_inference_steps=3,
                seed=7,
            )

            image_path = Path(result.image_path)
            audit_path = output_dir / "audit.json"

            self.assertTrue(image_path.exists())
            self.assertTrue(audit_path.exists())
            self.assertEqual(image_path.suffix, ".ppm")

            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["backend"], "fake")
            self.assertEqual(len(audit["states"]), 3)
            self.assertIn("add smile", audit["prompt"])


if __name__ == "__main__":
    unittest.main()
