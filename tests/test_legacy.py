import unittest
from pathlib import Path

from cci_diff.legacy import build_legacy_sd2_command
from cci_diff.prompts import ConceptPrompt


class TestLegacyCommandBuilder(unittest.TestCase):
    def test_build_legacy_sd2_command_uses_prompt_and_output_paths(self):
        command = build_legacy_sd2_command(
            legacy_script=Path("../thesis_2025/bld_reranking/bld/scripts/text_editing_SD2.py"),
            init_image=Path("data/1.jpg"),
            mask=Path("data/1_mask.png"),
            classifier_path=Path("models/classifier.pth"),
            output_dir=Path("outputs/sample_1"),
            prompt=ConceptPrompt(positive="add smile", negative="do not change hair"),
            batch_size=2,
            device="cuda",
        )

        self.assertEqual(command[0], "python3")
        self.assertIn("--prompt", command)
        self.assertIn("add smile, negative: do not change hair", command)
        self.assertIn("--output_path_1", command)
        self.assertIn("outputs/sample_1/res_1.jpg", command)
        self.assertIn("--batch_size", command)
        self.assertIn("2", command)


if __name__ == "__main__":
    unittest.main()
