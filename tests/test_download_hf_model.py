import argparse
import unittest

from scripts.download_hf_model import (
    build_arg_parser,
    format_download_error,
    validate_model_id,
)


class TestDownloadHFModelCLI(unittest.TestCase):
    def test_parser_defaults_to_sd2_checkpoint_dir(self):
        parser = build_arg_parser()

        args = parser.parse_args([])

        self.assertEqual(args.model_id, "stabilityai/stable-diffusion-2-base")
        self.assertEqual(args.local_dir, "checkpoints/sd2-base")

    def test_format_download_error_explains_gated_or_private_repo(self):
        args = argparse.Namespace(
            model_id="stabilityai/stable-diffusion-2-base",
            local_dir="checkpoints/sd2-base",
        )

        message = format_download_error(args, Exception("Repository Not Found"))

        self.assertIn("accept access", message)
        self.assertIn("huggingface-cli login", message)
        self.assertIn("--token", message)

    def test_validate_model_id_rejects_wrong_sd21_base_repo(self):
        with self.assertRaises(SystemExit) as ctx:
            validate_model_id("stabilityai/stable-diffusion-2-1-base")

        self.assertIn("stabilityai/stable-diffusion-2-1", str(ctx.exception))
        self.assertIn("stabilityai/stable-diffusion-2-base", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
