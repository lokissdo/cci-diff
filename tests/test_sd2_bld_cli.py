import tempfile
import unittest
from pathlib import Path

from scripts.run_sd2_bld_cci import build_arg_parser, resolve_prompt


class TestSD2BLDCLI(unittest.TestCase):
    def test_parser_accepts_required_image_mask_config_and_output_args(self):
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "--cci_config",
                "examples/smile_intervention.json",
                "--init_image",
                "data/1.jpg",
                "--mask",
                "outputs/sample_1/mask.png",
                "--output_dir",
                "outputs/sample_1",
                "--local_files_only",
            ]
        )

        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.model_path, "stabilityai/stable-diffusion-2-base")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.cci_hook, "none")
        self.assertTrue(args.local_files_only)

    def test_parser_accepts_latent_color_cci_hook_options(self):
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "--cci_config",
                "examples/hair_intervention.json",
                "--init_image",
                "data/1.jpg",
                "--mask",
                "outputs/generated_masks/1_hair.png",
                "--output_dir",
                "outputs/hair_hook",
                "--cci_hook",
                "latent_color",
                "--cci_step_size",
                "0.08",
                "--cci_every_n_steps",
                "2",
                "--cci_normalize_grad",
                "--cci_start_step",
                "3",
                "--cci_end_step",
                "15",
                "--cci_target_rgb",
                "0.9,0.7,0.3",
            ]
        )

        self.assertEqual(args.cci_hook, "latent_color")
        self.assertEqual(args.cci_step_size, 0.08)
        self.assertEqual(args.cci_every_n_steps, 2)
        self.assertTrue(args.cci_normalize_grad)
        self.assertEqual(args.cci_start_step, 3)
        self.assertEqual(args.cci_end_step, 15)
        self.assertEqual(args.cci_target_rgb, "0.9,0.7,0.3")

    def test_parser_accepts_latent_classifier_options(self):
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "--cci_config",
                "examples/remove_smile_intervention.json",
                "--init_image",
                "data/0.jpg",
                "--mask",
                "data/00000_mouth.png",
                "--output_dir",
                "outputs/remove_smile",
                "--cci_hook",
                "latent_classifier",
                "--classifier_path",
                "models/resnet50_multilabel_model.pth",
                "--classifier_label_index",
                "31",
                "--classifier_input_size",
                "512",
            ]
        )

        self.assertEqual(args.cci_hook, "latent_classifier")
        self.assertEqual(
            args.classifier_path,
            "models/resnet50_multilabel_model.pth",
        )
        self.assertEqual(args.classifier_label_index, 31)
        self.assertEqual(args.classifier_input_size, 512)

    def test_parser_accepts_robust_classifier_guidance_options(self):
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "--cci_config",
                "examples/remove_smile_intervention.json",
                "--init_image",
                "data/0.jpg",
                "--mask",
                "data/00000_mouth.png",
                "--output_dir",
                "outputs/robust",
                "--cci_hook",
                "latent_classifier",
                "--classifier_path",
                "models/resnet50_multilabel_model.pth",
                "--robust_classifier_guidance",
                "--generation_mask_component",
                "data/00000_mouth.png",
                "--generation_mask_component",
                "data/00000_u_lip.png",
                "--generation_mask_component",
                "data/00000_l_lip.png",
                "--generation_mask_feather",
                "3",
                "--classifier_scales",
                "256,384,512",
                "--classifier_blur_sigma",
                "1.0",
                "--boundary_weight",
                "0.3",
                "--tv_weight",
                "0.05",
            ]
        )

        self.assertTrue(args.robust_classifier_guidance)
        self.assertEqual(len(args.generation_mask_component), 3)
        self.assertEqual(args.generation_mask_feather, 3.0)
        self.assertEqual(args.classifier_scales, "256,384,512")
        self.assertEqual(args.boundary_weight, 0.3)
        self.assertEqual(args.tv_weight, 0.05)

    def test_parser_accepts_clip_guidance_options(self):
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "--cci_config",
                "examples/remove_smile_intervention.json",
                "--init_image",
                "data/3.jpg",
                "--mask",
                "outputs/generated_masks/3_mouth_lips.png",
                "--output_dir",
                "outputs/remove_smile_clip",
                "--cci_hook",
                "latent_classifier",
                "--classifier_path",
                "models/resnet50_multilabel_model.pth",
                "--clip_guidance_text",
                "a realistic portrait with a neutral facial expression",
                "--clip_model",
                "ViT-B-32",
                "--clip_pretrained",
                "laion2b_s34b_b79k",
                "--clip_input_size",
                "224",
            ]
        )

        self.assertEqual(
            args.clip_guidance_text,
            "a realistic portrait with a neutral facial expression",
        )
        self.assertEqual(args.clip_model, "ViT-B-32")
        self.assertEqual(args.clip_pretrained, "laion2b_s34b_b79k")
        self.assertEqual(args.clip_input_size, 224)

    def test_resolve_prompt_uses_override_when_present(self):
        prompt = resolve_prompt(
            config_path="examples/smile_intervention.json",
            override="custom prompt",
        )

        self.assertEqual(prompt, "custom prompt")

    def test_resolve_prompt_builds_prompt_from_cci_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "cci.json"
            config_path.write_text(
                """
                {
                  "target_concept": "smile",
                  "desired_value": 1,
                  "preserved_concepts": ["identity", "hair"],
                  "candidate_concepts": ["smile", "identity", "hair"]
                }
                """,
                encoding="utf-8",
            )

            prompt = resolve_prompt(config_path=config_path, override=None)

        self.assertIn("add smile", prompt)
        self.assertIn("preserve identity", prompt)

    def test_parse_target_rgb_requires_three_float_channels(self):
        from scripts.run_sd2_bld_cci import parse_target_rgb

        self.assertEqual(parse_target_rgb("0.9,0.7,0.3"), (0.9, 0.7, 0.3))
        with self.assertRaises(ValueError):
            parse_target_rgb("0.9,0.7")


if __name__ == "__main__":
    unittest.main()
