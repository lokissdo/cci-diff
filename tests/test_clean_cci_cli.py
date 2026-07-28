import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from scripts.run_sd2_bld_cci import build_arg_parser, validate_mode_args


class TestCleanCCICLI(unittest.TestCase):
    def parse(self, *extra):
        return build_arg_parser().parse_args(
            [
                "--output_dir",
                "outputs/clean",
                "--cci_hook",
                "clean_constraint",
                "--cci_graph",
                "examples/graphs/remove_smile_clean_cci.json",
                "--cci_sample_bindings",
                "examples/bindings/sample_0_mouth.json",
                "--classifier_path",
                "models/resnet50_multilabel_model.pth",
                "--identity_model_path",
                "models/facenet_vggface2.ts",
                "--batch_size",
                "1",
                *extra,
            ]
        )

    def test_clean_mode_accepts_graph_and_binding_without_legacy_paths(self):
        args = self.parse()
        validate_mode_args(args)

        self.assertEqual(args.cci_hook, "clean_constraint")
        self.assertIsNone(args.cci_config)
        self.assertIsNone(args.init_image)
        self.assertIsNone(args.mask)
        self.assertEqual(args.generation_mask_dilation, 0)

    def test_clean_mode_accepts_trust_region_modes(self):
        for mode in ("fixed_trust_matched", "trust_region"):
            with self.subTest(mode=mode):
                args = self.parse("--cci_controller_mode", mode)
                validate_mode_args(args)
                self.assertEqual(args.cci_controller_mode, mode)

    def test_trust_region_mode_disables_archived_target_only_final_hook(self):
        from scripts.run_sd2_bld_cci import (
            uses_archived_final_correction,
            uses_trust_region,
        )

        self.assertTrue(uses_trust_region("trust_region"))
        self.assertTrue(uses_trust_region("fixed_trust_matched"))
        self.assertFalse(uses_trust_region("feedback"))
        self.assertFalse(uses_archived_final_correction("trust_region"))
        self.assertFalse(
            uses_archived_final_correction("fixed_trust_matched")
        )
        self.assertTrue(uses_archived_final_correction("feedback"))

    def test_clean_mode_accepts_predicted_clean_frame_directory(self):
        args = self.parse("--cci_frame_dir", "outputs/frames")

        validate_mode_args(args)

        self.assertEqual(args.cci_frame_dir, "outputs/frames")

    def test_predicted_clean_frame_writer_saves_images_and_manifest(self):
        import json

        import torch

        from scripts.run_sd2_bld_cci import PredictedCleanFrameWriter

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "frames"
            writer = PredictedCleanFrameWriter(output_dir)
            writer(
                {
                    "step": 4,
                    "timestep": 500,
                    "progress": 0.4,
                    "before_image": torch.zeros((1, 3, 2, 2)),
                    "after_image": torch.ones((1, 3, 2, 2)),
                }
            )
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertTrue((output_dir / "step_04_before.png").is_file())
            self.assertTrue((output_dir / "step_04_after.png").is_file())

        self.assertEqual(
            manifest,
            [
                {
                    "step": 4,
                    "timestep": 500,
                    "progress": 0.4,
                    "before": "step_04_before.png",
                    "after": "step_04_after.png",
                }
            ],
        )

    def test_predicted_clean_frame_writer_rejects_stale_directory(self):
        from scripts.run_sd2_bld_cci import PredictedCleanFrameWriter

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "frames"
            output_dir.mkdir()
            (output_dir / "stale.png").write_bytes(b"old")

            with self.assertRaisesRegex(FileExistsError, "not empty"):
                PredictedCleanFrameWriter(output_dir)

    def test_clean_mode_accepts_component_ablation_switches(self):
        args = self.parse(
            "--cci_disable_target_guidance",
            "--cci_disable_gradient_normalization",
            "--cci_disable_target_budget",
            "--cci_disable_guidance_schedule",
            "--cci_disable_final_correction",
        )
        validate_mode_args(args)

        self.assertTrue(args.cci_disable_target_guidance)
        self.assertTrue(args.cci_disable_gradient_normalization)
        self.assertTrue(args.cci_disable_target_budget)
        self.assertTrue(args.cci_disable_guidance_schedule)
        self.assertTrue(args.cci_disable_final_correction)

    def test_clean_mode_selects_final_correction_mask(self):
        default_args = self.parse()
        semantic_args = self.parse(
            "--cci_final_correction_mask",
            "semantic",
        )

        self.assertEqual(
            default_args.cci_final_correction_mask,
            "semantic_attribution",
        )
        self.assertEqual(semantic_args.cci_final_correction_mask, "semantic")

    def test_clean_post_attack_defaults_to_disabled(self):
        args = self.parse()

        validate_mode_args(args)

        self.assertEqual(args.cci_post_attack, "none")
        self.assertIsNone(args.cci_post_attack_epsilon)
        self.assertEqual(
            args.cci_post_attack_epsilon_schedule,
            (0.05, 0.08, 0.1, 0.3, 0.5),
        )
        self.assertEqual(args.cci_post_attack_step_size, 0.005)
        self.assertEqual(args.cci_post_attack_boundary_margin, 0.03)

    def test_clean_mode_accepts_smooth_post_attack_settings(self):
        args = self.parse(
            "--cci_post_attack",
            "smooth_boundary",
            "--cci_post_attack_epsilon",
            "0.04",
            "--cci_post_attack_step_size",
            "0.002",
            "--cci_post_attack_max_steps",
            "200",
            "--cci_post_attack_boundary_margin",
            "0.02",
            "--cci_post_attack_boundary_steps",
            "12",
            "--cci_post_attack_gaussian_kernel_size",
            "7",
            "--cci_post_attack_gaussian_sigma",
            "1.5",
        )

        validate_mode_args(args)

        self.assertEqual(args.cci_post_attack, "smooth_boundary")
        self.assertEqual(args.cci_post_attack_epsilon, 0.04)
        self.assertEqual(args.cci_post_attack_epsilon_schedule, (0.04,))
        self.assertEqual(args.cci_post_attack_gaussian_kernel_size, 7)

    def test_clean_post_attack_rejects_invalid_numerical_settings(self):
        invalid_cases = (
            ("--cci_post_attack_epsilon", "0", "positive"),
            ("--cci_post_attack_step_size", "-0.1", "positive"),
            ("--cci_post_attack_max_steps", "0", "positive"),
            ("--cci_post_attack_boundary_margin", "0.5", "margin"),
            ("--cci_post_attack_boundary_steps", "0", "positive"),
            ("--cci_post_attack_gaussian_kernel_size", "4", "odd"),
            ("--cci_post_attack_gaussian_sigma", "0", "positive"),
        )
        for option, value, message in invalid_cases:
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, message):
                    validate_mode_args(
                        self.parse(
                            "--cci_post_attack",
                            "smooth_boundary",
                            option,
                            value,
                        )
                    )

    def test_clean_post_attack_rejects_invalid_epsilon_schedules(self):
        for schedule in ("", "0.05,0.05", "0.08,0.05", "nan,0.1"):
            with self.subTest(schedule=schedule):
                with self.assertRaisesRegex(ValueError, "epsilon schedule"):
                    validate_mode_args(
                        self.parse(
                            "--cci_post_attack",
                            "smooth_boundary",
                            "--cci_post_attack_epsilon_schedule",
                            schedule,
                        )
                    )

    def test_clean_mode_accepts_nonnegative_generation_mask_dilation(self):
        args = self.parse("--generation_mask_dilation", "4")
        validate_mode_args(args)
        self.assertEqual(args.generation_mask_dilation, 4)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_mode_args(self.parse("--generation_mask_dilation", "-1"))

    def test_clean_mode_accepts_anisotropic_dilation_and_feather_override(self):
        args = self.parse(
            "--generation_mask_dilation_x",
            "12",
            "--generation_mask_dilation_y",
            "6",
            "--generation_mask_feather",
            "7",
        )

        validate_mode_args(args)

        self.assertEqual(args.generation_mask_dilation_x, 12)
        self.assertEqual(args.generation_mask_dilation_y, 6)
        self.assertEqual(args.generation_mask_feather, 7.0)

        for option in ("--generation_mask_dilation_x", "--generation_mask_dilation_y"):
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    validate_mode_args(self.parse(option, "-1"))

    def test_clean_mode_rejects_duplicate_graph_owned_cli_values(self):
        duplicate_cases = (
            ("--cci_config", "examples/remove_smile_intervention.json"),
            ("--init_image", "data/0.jpg"),
            ("--mask", "data/00000_mouth.png"),
            ("--cci_step_size", "0.2"),
            ("--cci_every_n_steps", "2"),
            ("--classifier_label_index", "31"),
        )
        for option, value in duplicate_cases:
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "single source of truth"):
                    validate_mode_args(self.parse(option, value))

    def test_clean_mode_requires_models_graph_binding_batch_one_and_float32_mps(self):
        missing = build_arg_parser().parse_args(
            ["--output_dir", "outputs/clean", "--cci_hook", "clean_constraint"]
        )
        with self.assertRaisesRegex(ValueError, "clean_constraint requires"):
            validate_mode_args(missing)
        with self.assertRaisesRegex(ValueError, "batch_size 1"):
            validate_mode_args(self.parse("--batch_size", "2"))
        with self.assertRaisesRegex(ValueError, "requires --torch_dtype float32"):
            validate_mode_args(self.parse("--device", "mps"))

    def test_legacy_mode_still_requires_config_image_and_mask(self):
        args = build_arg_parser().parse_args(["--output_dir", "outputs/legacy"])
        with self.assertRaisesRegex(ValueError, "Legacy CCI modes require"):
            validate_mode_args(args)

    def test_legacy_mode_restores_old_guidance_defaults(self):
        args = build_arg_parser().parse_args(
            [
                "--output_dir",
                "outputs/legacy",
                "--cci_config",
                "examples/remove_smile_intervention.json",
                "--init_image",
                "data/0.jpg",
                "--mask",
                "data/00000_mouth.png",
            ]
        )
        validate_mode_args(args)

        self.assertEqual(args.cci_step_size, 0.03)
        self.assertEqual(args.cci_every_n_steps, 4)
        self.assertEqual(args.cci_start_step, 0)
        self.assertEqual(args.generation_mask_feather, 3.0)

    def test_clean_options_are_rejected_in_legacy_mode(self):
        args = build_arg_parser().parse_args(
            [
                "--output_dir",
                "outputs/legacy",
                "--cci_config",
                "examples/remove_smile_intervention.json",
                "--init_image",
                "data/0.jpg",
                "--mask",
                "data/00000_mouth.png",
                "--cci_graph",
                "graph.json",
            ]
        )
        with self.assertRaisesRegex(ValueError, "Clean graph options"):
            validate_mode_args(args)

    def test_frame_directory_is_rejected_in_legacy_mode(self):
        args = build_arg_parser().parse_args(
            [
                "--output_dir",
                "outputs/legacy",
                "--cci_config",
                "examples/remove_smile_intervention.json",
                "--init_image",
                "data/0.jpg",
                "--mask",
                "data/00000_mouth.png",
                "--cci_frame_dir",
                "outputs/frames",
            ]
        )

        with self.assertRaisesRegex(ValueError, "Clean graph options"):
            validate_mode_args(args)

    def test_post_attack_is_rejected_in_legacy_mode(self):
        args = build_arg_parser().parse_args(
            [
                "--output_dir",
                "outputs/legacy",
                "--cci_config",
                "examples/remove_smile_intervention.json",
                "--init_image",
                "data/0.jpg",
                "--mask",
                "data/00000_mouth.png",
                "--cci_post_attack",
                "smooth_boundary",
            ]
        )

        with self.assertRaisesRegex(ValueError, "Clean graph options"):
            validate_mode_args(args)

    def test_repository_examples_compile_and_prepare_distinct_masks(self):
        from scripts.run_sd2_bld_cci import prepare_clean_plan

        args = self.parse()
        with TemporaryDirectory() as tmpdir:
            plan, masks = prepare_clean_plan(args, Path(tmpdir))

        self.assertEqual(plan.target.attribute_index, 31)
        self.assertEqual(plan.audit_mask_path.endswith("00000_mouth.png"), True)
        self.assertNotEqual(masks.semantic_path, masks.generation_path)

    def test_evaluator_factory_dispatches_reviewed_constraints(self):
        from scripts.run_sd2_bld_cci import (
            build_clean_evaluators,
            prepare_clean_plan,
        )

        args = self.parse()
        with TemporaryDirectory() as tmpdir:
            plan, _ = prepare_clean_plan(args, Path(tmpdir))
        target, constraints = build_clean_evaluators(
            plan,
            classifier=object(),
            identity_model=object(),
            face_detector=object(),
            classifier_input_size=512,
        )

        self.assertEqual(type(target).__name__, "CelebAAttributeTarget")
        self.assertEqual(
            [type(value).__name__ for value in constraints],
            [
                "FaceNetIdentityConstraint",
                "OutsideL1Constraint",
                "MaskedResidualTVConstraint",
            ],
        )

    def test_robust_mask_components_accept_one_or_more_and_reject_zero(self):
        from scripts.run_sd2_bld_cci import validate_robust_mask_components

        args = SimpleNamespace(
            robust_classifier_guidance=True,
            generation_mask_component=["hair.png"],
        )
        validate_robust_mask_components(args)
        args.generation_mask_component = []
        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_robust_mask_components(args)

    def test_mocked_clean_run_wires_noise_hook_masks_and_audit_provenance(self):
        import hashlib
        import json

        from PIL import Image

        from scripts import run_sd2_bld_cci as runner

        class FakeResult:
            def __init__(self, image_path):
                self.image_path = str(image_path)

            def to_dict(self):
                return {"image_path": self.image_path, "backend": "fake", "states": []}

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.png"
            Image.new("RGB", (8, 8), "gray").save(source)
            mask_paths = {}
            for role in ("mouth", "upper_lip", "lower_lip"):
                path = root / f"{role}.png"
                image = Image.new("L", (8, 8), 0)
                image.putpixel((3, 3), 255)
                image.save(path)
                mask_paths[role] = str(path)
            graph = root / "graph.json"
            graph.write_text(
                Path("examples/graphs/remove_smile_clean_cci.json").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            bindings = root / "bindings.json"
            bindings.write_text(
                json.dumps({"source_image": str(source), "masks": mask_paths}),
                encoding="utf-8",
            )
            classifier = root / "classifier.pth"
            classifier.write_bytes(b"classifier")
            identity = root / "identity.ts"
            identity.write_bytes(b"identity")
            identity_manifest = {
                "facenet_pytorch_version": "2.6.0",
                "export_torch_version": "2.2.0",
                "sha256": hashlib.sha256(b"identity").hexdigest(),
            }
            output_dir = root / "output"
            args = self.parse(
                "--cci_graph",
                str(graph),
                "--cci_sample_bindings",
                str(bindings),
                "--classifier_path",
                str(classifier),
                "--identity_model_path",
                str(identity),
                "--output_dir",
                str(output_dir),
                "--generation_mask_dilation",
                "0",
                "--generation_mask_dilation_x",
                "12",
                "--generation_mask_dilation_y",
                "6",
                "--generation_mask_feather",
                "7",
            )
            backend = mock.Mock()
            backend.device = "cpu"
            backend.scheduler = object()
            backend.vae = object()
            backend.edit_image.return_value = FakeResult(output_dir / "result.png")
            clean_hook = mock.Mock()
            clean_hook.peak_mps_bytes = None

            with (
                mock.patch.object(
                    runner,
                    "BlendedLatentDiffusionSD2Backend",
                    return_value=backend,
                ),
                mock.patch.object(runner, "load_celeba_resnet50", return_value=object()),
                mock.patch.object(runner, "load_facenet_identity", return_value=object()),
                mock.patch.object(
                    runner,
                    "load_identity_export_manifest",
                    return_value=identity_manifest,
                ),
                mock.patch.object(runner, "build_face_detector", return_value=object()),
                mock.patch.object(
                    runner,
                    "build_clean_evaluators",
                    return_value=(object(), ()),
                ),
                mock.patch.object(
                    runner,
                    "CleanCCIGuidanceHook",
                    return_value=clean_hook,
                ),
                mock.patch.object(
                    runner,
                    "build_clean_postrun_metrics",
                    return_value={"feasible": True},
                ),
            ):
                result_path = runner.run(args)

            call = backend.edit_image.call_args.kwargs
            audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))

        self.assertEqual(result_path, str(output_dir / "result.png"))
        self.assertIs(call["cci_guidance_hook"], clean_hook)
        self.assertIsNotNone(call["cci_latent_guidance_hook"])
        self.assertEqual(call["cci_latent_guidance_hook"].max_steps, 12)
        self.assertEqual(call["init_image"], str(source))
        self.assertEqual(call["mask"], mask_paths["mouth"])
        self.assertNotEqual(call["semantic_mask"], call["generation_mask"])
        self.assertEqual(len(audit["cci"]["graph_sha256"]), 64)
        self.assertEqual(audit["cci"]["controller_mode"], "feedback")
        self.assertEqual(audit["cci"]["generation_mask_dilation"], 0)
        self.assertEqual(audit["cci"]["generation_mask_dilation_x"], 12)
        self.assertEqual(audit["cci"]["generation_mask_dilation_y"], 6)
        self.assertEqual(audit["cci"]["generation_mask_feather"], 7.0)
        self.assertIsNone(audit["cci"]["final_correction"])
        self.assertIsNone(audit["cci"]["post_attack"])

    def test_clean_post_attack_writes_separate_grid_and_skips_existing_success(self):
        import numpy as np
        import torch
        from PIL import Image

        from scripts import run_sd2_bld_cci as runner

        class TinyClassifier(torch.nn.Module):
            def forward_logits(self, images):
                return images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

            def forward(self, images):
                return torch.sigmoid(self.forward_logits(images))

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.png"
            semantic_path = root / "semantic.png"
            raw_path = root / "sd2_bld_grid.png"
            Image.new("RGB", (4, 4), (220, 220, 220)).save(source_path)
            Image.new("L", (4, 4), 255).save(semantic_path)
            raw = np.zeros((4, 8, 3), dtype=np.uint8)
            raw[:, 4:] = 204
            Image.fromarray(raw, mode="RGB").save(raw_path)
            raw_bytes = raw_path.read_bytes()
            args = SimpleNamespace(
                cci_post_attack="smooth_boundary",
                cci_post_attack_epsilon=None,
                cci_post_attack_epsilon_schedule=(0.05, 0.08, 0.1),
                cci_post_attack_step_size=0.005,
                cci_post_attack_max_steps=20,
                cci_post_attack_boundary_margin=0.01,
                cci_post_attack_boundary_steps=8,
                cci_post_attack_gaussian_kernel_size=5,
                cci_post_attack_gaussian_sigma=1.0,
                batch_size=2,
                width=4,
                height=4,
            )
            plan = SimpleNamespace(
                source_image=str(source_path),
                graph=SimpleNamespace(
                    intervention=SimpleNamespace(desired_value=0)
                ),
            )
            masks = SimpleNamespace(semantic_path=str(semantic_path))
            runtime = SimpleNamespace(
                model=TinyClassifier(),
                label_index=0,
                input_size=4,
                device="cpu",
            )
            attack_calls = []

            def fake_attack(model, image, mask, **kwargs):
                attack_calls.append(
                    (image.detach().clone(), kwargs["epsilon"])
                )
                value = 0.2 if kwargs["epsilon"] == 0.05 else -2.0
                corrected = torch.full_like(image, value)
                return corrected, {
                    "after_probability": float(
                        model(corrected)[:, 0].item()
                    ),
                    "iterations": 2,
                    "boundary_iterations": 8,
                    "margin_pass": True,
                }

            metadata = runner.run_clean_post_attack(
                args=args,
                output_dir=root,
                plan=plan,
                mask_artifacts=masks,
                classifier_runtime=runtime,
                identity_model=object(),
                face_detector=object(),
                raw_output_path=raw_path,
                saliency_fn=lambda *unused_args, **unused_kwargs: np.ones(
                    (4, 4),
                    dtype=np.float32,
                ),
                attack_fn=fake_attack,
                identity_score_fn=mock.Mock(
                    side_effect=ValueError("face not detected")
                ),
            )
            corrected_path = root / "sd2_bld_grid_corrected.png"
            corrected = np.asarray(Image.open(corrected_path).convert("RGB"))
            corrected_exists = corrected_path.is_file()
            raw_bytes_after = raw_path.read_bytes()
            raw_path_text = str(raw_path)
            corrected_path_text = str(corrected_path)

        self.assertEqual(raw_bytes_after, raw_bytes)
        self.assertTrue(corrected_exists)
        self.assertEqual(len(attack_calls), 2)
        self.assertEqual([item[1] for item in attack_calls], [0.05, 0.08])
        self.assertTrue(torch.equal(attack_calls[0][0], attack_calls[1][0]))
        self.assertTrue(np.array_equal(corrected[:, :4], raw[:, :4]))
        self.assertFalse(np.array_equal(corrected[:, 4:], raw[:, 4:]))
        self.assertEqual(metadata["raw_output_path"], raw_path_text)
        self.assertEqual(metadata["corrected_output_path"], corrected_path_text)
        self.assertEqual(len(metadata["candidates"]), 2)
        self.assertTrue(metadata["candidates"][0]["already_successful"])
        self.assertFalse(metadata["candidates"][1]["already_successful"])
        self.assertTrue(metadata["candidates"][1]["target_pass"])
        self.assertEqual(
            metadata["configuration"]["epsilon_schedule"],
            [0.05, 0.08, 0.1],
        )
        self.assertEqual(metadata["candidates"][1]["selected_epsilon"], 0.08)
        self.assertTrue(metadata["candidates"][1]["escalated"])
        self.assertEqual(len(metadata["candidates"][1]["attempts"]), 2)
        self.assertIsNone(metadata["candidates"][0]["identity_before"])
        self.assertIsNone(metadata["candidates"][1]["identity_after"])
        self.assertLess(
            metadata["candidates"][1]["after_probability"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
