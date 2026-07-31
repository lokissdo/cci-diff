import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image


class TestCleanCCIPilot(unittest.TestCase):
    def test_trust_region_variants_are_explicit_and_matched(self):
        from scripts.run_clean_cci_pilot import (
            CONTROLLER_VARIANTS,
            VARIANTS,
        )

        self.assertEqual(
            VARIANTS["A10"]["controller_mode"],
            "fixed_trust_matched",
        )
        self.assertEqual(
            VARIANTS["A11"]["controller_mode"],
            "trust_region",
        )
        self.assertEqual(
            CONTROLLER_VARIANTS["fixed_trust_matched"],
            "A10",
        )
        self.assertEqual(CONTROLLER_VARIANTS["trust_region"], "A11")

    def test_controller_mode_flags_resolve_fixed_and_adaptive_variants(self):
        from scripts.run_clean_cci_pilot import (
            build_arg_parser,
            resolve_requested_variants,
        )

        args = build_arg_parser().parse_args(
            [
                "--features",
                "smile",
                "--classifier_path",
                "classifier.pth",
                "--identity_model_path",
                "identity.ts",
                "--output_dir",
                "out",
                "--controller_modes",
                "fixed_equal",
                "feedback",
            ]
        )

        assert resolve_requested_variants(args) == ["A2", "A3"]
        assert args.python_executable == sys.executable
        assert args.torch_dtype == "auto"

    def test_variant_command_can_download_public_model(self):
        from scripts.run_clean_cci_pilot import (
            build_arg_parser,
            build_variant_command,
        )

        args = build_arg_parser().parse_args(
            [
                "--features",
                "smile",
                "--classifier_path",
                "classifier.pth",
                "--identity_model_path",
                "identity.ts",
                "--output_dir",
                "out",
                "--model_path",
                "sd2-community/stable-diffusion-2-1",
                "--allow_model_download",
            ]
        )
        command = build_variant_command(
            args,
            feature="smile",
            variant="A0",
            sample_id=0,
            source=Path("source.jpg"),
            masks={
                "mouth": Path("mouth.png"),
                "u_lip": Path("u_lip.png"),
                "l_lip": Path("l_lip.png"),
            },
            binding_path=Path("binding.json"),
            output_path=Path("output"),
        )

        self.assertNotIn("--local_files_only", command)

    def test_excluded_discovery_ids_are_loaded_per_feature(self):
        from scripts.run_clean_cci_pilot import load_excluded_ids

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "excluded.json"
            path.write_text(
                json.dumps({"smile": [1, 3], "hair": [2, 4]}),
                encoding="utf-8",
            )

            excluded = load_excluded_ids(path)

        self.assertEqual(excluded["smile"], {1, 3})
        self.assertEqual(excluded["hair"], {2, 4})

    def test_component_ablation_variants_forward_exactly_one_switch(self):
        from scripts.run_clean_cci_pilot import build_arg_parser, build_variant_command

        args = build_arg_parser().parse_args(
            [
                "--features", "smile",
                "--classifier_path", "classifier.pth",
                "--identity_model_path", "identity.pt",
                "--output_dir", "outputs/test",
            ]
        )
        expected = {
            "A4": "--cci_disable_target_projection",
            "A5": "--cci_disable_target_guidance",
            "A6": "--cci_disable_gradient_normalization",
            "A7": "--cci_disable_target_budget",
            "A8": "--cci_disable_guidance_schedule",
            "A9": "--cci_disable_final_correction",
        }
        all_switches = set(expected.values())

        for variant, switch in expected.items():
            with self.subTest(variant=variant):
                command = build_variant_command(
                    args,
                    feature="smile",
                    variant=variant,
                    sample_id=0,
                    source=Path("source.jpg"),
                    masks={
                        "mouth": Path("mouth.png"),
                        "u_lip": Path("u_lip.png"),
                        "l_lip": Path("l_lip.png"),
                    },
                    binding_path=Path("binding.json"),
                    output_path=Path("candidate"),
                    mask_candidate=None,
                )
                self.assertEqual(all_switches.intersection(command), {switch})
    def test_parser_accepts_selected_variants_dilations_and_continue(self):
        from scripts.run_clean_cci_pilot import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--features",
                "smile",
                "--classifier_path",
                "classifier.pth",
                "--identity_model_path",
                "identity.pt",
                "--output_dir",
                "outputs/test",
                "--variants",
                "A3",
                "--mask_dilations",
                "0",
                "4",
                "8",
                "--continue_on_error",
            ]
        )

        self.assertEqual(args.variants, ["A3"])
        self.assertEqual(args.mask_dilations, [0, 4, 8])
        self.assertTrue(args.continue_on_error)

    def test_parser_defaults_keep_all_variants_with_zero_dilation(self):
        from scripts.run_clean_cci_pilot import VARIANTS, build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--features",
                "smile",
                "--classifier_path",
                "classifier.pth",
                "--identity_model_path",
                "identity.pt",
                "--output_dir",
                "outputs/test",
            ]
        )

        self.assertEqual(args.variants, list(VARIANTS))
        self.assertEqual(args.mask_dilations, [0])
        self.assertIsNone(args.mask_shapes)
        self.assertFalse(args.continue_on_error)
        self.assertEqual(args.cci_post_attack, "none")

    def test_parser_accepts_explicit_sample_ids(self):
        from scripts.run_clean_cci_pilot import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--features",
                "smile",
                "--classifier_path",
                "classifier.pth",
                "--identity_model_path",
                "identity.pt",
                "--output_dir",
                "outputs/test",
                "--sample_ids",
                "9",
                "3",
            ]
        )

        self.assertEqual(args.sample_ids, [9, 3])

    def test_parser_accepts_canonical_smile_region_components(self):
        from scripts.run_clean_cci_pilot import (
            build_arg_parser,
            resolve_region_components,
        )

        args = build_arg_parser().parse_args(
            [
                "--features",
                "smile",
                "--classifier_path",
                "classifier.pth",
                "--identity_model_path",
                "identity.pt",
                "--output_dir",
                "outputs/test",
                "--region_components",
                "mouth",
                "upper_lip",
                "lower_lip",
            ]
        )

        self.assertEqual(
            args.region_components,
            ["mouth", "upper_lip", "lower_lip"],
        )
        self.assertEqual(
            resolve_region_components(["mouth"]),
            (("mouth",), {"mouth": "mouth"}),
        )

    def test_binding_roles_match_active_smile_region_components(self):
        from scripts.run_clean_cci_pilot import resolve_binding_roles

        self.assertEqual(
            resolve_binding_roles("smile", ["mouth"]),
            {"mouth": "mouth"},
        )
        self.assertEqual(
            resolve_binding_roles(
                "smile",
                ["mouth", "upper_lip", "lower_lip"],
            ),
            {
                "mouth": "mouth",
                "upper_lip": "u_lip",
                "lower_lip": "l_lip",
            },
        )
        self.assertEqual(
            resolve_binding_roles("hair", None),
            {"hair": "hair"},
        )

    def test_region_graph_override_does_not_modify_source_graph(self):
        from scripts.run_clean_cci_pilot import write_region_graph

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.json"
            destination = root / "generated" / "mouth.json"
            payload = {
                "version": 1,
                "region": {
                    "audit_role": "mouth",
                    "components": ["mouth", "upper_lip", "lower_lip"],
                },
            }
            source.write_text(json.dumps(payload), encoding="utf-8")

            result = write_region_graph(
                source,
                destination,
                ("mouth",),
            )

            self.assertEqual(result, destination)
            self.assertEqual(
                json.loads(destination.read_text())["region"]["components"],
                ["mouth"],
            )
            self.assertEqual(
                json.loads(source.read_text())["region"]["components"],
                ["mouth", "upper_lip", "lower_lip"],
            )

    def test_sample_ids_manifest_loads_unique_feature_cohort(self):
        from scripts.run_clean_cci_pilot import load_sample_ids_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pilot_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "features": {
                            "smile": {"selected_ids": [9, 3, 7]},
                        }
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                load_sample_ids_manifest(path, "smile"),
                [9, 3, 7],
            )

            path.write_text(
                json.dumps(
                    {
                        "features": {
                            "smile": {"selected_ids": [9, 9]},
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                load_sample_ids_manifest(path, "smile")

    def test_sampling_sources_are_mutually_exclusive(self):
        from scripts.run_clean_cci_pilot import (
            build_arg_parser,
            validate_pilot_args,
        )

        args = build_arg_parser().parse_args(
            [
                "--features",
                "smile",
                "--classifier_path",
                "classifier.pth",
                "--identity_model_path",
                "identity.pt",
                "--output_dir",
                "outputs/test",
                "--random_sample_seed",
                "42",
                "--sample_ids_manifest",
                "prior.json",
            ]
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_pilot_args(args)

    def test_region_override_rejects_non_smile_feature(self):
        from scripts.run_clean_cci_pilot import (
            build_arg_parser,
            validate_pilot_args,
        )

        args = build_arg_parser().parse_args(
            [
                "--features",
                "hair",
                "--classifier_path",
                "classifier.pth",
                "--identity_model_path",
                "identity.pt",
                "--output_dir",
                "outputs/test",
                "--region_components",
                "mouth",
            ]
        )

        with self.assertRaisesRegex(ValueError, "smile"):
            validate_pilot_args(args)

    def test_random_sample_seed_shuffles_candidate_scan_deterministically(self):
        import random

        from scripts.run_clean_cci_pilot import select_eligible_samples

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for sample_id in range(5):
                Image.new("RGB", (4, 4), "black").save(
                    root / f"{sample_id}.jpg"
                )
            mask = root / "mask.png"
            Image.new("L", (4, 4), 255).save(mask)
            args = SimpleNamespace(
                sample_ids=None,
                random_sample_seed=7,
                max_image_id=5,
                limit=2,
                image_root=str(root),
                mask_root=str(root),
                classifier_input_size=512,
                device="cpu",
                excluded_ids_by_feature={},
            )
            expected = list(range(5))
            random.Random(7).shuffle(expected)
            with mock.patch(
                "scripts.run_clean_cci_pilot.annotation_paths",
                return_value={"mouth": mask, "u_lip": mask, "l_lip": mask},
            ), mock.patch(
                "scripts.run_sd2_bld_cci.score_classifier_image_grid",
                return_value=[0.9],
            ), mock.patch(
                "scripts.run_sd2_bld_cci.load_rgb_image_tensor",
                return_value=object(),
            ), mock.patch(
                "cci_diff.identity.facenet.detect_largest_face_box",
                return_value=(0, 0, 4, 4),
            ):
                selected, _ = select_eligible_samples(
                    args,
                    feature="smile",
                    classifier=object(),
                    detector=object(),
                )

        self.assertEqual(
            [sample[0] for sample in selected],
            expected[:2],
        )

    def test_explicit_sample_ids_limit_selection_to_sorted_shard(self):
        from scripts.run_clean_cci_pilot import select_eligible_samples

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for sample_id in (3, 9):
                Image.new("RGB", (4, 4), "black").save(
                    root / f"{sample_id}.jpg"
                )
            mask = root / "mask.png"
            Image.new("L", (4, 4), 255).save(mask)
            args = SimpleNamespace(
                sample_ids=[9, 3],
                max_image_id=100,
                limit=2,
                image_root=str(root),
                mask_root=str(root),
                classifier_input_size=512,
                device="cpu",
                excluded_ids_by_feature={},
            )
            with mock.patch(
                "scripts.run_clean_cci_pilot.annotation_paths",
                return_value={"mouth": mask, "u_lip": mask, "l_lip": mask},
            ), mock.patch(
                "scripts.run_sd2_bld_cci.score_classifier_image_grid",
                return_value=[0.9],
            ), mock.patch(
                "scripts.run_sd2_bld_cci.load_rgb_image_tensor",
                return_value=object(),
            ), mock.patch(
                "cci_diff.identity.facenet.detect_largest_face_box",
                return_value=(0, 0, 4, 4),
            ):
                selected, decisions = select_eligible_samples(
                    args,
                    feature="smile",
                    classifier=object(),
                    detector=object(),
                )

        self.assertEqual([sample[0] for sample in selected], [3, 9])
        self.assertEqual([decision["image_id"] for decision in decisions], [3, 9])

    def test_raw_and_adaptive_variants_forward_post_attack(self):
        from scripts.run_clean_cci_pilot import (
            build_arg_parser,
            build_variant_command,
        )

        args = build_arg_parser().parse_args(
            [
                "--features",
                "smile",
                "--classifier_path",
                "classifier.pth",
                "--identity_model_path",
                "identity.pt",
                "--output_dir",
                "outputs/test",
                "--variants",
                "A0",
                "A11",
                "--cci_post_attack",
                "smooth_boundary",
                "--cci_post_attack_epsilon_schedule",
                "0.05,0.08,0.10,0.30,0.50",
                "--cci_post_attack_boundary_margin",
                "0.03",
            ]
        )

        for variant in ("A0", "A11"):
            with self.subTest(variant=variant):
                command = build_variant_command(
                    args,
                    feature="smile",
                    variant=variant,
                    sample_id=0,
                    source=Path("source.jpg"),
                    masks={
                        "mouth": Path("mouth.png"),
                        "u_lip": Path("u_lip.png"),
                        "l_lip": Path("l_lip.png"),
                    },
                    binding_path=Path("binding.json"),
                    output_path=Path("candidate"),
                )

                index = command.index("--cci_post_attack")
                self.assertEqual(command[index + 1], "smooth_boundary")
                schedule_index = command.index(
                    "--cci_post_attack_epsilon_schedule"
                )
                self.assertEqual(
                    command[schedule_index + 1],
                    "0.05,0.08,0.10,0.30,0.50",
                )
                margin_index = command.index(
                    "--cci_post_attack_boundary_margin"
                )
                self.assertEqual(command[margin_index + 1], "0.03")

    def test_candidate_row_scores_corrected_post_attack_artifact(self):
        from scripts.run_clean_cci_pilot import MaskCandidate, _candidate_row

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, _ = self._sample_files(root, 0)
            run_dir = root / "candidate"
            self._write_candidate(run_dir, 0.2, changed_pixels=1)
            corrected = run_dir / "sd2_bld_grid_corrected.png"
            Image.new("RGB", (4, 4), "white").save(corrected)
            audit_path = run_dir / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["cci"]["post_attack"] = {
                "corrected_output_path": str(corrected),
                "candidates": [
                    {
                        "after_probability": 0.42,
                        "target_pass": True,
                        "margin_pass": True,
                        "identity_after": 0.93,
                        "selected_epsilon": 0.08,
                        "escalated": True,
                        "mean_abs_change": 0.001,
                        "linf": 0.02,
                        "changed_fraction": 0.03,
                    }
                ],
            }
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            row = _candidate_row(
                run_dir,
                feature="smile",
                sample_id=0,
                variant="A3",
                candidate=MaskCandidate("x4_y4_f3", 0, 4, 4, 3),
                source=source,
            )

        self.assertEqual(row["output_path"], str(corrected))
        self.assertAlmostEqual(row["desired_probability"], 0.58)
        self.assertTrue(row["target_pass"])
        self.assertEqual(row["identity_cosine"], 0.93)
        self.assertEqual(row["post_attack_selected_epsilon"], 0.08)

    def test_candidate_row_falls_back_when_post_attack_identity_is_missing(self):
        from scripts.run_clean_cci_pilot import MaskCandidate, _candidate_row

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, _ = self._sample_files(root, 0)
            run_dir = root / "candidate"
            self._write_candidate(run_dir, 0.2, changed_pixels=1)
            audit_path = run_dir / "audit.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["cci"]["metrics"]["identity_cosine"] = 0.91
            audit["cci"]["post_attack"] = {
                "corrected_output_path": str(run_dir / "sd2_bld_grid.png"),
                "candidates": [
                    {
                        "after_probability": 0.42,
                        "target_pass": True,
                        "identity_after": None,
                    }
                ],
            }
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            row = _candidate_row(
                run_dir,
                feature="smile",
                sample_id=0,
                variant="A3",
                candidate=MaskCandidate("x4_y4_f3", 0, 4, 4, 3),
                source=source,
            )

        self.assertEqual(row["identity_cosine"], 0.91)

    def test_mask_shapes_replace_scalar_candidates_and_forward_geometry(self):
        from scripts.run_clean_cci_pilot import (
            build_arg_parser,
            build_variant_command,
            resolve_mask_candidates,
        )

        args = build_arg_parser().parse_args(
            [
                "--features",
                "smile",
                "--classifier_path",
                "classifier.pth",
                "--identity_model_path",
                "identity.pt",
                "--output_dir",
                "outputs/test",
                "--variants",
                "A3",
                "--mask_shapes",
                "4,4,3",
                "8,4,5",
                "12,6,7",
                "16,8,9",
            ]
        )

        candidates = resolve_mask_candidates(args)

        self.assertEqual(
            [candidate.label for candidate in candidates],
            ["x4_y4_f3", "x8_y4_f5", "x12_y6_f7", "x16_y8_f9"],
        )
        command = build_variant_command(
            args,
            feature="smile",
            variant="A3",
            sample_id=0,
            source=Path("source.jpg"),
            masks={
                "mouth": Path("mouth.png"),
                "u_lip": Path("u_lip.png"),
                "l_lip": Path("l_lip.png"),
            },
            binding_path=Path("binding.json"),
            output_path=Path("candidate"),
            mask_candidate=candidates[2],
        )
        expected = {
            "--generation_mask_dilation": "0",
            "--generation_mask_dilation_x": "12",
            "--generation_mask_dilation_y": "6",
            "--generation_mask_feather": "7.0",
        }
        for option, value in expected.items():
            with self.subTest(option=option):
                self.assertEqual(command[command.index(option) + 1], value)

    def test_variant_command_passes_generation_mask_dilation(self):
        from scripts.run_clean_cci_pilot import build_arg_parser, build_variant_command

        args = build_arg_parser().parse_args(
            [
                "--features",
                "smile",
                "--classifier_path",
                "classifier.pth",
                "--identity_model_path",
                "identity.pt",
                "--output_dir",
                "outputs/test",
            ]
        )
        command = build_variant_command(
            args,
            feature="smile",
            variant="A3",
            sample_id=0,
            source=Path("source.jpg"),
            masks={
                "mouth": Path("mouth.png"),
                "u_lip": Path("u_lip.png"),
                "l_lip": Path("l_lip.png"),
            },
            binding_path=Path("binding.json"),
            output_path=Path("candidate"),
            dilation=4,
        )

        index = command.index("--generation_mask_dilation")
        self.assertEqual(command[index + 1], "4")

    def test_a3_candidates_resume_and_materialize_target_first_selection(self):
        from scripts.run_clean_cci_pilot import build_arg_parser, run_pilot

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source, masks = self._sample_files(root, 0)
            args = build_arg_parser().parse_args(
                [
                    "--features",
                    "smile",
                    "--limit",
                    "1",
                    "--classifier_path",
                    "classifier.pth",
                    "--identity_model_path",
                    "identity.pt",
                    "--output_dir",
                    str(root / "outputs"),
                    "--variants",
                    "A3",
                    "--mask_dilations",
                    "0",
                    "4",
                    "8",
                ]
            )
            calls = []

            def generation(command, check=False):
                calls.append(command)
                dilation = int(command[command.index("--generation_mask_dilation") + 1])
                run_dir = Path(command[command.index("--output_dir") + 1])
                desired = {0: 0.79, 4: 0.82, 8: 0.91}[dilation]
                self._write_candidate(run_dir, desired, changed_pixels={0: 1, 4: 2, 8: 3}[dilation])
                return SimpleNamespace(returncode=0)

            patches = self._pilot_patches(source, masks, generation)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                run_pilot(args)

            self.assertEqual(len(calls), 3)
            result_dir = root / "outputs" / "smile" / "00000" / "A3"
            selection = json.loads((result_dir / "selected.json").read_text())
            self.assertEqual(selection["dilation"], 4)
            self.assertTrue((result_dir / "input.jpg").is_file())
            self.assertTrue((result_dir / "input_output.jpg").is_file())
            self.assertTrue((root / "outputs" / "candidate_results.csv").is_file())

            resumed = mock.Mock(side_effect=AssertionError("generation should resume"))
            patches = self._pilot_patches(source, masks, resumed)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                run_pilot(args)
            resumed.assert_not_called()

    def test_continue_on_error_runs_later_candidates_and_samples(self):
        from scripts.run_clean_cci_pilot import build_arg_parser, run_pilot

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source0, masks0 = self._sample_files(root, 0)
            source1, masks1 = self._sample_files(root, 1)
            args = build_arg_parser().parse_args(
                [
                    "--features",
                    "smile",
                    "--limit",
                    "2",
                    "--classifier_path",
                    "classifier.pth",
                    "--identity_model_path",
                    "identity.pt",
                    "--output_dir",
                    str(root / "outputs"),
                    "--variants",
                    "A3",
                    "--mask_dilations",
                    "0",
                    "4",
                    "8",
                    "--continue_on_error",
                ]
            )
            calls = []

            def generation(command, check=False):
                calls.append(command)
                run_dir = Path(command[command.index("--output_dir") + 1])
                dilation = int(command[command.index("--generation_mask_dilation") + 1])
                if "00000" in run_dir.parts and dilation == 4:
                    return SimpleNamespace(returncode=9)
                self._write_candidate(run_dir, 0.85, changed_pixels=2)
                return SimpleNamespace(returncode=0)

            selected = [(0, source0, masks0), (1, source1, masks1)]
            with (
                mock.patch(
                    "scripts.run_clean_cci_pilot.select_eligible_samples",
                    return_value=(selected, []),
                ),
                mock.patch(
                    "cci_diff.classifiers.celeba_resnet50.load_celeba_resnet50",
                    return_value=object(),
                ),
                mock.patch(
                    "cci_diff.identity.facenet.build_face_detector",
                    return_value=object(),
                ),
                mock.patch("cci_diff.concept_graph.sha256_file", return_value="sha"),
                mock.patch(
                    "scripts.run_clean_cci_pilot.subprocess.run",
                    side_effect=generation,
                ),
            ):
                summary = run_pilot(args)

            self.assertEqual(len(calls), 6)
            self.assertEqual(len(summary["unresolved_candidates"]), 1)
            self.assertTrue(
                (root / "outputs" / "smile" / "00001" / "A3" / "selected.json").is_file()
            )
            failures = (root / "outputs" / "failures.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(failures[0])["dilation"], 4)

    def test_direct_script_can_import_generation_helpers_outside_repo_cwd(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_clean_cci_pilot.py"
        code = (
            "import runpy; from types import SimpleNamespace; "
            f"ns=runpy.run_path({str(script)!r}, run_name='pilot_module'); "
            "args=SimpleNamespace(max_image_id=0, limit=0, "
            "image_root='unused', mask_root='unused', "
            "classifier_input_size=512, device='cpu'); "
            "ns['select_eligible_samples'](args, feature='smile', "
            "classifier=None, detector=None)"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root / "src")

        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_annotation_paths_use_celeba_partition_directory(self):
        from scripts.run_clean_cci_pilot import annotation_paths

        root = Path("masks")
        paths = annotation_paths(root, 2001, ("mouth", "u_lip"))

        self.assertEqual(paths["mouth"], root / "1" / "02001_mouth.png")
        self.assertEqual(paths["u_lip"], root / "1" / "02001_u_lip.png")

    def test_discovery_skips_incomplete_samples_and_stops_at_limit(self):
        from scripts.run_clean_cci_pilot import discover_samples

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            for image_id in (0, 1, 2001):
                (images / f"{image_id}.jpg").write_bytes(b"image")
                directory = masks / str(image_id // 2000)
                directory.mkdir(parents=True, exist_ok=True)
                stem = f"{image_id:05d}"
                (directory / f"{stem}_mouth.png").write_bytes(b"mask")
                if image_id != 1:
                    (directory / f"{stem}_u_lip.png").write_bytes(b"mask")

            selected = discover_samples(
                images,
                masks,
                ("mouth", "u_lip"),
                limit=2,
                max_image_id=2002,
            )

        self.assertEqual([sample[0] for sample in selected], [0, 2001])

    def test_discovery_rejects_too_few_complete_samples(self):
        from scripts.run_clean_cci_pilot import discover_samples

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "Found only 0 complete samples"):
                discover_samples(
                    root / "images",
                    root / "masks",
                    ("hair",),
                    limit=1,
                    max_image_id=2,
                )

    def test_variants_preserve_exact_ablation_meanings(self):
        from scripts.run_clean_cci_pilot import VARIANTS

        self.assertEqual(VARIANTS["A0"]["controller_mode"], "disabled")
        self.assertTrue(VARIANTS["A1"]["robust"])
        self.assertEqual(VARIANTS["A2"]["controller_mode"], "fixed_equal")
        self.assertEqual(VARIANTS["A3"]["controller_mode"], "feedback")
        self.assertFalse(VARIANTS["A4"]["projection"])

    def test_graph_prompt_excludes_measurement_constraint_ids(self):
        from scripts.run_clean_cci_pilot import FEATURES, _prompt_for_graph

        prompt = _prompt_for_graph(FEATURES["smile"]["graph"])

        self.assertIn("closed relaxed lips", prompt)
        self.assertIn("preserve identity", prompt)
        self.assertNotIn("outside_locality", prompt)
        self.assertNotIn("residual_tv", prompt)

    def test_summary_supports_adaptive_when_flip_rate_is_higher(self):
        from scripts.run_clean_cci_pilot import summarize_results

        rows = self.rows(a2_flips=(True, False), a3_flips=(True, True))
        summary = summarize_results(rows)

        self.assertTrue(summary["adaptive_supported"])
        self.assertGreater(
            summary["features"]["smile"]["variants"]["A3"]["flip_rate"],
            summary["features"]["smile"]["variants"]["A2"]["flip_rate"],
        )

    def test_summary_supports_equal_flips_with_better_identity_or_locality(self):
        from scripts.run_clean_cci_pilot import summarize_results

        rows = self.rows(a2_flips=(True, True), a3_flips=(True, True))
        for row in rows:
            if row["variant"] == "A2":
                row["identity_cosine"] = 0.90
                row["semantic_outside_mae"] = 0.04
            if row["variant"] == "A3":
                row["identity_cosine"] = 0.94
                row["semantic_outside_mae"] = 0.03

        summary = summarize_results(rows)

        self.assertTrue(summary["adaptive_supported"])

    def test_summary_rejects_adaptive_when_no_target_flip_succeeds(self):
        from scripts.run_clean_cci_pilot import summarize_results

        rows = self.rows(a2_flips=(False, False), a3_flips=(False, False))
        for row in rows:
            if row["variant"] == "A3":
                row["identity_cosine"] = 0.99

        summary = summarize_results(rows)

        self.assertFalse(summary["adaptive_supported"])
        self.assertEqual(
            summary["features"]["smile"]["conclusion"],
            "adaptive CCI not supported",
        )

    def test_remove_smile_row_reports_probability_of_desired_zero_value(self):
        from scripts.run_clean_cci_pilot import extract_audit_row

        source = [0.0] * 40
        output = [0.0] * 40
        source[31] = 0.95
        output[31] = 0.90
        audit = {
            "cci": {
                "metrics": {
                    "attributes": {
                        "source_probabilities": source,
                        "output_probabilities": output,
                    }
                }
            }
        }

        row = extract_audit_row(
            audit,
            feature="smile",
            sample_id=0,
            variant="A1",
            output_path=Path("output.png"),
        )

        self.assertAlmostEqual(row["source_probability"], 0.95)
        self.assertAlmostEqual(row["desired_probability"], 0.10)
        self.assertFalse(row["target_pass"])

    @staticmethod
    def rows(*, a2_flips, a3_flips):
        rows = []
        for sample_id in range(2):
            rows.append(
                {
                    "feature": "smile",
                    "sample_id": sample_id,
                    "variant": "A0",
                    "target_pass": False,
                    "feasible": False,
                    "identity_cosine": 0.96,
                    "semantic_outside_mae": 0.02,
                    "residual_tv": 0.01,
                    "runtime_seconds": 1.0,
                }
            )
            for variant, flips in (("A2", a2_flips), ("A3", a3_flips)):
                rows.append(
                    {
                        "feature": "smile",
                        "sample_id": sample_id,
                        "variant": variant,
                        "target_pass": flips[sample_id],
                        "feasible": flips[sample_id],
                        "identity_cosine": 0.92,
                        "semantic_outside_mae": 0.03,
                        "residual_tv": 0.02,
                        "runtime_seconds": 2.0,
                    }
                )
        return rows

    @staticmethod
    def _sample_files(root, sample_id):
        source = root / f"{sample_id}.jpg"
        Image.new("RGB", (4, 4), "black").save(source)
        mask_dir = root / "masks"
        mask_dir.mkdir(exist_ok=True)
        masks = {}
        for component in ("mouth", "u_lip", "l_lip"):
            path = mask_dir / f"{component}.png"
            Image.new("L", (4, 4), 255).save(path)
            masks[component] = path
        return source, masks

    @staticmethod
    def _write_candidate(run_dir, desired_probability, changed_pixels):
        run_dir.mkdir(parents=True, exist_ok=True)
        output = Image.new("RGB", (4, 4), "black")
        for index in range(changed_pixels):
            output.putpixel((index, 0), (255, 255, 255))
        output_path = run_dir / "sd2_bld_grid.png"
        output.save(output_path)
        Image.new("L", (4, 4), 255).save(run_dir / "semantic_mask.png")
        Image.new("L", (4, 4), 255).save(run_dir / "generation_mask.png")
        probabilities = [0.0] * 40
        probabilities[31] = 1.0 - desired_probability
        audit = {
            "image_path": str(output_path),
            "cci": {
                "metrics": {
                    "attributes": {
                        "source_probabilities": [0.95] * 40,
                        "output_probabilities": probabilities,
                    },
                    "identity_cosine": 0.95,
                }
            },
        }
        (run_dir / "audit.json").write_text(json.dumps(audit), encoding="utf-8")

    @staticmethod
    def _pilot_patches(source, masks, generation):
        selected = [(0, source, masks)]
        return (
            mock.patch(
                "scripts.run_clean_cci_pilot.select_eligible_samples",
                return_value=(selected, [{"image_id": 0, "eligible": True}]),
            ),
            mock.patch(
                "cci_diff.classifiers.celeba_resnet50.load_celeba_resnet50",
                return_value=object(),
            ),
            mock.patch(
                "cci_diff.identity.facenet.build_face_detector",
                return_value=object(),
            ),
            mock.patch("cci_diff.concept_graph.sha256_file", return_value="sha"),
            mock.patch("scripts.run_clean_cci_pilot.subprocess.run", side_effect=generation),
        )


if __name__ == "__main__":
    unittest.main()
