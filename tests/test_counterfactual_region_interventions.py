import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.run_counterfactual_region_interventions import (
    build_arg_parser,
    build_intervention_command,
    deduplicate_region_sets,
    load_completed_observation,
    materialize_region_policy,
    summarize_cardinality,
)


def test_parser_keeps_early_stop_by_default_and_can_disable_it():
    parser = build_arg_parser()

    defaults = parser.parse_args(
        [
            "--template_graph",
            "graph.json",
            "--sample_ids",
            "1",
            "--candidate_regions",
            "mouth",
            "--classifier_path",
            "classifier.pth",
            "--identity_model_path",
            "identity.ts",
            "--output_dir",
            "output",
        ]
    )
    disabled = parser.parse_args(
        [
            "--template_graph",
            "graph.json",
            "--sample_ids",
            "1",
            "--candidate_regions",
            "mouth",
            "--classifier_path",
            "classifier.pth",
            "--identity_model_path",
            "identity.ts",
            "--output_dir",
            "output",
            "--disable_early_stop",
        ]
    )

    assert defaults.disable_early_stop is False
    assert disabled.disable_early_stop is True


def graph_payload():
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


def make_policy_inputs(tmp_path):
    source = tmp_path / "0.jpg"
    Image.new("RGB", (4, 4), "gray").save(source)
    component_paths = {}
    for index, region in enumerate(("mouth", "upper_lip", "lower_lip")):
        array = np.zeros((4, 4), dtype=np.uint8)
        array[index, index] = 255
        path = tmp_path / f"{region}.png"
        Image.fromarray(array).save(path)
        component_paths[region] = path
    graph = tmp_path / "template.json"
    graph.write_text(json.dumps(graph_payload()), encoding="utf-8")
    return graph, source, component_paths


def test_materialize_region_policy_preserves_graph_and_builds_union(tmp_path):
    graph, source, components = make_policy_inputs(tmp_path)

    graph_path, binding_path, union_path = materialize_region_policy(
        graph,
        source,
        components,
        ("mouth", "lower_lip"),
        tmp_path / "policy",
    )

    generated_graph = json.loads(graph_path.read_text())
    binding = json.loads(binding_path.read_text())
    union = np.asarray(Image.open(union_path))
    assert generated_graph["region"]["audit_role"] == "target_region"
    assert generated_graph["region"]["components"] == ["lower_lip", "mouth"]
    assert generated_graph["controller"] == graph_payload()["controller"]
    assert binding["source_image"] == str(source)
    assert set(binding["masks"]) == {"lower_lip", "mouth", "target_region"}
    assert np.count_nonzero(union) == 2


def test_deduplicate_region_sets_uses_cohort_wide_exact_unions(tmp_path):
    mask_root = tmp_path / "masks"
    group = mask_root / "0"
    group.mkdir(parents=True)
    for sample_id in (0, 1):
        skin = np.zeros((4, 4), dtype=np.uint8)
        skin[0:3, 0:3] = 255
        mouth = np.zeros((4, 4), dtype=np.uint8)
        mouth[1, 1] = 255
        lower_lip = np.zeros((4, 4), dtype=np.uint8)
        lower_lip[3, 3] = 255
        Image.fromarray(skin).save(group / f"{sample_id:05d}_skin.png")
        Image.fromarray(mouth).save(group / f"{sample_id:05d}_mouth.png")
        Image.fromarray(lower_lip).save(
            group / f"{sample_id:05d}_l_lip.png"
        )

    region_sets = (
        ("skin",),
        ("mouth",),
        ("lower_lip",),
        ("mouth", "skin"),
        ("lower_lip", "mouth"),
    )
    canonical, aliases, signatures = deduplicate_region_sets(
        region_sets,
        sample_ids=(0, 1),
        mask_root=mask_root,
    )

    assert canonical == (
        ("lower_lip",),
        ("mouth",),
        ("skin",),
        ("lower_lip", "mouth"),
    )
    assert aliases == {("mouth", "skin"): ("skin",)}
    assert signatures[("mouth", "skin")] == signatures[("skin",)]
    assert signatures[("lower_lip", "mouth")] != signatures[("skin",)]


def test_build_intervention_command_uses_clean_feedback_without_post_attack(
    tmp_path,
):
    args = Namespace(
        python_executable=".venv-ml/bin/python",
        model_path="checkpoints/sd2-1-base",
        device="mps",
        num_inference_steps=35,
        guidance_scale=5.0,
        blending_start_percentage=0.25,
        generation_mask_dilation=0,
        generation_mask_feather=3.0,
        classifier_path="models/classifier.pth",
        identity_model_path="models/facenet.ts",
    )

    command = build_intervention_command(
        args,
        graph_path=tmp_path / "graph.json",
        binding_path=tmp_path / "binding.json",
        output_dir=tmp_path / "run",
        seed=43,
        prompt="same person with a neutral expression",
    )

    joined = " ".join(command)
    assert "--seed 43" in joined
    assert "--cci_hook clean_constraint" in joined
    assert "--cci_controller_mode feedback" in joined
    assert "--cci_graph" in command
    assert "--cci_sample_bindings" in command
    assert "--classifier_path models/classifier.pth" in joined
    assert "--identity_model_path models/facenet.ts" in joined
    assert "--torch_dtype float32" in joined
    assert "post_attack" not in joined


def test_build_intervention_command_can_download_public_model(tmp_path):
    args = Namespace(
        python_executable="python",
        model_path="sd2-community/stable-diffusion-2-1",
        allow_model_download=True,
        device="cuda",
        torch_dtype="float16",
        num_inference_steps=35,
        guidance_scale=5.0,
        blending_start_percentage=0.25,
        generation_mask_dilation=0,
        generation_mask_feather=3.0,
        classifier_path="classifier.pth",
        identity_model_path="identity.ts",
    )

    command = build_intervention_command(
        args,
        graph_path=tmp_path / "graph.json",
        binding_path=tmp_path / "binding.json",
        output_dir=tmp_path / "run",
        seed=42,
        prompt="neutral expression",
    )

    assert "--local_files_only" not in command


def test_load_completed_observation_reads_audit_and_spatial_metrics(tmp_path):
    graph, source, components = make_policy_inputs(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output = run_dir / "sd2_bld_grid.png"
    changed = np.asarray(Image.open(source).convert("RGB")).copy()
    changed[0, 0] = [255, 255, 255]
    Image.fromarray(changed).save(output)
    semantic = run_dir / "semantic_mask.png"
    generation = run_dir / "generation_mask.png"
    Image.open(components["mouth"]).save(semantic)
    Image.open(components["mouth"]).save(generation)
    source_probabilities = [0.0] * 40
    output_probabilities = [0.0] * 40
    source_probabilities[31] = 0.9
    output_probabilities[31] = 0.4
    audit = {
        "image_path": str(output),
        "cci": {
            "source_image": str(source),
            "mask_artifacts": {
                "semantic_path": str(semantic),
                "generation_path": str(generation),
                "semantic_fraction": 1 / 16,
            },
            "metrics": {
                "attributes": {
                    "source_probabilities": source_probabilities,
                    "output_probabilities": output_probabilities,
                    "mean_non_target_drift": 0.02,
                },
                "identity_cosine": 0.94,
                "locality": {
                    "semantic_union": {
                        "outside_mae": 2.5,
                        "mask_fraction": 1 / 16,
                    }
                },
            },
        },
    }
    (run_dir / "audit.json").write_text(json.dumps(audit), encoding="utf-8")

    row = load_completed_observation(
        run_dir,
        target="Smiling",
        label_index=31,
        desired_value=0,
        sample_id=0,
        seed=42,
        regions=("mouth",),
    )

    assert row.source_probability == 0.9
    assert row.output_probability == 0.4
    assert row.target_pass
    assert row.identity_cosine == 0.94
    assert row.non_target_drift == 0.02
    assert row.outside_l1 == 2.5
    assert row.mask_fraction == 1 / 16
    assert row.changed_fraction is not None
    assert row.audit_path == str(run_dir / "audit.json")


def test_cardinality_summary_stops_on_target_effect_ranked_sufficient_set():
    from cci_diff.counterfactual_graph import InterventionObservation

    rows = []
    for regions, output_probability in (
        (("mouth",), 0.60),
        (("upper_lip",), 0.40),
    ):
        for sample_id in range(4):
            rows.append(
                InterventionObservation(
                    target="Smiling",
                    desired_value=0,
                    sample_id=sample_id,
                    seed=42,
                    regions=regions,
                    source_probability=0.9,
                    output_probability=output_probability,
                    mask_fraction=0.01,
                )
            )

    summary = summarize_cardinality(
        rows,
        (("mouth",), ("upper_lip",)),
        expected_rows_per_set=4,
        stop_flip_rate=0.95,
    )

    assert summary["complete"]
    assert summary["best_regions"] == ["upper_lip"]
    assert summary["best_mean_target_effect"] == 0.5
    assert summary["threshold_reached"]


def test_intervention_cli_defaults_to_auto_dtype_for_cuda_portability():
    args = build_arg_parser().parse_args(
        [
            "--template_graph",
            "graph.json",
            "--sample_ids",
            "1",
            "--candidate_regions",
            "mouth",
            "--classifier_path",
            "classifier.pth",
            "--identity_model_path",
            "identity.ts",
            "--output_dir",
            "out",
        ]
    )

    assert args.torch_dtype == "auto"
