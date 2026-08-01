import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cci_diff.counterfactual_graph import InterventionObservation
from cci_diff.individual_region_selection import (
    load_frozen_influence_policy,
)
from cci_diff.risk_controlled_selection import (
    FEATURE_NAMES,
    FrozenSelectorArtifact,
    LogisticModel,
    PlattCalibrator,
    RiskThreshold,
    SafeSuccessThresholds,
)
from scripts.run_individual_region_cci import (
    generation_policy_signature,
    prepare_individual_policy,
    run_individual_cci,
    selector_feature_signature,
    source_requires_flip,
)


def template_graph_payload():
    return {
        "version": 1,
        "intervention": {
            "concept": "Smiling",
            "desired_value": 0,
            "target_probability": 0.8,
        },
        "region": {
            "audit_role": "mouth",
            "components": ["mouth"],
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


def influence_graph_payload(regions=("mouth", "lower_lip")):
    candidates = (("mouth",), tuple(sorted(regions)))
    return {
        "version": 1,
        "type": "classifier_counterfactual_influence",
        "graph_type": "classifier_counterfactual_influence",
        "target": "Smiling",
        "desired_value": 0,
        "selected_regions": list(regions),
        "generation_regions": list(regions),
        "candidate_region_sets": [list(item) for item in candidates],
        "fallback_regions": list(regions),
        "verified_edges": [
            {
                "source": "Smiling",
                "target": region,
                "relation": "classifier_counterfactual_influence",
            }
            for region in regions
        ],
        "region_set_evidence": [
            {
                "regions": [region],
                "mean_effect": 0.2,
                "flip_rate": 0.7,
                "effect_ci_low": 0.05,
                "mean_mask_fraction": 0.02,
            }
            for region in regions
        ]
        + [
            {
                "regions": list(regions),
                "mean_effect": 0.4,
                "flip_rate": 0.97,
                "effect_ci_low": 0.15,
                "mean_mask_fraction": 0.05,
            }
        ],
    }


def make_file_tree(tmp_path, sample_ids=(0,)):
    template = tmp_path / "template.json"
    template.write_text(json.dumps(template_graph_payload()), encoding="utf-8")
    influence = tmp_path / "influence.json"
    influence.write_text(json.dumps(influence_graph_payload()), encoding="utf-8")
    image_root = tmp_path / "images"
    image_root.mkdir()
    mask_root = tmp_path / "masks"
    (mask_root / "0").mkdir(parents=True)
    for sample_id in sample_ids:
        Image.new("RGB", (4, 4), "gray").save(
            image_root / f"{sample_id}.jpg"
        )
        mouth = np.zeros((4, 4), dtype=np.uint8)
        mouth[1, 1] = 255
        lower = np.zeros((4, 4), dtype=np.uint8)
        lower[2, 2] = 255
        Image.fromarray(mouth).save(
            mask_root / "0" / f"{sample_id:05d}_mouth.png"
        )
        Image.fromarray(lower).save(
            mask_root / "0" / f"{sample_id:05d}_l_lip.png"
        )
    classifier = tmp_path / "classifier.pth"
    classifier.write_bytes(b"classifier")
    identity = tmp_path / "identity.ts"
    identity.write_bytes(b"identity")
    model = tmp_path / "sd2"
    model.mkdir()
    return template, influence, image_root, mask_root, classifier, identity, model


def write_selector(path, args, *, evaluation_ids=()):
    policy = load_frozen_influence_policy(args.influence_graph)
    coefficients = (0.0,) * len(FEATURE_NAMES)
    artifact = FrozenSelectorArtifact(
        protocol_version=1,
        target=policy.target,
        desired_value=policy.desired_value,
        graph_sha256=policy.graph_sha256,
        candidate_region_sets=policy.candidate_region_sets,
        fallback_regions=policy.fallback_regions,
        feature_names=FEATURE_NAMES,
        feature_signature=selector_feature_signature(args, policy),
        classifier_sha256=hashlib.sha256(
            Path(args.classifier_path).read_bytes()
        ).hexdigest(),
        generation_policy_signature=generation_policy_signature(args, policy),
        model=LogisticModel(
            mean=(0.0,) * len(FEATURE_NAMES),
            scale=(1.0,) * len(FEATURE_NAMES),
            intercept=4.0,
            coefficients=coefficients,
            l2=0.01,
            iterations=1,
        ),
        calibrator=PlattCalibrator(0.0, 1.0, 1),
        risk_calibration=RiskThreshold(0.8, 60, 0, 0.04),
        coverage_threshold=0.8,
        safe_success_thresholds=SafeSuccessThresholds(),
        evaluation_sample_ids=tuple(evaluation_ids),
    )
    path.write_text(json.dumps(artifact.to_dict()), encoding="utf-8")
    return path


def test_source_requires_flip_uses_opposite_classifier_side():
    assert source_requires_flip(0.9, desired_value=0)
    assert not source_requires_flip(0.2, desired_value=0)
    assert source_requires_flip(0.2, desired_value=1)
    assert not source_requires_flip(0.9, desired_value=1)


def test_generation_policy_signature_can_bind_external_replay_policy(tmp_path):
    (
        template,
        influence,
        image_root,
        mask_root,
        classifier,
        identity,
        model,
    ) = make_file_tree(tmp_path)
    policy = load_frozen_influence_policy(influence)
    external = tmp_path / "replay_policy.json"
    external.write_text(
        json.dumps(
            {
                "variant": "A11",
                "controller": "trust_region",
                "post_attack": "smooth_boundary",
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        generation_policy_manifest=str(external),
        model_path=str(model),
        template_graph=str(template),
        seed=42,
        num_inference_steps=35,
        guidance_scale=5.0,
        blending_start_percentage=0.25,
        generation_mask_dilation=8,
        generation_mask_feather=3.0,
    )

    first = generation_policy_signature(args, policy)
    external.write_text(
        json.dumps(
            {
                "variant": "A11",
                "controller": "disabled",
                "post_attack": "smooth_boundary",
            }
        ),
        encoding="utf-8",
    )
    second = generation_policy_signature(args, policy)

    assert first != second


def test_prepare_individual_policy_materializes_selected_regions(tmp_path):
    (
        template,
        influence,
        image_root,
        mask_root,
        _,
        _,
        _,
    ) = make_file_tree(tmp_path)
    policy = load_frozen_influence_policy(influence)
    mouth = mask_root / "0" / "00000_mouth.png"
    lower = mask_root / "0" / "00000_l_lip.png"
    saliency = np.zeros((4, 4), dtype=np.float32)
    saliency[1, 1] = 4
    saliency[2, 2] = 1

    selection, graph_path, binding_path, union_path = (
        prepare_individual_policy(
            source_path=image_root / "0.jpg",
            sample_id=0,
            source_probability=0.91,
            saliency=saliency,
            component_paths={"mouth": mouth, "lower_lip": lower},
            frozen_policy=policy,
            template_graph_path=template,
            coverage_threshold=0.80,
            seed=42,
            output_dir=tmp_path / "policy",
        )
    )

    graph = json.loads(graph_path.read_text())
    binding = json.loads(binding_path.read_text())
    record = json.loads((tmp_path / "policy" / "selection.json").read_text())
    assert selection.selected_regions == ("mouth",)
    assert graph["region"]["components"] == ["mouth"]
    assert set(binding["masks"]) == {"mouth", "target_region"}
    assert np.count_nonzero(np.asarray(Image.open(union_path))) == 1
    assert record["source_probability"] == 0.91
    assert record["selected_regions"] == ["mouth"]
    assert record["influence_graph_sha256"] == policy.graph_sha256
    assert record["selection_uses_generated_output"] is False


def test_runner_calls_diffusion_at_most_once_per_image(monkeypatch, tmp_path):
    (
        template,
        influence,
        image_root,
        mask_root,
        classifier,
        identity,
        model,
    ) = make_file_tree(tmp_path, sample_ids=(0, 1))
    output_dir = tmp_path / "output"
    commands = []

    monkeypatch.setattr(
        "scripts.run_individual_region_cci.load_celeba_resnet50",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "scripts.run_individual_region_cci._compute_source_saliency",
        lambda *args, **kwargs: (
            0.9,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if len(commands) == 1 else 0)

    monkeypatch.setattr(
        "scripts.run_individual_region_cci.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "scripts.run_individual_region_cci.load_completed_observation",
        lambda run_dir, **kwargs: InterventionObservation(
            target="Smiling",
            desired_value=0,
            sample_id=kwargs["sample_id"],
            seed=kwargs["seed"],
            regions=kwargs["regions"],
            source_probability=0.9,
            output_probability=0.4,
        ),
    )
    args = Namespace(
        influence_graph=str(influence),
        template_graph=str(template),
        sample_ids=[0, 1],
        coverage_threshold=0.8,
        seed=42,
        image_root=str(image_root),
        mask_root=str(mask_root),
        model_path=str(model),
        classifier_path=str(classifier),
        identity_model_path=str(identity),
        output_dir=str(output_dir),
        device="cpu",
        classifier_input_size=4,
        num_inference_steps=5,
        guidance_scale=5.0,
        blending_start_percentage=0.25,
        generation_mask_dilation=0,
        generation_mask_feather=3.0,
        python_executable=".venv-ml/bin/python",
        continue_on_error=True,
        dry_run=False,
        discovery_manifest=None,
    )

    manifest = run_individual_cci(args)

    assert len(commands) == 2
    assert manifest["attempted_generations"] == 2
    assert manifest["failed_generations"] == 1
    assert manifest["completed_generations"] == 1
    for command in commands:
        joined = " ".join(command)
        assert "--cci_hook clean_constraint" in joined
        assert "--cci_controller_mode feedback" in joined
        assert "post_attack" not in joined
        assert "candidate" not in joined
        assert "rerank" not in joined
        assert "retry" not in joined


def test_adaptive_manifest_is_complete_before_any_generation(
    monkeypatch, tmp_path
):
    (
        template,
        influence,
        image_root,
        mask_root,
        classifier,
        identity,
        model,
    ) = make_file_tree(tmp_path, sample_ids=(0, 1))
    output_dir = tmp_path / "adaptive"
    args = Namespace(
        influence_graph=str(influence),
        template_graph=str(template),
        sample_ids=[0, 1],
        coverage_threshold=0.8,
        seed=42,
        image_root=str(image_root),
        mask_root=str(mask_root),
        model_path=str(model),
        classifier_path=str(classifier),
        identity_model_path=str(identity),
        output_dir=str(output_dir),
        device="cpu",
        classifier_input_size=4,
        num_inference_steps=5,
        guidance_scale=5.0,
        blending_start_percentage=0.25,
        generation_mask_dilation=0,
        generation_mask_feather=3.0,
        python_executable=".venv-ml/bin/python",
        continue_on_error=False,
        dry_run=False,
        selection_only=False,
        discovery_manifest=None,
        exploratory=False,
    )
    args.selector_model = str(write_selector(tmp_path / "selector.json", args))
    monkeypatch.setattr(
        "scripts.run_individual_region_cci.load_celeba_resnet50",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "scripts.run_individual_region_cci._compute_source_saliency",
        lambda *args, **kwargs: (
            0.9,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
    )
    commands = []

    def fake_run(command, **kwargs):
        manifest = json.loads(
            (output_dir / "adaptive_selection_manifest.json").read_text()
        )
        assert len(manifest["decisions"]) == 2
        assert len(manifest["manifest_sha256"]) == 64
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "scripts.run_individual_region_cci.subprocess.run", fake_run
    )
    monkeypatch.setattr(
        "scripts.run_individual_region_cci.load_completed_observation",
        lambda run_dir, **kwargs: InterventionObservation(
            target="Smiling",
            desired_value=0,
            sample_id=kwargs["sample_id"],
            seed=kwargs["seed"],
            regions=kwargs["regions"],
            source_probability=0.9,
            output_probability=0.4,
        ),
    )

    manifest = run_individual_cci(args)

    assert len(commands) == 2
    assert manifest["selection_count"] == 2
    decisions = json.loads(
        (output_dir / "adaptive_selection_manifest.json").read_text()
    )["decisions"]
    assert all("output_path" not in decision for decision in decisions)
    assert all(decision["selected_regions"] == ["mouth"] for decision in decisions)


def test_selection_only_writes_manifest_without_diffusion(monkeypatch, tmp_path):
    (
        template,
        influence,
        image_root,
        mask_root,
        classifier,
        identity,
        model,
    ) = make_file_tree(tmp_path)
    args = Namespace(
        influence_graph=str(influence),
        template_graph=str(template),
        sample_ids=[0],
        coverage_threshold=0.8,
        seed=42,
        image_root=str(image_root),
        mask_root=str(mask_root),
        model_path=str(model),
        classifier_path=str(classifier),
        identity_model_path=str(identity),
        output_dir=str(tmp_path / "selection_only"),
        device="cpu",
        classifier_input_size=4,
        num_inference_steps=5,
        guidance_scale=5.0,
        blending_start_percentage=0.25,
        generation_mask_dilation=0,
        generation_mask_feather=3.0,
        python_executable=".venv-ml/bin/python",
        continue_on_error=False,
        dry_run=False,
        selection_only=True,
        discovery_manifest=None,
        exploratory=True,
    )
    args.selector_model = str(
        write_selector(
            tmp_path / "selector.json", args, evaluation_ids=(99,)
        )
    )
    monkeypatch.setattr(
        "scripts.run_individual_region_cci.load_celeba_resnet50",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "scripts.run_individual_region_cci._compute_source_saliency",
        lambda *args, **kwargs: (0.9, np.ones((4, 4))),
    )
    monkeypatch.setattr(
        "scripts.run_individual_region_cci.subprocess.run",
        lambda *args, **kwargs: pytest.fail("diffusion must not run"),
    )

    manifest = run_individual_cci(args)

    assert manifest["attempted_generations"] == 0
    assert manifest["selection_only"] is True
    assert Path(manifest["selection_manifest_path"]).is_file()


def test_source_features_only_writes_candidate_rows_without_diffusion(
    monkeypatch, tmp_path
):
    (
        template,
        influence,
        image_root,
        mask_root,
        classifier,
        identity,
        model,
    ) = make_file_tree(tmp_path)
    args = Namespace(
        influence_graph=str(influence),
        template_graph=str(template),
        sample_ids=[0],
        coverage_threshold=0.8,
        seed=42,
        image_root=str(image_root),
        mask_root=str(mask_root),
        model_path=str(model),
        classifier_path=str(classifier),
        identity_model_path=str(identity),
        output_dir=str(tmp_path / "features"),
        device="cpu",
        classifier_input_size=4,
        num_inference_steps=5,
        guidance_scale=5.0,
        blending_start_percentage=0.25,
        generation_mask_dilation=0,
        generation_mask_feather=3.0,
        python_executable=".venv-ml/bin/python",
        continue_on_error=False,
        dry_run=False,
        source_features_only=True,
        selection_only=False,
        selector_model=None,
        discovery_manifest=None,
        exploratory=False,
    )
    monkeypatch.setattr(
        "scripts.run_individual_region_cci.load_celeba_resnet50",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "scripts.run_individual_region_cci._compute_source_saliency",
        lambda *args, **kwargs: (
            0.9,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
    )
    monkeypatch.setattr(
        "scripts.run_individual_region_cci.subprocess.run",
        lambda *args, **kwargs: pytest.fail("diffusion must not run"),
    )

    manifest = run_individual_cci(args)

    rows = list(
        __import__("csv").DictReader(
            (tmp_path / "features/selector_source_features.csv").open()
        )
    )
    assert len(rows) == 2
    assert {tuple(json.loads(row["regions"])) for row in rows} == {
        ("mouth",),
        ("lower_lip", "mouth"),
    }
    assert set(FEATURE_NAMES).issubset(rows[0])
    assert "output_path" not in rows[0]
    assert manifest["source_features_only"] is True
    assert manifest["source_feature_row_count"] == 2
    feature_manifest = json.loads(
        (tmp_path / "features/source_feature_manifest.json").read_text()
    )
    assert len(feature_manifest["generation_policy_signature"]) == 64
