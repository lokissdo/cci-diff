import json
from pathlib import Path

import pytest
from PIL import Image


def _ablation_args(tmp_path):
    from scripts.run_final_restoration_ablation import build_arg_parser

    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    model_path = tmp_path / "model"
    image_root.mkdir()
    (mask_root / "13").mkdir(parents=True)
    model_path.mkdir()
    Image.new("RGB", (4, 4), "white").save(image_root / "26811.jpg")
    for component in ("mouth", "u_lip", "l_lip"):
        Image.new("L", (4, 4), 255).save(
            mask_root / "13" / f"26811_{component}.png"
        )
    classifier_path = tmp_path / "classifier.pth"
    identity_path = tmp_path / "identity.ts"
    classifier_path.write_bytes(b"classifier")
    identity_path.write_bytes(b"identity")
    return build_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "output"),
            "--image-root",
            str(image_root),
            "--mask-root",
            str(mask_root),
            "--model-path",
            str(model_path),
            "--classifier-path",
            str(classifier_path),
            "--identity-model-path",
            str(identity_path),
        ]
    )


def _normalized_command(command):
    normalized = list(command)
    normalized[normalized.index("--output_dir") + 1] = "<OUTPUT>"
    return normalized


def test_prepare_ablation_changes_only_output_and_final_restoration_flag(tmp_path):
    from scripts.run_final_restoration_ablation import prepare_ablation

    prepared = prepare_ablation(_ablation_args(tmp_path))
    enabled = prepared["enabled_command"]
    disabled = prepared["disabled_command"]

    assert "--cci_post_attack" not in enabled
    assert "--cci_disable_final_correction" not in enabled
    assert disabled[-1] == "--cci_disable_final_correction"
    assert _normalized_command(disabled[:-1]) == _normalized_command(enabled)

    graph = json.loads(Path(prepared["graph"]).read_text(encoding="utf-8"))
    binding = json.loads(
        Path(prepared["binding"]).read_text(encoding="utf-8")
    )
    assert graph["region"]["components"] == ["mouth"]
    assert set(binding["masks"]) == {"mouth"}


def _write_case(
    path,
    *,
    pixel_value,
    smiling_probability,
    restoration,
    post_attack=None,
    identity=0.95,
    drift=0.02,
):
    path.mkdir(parents=True)
    Image.new("RGB", (4, 4), (pixel_value,) * 3).save(
        path / "sd2_bld_grid.png"
    )
    probabilities = [0.1] * 40
    probabilities[31] = smiling_probability
    audit = {
        "cci": {
            "post_attack": post_attack,
            "trust_region_final_restoration": restoration,
            "wall_seconds": 12.5,
            "metrics": {
                "attributes": {
                    "output_probabilities": probabilities,
                    "mean_non_target_drift": drift,
                },
                "identity_cosine": identity,
            },
        }
    }
    (path / "audit.json").write_text(
        json.dumps(audit),
        encoding="utf-8",
    )


def test_compare_ablation_writes_metrics_and_visual_artifacts(tmp_path):
    from scripts.run_final_restoration_ablation import compare_ablation

    before = tmp_path / "before"
    after = tmp_path / "after"
    report = tmp_path / "report"
    _write_case(
        before,
        pixel_value=0,
        smiling_probability=0.8,
        restoration=None,
    )
    _write_case(
        after,
        pixel_value=10,
        smiling_probability=0.2,
        restoration={
            "initial_probability": 0.2,
            "final_probability": 0.8,
            "accepted_steps": 3,
            "attempts": [],
        },
    )

    result = compare_ablation(
        before,
        after,
        report,
        consistency_tolerance=0.01,
        commands={"disabled": ["before"], "enabled": ["after"]},
    )

    assert result["pixel"]["mean_absolute_difference"] == pytest.approx(
        10 / 255
    )
    assert result["pixel"]["maximum_absolute_difference"] == pytest.approx(
        10 / 255
    )
    assert result["pixel"]["changed_fraction"] == 1.0
    assert result["restoration"]["initial_probability"] == pytest.approx(0.2)
    assert result["restoration"]["final_probability"] == pytest.approx(0.8)
    assert result["consistency"]["passed"]
    assert Path(result["artifacts"]["side_by_side"]).is_file()
    assert Path(result["artifacts"]["difference_amplified"]).is_file()
    assert (report / "comparison.json").is_file()
    assert (report / "comparison.md").is_file()


def test_compare_ablation_rejects_post_attack_output(tmp_path):
    from scripts.run_final_restoration_ablation import compare_ablation

    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_case(
        before,
        pixel_value=0,
        smiling_probability=0.8,
        restoration=None,
    )
    _write_case(
        after,
        pixel_value=10,
        smiling_probability=0.2,
        restoration={
            "initial_probability": 0.2,
            "final_probability": 0.8,
            "accepted_steps": 3,
            "attempts": [],
        },
        post_attack={"mode": "smooth_boundary"},
    )

    with pytest.raises(ValueError, match="post-attack"):
        compare_ablation(
            before,
            after,
            tmp_path / "report",
            consistency_tolerance=0.01,
            commands={},
        )


def test_compare_ablation_rejects_inconsistent_separate_runs(tmp_path):
    from scripts.run_final_restoration_ablation import compare_ablation

    before = tmp_path / "before"
    after = tmp_path / "after"
    _write_case(
        before,
        pixel_value=0,
        smiling_probability=0.8,
        restoration=None,
    )
    _write_case(
        after,
        pixel_value=10,
        smiling_probability=0.2,
        restoration={
            "initial_probability": 0.4,
            "final_probability": 0.8,
            "accepted_steps": 3,
            "attempts": [],
        },
    )

    with pytest.raises(ValueError, match="consistency"):
        compare_ablation(
            before,
            after,
            tmp_path / "report",
            consistency_tolerance=0.01,
            commands={},
        )


def test_run_executes_disabled_then_enabled_and_writes_comparison(
    tmp_path,
    monkeypatch,
):
    from scripts.run_final_restoration_ablation import run

    args = _ablation_args(tmp_path)
    calls = []

    def fake_run(command, *, check, cwd):
        assert check
        calls.append(list(command))
        case_dir = Path(command[command.index("--output_dir") + 1])
        disabled = "--cci_disable_final_correction" in command
        _write_case(
            case_dir,
            pixel_value=0 if disabled else 10,
            smiling_probability=0.8 if disabled else 0.2,
            restoration=(
                None
                if disabled
                else {
                    "initial_probability": 0.2,
                    "final_probability": 0.8,
                    "accepted_steps": 3,
                    "attempts": [],
                }
            ),
        )

    monkeypatch.setattr(
        "scripts.run_final_restoration_ablation.subprocess.run",
        fake_run,
    )

    result = run(args)

    assert len(calls) == 2
    assert "--cci_disable_final_correction" in calls[0]
    assert "--cci_disable_final_correction" not in calls[1]
    assert result["consistency"]["passed"]
