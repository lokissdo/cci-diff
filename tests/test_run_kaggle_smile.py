import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_kaggle_smile.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_kaggle_smile", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_requires_explicit_mode_and_defaults_to_smile():
    module = load_script()

    args = module.build_parser().parse_args(["--mode", "evaluation"])

    assert args.mode == "evaluation"
    assert args.task == "smile"
    assert args.sample_count == 300
    assert args.max_workers == 0
    assert args.classifier_path is None
    assert args.model_path == "sd2-community/stable-diffusion-2-1"
    assert str(module.CLASSIFIER_PATH) == (
        "/kaggle/input/datasets/a210462khihng/cci-assets/"
        "resnet50_multilabel_model.pth"
    )
    assert str(module.IDENTITY_MODEL_PATH) == (
        "/kaggle/input/datasets/a210462khihng/cci-assets/"
        "facenet_vggface2.ts"
    )
    assert str(module.IMAGE_ROOT) == (
        "/kaggle/input/datasets/ipythonx/celebamaskhq/"
        "CelebAMask-HQ/CelebA-HQ-img"
    )
    assert str(module.MASK_ROOT) == (
        "/kaggle/input/datasets/ipythonx/celebamaskhq/"
        "CelebAMask-HQ/CelebAMask-HQ-mask-anno"
    )


def test_blond_hair_discovery_task_uses_hair_graph_and_regions():
    module = load_script()

    task = module.resolve_discovery_task("blond_hair")

    assert task.feature == "hair"
    assert task.output_key == "blond_hair"
    assert task.template == Path("examples/graphs/blond_hair_clean_cci.json")
    assert task.candidate_regions[0] == "hair"
    assert set(task.candidate_regions) == {
        "hair",
        "skin",
        "left_brow",
        "right_brow",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "hat",
    }


def test_parser_accepts_blond_hair_only_for_discovery():
    module = load_script()

    args = module.build_parser().parse_args(
        ["--mode", "discovery", "--task", "blond_hair"]
    )
    assert args.task == "blond_hair"

    try:
        module.validate_args(
            module.build_parser().parse_args(
                ["--mode", "evaluation", "--task", "blond_hair"]
            )
        )
    except ValueError as error:
        assert "smile-only" in str(error)
    else:
        raise AssertionError("Expected blond-hair evaluation to be rejected")


def test_discovery_paths_are_task_specific(tmp_path):
    module = load_script()
    task = module.resolve_discovery_task("blond_hair")

    paths = module.discovery_paths(tmp_path, task)

    assert paths == {
        "screening": tmp_path / "blond_hair" / "screening",
        "interventions": tmp_path / "blond_hair" / "interventions",
        "graph": tmp_path / "blond_hair" / "graph",
    }
    assert all("smile" not in str(path) for path in paths.values())


def test_standalone_runner_exposes_setup_and_model_download_logs():
    source = SCRIPT.read_text()

    assert "--allow_model_download" in source
    assert "--progress-bar" in source
    assert "HF_HUB_VERBOSITY" in source
    assert "DIFFUSERS_VERBOSITY" in source
    assert "Kaggle assets:" in source
    assert "FOUND {key}" in source
    assert "capture_output" not in source
    assert ".rglob(" not in source


def test_run_logged_command_streams_stage_and_child_output(capfd):
    module = load_script()

    module.run_logged_command(
        "test child",
        [sys.executable, "-c", "print('visible-child-output', flush=True)"],
    )

    output = capfd.readouterr().out
    assert "START test child" in output
    assert "visible-child-output" in output
    assert "DONE  test child" in output


def test_evaluation_jobs_are_disjoint_and_bind_one_device_each(tmp_path):
    module = load_script()
    args = Namespace(
        sample_count=4,
        model_path="model",
        seed=42,
        num_inference_steps=35,
    )
    assets = {
        "classifier": Path("classifier.pth"),
        "identity": Path("identity.ts"),
        "images": Path("images"),
        "masks": Path("masks"),
    }

    jobs = module.build_evaluation_shard_jobs(
        args,
        assets,
        sample_ids=[4, 1, 3, 2],
        devices=("cuda:0", "cuda:1"),
        output_root=tmp_path,
    )

    assert [job["sample_ids"] for job in jobs] == [(1, 3), (2, 4)]
    assert [job["device"] for job in jobs] == ["cuda:0", "cuda:1"]
    for index, job in enumerate(jobs):
        command = job["command"]
        sample_index = command.index("--sample_ids")
        device_index = command.index("--device")
        output_index = command.index("--output_dir")
        assert command[sample_index + 1 : sample_index + 3] == [
            str(value) for value in job["sample_ids"]
        ]
        assert command[device_index + 1] == job["device"]
        assert command[output_index + 1] == str(
            tmp_path / "shards" / f"worker_{index:02d}"
        )


def test_discovery_jobs_partition_same_region_grid_across_devices(tmp_path):
    module = load_script()
    args = Namespace(
        model_path="model",
        seed=42,
        num_inference_steps=35,
    )
    assets = {
        "classifier": Path("classifier.pth"),
        "identity": Path("identity.ts"),
        "images": Path("images"),
        "masks": Path("masks"),
    }

    jobs = module.build_discovery_shard_jobs(
        args,
        assets,
        sample_ids=[1, 2, 3, 4],
        candidate_regions=["mouth", "lower_lip"],
        devices=("cuda:0", "cuda:1"),
        output_root=tmp_path,
        template=Path("graph.json"),
    )

    assert [job["sample_ids"] for job in jobs] == [(1, 3), (2, 4)]
    for job in jobs:
        command = job["command"]
        region_index = command.index("--candidate_regions")
        assert command[region_index + 1 : region_index + 3] == [
            "mouth",
            "lower_lip",
        ]
        assert command[command.index("--device") + 1] == job["device"]
        assert "--disable_early_stop" in command


def test_sharded_commands_run_concurrently_and_write_manifest(tmp_path):
    module = load_script()
    jobs = [
        {
            "index": index,
            "device": f"cuda:{index}",
            "sample_ids": (index,),
            "output_dir": str(tmp_path / f"worker_{index}"),
            "command": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(tmp_path / f'done_{index}')!r}).write_text('ok')"
                ),
            ],
        }
        for index in range(2)
    ]

    records = module.run_sharded_commands(
        "test shards",
        jobs,
        manifest_path=tmp_path / "shard_manifest.json",
    )

    assert [record["returncode"] for record in records] == [0, 0]
    assert (tmp_path / "done_0").is_file()
    assert (tmp_path / "done_1").is_file()
    manifest = json.loads(
        (tmp_path / "shard_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["complete"] is True
    assert [worker["device"] for worker in manifest["workers"]] == [
        "cuda:0",
        "cuda:1",
    ]


def test_evaluation_merge_rejects_recorded_child_failures(tmp_path):
    module = load_script()
    shard = tmp_path / "shard"
    shard.mkdir()
    for name in ("candidate_results.csv", "pilot_results.csv", "pilot_ranked.csv"):
        (shard / name).write_text("", encoding="utf-8")
    (shard / "failures.jsonl").write_text(
        '{"sample_id": 1, "variant": "A0"}\n',
        encoding="utf-8",
    )
    jobs = [{"output_dir": str(shard)}]

    try:
        module.merge_evaluation_shards(
            tmp_path / "merged",
            jobs,
            expected_selected_rows=3,
        )
    except RuntimeError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("Expected child failures to reject the merged run")
    summary = json.loads(
        (tmp_path / "merged" / "shard_summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] is False
    assert summary["counts"]["failures.jsonl"] == 1


def test_discovery_merge_rejects_recorded_child_failures(tmp_path):
    module = load_script()
    shard = tmp_path / "shard"
    shard.mkdir()
    (shard / "intervention_results.csv").write_text(
        "sample_id,regions\n1,mouth\n",
        encoding="utf-8",
    )
    (shard / "failures.jsonl").write_text(
        '{"sample_id": 2, "regions": ["mouth"]}\n',
        encoding="utf-8",
    )

    try:
        module.merge_discovery_shards(
            tmp_path / "merged",
            [{"output_dir": str(shard), "index": 0, "device": "cuda:0", "sample_ids": [1, 2]}],
        )
    except RuntimeError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("Expected child failures to reject the merged run")


def test_configure_runtime_exports_repo_paths_to_child_processes(monkeypatch):
    module = load_script()
    monkeypatch.setenv("PYTHONPATH", "/existing/packages")

    module.configure_runtime()

    python_paths = os.environ["PYTHONPATH"].split(os.pathsep)
    assert python_paths[:2] == [
        str(module.REPO_ROOT / "src"),
        str(module.REPO_ROOT),
    ]
    assert python_paths[2:] == ["/existing/packages"]


def test_kaggle_assets_uses_and_validates_hardcoded_paths(tmp_path):
    module = load_script()
    module.CLASSIFIER_PATH = tmp_path / "classifier.pth"
    module.IDENTITY_MODEL_PATH = tmp_path / "identity.ts"
    module.IMAGE_ROOT = tmp_path / "images"
    module.MASK_ROOT = tmp_path / "masks"
    module.CLASSIFIER_PATH.write_bytes(b"classifier")
    module.IDENTITY_MODEL_PATH.write_bytes(b"identity")
    module.IMAGE_ROOT.mkdir()
    module.MASK_ROOT.mkdir()

    assert module.kaggle_assets() == {
        "classifier": module.CLASSIFIER_PATH,
        "identity": module.IDENTITY_MODEL_PATH,
        "images": module.IMAGE_ROOT,
        "masks": module.MASK_ROOT,
    }


def test_kaggle_assets_accepts_legacy_unversioned_private_dataset_mount(tmp_path):
    module = load_script()
    module.CLASSIFIER_PATH = tmp_path / "missing-classifier.pth"
    module.IDENTITY_MODEL_PATH = tmp_path / "missing-identity.ts"
    module.LEGACY_CLASSIFIER_PATH = tmp_path / "cci-assets" / "classifier.pth"
    module.LEGACY_IDENTITY_MODEL_PATH = tmp_path / "cci-assets" / "identity.ts"
    module.IMAGE_ROOT = tmp_path / "images"
    module.MASK_ROOT = tmp_path / "masks"
    module.LEGACY_CLASSIFIER_PATH.parent.mkdir()
    module.LEGACY_CLASSIFIER_PATH.write_bytes(b"classifier")
    module.LEGACY_IDENTITY_MODEL_PATH.write_bytes(b"identity")
    module.IMAGE_ROOT.mkdir()
    module.MASK_ROOT.mkdir()

    assets = module.kaggle_assets()

    assert assets["classifier"] == module.LEGACY_CLASSIFIER_PATH
    assert assets["identity"] == module.LEGACY_IDENTITY_MODEL_PATH


def test_kaggle_assets_rejects_invalid_hardcoded_path(tmp_path):
    module = load_script()
    module.CLASSIFIER_PATH = tmp_path / "missing.pth"

    try:
        module.kaggle_assets()
    except FileNotFoundError as error:
        assert "classifier" in str(error)
    else:
        raise AssertionError("Expected missing hardcoded path to fail")


def test_resolve_assets_accepts_complete_explicit_local_paths(tmp_path):
    module = load_script()
    paths = {
        "classifier_path": tmp_path / "classifier.pth",
        "identity_model_path": tmp_path / "identity.ts",
        "image_root": tmp_path / "images",
        "mask_root": tmp_path / "masks",
    }
    paths["classifier_path"].write_bytes(b"classifier")
    paths["identity_model_path"].write_bytes(b"identity")
    paths["image_root"].mkdir()
    paths["mask_root"].mkdir()
    args = Namespace(**{key: str(value) for key, value in paths.items()})

    assert module.resolve_assets(args) == {
        "classifier": paths["classifier_path"],
        "identity": paths["identity_model_path"],
        "images": paths["image_root"],
        "masks": paths["mask_root"],
    }


def test_resolve_assets_rejects_partial_explicit_paths():
    module = load_script()
    args = Namespace(
        classifier_path="classifier.pth",
        identity_model_path=None,
        image_root=None,
        mask_root=None,
    )

    try:
        module.resolve_assets(args)
    except ValueError as error:
        assert "supplied together" in str(error)
    else:
        raise AssertionError("Expected partial explicit asset paths to fail")
