import importlib.util
import os
import sys
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
    assert args.sample_count == 300
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


def test_kaggle_assets_rejects_invalid_hardcoded_path(tmp_path):
    module = load_script()
    module.CLASSIFIER_PATH = tmp_path / "missing.pth"

    try:
        module.kaggle_assets()
    except FileNotFoundError as error:
        assert "classifier" in str(error)
    else:
        raise AssertionError("Expected missing hardcoded path to fail")
