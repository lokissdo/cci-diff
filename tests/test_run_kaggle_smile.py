import importlib.util
import sys
from pathlib import Path

import pytest


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


def test_standalone_runner_exposes_setup_and_model_download_logs():
    source = SCRIPT.read_text()

    assert "--allow_model_download" in source
    assert "--progress-bar" in source
    assert "HF_HUB_VERBOSITY" in source
    assert "DIFFUSERS_VERBOSITY" in source
    assert "START setup:" in source
    assert "FOUND {key}" in source
    assert "capture_output" not in source


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


def test_resolve_unique_finds_nested_kaggle_asset(tmp_path):
    module = load_script()
    asset = tmp_path / "mounted-dataset" / "checkpoint.pth"
    asset.parent.mkdir()
    asset.write_bytes(b"model")

    assert module.resolve_unique(tmp_path, "checkpoint.pth") == asset


def test_resolve_unique_rejects_missing_asset(tmp_path):
    module = load_script()

    with pytest.raises(FileNotFoundError, match="checkpoint.pth"):
        module.resolve_unique(tmp_path, "checkpoint.pth")
