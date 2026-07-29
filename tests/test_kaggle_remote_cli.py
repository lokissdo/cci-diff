import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_kaggle_two_stage.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_kaggle_two_stage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_upload_creates_missing_and_versions_existing():
    module = load_script()

    create = module.dataset_upload_command(
        "kaggle", Path("/tmp/source"), exists=False, message="initial"
    )
    version = module.dataset_upload_command(
        "kaggle", Path("/tmp/source"), exists=True, message="update"
    )

    assert create == [
        "kaggle",
        "datasets",
        "create",
        "-p",
        "/tmp/source",
        "--dir-mode",
        "zip",
    ]
    assert version == [
        "kaggle",
        "datasets",
        "version",
        "-p",
        "/tmp/source",
        "-m",
        "update",
        "--dir-mode",
        "zip",
    ]


def test_kernel_push_requests_t4():
    module = load_script()

    command = module.kernel_push_command(
        "kaggle", Path("/tmp/kernel"), timeout=43200
    )

    assert command == [
        "kaggle",
        "kernels",
        "push",
        "-p",
        "/tmp/kernel",
        "--accelerator",
        "NvidiaTeslaT4",
        "--timeout",
        "43200",
    ]


def test_evaluation_slug_matches_kaggle_title():
    module = load_script()

    assert module.EVALUATION_SLUG == "cci-raw-bld-fixed-adaptive"


def test_parser_defaults_to_authenticated_owner_and_both_stages():
    module = load_script()
    args = module.build_parser().parse_args([])

    assert args.owner == "a210462khihng"
    assert args.start_at == "discovery"
    assert args.sample_count == 300
    assert args.kaggle.endswith(".venv-kaggle/bin/kaggle")
    assert args.repo_url == "https://github.com/lokissdo/cci-diff.git"


def test_prepare_kernel_pins_requested_git_revision(tmp_path):
    module = load_script()
    notebook = tmp_path / "input.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [
                            "GIT_REF = 'main'\n",
                            "SAMPLE_COUNT = 100\n",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "code_file": "remote.ipynb",
        "id": "owner/kernel",
    }

    module.prepare_kernel(
        notebook=notebook,
        destination=tmp_path / "kernel",
        metadata=metadata,
        sample_count=25,
        git_ref="abc123",
    )

    payload = (tmp_path / "kernel" / "remote.ipynb").read_text()
    assert "GIT_REF = 'abc123'" in payload
    assert "SAMPLE_COUNT = 25" in payload
