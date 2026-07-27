import json

import pytest

from cci_diff.kaggle_remote import (
    REQUIRED_MODEL_FILES,
    is_ready_dataset_status,
    is_terminal_kernel_status,
    kernel_metadata,
    prepare_diffusion_model_dataset,
    prepare_model_dataset,
)


def test_prepare_model_dataset_requires_and_copies_local_models(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    for name in REQUIRED_MODEL_FILES:
        (models / name).write_bytes(name.encode())

    destination = tmp_path / "staging"
    prepare_model_dataset(models, destination, owner="owner")

    assert {
        path.name
        for path in destination.iterdir()
        if path.name != "dataset-metadata.json"
    } == set(REQUIRED_MODEL_FILES)
    metadata = json.loads(
        (destination / "dataset-metadata.json").read_text()
    )
    assert metadata["id"] == "owner/cci-assets"


def test_prepare_model_dataset_excludes_diffusion_snapshot(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    for name in REQUIRED_MODEL_FILES:
        (models / name).write_bytes(name.encode())
    snapshot = models / "stable-diffusion-2-1"
    snapshot.mkdir()
    (snapshot / "model_index.json").write_text("{}")

    destination = tmp_path / "staging"
    prepare_model_dataset(models, destination, owner="owner")

    assert not (destination / "stable-diffusion-2-1").exists()


def test_prepare_model_dataset_rejects_missing_classifier(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    with pytest.raises(FileNotFoundError, match="resnet50"):
        prepare_model_dataset(models, tmp_path / "staging", owner="owner")


def test_prepare_diffusion_model_dataset_copies_complete_local_snapshot(tmp_path):
    snapshot = tmp_path / "stable-diffusion-2-1"
    snapshot.mkdir()
    (snapshot / "model_index.json").write_text("{}")
    for component in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        directory = snapshot / component
        directory.mkdir()
        (directory / "config.json").write_text("{}")

    destination = tmp_path / "staging"
    prepare_diffusion_model_dataset(snapshot, destination, owner="owner")

    assert (destination / "stable-diffusion-2-1" / "model_index.json").is_file()
    metadata = json.loads((destination / "dataset-metadata.json").read_text())
    assert metadata["id"] == "owner/cci-sd2-assets"


def test_prepare_diffusion_model_dataset_rejects_incomplete_snapshot(tmp_path):
    snapshot = tmp_path / "stable-diffusion-2-1"
    snapshot.mkdir()
    (snapshot / "model_index.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="unet"):
        prepare_diffusion_model_dataset(snapshot, tmp_path / "staging", owner="owner")


def test_kernel_metadata_is_private_and_declares_sources():
    metadata = kernel_metadata(
        owner="owner",
        slug="graph-discovery",
        title="Graph Discovery",
        code_file="graph.ipynb",
        dataset_sources=("ipythonx/celebamaskhq", "owner/cci-assets"),
        kernel_sources=("owner/upstream",),
    )

    assert metadata["id"] == "owner/graph-discovery"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["dataset_sources"] == [
        "ipythonx/celebamaskhq",
        "owner/cci-assets",
    ]
    assert metadata["kernel_sources"] == ["owner/upstream"]


def test_standalone_evaluation_metadata_needs_no_kernel_source():
    metadata = kernel_metadata(
        owner="owner",
        slug="evaluation",
        title="Evaluation",
        code_file="evaluation.ipynb",
        dataset_sources=("ipythonx/celebamaskhq", "owner/cci-assets"),
    )

    assert metadata["kernel_sources"] == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("KernelWorkerStatus.RUNNING", (False, False)),
        ("complete", (True, True)),
        ("ERROR", (True, False)),
        ("cancelled", (True, False)),
    ],
)
def test_terminal_kernel_status(status, expected):
    assert is_terminal_kernel_status(status) == expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("ready", True),
        ("Dataset status: READY", True),
        ("pending", False),
        ("creating", False),
    ],
)
def test_ready_dataset_status(status, expected):
    assert is_ready_dataset_status(status) is expected
