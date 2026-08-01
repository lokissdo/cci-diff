from pathlib import Path
from types import SimpleNamespace

import pytest

from cci_diff.runtime_environment import resolve_device, validate_local_artifacts


def fake_torch(cuda, mps):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps)
        ),
    )


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_precedence(cuda, mps, expected):
    assert resolve_device("auto", fake_torch(cuda, mps)) == expected


def test_explicit_unavailable_accelerator_is_rejected():
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        resolve_device("cuda", fake_torch(False, True))


def test_validate_local_artifacts_returns_resolved_paths(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model")

    resolved = validate_local_artifacts({"classifier": artifact})

    assert resolved == {"classifier": artifact.resolve()}
    with pytest.raises(FileNotFoundError, match="identity"):
        validate_local_artifacts({"identity": tmp_path / "missing.bin"})
