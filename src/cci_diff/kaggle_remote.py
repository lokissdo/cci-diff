"""Deterministic packaging helpers for remote Kaggle CCI runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable


REQUIRED_MODEL_FILES = (
    "resnet50_multilabel_model.pth",
    "facenet_vggface2.ts",
    "facenet_vggface2.ts.json",
)

def prepare_model_dataset(
    models_root: str | Path,
    destination: str | Path,
    *,
    owner: str,
) -> Path:
    """Create a private-dataset payload from the local evaluator models."""

    root = Path(models_root).resolve()
    missing = [name for name in REQUIRED_MODEL_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required evaluator models: " + ", ".join(missing)
        )
    output = _reset_directory(destination)
    for source in sorted(root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(root)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _write_json(
        output / "dataset-metadata.json",
        _dataset_metadata(
            owner=owner,
            slug="cci-assets",
            title="CCI Evaluator Assets",
        ),
    )
    return output


def kernel_metadata(
    *,
    owner: str,
    slug: str,
    title: str,
    code_file: str,
    dataset_sources: Iterable[str],
    kernel_sources: Iterable[str] = (),
) -> dict[str, object]:
    """Return private, internet-enabled GPU notebook metadata."""

    return {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": list(dataset_sources),
        "competition_sources": [],
        "kernel_sources": list(kernel_sources),
    }


def is_terminal_kernel_status(status_output: str) -> tuple[bool, bool]:
    """Return ``(terminal, successful)`` from Kaggle status text."""

    normalized = status_output.strip().lower()
    if any(word in normalized for word in ("complete", "success")):
        return True, True
    if any(
        word in normalized
        for word in ("error", "fail", "cancel", "stopped", "expired")
    ):
        return True, False
    return False, False


def is_ready_dataset_status(status_output: str) -> bool:
    """Return whether Kaggle reports a dataset version as ready."""

    return "ready" in status_output.strip().lower()


def _dataset_metadata(*, owner: str, slug: str, title: str) -> dict[str, object]:
    return {
        "title": title,
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "CC-BY-4.0"}],
    }


def _reset_directory(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    return destination


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
