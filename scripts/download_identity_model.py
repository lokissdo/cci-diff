#!/usr/bin/env python3
"""Explicitly acquire and export facenet-pytorch VGGFace2 as TorchScript."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_torchscript(model, example):
    """Trace without freezing so runtime can move parameters to MPS."""

    import torch

    return torch.jit.trace(model, example, strict=False).eval()


def download(output: str | Path) -> tuple[str, str, str]:
    import torch
    from facenet_pytorch import InceptionResnetV1

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = InceptionResnetV1(pretrained="vggface2", classify=False).eval()
    example = torch.zeros((1, 3, 160, 160), dtype=torch.float32)
    exported = export_torchscript(model, example)
    exported.save(str(destination))
    digest = sha256_file(destination)
    manifest_path = Path(str(destination) + ".json")
    manifest_path.write_text(
        json.dumps(
            {
                "facenet_pytorch_version": importlib.metadata.version(
                    "facenet-pytorch"
                ),
                "export_torch_version": torch.__version__,
                "frozen": False,
                "sha256": digest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(destination), digest, str(manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/facenet_vggface2.ts")
    args = parser.parse_args()
    path, digest, manifest = download(args.output)
    print(f"saved={path}")
    print(f"sha256={digest}")
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
