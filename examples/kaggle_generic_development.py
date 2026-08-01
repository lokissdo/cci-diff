"""Thin Kaggle launcher for the shared generic development CLI."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KagglePaths:
    repo_root: Path
    image_root: Path
    mask_root: Path
    model_path: Path
    classifier_path: Path
    identity_model_path: Path
    template_graph: Path
    generation_policy: Path
    eligible_ids_manifest: Path
    evaluation_ids_manifest: Path
    working_root: Path


def build_kaggle_command(
    *, data_size: int, paths: KagglePaths
) -> list[str]:
    """Build the same command for a 30- or 300-image development run."""

    if isinstance(data_size, bool) or not isinstance(data_size, int):
        raise ValueError("data_size must be an integer")
    output_dir = paths.working_root / "runs" / f"n{data_size}"
    cache_dir = paths.working_root / "a11-cache"
    return [
        sys.executable,
        str(paths.repo_root / "scripts/run_generic_region_development.py"),
        "--data_size",
        str(data_size),
        "--eligible_ids_manifest",
        str(paths.eligible_ids_manifest),
        "--evaluation_ids_manifest",
        str(paths.evaluation_ids_manifest),
        "--template_graph",
        str(paths.template_graph),
        "--generation_policy",
        str(paths.generation_policy),
        "--image_root",
        str(paths.image_root),
        "--mask_root",
        str(paths.mask_root),
        "--model_path",
        str(paths.model_path),
        "--classifier_path",
        str(paths.classifier_path),
        "--identity_model_path",
        str(paths.identity_model_path),
        "--cache_dir",
        str(cache_dir),
        "--output_dir",
        str(output_dir),
        "--device",
        "auto",
    ]


def main() -> int:
    paths = KagglePaths(
        repo_root=Path("/kaggle/working/cci-diff"),
        image_root=Path("/kaggle/input/celebamask-hq/CelebA-HQ-img"),
        mask_root=Path(
            "/kaggle/input/celebamask-hq/CelebAMask-HQ-mask-anno"
        ),
        model_path=Path("/kaggle/input/sd2-1-base/sd2-1-base"),
        classifier_path=Path("/kaggle/input/cci-models/classifier.pth"),
        identity_model_path=Path("/kaggle/input/cci-models/facenet.ts"),
        template_graph=Path("/kaggle/input/cci-config/graph.json"),
        generation_policy=Path(
            "/kaggle/input/cci-config/a11-generation-policy.json"
        ),
        eligible_ids_manifest=Path(
            "/kaggle/input/cci-config/candidate-source-ids.json"
        ),
        evaluation_ids_manifest=Path(
            "/kaggle/input/cci-config/evaluation-ids.json"
        ),
        working_root=Path("/kaggle/working/cci-generic"),
    )
    subprocess.run(build_kaggle_command(data_size=30, paths=paths), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
