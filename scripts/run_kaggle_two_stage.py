#!/usr/bin/env python3
"""Prepare, upload, run, and retrieve the two-stage Kaggle CCI experiment."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cci_diff.kaggle_remote import (
    is_ready_dataset_status,
    is_terminal_kernel_status,
    kernel_metadata,
    prepare_model_dataset,
)


DEFAULT_KAGGLE = REPO_ROOT / ".venv-kaggle" / "bin" / "kaggle"
PUBLIC_DATASET = "ipythonx/celebamaskhq"
DEFAULT_REPO_URL = "https://github.com/lokissdo/cci-diff.git"
DISCOVERY_SLUG = "cci-global-graph-discovery"
EVALUATION_SLUG = "cci-raw-bld-fixed-adaptive"


def dataset_upload_command(
    kaggle: str,
    source: Path,
    *,
    exists: bool,
    message: str,
) -> list[str]:
    if exists:
        return [
            kaggle,
            "datasets",
            "version",
            "-p",
            str(source),
            "-m",
            message,
            "--dir-mode",
            "zip",
        ]
    return [
        kaggle,
        "datasets",
        "create",
        "-p",
        str(source),
        "--dir-mode",
        "zip",
    ]


def kernel_push_command(
    kaggle: str,
    kernel_dir: Path,
    *,
    timeout: int,
) -> list[str]:
    return [
        kaggle,
        "kernels",
        "push",
        "-p",
        str(kernel_dir),
        "--accelerator",
        "NvidiaTeslaT4",
        "--timeout",
        str(timeout),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="a210462khihng")
    parser.add_argument("--kaggle", default=str(DEFAULT_KAGGLE))
    parser.add_argument("--repo_url", default=DEFAULT_REPO_URL)
    parser.add_argument("--git_ref", default=None)
    parser.add_argument(
        "--staging_dir",
        type=Path,
        default=Path("/tmp/cci-diff-kaggle-remote"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "kaggle_remote",
    )
    parser.add_argument("--sample_count", type=int, default=300)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=43200)
    parser.add_argument(
        "--start_at",
        choices=("discovery", "evaluation"),
        default="discovery",
    )
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--skip_datasets", action="store_true")
    parser.add_argument("--no_wait", action="store_true")
    return parser


def run(command: Sequence[str], *, capture: bool = False) -> str:
    print("+", " ".join(str(item) for item in command), flush=True)
    completed = subprocess.run(
        [str(item) for item in command],
        check=True,
        text=True,
        capture_output=capture,
    )
    if capture:
        output = (completed.stdout or "") + (completed.stderr or "")
        print(output.strip(), flush=True)
        return output
    return ""


def kaggle_resource_exists(kaggle: str, kind: str, reference: str) -> bool:
    completed = subprocess.run(
        [kaggle, kind, "status", reference],
        text=True,
        capture_output=True,
    )
    return completed.returncode == 0


def prepare_kernel(
    *,
    notebook: Path,
    destination: Path,
    metadata: dict[str, object],
    sample_count: int,
    git_ref: str,
    repo_url: str = DEFAULT_REPO_URL,
) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = re.sub(
            r"(?m)^SAMPLE_COUNT = \d+$",
            f"SAMPLE_COUNT = {sample_count}",
            source,
        )
        source = source.replace("GIT_REF = 'main'", f"GIT_REF = '{git_ref}'")
        source = source.replace(
            f"REPO_URL = '{DEFAULT_REPO_URL}'",
            f"REPO_URL = '{repo_url}'",
        )
        cell["source"] = source.splitlines(keepends=True)
    code_file = str(metadata["code_file"])
    (destination / code_file).write_text(
        json.dumps(payload, indent=1) + "\n",
        encoding="utf-8",
    )
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def wait_for_kernel(
    kaggle: str,
    reference: str,
    *,
    poll_seconds: int,
    timeout: int,
) -> None:
    started = time.monotonic()
    while True:
        output = run(
            [kaggle, "kernels", "status", reference],
            capture=True,
        )
        terminal, successful = is_terminal_kernel_status(output)
        if terminal:
            if successful:
                return
            run([kaggle, "kernels", "logs", reference], capture=True)
            raise RuntimeError(f"Kaggle kernel failed: {reference}")
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"Timed out waiting for {reference}")
        time.sleep(poll_seconds)


def upload_dataset(
    kaggle: str,
    reference: str,
    source: Path,
    *,
    message: str,
) -> None:
    exists = kaggle_resource_exists(kaggle, "datasets", reference)
    run(
        dataset_upload_command(
            kaggle,
            source,
            exists=exists,
            message=message,
        )
    )
    wait_for_dataset(kaggle, reference, poll_seconds=5, timeout=900)


def wait_for_dataset(
    kaggle: str,
    reference: str,
    *,
    poll_seconds: int,
    timeout: int,
) -> None:
    started = time.monotonic()
    while True:
        output = run(
            [kaggle, "datasets", "status", reference],
            capture=True,
        )
        if is_ready_dataset_status(output):
            return
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"Timed out waiting for dataset {reference}")
        time.sleep(poll_seconds)


def download_output(
    kaggle: str,
    reference: str,
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            kaggle,
            "kernels",
            "output",
            reference,
            "-p",
            str(destination),
            "--force",
        ]
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    kaggle = str(Path(args.kaggle).resolve())
    if not Path(kaggle).is_file():
        raise FileNotFoundError(f"Kaggle CLI not found: {kaggle}")

    staging = args.staging_dir.resolve()
    model_dataset = prepare_model_dataset(
        REPO_ROOT / "models",
        staging / "model_dataset",
        owner=args.owner,
    )

    git_ref = args.git_ref or run(
        ["git", "rev-parse", "HEAD"],
        capture=True,
    ).strip()
    datasets = (PUBLIC_DATASET, f"{args.owner}/cci-assets")
    discovery_code = "01_global_graph_discovery.ipynb"
    evaluation_code = "02_full_cci_fixed_vs_adaptive.ipynb"
    discovery = prepare_kernel(
        notebook=REPO_ROOT / "notebooks" / discovery_code,
        destination=staging / "discovery_kernel",
        metadata=kernel_metadata(
            owner=args.owner,
            slug=DISCOVERY_SLUG,
            title="CCI Global Graph Discovery",
            code_file=discovery_code,
            dataset_sources=datasets,
        ),
        sample_count=args.sample_count,
        git_ref=git_ref,
        repo_url=args.repo_url,
    )
    evaluation = prepare_kernel(
        notebook=REPO_ROOT / "notebooks" / evaluation_code,
        destination=staging / "evaluation_kernel",
        metadata=kernel_metadata(
            owner=args.owner,
            slug=EVALUATION_SLUG,
            title="CCI Raw BLD Fixed Adaptive",
            code_file=evaluation_code,
            dataset_sources=datasets,
        ),
        sample_count=args.sample_count,
        git_ref=git_ref,
        repo_url=args.repo_url,
    )
    print(f"Prepared remote payloads in {staging}")
    if args.prepare_only:
        return 0

    run([kaggle, "kernels", "list", "--mine", "--page-size", "1"])
    if not args.skip_datasets:
        upload_dataset(
            kaggle,
            f"{args.owner}/cci-assets",
            model_dataset,
            message="Update CCI evaluator assets",
        )

    if args.start_at == "discovery":
        run(kernel_push_command(kaggle, discovery, timeout=args.timeout))
        if args.no_wait:
            print(f"Started {args.owner}/{DISCOVERY_SLUG}")
            return 0
        wait_for_kernel(
            kaggle,
            f"{args.owner}/{DISCOVERY_SLUG}",
            poll_seconds=args.poll_seconds,
            timeout=args.timeout,
        )
        download_output(
            kaggle,
            f"{args.owner}/{DISCOVERY_SLUG}",
            args.output_dir / "discovery",
        )

    run(kernel_push_command(kaggle, evaluation, timeout=args.timeout))
    if args.no_wait:
        print(f"Started {args.owner}/{EVALUATION_SLUG}")
        return 0
    wait_for_kernel(
        kaggle,
        f"{args.owner}/{EVALUATION_SLUG}",
        poll_seconds=args.poll_seconds,
        timeout=args.timeout,
    )
    download_output(
        kaggle,
        f"{args.owner}/{EVALUATION_SLUG}",
        args.output_dir / "evaluation",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
