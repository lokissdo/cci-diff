#!/usr/bin/env python3
"""Run counterfactual graph discovery or smile CCI evaluation on Kaggle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import NamedTuple, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "sd2-community/stable-diffusion-2-1"
DEFAULT_CCI_ASSET_ROOT = Path("/kaggle/input/datasets/a210462khihng/cci-assets")
DEFAULT_CELEBA_ROOT = Path(
    "/kaggle/input/datasets/ipythonx/celebamaskhq/CelebAMask-HQ"
)
CLASSIFIER_PATH = DEFAULT_CCI_ASSET_ROOT / "resnet50_multilabel_model.pth"
IDENTITY_MODEL_PATH = DEFAULT_CCI_ASSET_ROOT / "facenet_vggface2.ts"
LEGACY_CCI_ASSET_ROOT = Path("/kaggle/input/cci-assets")
LEGACY_CLASSIFIER_PATH = LEGACY_CCI_ASSET_ROOT / "resnet50_multilabel_model.pth"
LEGACY_IDENTITY_MODEL_PATH = LEGACY_CCI_ASSET_ROOT / "facenet_vggface2.ts"
IMAGE_ROOT = DEFAULT_CELEBA_ROOT / "CelebA-HQ-img"
MASK_ROOT = DEFAULT_CELEBA_ROOT / "CelebAMask-HQ-mask-anno"
PACKAGE_MODULES = {
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
}


class DiscoveryTask(NamedTuple):
    feature: str
    output_key: str
    template: Path
    candidate_regions: tuple[str, ...]


DISCOVERY_TASKS = {
    "smile": DiscoveryTask(
        feature="smile",
        output_key="smile",
        template=Path("examples/graphs/remove_smile_clean_cci.json"),
        candidate_regions=(
            "skin",
            "nose",
            "mouth",
            "upper_lip",
            "lower_lip",
            "left_eye",
            "right_eye",
            "left_brow",
            "right_brow",
        ),
    ),
    "blond_hair": DiscoveryTask(
        feature="hair",
        output_key="blond_hair",
        template=Path("examples/graphs/blond_hair_clean_cci.json"),
        candidate_regions=(
            "hair",
            "skin",
            "left_brow",
            "right_brow",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "hat",
        ),
    ),
}


def resolve_discovery_task(name: str) -> DiscoveryTask:
    try:
        return DISCOVERY_TASKS[name]
    except KeyError as error:
        raise ValueError(f"Unsupported discovery task: {name}") from error


def discovery_paths(output_root: Path, task: DiscoveryTask) -> dict[str, Path]:
    task_root = output_root / task.output_key
    return {
        "screening": task_root / "screening",
        "interventions": task_root / "interventions",
        "graph": task_root / "graph",
    }


def timestamp() -> str:
    return time.strftime("%H:%M:%S")


def run_logged_command(
    label: str,
    command: Sequence[str | Path],
    *,
    cwd: str | Path | None = None,
) -> None:
    rendered = [str(item) for item in command]
    print(f"[{timestamp()}] START {label}", flush=True)
    print("+ " + " ".join(rendered), flush=True)
    started = time.monotonic()
    try:
        subprocess.run(rendered, cwd=cwd, check=True)
    except BaseException:
        print(
            f"[{timestamp()}] FAIL  {label} after "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )
        raise
    print(
        f"[{timestamp()}] DONE  {label} in {time.monotonic() - started:.1f}s",
        flush=True,
    )


def ensure_runtime_packages() -> None:
    missing = [
        package
        for package, module in PACKAGE_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        print(f"[{timestamp()}] SKIP  setup: all runtime packages available", flush=True)
        return
    run_logged_command(
        "setup: install missing runtime packages",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--progress-bar",
            "on",
            *missing,
        ],
    )


def configure_runtime() -> None:
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    os.environ["HF_HUB_VERBOSITY"] = "info"
    os.environ["DIFFUSERS_VERBOSITY"] = "info"
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    repo_paths = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
    existing_paths = [
        value
        for value in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if value and value not in repo_paths
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join([*repo_paths, *existing_paths])
    for value in repo_paths:
        if value not in sys.path:
            sys.path.insert(0, value)


def kaggle_assets() -> dict[str, Path]:
    assets = {
        "classifier": (
            CLASSIFIER_PATH if CLASSIFIER_PATH.exists() else LEGACY_CLASSIFIER_PATH
        ),
        "identity": (
            IDENTITY_MODEL_PATH
            if IDENTITY_MODEL_PATH.exists()
            else LEGACY_IDENTITY_MODEL_PATH
        ),
        "images": IMAGE_ROOT,
        "masks": MASK_ROOT,
    }
    missing = [key for key, path in assets.items() if not path.exists()]
    if missing:
        details = ", ".join(f"{key}={assets[key]}" for key in missing)
        raise FileNotFoundError(f"Invalid hardcoded Kaggle asset paths: {details}")
    print(f"[{timestamp()}] Kaggle assets:", flush=True)
    for key, path in assets.items():
        print(f"[{timestamp()}] FOUND {key}: {path}", flush=True)
    return assets


def resolved_worker_devices(args: argparse.Namespace) -> tuple[str, ...]:
    import torch

    from cci_diff.execution_shards import resolve_worker_devices

    cuda_count = torch.cuda.device_count() if args.device.startswith("cuda") else 0
    return resolve_worker_devices(
        args.device,
        args.max_workers,
        cuda_device_count=cuda_count,
    )


def build_evaluation_shard_jobs(
    args: argparse.Namespace,
    assets: dict[str, Path],
    *,
    sample_ids: Sequence[int],
    devices: Sequence[str],
    output_root: Path,
) -> list[dict]:
    from cci_diff.execution_shards import partition_ids

    shards = partition_ids(sample_ids, len(devices))
    jobs = []
    for index, (device, shard) in enumerate(zip(devices, shards)):
        shard_output = output_root / "shards" / f"worker_{index:02d}"
        command = [
            sys.executable,
            "-u",
            REPO_ROOT / "scripts" / "run_clean_cci_pilot.py",
            "--features",
            "smile",
            "--limit",
            str(len(shard)),
            "--sample_ids",
            *[str(sample_id) for sample_id in shard],
            "--controller_modes",
            "disabled",
            "fixed_equal",
            "feedback",
            "--model_path",
            args.model_path,
            "--classifier_path",
            assets["classifier"],
            "--allow_model_download",
            "--identity_model_path",
            assets["identity"],
            "--image_root",
            assets["images"],
            "--mask_root",
            assets["masks"],
            "--device",
            device,
            "--torch_dtype",
            "auto",
            "--python_executable",
            sys.executable,
            "--seed",
            str(args.seed),
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--mask_shapes",
            "4,4,3",
            "--continue_on_error",
            "--output_dir",
            shard_output,
        ]
        jobs.append(
            {
                "index": index,
                "device": device,
                "sample_ids": shard,
                "output_dir": str(shard_output),
                "command": [str(value) for value in command],
            }
        )
    return jobs


def run_sharded_commands(
    label: str,
    jobs: Sequence[dict],
    *,
    manifest_path: str | Path,
) -> list[dict]:
    """Launch independent workers concurrently and persist their outcomes."""

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    workers = [
        {
            "index": job["index"],
            "device": job["device"],
            "sample_ids": list(job["sample_ids"]),
            "output_dir": str(job["output_dir"]),
            "command": [str(value) for value in job["command"]],
            "status": "pending",
            "returncode": None,
        }
        for job in jobs
    ]
    manifest_path.write_text(
        json.dumps({"label": label, "complete": False, "workers": workers}, indent=2),
        encoding="utf-8",
    )
    processes = []
    for job, worker in zip(jobs, workers):
        command = [str(value) for value in job["command"]]
        print(
            f"[{timestamp()}] START {label} worker={job['index']} "
            f"device={job['device']} samples={len(job['sample_ids'])}",
            flush=True,
        )
        print("+ " + " ".join(command), flush=True)
        worker["status"] = "running"
        worker["started_at"] = time.time()
        processes.append(subprocess.Popen(command, cwd=REPO_ROOT))
    failed = []
    for process, worker in zip(processes, workers):
        returncode = process.wait()
        worker["returncode"] = returncode
        worker["elapsed_seconds"] = time.time() - worker["started_at"]
        worker["status"] = "complete" if returncode == 0 else "failed"
        print(
            f"[{timestamp()}] {worker['status'].upper()} {label} "
            f"worker={worker['index']} returncode={returncode}",
            flush=True,
        )
        if returncode:
            failed.append(worker)
    manifest = {"label": label, "complete": not failed, "workers": workers}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if failed:
        indexes = ", ".join(str(worker["index"]) for worker in failed)
        raise RuntimeError(f"{label} failed for workers: {indexes}")
    return workers


def merge_evaluation_shards(
    output_root: Path,
    jobs: Sequence[dict],
    *,
    expected_selected_rows: int,
) -> None:
    from cci_diff.execution_shards import merge_csv_files, merge_jsonl_files

    shard_dirs = [Path(job["output_dir"]) for job in jobs]
    counts = {
        name: merge_csv_files(
            [shard / name for shard in shard_dirs],
            output_root / name,
        )
        for name in ("candidate_results.csv", "pilot_results.csv", "pilot_ranked.csv")
    }
    failure_count = merge_jsonl_files(
        [shard / "failures.jsonl" for shard in shard_dirs],
        output_root / "failures.jsonl",
    )
    summaries = []
    for shard in shard_dirs:
        path = shard / "pilot_summary.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    complete = (
        failure_count == 0
        and counts["pilot_results.csv"] == expected_selected_rows
        and counts["pilot_ranked.csv"] == expected_selected_rows
    )
    summary = {
        "complete": complete,
        "expected_selected_rows": expected_selected_rows,
        "counts": {**counts, "failures.jsonl": failure_count},
        "shard_summaries": summaries,
    }
    (output_root / "shard_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if not complete:
        raise RuntimeError(
            "Evaluation shards are incomplete: "
            f"expected {expected_selected_rows} selected rows, "
            f"got {counts['pilot_results.csv']} with {failure_count} failures"
        )


def run_evaluation(args: argparse.Namespace, assets: dict[str, Path]) -> None:
    output_root = Path(args.output_dir or "/kaggle/working/cci_fixed_vs_adaptive")
    output_root.mkdir(parents=True, exist_ok=True)
    ids = select_discovery_ids(args, assets, output_root)
    devices = resolved_worker_devices(args)
    print(f"[{timestamp()}] WORKER devices: {list(devices)}", flush=True)
    jobs = build_evaluation_shard_jobs(
        args,
        assets,
        sample_ids=ids,
        devices=devices,
        output_root=output_root,
    )
    run_sharded_commands(
        "smile evaluation",
        jobs,
        manifest_path=output_root / "shard_manifest.json",
    )
    merge_evaluation_shards(
        output_root,
        jobs,
        expected_selected_rows=len(ids) * 3,
    )


def select_discovery_ids(
    args: argparse.Namespace,
    assets: dict[str, Path],
    output_root: Path,
    task: DiscoveryTask,
) -> list[int]:
    ids_path = output_root / "discovery_ids.json"
    if ids_path.is_file():
        payload = json.loads(ids_path.read_text())
        if task.output_key in payload:
            ids = payload[task.output_key]
            print(
                f"[{timestamp()}] SKIP  {task.output_key} discovery cohort: "
                f"reusing {len(ids)} IDs",
                flush=True,
            )
            return ids
    else:
        payload = {}

    print(
        f"[{timestamp()}] START {task.output_key} discovery cohort selection",
        flush=True,
    )
    import torch

    from cci_diff.classifiers.celeba_resnet50 import load_celeba_resnet50
    from cci_diff.identity.facenet import build_face_detector
    from scripts.run_clean_cci_pilot import select_eligible_samples

    classifier = load_celeba_resnet50(
        str(assets["classifier"]),
        device=args.device,
        dtype=torch.float32,
    )
    detector = build_face_detector()
    selection_args = Namespace(
        max_image_id=30000,
        image_root=str(assets["images"]),
        mask_root=str(assets["masks"]),
        classifier_input_size=512,
        device=args.device,
        limit=args.sample_count,
        excluded_ids_by_feature={},
    )
    selected, _ = select_eligible_samples(
        selection_args,
        feature=task.feature,
        classifier=classifier,
        detector=detector,
    )
    ids = [sample_id for sample_id, _, _ in selected]
    payload[task.output_key] = ids
    ids_path.write_text(json.dumps(payload, indent=2))
    print(
        f"[{timestamp()}] DONE  {task.output_key} discovery cohort selection: "
        f"{len(ids)} IDs",
        flush=True,
    )
    return ids


def build_discovery_shard_jobs(
    args: argparse.Namespace,
    assets: dict[str, Path],
    *,
    sample_ids: Sequence[int],
    candidate_regions: Sequence[str],
    devices: Sequence[str],
    output_root: Path,
    template: Path,
) -> list[dict]:
    from cci_diff.execution_shards import partition_ids

    shards = partition_ids(sample_ids, len(devices))
    jobs = []
    for index, (device, shard) in enumerate(zip(devices, shards)):
        shard_output = output_root / "shards" / f"worker_{index:02d}"
        command = [
            sys.executable,
            "-u",
            REPO_ROOT / "scripts" / "run_counterfactual_region_interventions.py",
            "--template_graph",
            template,
            "--sample_ids",
            *[str(sample_id) for sample_id in shard],
            "--candidate_regions",
            *candidate_regions,
            "--max_set_size",
            str(len(candidate_regions)),
            "--stop_flip_rate",
            "1.0",
            "--disable_early_stop",
            "--seeds",
            str(args.seed),
            "--image_root",
            assets["images"],
            "--mask_root",
            assets["masks"],
            "--model_path",
            args.model_path,
            "--classifier_path",
            assets["classifier"],
            "--allow_model_download",
            "--identity_model_path",
            assets["identity"],
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--device",
            device,
            "--torch_dtype",
            "auto",
            "--python_executable",
            sys.executable,
            "--continue_on_error",
            "--output_dir",
            shard_output,
        ]
        jobs.append(
            {
                "index": index,
                "device": device,
                "sample_ids": shard,
                "output_dir": str(shard_output),
                "command": [str(value) for value in command],
            }
        )
    return jobs


def merge_discovery_shards(output_root: Path, jobs: Sequence[dict]) -> None:
    from cci_diff.execution_shards import merge_csv_files, merge_jsonl_files

    shard_dirs = [Path(job["output_dir"]) for job in jobs]
    result_count = merge_csv_files(
        [shard / "intervention_results.csv" for shard in shard_dirs],
        output_root / "intervention_results.csv",
    )
    failure_count = merge_jsonl_files(
        [shard / "failures.jsonl" for shard in shard_dirs],
        output_root / "failures.jsonl",
    )
    complete = result_count > 0 and failure_count == 0
    summary = {
        "complete": complete,
        "result_count": result_count,
        "failure_count": failure_count,
        "shards": [
            {
                "index": job["index"],
                "device": job["device"],
                "sample_ids": list(job["sample_ids"]),
                "output_dir": job["output_dir"],
            }
            for job in jobs
        ],
    }
    (output_root / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if not complete:
        raise RuntimeError(
            "Discovery shards are incomplete: "
            f"{result_count} results with {failure_count} failures"
        )


def run_discovery(args: argparse.Namespace, assets: dict[str, Path]) -> None:
    output_root = Path(args.output_dir or "/kaggle/working/cci_graph_discovery")
    output_root.mkdir(parents=True, exist_ok=True)
    task = resolve_discovery_task(args.task)
    ids = select_discovery_ids(args, assets, output_root, task)
    template = REPO_ROOT / task.template
    paths = discovery_paths(output_root, task)
    screening = paths["screening"]
    interventions = paths["interventions"]
    graph = paths["graph"]

    run_logged_command(
        f"{task.output_key} discovery: Grad-CAM++ region screening",
        [
            sys.executable,
            "-u",
            REPO_ROOT / "scripts" / "screen_counterfactual_regions.py",
            "--template_graph",
            template,
            "--classifier_path",
            assets["classifier"],
            "--sample_ids",
            *ids,
            "--candidate_regions",
            *task.candidate_regions,
            "--max_selected_regions",
            "4",
            "--saliency_coverage_threshold",
            "0.80",
            "--cohort_frequency_threshold",
            "0.90",
            "--minimum_captured_saliency",
            "0.0",
            "--image_root",
            assets["images"],
            "--mask_root",
            assets["masks"],
            "--device",
            args.device,
            "--output_dir",
            screening,
        ],
        cwd=REPO_ROOT,
    )
    manifest = json.loads((screening / "screening_manifest.json").read_text())
    candidates = manifest["selected_candidate_regions"][:4]
    print(f"[{timestamp()}] SELECTED discovery regions: {candidates}", flush=True)
    devices = resolved_worker_devices(args)
    print(f"[{timestamp()}] WORKER devices: {list(devices)}", flush=True)
    jobs = build_discovery_shard_jobs(
        args,
        assets,
        sample_ids=ids,
        candidate_regions=candidates,
        devices=devices,
        output_root=interventions,
        template=template,
    )
    run_sharded_commands(
        f"{task.output_key} discovery interventions",
        jobs,
        manifest_path=interventions / "shard_manifest.json",
    )
    merge_discovery_shards(interventions, jobs)
    run_logged_command(
        f"{task.output_key} discovery: freeze global graph",
        [
            sys.executable,
            "-u",
            REPO_ROOT / "scripts" / "discover_counterfactual_graph.py",
            "--results",
            interventions / "intervention_results.csv",
            "--template_graph",
            template,
            "--required_flip_rate",
            "0.96",
            "--minimum_samples",
            str(args.sample_count),
            "--bootstrap_samples",
            "2000",
            "--confidence",
            "0.95",
            "--random_seed",
            str(args.seed),
            "--output_dir",
            graph,
        ],
        cwd=REPO_ROOT,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("discovery", "evaluation"), required=True)
    parser.add_argument(
        "--task",
        choices=tuple(DISCOVERY_TASKS),
        default="smile",
    )
    parser.add_argument("--sample_count", type=int, default=300)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--classifier_path", default=None)
    parser.add_argument("--identity_model_path", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--mask_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=0,
        help="Maximum concurrent device workers; zero uses every requested CUDA GPU.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if args.max_workers < 0:
        raise ValueError("max_workers must be non-negative")
    if args.mode == "evaluation" and args.task != "smile":
        raise ValueError("CCI evaluation is currently smile-only")


def resolve_assets(args: argparse.Namespace) -> dict[str, Path]:
    explicit = {
        "classifier": args.classifier_path,
        "identity": args.identity_model_path,
        "images": args.image_root,
        "masks": args.mask_root,
    }
    supplied = [value is not None for value in explicit.values()]
    if not any(supplied):
        return kaggle_assets()
    if not all(supplied):
        missing = ", ".join(
            key for key, value in explicit.items() if value is None
        )
        raise ValueError(
            f"Explicit asset paths must be supplied together; missing: {missing}"
        )
    assets = {key: Path(value) for key, value in explicit.items()}
    missing = [key for key, path in assets.items() if not path.exists()]
    if missing:
        details = ", ".join(f"{key}={assets[key]}" for key in missing)
        raise FileNotFoundError(f"Invalid explicit asset paths: {details}")
    print(f"[{timestamp()}] Explicit assets:", flush=True)
    for key, path in assets.items():
        print(f"[{timestamp()}] FOUND {key}: {path}", flush=True)
    return assets


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    configure_runtime()
    print(
        f"[{timestamp()}] CONFIG mode={args.mode} samples={args.sample_count} "
        f"task={args.task} model={args.model_path} device={args.device}",
        flush=True,
    )
    ensure_runtime_packages()
    assets = resolve_assets(args)
    if args.mode == "discovery":
        run_discovery(args, assets)
    else:
        run_evaluation(args, assets)
    print(f"[{timestamp()}] COMPLETE mode={args.mode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
