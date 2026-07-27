#!/usr/bin/env python3
"""Run the smile graph-discovery or CCI evaluation workflow on Kaggle."""

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
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "sd2-community/stable-diffusion-2-1"
DEFAULT_CCI_ASSET_ROOT = Path("/kaggle/input/datasets/a210462khihng/cci-assets")
DEFAULT_CELEBA_ROOT = Path(
    "/kaggle/input/datasets/ipythonx/celebamaskhq/CelebAMask-HQ"
)
CLASSIFIER_PATH = DEFAULT_CCI_ASSET_ROOT / "resnet50_multilabel_model.pth"
IDENTITY_MODEL_PATH = DEFAULT_CCI_ASSET_ROOT / "facenet_vggface2.ts"
IMAGE_ROOT = DEFAULT_CELEBA_ROOT / "CelebA-HQ-img"
MASK_ROOT = DEFAULT_CELEBA_ROOT / "CelebAMask-HQ-mask-anno"
PACKAGE_MODULES = {
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "open-clip-torch": "open_clip",
    "grad-cam": "pytorch_grad_cam",
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
        "classifier": CLASSIFIER_PATH,
        "identity": IDENTITY_MODEL_PATH,
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


def run_evaluation(args: argparse.Namespace, assets: dict[str, Path]) -> None:
    output_root = Path(args.output_dir or "/kaggle/working/cci_fixed_vs_adaptive")
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        REPO_ROOT / "scripts" / "run_clean_cci_pilot.py",
        "--features",
        "smile",
        "--limit",
        str(args.sample_count),
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
        args.device,
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
        output_root,
    ]
    run_logged_command("smile evaluation: raw BLD / fixed CCI / adaptive CCI", command, cwd=REPO_ROOT)


def select_discovery_ids(
    args: argparse.Namespace,
    assets: dict[str, Path],
    output_root: Path,
) -> list[int]:
    ids_path = output_root / "discovery_ids.json"
    if ids_path.is_file():
        ids = json.loads(ids_path.read_text())["smile"]
        print(f"[{timestamp()}] SKIP  discovery cohort: reusing {len(ids)} IDs", flush=True)
        return ids

    print(f"[{timestamp()}] START discovery cohort selection", flush=True)
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
        feature="smile",
        classifier=classifier,
        detector=detector,
    )
    ids = [sample_id for sample_id, _, _ in selected]
    ids_path.write_text(json.dumps({"smile": ids}, indent=2))
    print(f"[{timestamp()}] DONE  discovery cohort selection: {len(ids)} IDs", flush=True)
    return ids


def run_discovery(args: argparse.Namespace, assets: dict[str, Path]) -> None:
    output_root = Path(args.output_dir or "/kaggle/working/cci_graph_discovery")
    output_root.mkdir(parents=True, exist_ok=True)
    ids = select_discovery_ids(args, assets, output_root)
    template = REPO_ROOT / "examples" / "graphs" / "remove_smile_clean_cci.json"
    screening = output_root / "smile" / "screening"
    interventions = output_root / "smile" / "interventions"
    graph = output_root / "smile" / "graph"

    run_logged_command(
        "smile discovery: Grad-CAM++ region screening",
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
            "skin",
            "nose",
            "mouth",
            "upper_lip",
            "lower_lip",
            "left_eye",
            "right_eye",
            "left_brow",
            "right_brow",
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

    run_logged_command(
        "smile discovery: diffusion region interventions",
        [
            sys.executable,
            "-u",
            REPO_ROOT / "scripts" / "run_counterfactual_region_interventions.py",
            "--template_graph",
            template,
            "--sample_ids",
            *ids,
            "--candidate_regions",
            *candidates,
            "--max_set_size",
            str(len(candidates)),
            "--stop_flip_rate",
            "0.96",
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
            args.device,
            "--torch_dtype",
            "auto",
            "--python_executable",
            sys.executable,
            "--continue_on_error",
            "--output_dir",
            interventions,
        ],
        cwd=REPO_ROOT,
    )
    run_logged_command(
        "smile discovery: freeze global graph",
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
    parser.add_argument("--sample_count", type=int, default=300)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    configure_runtime()
    print(
        f"[{timestamp()}] CONFIG mode={args.mode} samples={args.sample_count} "
        f"model={args.model_path} device={args.device}",
        flush=True,
    )
    ensure_runtime_packages()
    assets = kaggle_assets()
    if args.mode == "discovery":
        run_discovery(args, assets)
    else:
        run_evaluation(args, assets)
    print(f"[{timestamp()}] COMPLETE mode={args.mode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
