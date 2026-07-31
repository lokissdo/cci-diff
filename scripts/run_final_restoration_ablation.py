#!/usr/bin/env python3
"""Run an isolated A11 final-restoration visual ablation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from scripts.run_clean_cci_pilot import (
        FEATURES,
        MaskCandidate,
        annotation_paths,
        build_variant_command,
        resolve_binding_roles,
        write_binding,
        write_region_graph,
    )
except ModuleNotFoundError:
    from run_clean_cci_pilot import (
        FEATURES,
        MaskCandidate,
        annotation_paths,
        build_variant_command,
        resolve_binding_roles,
        write_binding,
        write_region_graph,
    )


ROOT = Path(__file__).resolve().parents[1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", type=int, default=26811)
    parser.add_argument(
        "--output-dir",
        default="outputs/final_restoration_ablation_26811",
    )
    parser.add_argument(
        "--image-root",
        default=str(ROOT / "data/CelebAMask-HQ/CelebA-HQ-img"),
    )
    parser.add_argument(
        "--mask-root",
        default=str(ROOT / "data/CelebAMask-HQ/CelebAMask-HQ-mask-anno"),
    )
    parser.add_argument(
        "--model-path",
        default=str(ROOT / "checkpoints/sd2-1-base"),
    )
    parser.add_argument(
        "--classifier-path",
        default=str(ROOT / "models/resnet50_multilabel_model.pth"),
    )
    parser.add_argument(
        "--identity-model-path",
        default=str(ROOT / "models/facenet_vggface2.ts"),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--num-inference-steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-dilation", type=int, default=8)
    parser.add_argument("--consistency-tolerance", type=float, default=0.02)
    parser.add_argument("--reuse", action="store_true")
    return parser


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _pilot_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        python_executable=args.python_executable,
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        allow_model_download=False,
        classifier_path=args.classifier_path,
        identity_model_path=args.identity_model_path,
        cci_post_attack="none",
        cci_post_attack_epsilon_schedule="0.05,0.08,0.10,0.30,0.50",
        cci_post_attack_boundary_margin=0.03,
    )


def _replace_output_dir(command: list[str], output_dir: Path) -> list[str]:
    replaced = list(command)
    replaced[replaced.index("--output_dir") + 1] = str(output_dir)
    return replaced


def prepare_ablation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    source = Path(args.image_root) / f"{args.sample_id}.jpg"
    masks = annotation_paths(
        Path(args.mask_root),
        args.sample_id,
        FEATURES["smile"]["components"],
    )
    _require_file(Path(args.python_executable), "Python executable")
    _require_file(source, "source image")
    for component, path in masks.items():
        _require_file(path, f"{component} mask")
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(f"Missing SD2 model directory: {args.model_path}")
    _require_file(Path(args.classifier_path), "classifier checkpoint")
    _require_file(Path(args.identity_model_path), "identity checkpoint")

    graph = write_region_graph(
        ROOT / FEATURES["smile"]["graph"],
        output_dir / "config" / "remove_smile_mouth_only.json",
        ("mouth",),
    )
    binding = output_dir / "config" / f"smile_{args.sample_id:05d}.json"
    write_binding(
        binding,
        source,
        masks,
        resolve_binding_roles("smile", ["mouth"]),
    )
    enabled_dir = output_dir / "with_final_restoration"
    disabled_dir = output_dir / "without_final_restoration"
    enabled = build_variant_command(
        _pilot_args(args),
        feature="smile",
        variant="A11",
        sample_id=args.sample_id,
        source=source,
        masks=masks,
        binding_path=binding,
        output_path=enabled_dir,
        mask_candidate=MaskCandidate("d8", args.mask_dilation),
        graph_path=graph,
    )
    disabled = _replace_output_dir(enabled, disabled_dir)
    disabled.append("--cci_disable_final_correction")
    return {
        "source": source,
        "masks": masks,
        "graph": graph,
        "binding": binding,
        "enabled_dir": enabled_dir,
        "disabled_dir": disabled_dir,
        "enabled_command": enabled,
        "disabled_command": disabled,
    }


def _load_case(case_dir: Path) -> tuple[dict[str, Any], Image.Image]:
    audit_path = case_dir / "audit.json"
    image_path = case_dir / "sd2_bld_grid.png"
    _require_file(audit_path, "ablation audit")
    _require_file(image_path, "ablation image")
    return (
        json.loads(audit_path.read_text(encoding="utf-8")),
        Image.open(image_path).convert("RGB"),
    )


def _desired_probability(audit: dict[str, Any]) -> float:
    probabilities = audit["cci"]["metrics"]["attributes"][
        "output_probabilities"
    ]
    return 1.0 - float(probabilities[FEATURES["smile"]["label_index"]])


def _case_metrics(audit: dict[str, Any]) -> dict[str, float | None]:
    metrics = audit["cci"]["metrics"]
    return {
        "desired_probability": _desired_probability(audit),
        "identity_cosine": (
            None
            if metrics.get("identity_cosine") is None
            else float(metrics["identity_cosine"])
        ),
        "mean_non_target_drift": float(
            metrics["attributes"]["mean_non_target_drift"]
        ),
        "wall_seconds": float(audit["cci"]["wall_seconds"]),
    }


def _labelled_side_by_side(
    before: Image.Image,
    after: Image.Image,
) -> Image.Image:
    banner_height = 32
    canvas = Image.new(
        "RGB",
        (before.width + after.width, before.height + banner_height),
        "white",
    )
    canvas.paste(before, (0, banner_height))
    canvas.paste(after, (before.width, banner_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), "Without final restoration", fill="black")
    draw.text(
        (before.width + 8, 9),
        "With final restoration",
        fill="black",
    )
    return canvas


def compare_ablation(
    before_dir: str | Path,
    after_dir: str | Path,
    output_dir: str | Path,
    *,
    consistency_tolerance: float,
    commands: dict[str, list[str]],
    sample_id: int = 26811,
) -> dict[str, Any]:
    before_dir = Path(before_dir)
    after_dir = Path(after_dir)
    output_dir = Path(output_dir)
    before_audit, before_image = _load_case(before_dir)
    after_audit, after_image = _load_case(after_dir)
    before_cci = before_audit["cci"]
    after_cci = after_audit["cci"]
    if before_cci.get("post_attack") is not None or after_cci.get(
        "post_attack"
    ) is not None:
        raise ValueError("Final-restoration ablation contains post-attack data")
    if before_cci.get("trust_region_final_restoration") is not None:
        raise ValueError("Disabled case unexpectedly contains restoration data")
    restoration = after_cci.get("trust_region_final_restoration")
    if not isinstance(restoration, dict):
        raise ValueError("Enabled case is missing final-restoration data")
    if before_image.size != after_image.size:
        raise ValueError("Ablation images must have equal dimensions")

    before_metrics = _case_metrics(before_audit)
    after_metrics = _case_metrics(after_audit)
    consistency_gap = abs(
        float(restoration["initial_probability"])
        - float(before_metrics["desired_probability"])
    )
    if consistency_gap > consistency_tolerance:
        raise ValueError(
            "Separate-run consistency check failed: "
            f"gap={consistency_gap:.6f}, "
            f"tolerance={consistency_tolerance:.6f}"
        )

    before_values = np.asarray(before_image, dtype=np.float32) / 255.0
    after_values = np.asarray(after_image, dtype=np.float32) / 255.0
    difference = np.abs(after_values - before_values)
    output_dir.mkdir(parents=True, exist_ok=True)
    before_path = output_dir / "before_without_final_restoration.png"
    after_path = output_dir / "after_with_final_restoration.png"
    side_by_side_path = output_dir / "before_after_side_by_side.png"
    difference_path = output_dir / "difference_amplified.png"
    shutil.copyfile(before_dir / "sd2_bld_grid.png", before_path)
    shutil.copyfile(after_dir / "sd2_bld_grid.png", after_path)
    _labelled_side_by_side(before_image, after_image).save(side_by_side_path)
    Image.fromarray(
        np.uint8(np.clip(difference * 8.0, 0.0, 1.0) * 255.0),
        mode="RGB",
    ).save(difference_path)

    payload = {
        "sample_id": sample_id,
        "before": before_metrics,
        "after": after_metrics,
        "delta": {
            key: (
                None
                if before_metrics[key] is None or after_metrics[key] is None
                else float(after_metrics[key]) - float(before_metrics[key])
            )
            for key in before_metrics
        },
        "pixel": {
            "mean_absolute_difference": float(difference.mean()),
            "maximum_absolute_difference": float(difference.max()),
            "changed_fraction": float(
                np.any(difference > (1.0 / 255.0), axis=2).mean()
            ),
        },
        "restoration": restoration,
        "consistency": {
            "gap": consistency_gap,
            "tolerance": consistency_tolerance,
            "passed": True,
        },
        "commands": commands,
        "artifacts": {
            "before": str(before_path),
            "after": str(after_path),
            "side_by_side": str(side_by_side_path),
            "difference_amplified": str(difference_path),
        },
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Final-Restoration Visual Ablation",
        "",
        "| Metric | Without restoration | With restoration | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in before_metrics:
        before_value = before_metrics[key]
        after_value = after_metrics[key]
        delta = payload["delta"][key]
        lines.append(
            f"| {key} | {before_value} | {after_value} | {delta} |"
        )
    lines.extend(
        [
            "",
            f"- Pixel MAE: {payload['pixel']['mean_absolute_difference']}",
            f"- Pixel maximum change: {payload['pixel']['maximum_absolute_difference']}",
            f"- Changed fraction: {payload['pixel']['changed_fraction']}",
            f"- Consistency gap: {consistency_gap}",
        ]
    )
    (output_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return payload


def _case_is_complete(case_dir: Path) -> bool:
    return (case_dir / "audit.json").is_file() and (
        case_dir / "sd2_bld_grid.png"
    ).is_file()


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepared = prepare_ablation(args)
    cases = (
        ("disabled", prepared["disabled_dir"], prepared["disabled_command"]),
        ("enabled", prepared["enabled_dir"], prepared["enabled_command"]),
    )
    for _, case_dir, command in cases:
        if args.reuse and _case_is_complete(case_dir):
            continue
        subprocess.run(command, check=True, cwd=ROOT)
    payload = compare_ablation(
        prepared["disabled_dir"],
        prepared["enabled_dir"],
        args.output_dir,
        consistency_tolerance=args.consistency_tolerance,
        commands={
            "disabled": prepared["disabled_command"],
            "enabled": prepared["enabled_command"],
        },
        sample_id=args.sample_id,
    )
    print(payload["artifacts"]["side_by_side"])
    return payload


def main() -> int:
    run(build_arg_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
