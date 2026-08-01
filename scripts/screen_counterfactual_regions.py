#!/usr/bin/env python3
"""Rank semantic component proposals using classifier Grad-CAM++ overlap."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cci_diff.classifiers.celeba_resnet50 import (  # noqa: E402
    load_celeba_resnet50,
    preprocess_classifier_images,
    resolve_celeba_attribute_index,
)
from cci_diff.concept_graph import load_concept_graph, sha256_file  # noqa: E402
from cci_diff.post_attack import gradcam_pp_saliency  # noqa: E402
from cci_diff.region_screening import (  # noqa: E402
    celebamask_component_path,
    score_region_masks,
    select_saliency_covering_regions,
)
from cci_diff.runtime_environment import resolve_device  # noqa: E402


def aggregate_screening_rows(
    rows: Iterable[dict[str, Any]],
    *,
    sample_count: int,
) -> list[dict[str, Any]]:
    """Aggregate robust Grad-CAM statistics and component availability."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["region"]), []).append(row)
    summary = []
    for region, region_rows in grouped.items():
        metrics = {
            name: [float(row[name]) for row in region_rows]
            for name in (
                "proposal_score",
                "captured_mass",
                "region_density",
                "mask_fraction",
            )
        }
        coverage_count = len(
            {int(row["sample_id"]) for row in region_rows}
        )
        summary.append(
            {
                "region": region,
                **{
                    f"mean_{name}": sum(values) / len(values)
                    for name, values in metrics.items()
                },
                **{
                    f"median_{name}": statistics.median(values)
                    for name, values in metrics.items()
                },
                "coverage_count": coverage_count,
                "coverage_frequency": coverage_count / sample_count,
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            -row["median_region_density"],
            -row["median_captured_mass"],
            -row["coverage_frequency"],
            row["median_mask_fraction"],
            row["region"],
        ),
    )


def screen_regions(args: argparse.Namespace) -> dict[str, Any]:
    """Compute Grad-CAM proposal evidence for candidate semantic regions."""

    import torch

    args.device = resolve_device(args.device, torch)
    if len(args.sample_ids) != len(set(args.sample_ids)):
        raise ValueError("sample_ids must be unique")
    candidate_regions = tuple(
        sorted({str(region).strip() for region in args.candidate_regions})
    )
    if not candidate_regions or any(not region for region in candidate_regions):
        raise ValueError("candidate_regions must be non-empty")
    graph = load_concept_graph(args.template_graph)
    label_index = resolve_celeba_attribute_index(
        graph.intervention.concept
    )
    model = load_celeba_resnet50(
        args.classifier_path,
        device=args.device,
        dtype=torch.float32,
    )
    rows = []
    for sample_id in args.sample_ids:
        source_path = Path(args.image_root) / f"{sample_id}.jpg"
        source = _load_rgb_tensor(
            source_path,
            size=args.classifier_input_size,
            device=args.device,
        )
        normalized = preprocess_classifier_images(
            source, size=args.classifier_input_size
        )
        with torch.no_grad():
            source_probability = float(
                model(normalized)[:, label_index].item()
            )
        saliency = gradcam_pp_saliency(
            model,
            normalized,
            label_index=label_index,
            original_present=source_probability >= 0.5,
        )
        masks = {}
        for region in candidate_regions:
            path = celebamask_component_path(
                args.mask_root, sample_id, region
            )
            if not path.is_file():
                continue
            with Image.open(path) as image:
                masks[region] = np.asarray(
                    image.convert("L").resize(
                        (saliency.shape[1], saliency.shape[0]),
                        Image.Resampling.NEAREST,
                    )
                )
        if not masks:
            raise FileNotFoundError(
                f"No candidate masks found for sample {sample_id}"
            )
        scores = score_region_masks(saliency, masks)
        for score in scores:
            rows.append(
                {
                    "sample_id": sample_id,
                    "target": graph.intervention.concept,
                    "desired_value": graph.intervention.desired_value,
                    "source_probability": source_probability,
                    "region": score.region,
                    "captured_mass": score.captured_mass,
                    "region_density": score.region_density,
                    "mask_fraction": score.mask_fraction,
                    "proposal_score": score.proposal_score,
                }
            )

    summary = aggregate_screening_rows(
        rows, sample_count=len(args.sample_ids)
    )
    eligible_rows = [
        row
        for row in rows
        if float(row["captured_mass"]) >= args.minimum_captured_saliency
    ]
    selected_regions, subset_evidence, selection_status = (
        select_saliency_covering_regions(
            eligible_rows,
            saliency_coverage_threshold=args.saliency_coverage_threshold,
            cohort_frequency_threshold=args.cohort_frequency_threshold,
            max_regions=args.max_selected_regions,
        )
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "screening_rows.csv", rows)
    _write_rows(output_dir / "screening_summary.csv", summary)
    manifest = {
        "version": 2,
        "evidence_type": "gradcam_proposal_only",
        "target": graph.intervention.concept,
        "desired_value": graph.intervention.desired_value,
        "label_index": label_index,
        "sample_ids": args.sample_ids,
        "candidate_regions": list(candidate_regions),
        "selected_candidate_regions": list(selected_regions),
        "selection": {
            "method": "minimal_area_saliency_coverage",
            "status": selection_status,
            "max_selected_regions": args.max_selected_regions,
            "saliency_coverage_threshold": args.saliency_coverage_threshold,
            "cohort_frequency_threshold": args.cohort_frequency_threshold,
            "minimum_coverage_frequency": args.minimum_coverage_frequency,
            "minimum_captured_saliency": args.minimum_captured_saliency,
            "subset_metrics": [
                {
                    "regions": list(item.regions),
                    "mean_saliency_coverage": item.mean_saliency_coverage,
                    "cohort_frequency": item.cohort_frequency,
                    "mean_mask_fraction": item.mean_mask_fraction,
                    "passes": item.passes,
                }
                for item in subset_evidence
            ],
        },
        "classifier_path": args.classifier_path,
        "classifier_sha256": sha256_file(args.classifier_path),
        "device": args.device,
        "template_graph": args.template_graph,
        "template_graph_sha256": sha256_file(args.template_graph),
        "ranking": summary,
        "warning": (
            "Grad-CAM screening proposes regions but does not verify "
            "counterfactual influence."
        ),
    }
    (output_dir / "screening_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_graph", required=True)
    parser.add_argument("--classifier_path", required=True)
    parser.add_argument("--sample_ids", nargs="+", type=int, required=True)
    parser.add_argument("--candidate_regions", nargs="+", required=True)
    parser.add_argument("--max_selected_regions", type=int, default=4)
    parser.add_argument(
        "--saliency_coverage_threshold",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--cohort_frequency_threshold",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--minimum_coverage_frequency",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum_captured_saliency",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--image_root",
        default="data/CelebAMask-HQ/CelebA-HQ-img",
    )
    parser.add_argument(
        "--mask_root",
        default="data/CelebAMask-HQ/CelebAMask-HQ-mask-anno",
    )
    parser.add_argument("--classifier_input_size", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--output_dir", required=True)
    return parser


def _load_rgb_tensor(path: Path, *, size: int, device: str):
    import torch
    import torch.nn.functional as functional

    if not path.is_file():
        raise FileNotFoundError(f"Source image not found: {path}")
    with Image.open(path) as image:
        array = (
            np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        )
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = functional.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return tensor.to(device=device, dtype=torch.float32)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    screen_regions(build_arg_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
