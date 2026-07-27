#!/usr/bin/env python3
"""Run one source-selected CCI generation per held-out image."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

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
from cci_diff.counterfactual_graph import InterventionObservation  # noqa: E402
from cci_diff.individual_region_selection import (  # noqa: E402
    FrozenInfluencePolicy,
    IndividualRegionSelection,
    load_frozen_influence_policy,
    select_individual_region_set,
)
from cci_diff.post_attack import gradcam_pp_saliency  # noqa: E402
from cci_diff.region_screening import celebamask_component_path  # noqa: E402
from scripts.run_counterfactual_region_interventions import (  # noqa: E402
    _prompt_for_graph,
    build_intervention_command,
    load_completed_observation,
    materialize_region_policy,
)


def source_requires_flip(
    probability: float,
    desired_value: int,
    threshold: float = 0.5,
) -> bool:
    """Return whether the source lies on the opposite target decision side."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if desired_value not in (0, 1) or isinstance(desired_value, bool):
        raise ValueError("desired_value must be 0 or 1")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return probability >= threshold if desired_value == 0 else probability < threshold


def prepare_individual_policy(
    *,
    source_path: str | Path,
    sample_id: int,
    source_probability: float,
    saliency: np.ndarray,
    component_paths: Mapping[str, str | Path],
    frozen_policy: FrozenInfluencePolicy,
    template_graph_path: str | Path,
    coverage_threshold: float,
    seed: int,
    output_dir: str | Path,
) -> tuple[IndividualRegionSelection, Path, Path, Path]:
    """Select source-only regions and materialize one execution policy."""

    if not source_requires_flip(
        source_probability,
        frozen_policy.desired_value,
    ):
        raise ValueError(
            f"Source {sample_id} already satisfies the desired target"
        )
    saliency_array = np.asarray(saliency)
    masks = {}
    for region, path_value in component_paths.items():
        path = Path(path_value)
        if not path.is_file():
            continue
        with Image.open(path) as image:
            masks[region] = np.asarray(
                image.convert("L").resize(
                    (saliency_array.shape[1], saliency_array.shape[0]),
                    Image.Resampling.NEAREST,
                )
            )
    selection = select_individual_region_set(
        saliency_array,
        masks,
        frozen_policy,
        coverage_threshold=coverage_threshold,
    )
    destination = Path(output_dir)
    graph_path, binding_path, union_path = materialize_region_policy(
        template_graph_path,
        source_path,
        component_paths,
        selection.selected_regions,
        destination,
    )
    record = _selection_payload(
        selection,
        sample_id=sample_id,
        source_path=source_path,
        source_probability=source_probability,
        seed=seed,
        frozen_policy=frozen_policy,
        execution_graph_path=graph_path,
    )
    (destination / "selection.json").write_text(
        json.dumps(record, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return selection, graph_path, binding_path, union_path


def run_individual_cci(args: argparse.Namespace) -> dict[str, Any]:
    """Select and generate exactly once for every held-out source image."""

    import torch

    validate_args(args)
    frozen = load_frozen_influence_policy(args.influence_graph)
    template = load_concept_graph(args.template_graph)
    if (
        template.intervention.concept != frozen.target
        or template.intervention.desired_value != frozen.desired_value
    ):
        raise ValueError("Template and frozen influence graph targets disagree")
    _validate_discovery_separation(args)

    label_index = resolve_celeba_attribute_index(frozen.target)
    classifier = load_celeba_resnet50(
        args.classifier_path,
        device=args.device,
        dtype=torch.float32,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_config = {
        "version": 1,
        "policy_type": "source_gradcam_individual_region",
        "influence_graph": args.influence_graph,
        "influence_graph_sha256": frozen.graph_sha256,
        "template_graph": args.template_graph,
        "template_graph_sha256": sha256_file(args.template_graph),
        "target": frozen.target,
        "desired_value": frozen.desired_value,
        "label_index": label_index,
        "verified_regions": list(frozen.verified_regions),
        "fallback_regions": list(frozen.fallback_regions),
        "coverage_threshold": args.coverage_threshold,
        "classifier_path": args.classifier_path,
        "classifier_sha256": sha256_file(args.classifier_path),
        "seed": args.seed,
        "generation": {
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "blending_start_percentage": args.blending_start_percentage,
            "generation_mask_dilation": args.generation_mask_dilation,
            "generation_mask_feather": args.generation_mask_feather,
        },
        "one_generation_per_image": True,
        "post_attack": "disabled",
        "output_reranking": "disabled",
        "region_escalation": "disabled",
    }
    (output_dir / "individual_policy.json").write_text(
        json.dumps(frozen_config, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    selections: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    failures = []
    attempted = 0
    prompt = _prompt_for_graph(args.template_graph)
    for sample_id in args.sample_ids:
        source = Path(args.image_root) / f"{sample_id}.jpg"
        try:
            source_probability, saliency = _compute_source_saliency(
                classifier,
                source,
                label_index=label_index,
                input_size=args.classifier_input_size,
                device=args.device,
            )
            component_paths = {
                region: path
                for region in frozen.verified_regions
                if (
                    path := celebamask_component_path(
                        args.mask_root,
                        sample_id,
                        region,
                    )
                ).is_file()
            }
            policy_dir = output_dir / "policies" / f"{sample_id:05d}"
            selection, graph_path, binding_path, _ = (
                prepare_individual_policy(
                    source_path=source,
                    sample_id=sample_id,
                    source_probability=source_probability,
                    saliency=saliency,
                    component_paths=component_paths,
                    frozen_policy=frozen,
                    template_graph_path=args.template_graph,
                    coverage_threshold=args.coverage_threshold,
                    seed=args.seed,
                    output_dir=policy_dir,
                )
            )
            selection_row = _selection_payload(
                selection,
                sample_id=sample_id,
                source_path=source,
                source_probability=source_probability,
                seed=args.seed,
                frozen_policy=frozen,
                execution_graph_path=graph_path,
            )
            selections.append(selection_row)
            _write_rows(
                output_dir / "individual_selections.csv",
                selections,
            )
            if args.dry_run:
                continue

            run_dir = output_dir / "runs" / f"{sample_id:05d}"
            observation = None
            if _audit_matches_execution_graph(run_dir, graph_path):
                try:
                    observation = load_completed_observation(
                        run_dir,
                        target=frozen.target,
                        label_index=label_index,
                        desired_value=frozen.desired_value,
                        sample_id=sample_id,
                        seed=args.seed,
                        regions=selection.selected_regions,
                    )
                except (
                    FileNotFoundError,
                    ValueError,
                    KeyError,
                    json.JSONDecodeError,
                ):
                    observation = None
            if observation is None:
                attempted += 1
                command = build_intervention_command(
                    args,
                    graph_path=graph_path,
                    binding_path=binding_path,
                    output_dir=run_dir,
                    seed=args.seed,
                    prompt=prompt,
                )
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    check=False,
                )
                if completed.returncode != 0:
                    failure = {
                        "sample_id": sample_id,
                        "stage": "generation",
                        "returncode": completed.returncode,
                        "selected_regions": list(selection.selected_regions),
                    }
                    failures.append(failure)
                    _append_jsonl(output_dir / "failures.jsonl", failure)
                    if args.continue_on_error:
                        continue
                    raise RuntimeError(
                        f"Individual generation failed: {failure}"
                    )
                observation = load_completed_observation(
                    run_dir,
                    target=frozen.target,
                    label_index=label_index,
                    desired_value=frozen.desired_value,
                    sample_id=sample_id,
                    seed=args.seed,
                    regions=selection.selected_regions,
                )
            result = _observation_payload(observation)
            result.update(
                {
                    "coverage": selection.coverage,
                    "coverage_threshold": selection.coverage_threshold,
                    "selected_mask_fraction": selection.mask_fraction,
                    "fallback_used": selection.fallback_used,
                }
            )
            results.append(result)
            _write_rows(output_dir / "individual_results.csv", results)
        except (
            FileNotFoundError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            failure = {
                "sample_id": sample_id,
                "stage": "selection_or_validation",
                "error": str(error),
            }
            failures.append(failure)
            _append_jsonl(output_dir / "failures.jsonl", failure)
            if not args.continue_on_error:
                raise

    manifest = {
        **frozen_config,
        "sample_ids": args.sample_ids,
        "selection_count": len(selections),
        "attempted_generations": attempted,
        "completed_generations": len(results),
        "failed_generations": sum(
            failure["stage"] == "generation" for failure in failures
        ),
        "failures": failures,
        "dry_run": args.dry_run,
    }
    (output_dir / "individual_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


def validate_args(args: argparse.Namespace) -> None:
    if not args.sample_ids or len(args.sample_ids) != len(set(args.sample_ids)):
        raise ValueError("sample_ids must be non-empty and unique")
    if any(sample_id < 0 for sample_id in args.sample_ids):
        raise ValueError("sample_ids must be non-negative")
    if not 0.0 < args.coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold must be in (0, 1]")
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    for name in (
        "influence_graph",
        "template_graph",
        "classifier_path",
        "identity_model_path",
    ):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"model_path not found: {args.model_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--influence_graph", required=True)
    parser.add_argument("--template_graph", required=True)
    parser.add_argument("--sample_ids", nargs="+", type=int, required=True)
    parser.add_argument("--coverage_threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image_root",
        default="data/CelebAMask-HQ/CelebA-HQ-img",
    )
    parser.add_argument(
        "--mask_root",
        default="data/CelebAMask-HQ/CelebAMask-HQ-mask-anno",
    )
    parser.add_argument("--model_path", default="checkpoints/sd2-1-base")
    parser.add_argument("--classifier_path", required=True)
    parser.add_argument("--identity_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--classifier_input_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument(
        "--blending_start_percentage", type=float, default=0.25
    )
    parser.add_argument("--generation_mask_dilation", type=int, default=0)
    parser.add_argument("--generation_mask_feather", type=float, default=3.0)
    parser.add_argument(
        "--python_executable",
        default=".venv-ml/bin/python",
    )
    parser.add_argument("--discovery_manifest", default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _compute_source_saliency(
    classifier: Any,
    source_path: Path,
    *,
    label_index: int,
    input_size: int,
    device: str,
) -> tuple[float, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    if not source_path.is_file():
        raise FileNotFoundError(f"Source image not found: {source_path}")
    with Image.open(source_path) as image:
        array = (
            np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        )
    source = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    source = functional.interpolate(
        source,
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
    ).to(device=device, dtype=torch.float32)
    normalized = preprocess_classifier_images(source, size=input_size)
    with torch.no_grad():
        probability = float(classifier(normalized)[:, label_index].item())
    saliency = gradcam_pp_saliency(
        classifier,
        normalized,
        label_index=label_index,
        original_present=probability >= 0.5,
    )
    return probability, saliency


def _selection_payload(
    selection: IndividualRegionSelection,
    *,
    sample_id: int,
    source_path: str | Path,
    source_probability: float,
    seed: int,
    frozen_policy: FrozenInfluencePolicy,
    execution_graph_path: str | Path,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_path": str(source_path),
        "source_probability": source_probability,
        "seed": seed,
        "target": frozen_policy.target,
        "desired_value": frozen_policy.desired_value,
        "available_regions": list(selection.available_regions),
        "missing_regions": list(selection.missing_regions),
        "selected_regions": list(selection.selected_regions),
        "coverage": selection.coverage,
        "coverage_threshold": selection.coverage_threshold,
        "mask_fraction": selection.mask_fraction,
        "region_importance": dict(selection.region_importance),
        "candidate_count": selection.candidate_count,
        "fallback_used": selection.fallback_used,
        "fallback_reason": selection.fallback_reason,
        "selection_uses_generated_output": False,
        "influence_graph_sha256": frozen_policy.graph_sha256,
        "execution_graph_sha256": sha256_file(execution_graph_path),
    }


def _observation_payload(row: InterventionObservation) -> dict[str, Any]:
    return {
        "target": row.target,
        "desired_value": row.desired_value,
        "sample_id": row.sample_id,
        "seed": row.seed,
        "regions": list(row.regions),
        "source_probability": row.source_probability,
        "output_probability": row.output_probability,
        "desired_probability": row.output_desired_probability,
        "target_effect": row.target_effect,
        "target_pass": row.target_pass,
        "mask_fraction": row.mask_fraction,
        "identity_cosine": row.identity_cosine,
        "non_target_drift": row.non_target_drift,
        "outside_l1": row.outside_l1,
        "changed_fraction": row.changed_fraction,
        "output_path": row.output_path,
        "audit_path": row.audit_path,
    }


def _audit_matches_execution_graph(run_dir: Path, graph_path: Path) -> bool:
    audit_path = run_dir / "audit.json"
    if not audit_path.is_file():
        return False
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (audit.get("cci") or {}).get("graph_sha256") == sha256_file(
        graph_path
    )


def _validate_discovery_separation(args: argparse.Namespace) -> None:
    if args.discovery_manifest is None:
        return
    path = Path(args.discovery_manifest)
    if not path.is_file():
        raise FileNotFoundError(f"discovery_manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    discovery_ids = set(payload.get("sample_ids") or [])
    overlap = sorted(discovery_ids.intersection(args.sample_ids))
    if overlap:
        raise ValueError(
            f"Held-out sample IDs overlap discovery cohort: {overlap}"
        )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, allow_nan=False) + "\n")


def main() -> int:
    run_individual_cci(build_arg_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
