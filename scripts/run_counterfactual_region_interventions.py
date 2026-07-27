#!/usr/bin/env python3
"""Run paired same-seed semantic-region interventions for graph discovery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cci_diff.classifiers.celeba_resnet50 import (  # noqa: E402
    resolve_celeba_attribute_index,
)
from cci_diff.concept_graph import load_concept_graph, sha256_file  # noqa: E402
from cci_diff.counterfactual_graph import InterventionObservation  # noqa: E402
from cci_diff.region_screening import (  # noqa: E402
    build_union_mask,
    canonical_region_sets,
    celebamask_component_path,
)
from cci_diff.spatial_selection import measure_spatial_change  # noqa: E402


def deduplicate_region_sets(
    region_sets: Sequence[tuple[str, ...]],
    *,
    sample_ids: Sequence[int],
    mask_root: str | Path,
) -> tuple[
    tuple[tuple[str, ...], ...],
    dict[tuple[str, ...], tuple[str, ...]],
    dict[tuple[str, ...], str],
]:
    """Collapse region sets with identical union masks over the cohort."""

    canonical_sets = tuple(
        tuple(sorted({str(region).strip() for region in regions}))
        for regions in region_sets
    )
    if not canonical_sets or any(not regions for regions in canonical_sets):
        raise ValueError("region_sets must contain non-empty region sets")
    if len(canonical_sets) != len(set(canonical_sets)):
        raise ValueError("region_sets must be unique")
    cohort_ids = tuple(int(sample_id) for sample_id in sample_ids)
    if not cohort_ids or len(cohort_ids) != len(set(cohort_ids)):
        raise ValueError("sample_ids must be non-empty and unique")

    exact_keys: dict[
        tuple[str, ...],
        list[tuple[int, int, bytes]],
    ] = {regions: [] for regions in canonical_sets}
    for sample_id in cohort_ids:
        needed_regions = sorted(
            {region for regions in canonical_sets for region in regions}
        )
        components = {
            region: _load_binary_component(
                celebamask_component_path(mask_root, sample_id, region)
            )
            for region in needed_regions
        }
        shapes = {array.shape for array in components.values()}
        if len(shapes) != 1:
            raise ValueError(
                f"Component masks differ in shape for sample {sample_id}"
            )
        for regions in canonical_sets:
            union = np.logical_or.reduce(
                [components[region] for region in regions]
            )
            height, width = union.shape
            exact_keys[regions].append(
                (
                    int(height),
                    int(width),
                    np.packbits(union, axis=None).tobytes(),
                )
            )

    groups: dict[
        tuple[tuple[int, int, bytes], ...],
        list[tuple[str, ...]],
    ] = {}
    signatures: dict[tuple[str, ...], str] = {}
    for regions, image_keys in exact_keys.items():
        cohort_key = tuple(image_keys)
        groups.setdefault(cohort_key, []).append(regions)
        digest = hashlib.sha256()
        for sample_id, (height, width, packed) in zip(
            cohort_ids, cohort_key
        ):
            digest.update(
                f"{sample_id}:{height}:{width}:".encode("ascii")
            )
            digest.update(packed)
        signatures[regions] = digest.hexdigest()

    aliases: dict[tuple[str, ...], tuple[str, ...]] = {}
    representatives = []
    for equivalent_sets in groups.values():
        representative = min(
            equivalent_sets,
            key=lambda regions: (len(regions), regions),
        )
        representatives.append(representative)
        aliases.update(
            {
                regions: representative
                for regions in equivalent_sets
                if regions != representative
            }
        )
    canonical = tuple(
        sorted(representatives, key=lambda regions: (len(regions), regions))
    )
    return canonical, aliases, signatures


def materialize_region_policy(
    template_graph_path: str | Path,
    source_path: str | Path,
    component_paths: Mapping[str, str | Path],
    regions: tuple[str, ...],
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Create an execution graph, binding, and strict union audit mask."""

    template_path = Path(template_graph_path)
    source = Path(source_path)
    if not template_path.is_file():
        raise FileNotFoundError(f"Template graph not found: {template_path}")
    if not source.is_file():
        raise FileNotFoundError(f"Source image not found: {source}")
    canonical = tuple(sorted(set(regions)))
    missing = [region for region in canonical if region not in component_paths]
    if missing:
        raise ValueError(f"Missing component paths: {missing}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    union_path = destination / "target_region.png"
    build_union_mask(component_paths, canonical, output_path=union_path)

    graph_payload = json.loads(template_path.read_text(encoding="utf-8"))
    graph_payload["region"]["audit_role"] = "target_region"
    graph_payload["region"]["components"] = list(canonical)
    graph_path = destination / "graph.json"
    graph_path.write_text(
        json.dumps(graph_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    binding_payload = {
        "source_image": str(source),
        "masks": {
            **{region: str(component_paths[region]) for region in canonical},
            "target_region": str(union_path),
        },
    }
    binding_path = destination / "binding.json"
    binding_path.write_text(
        json.dumps(binding_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return graph_path, binding_path, union_path


def build_intervention_command(
    args: argparse.Namespace,
    *,
    graph_path: Path,
    binding_path: Path,
    output_dir: Path,
    seed: int,
    prompt: str,
) -> list[str]:
    """Build a clean-CCI command without any post-generation attack."""

    return [
        args.python_executable,
        "scripts/run_sd2_bld_cci.py",
        "--output_dir",
        str(output_dir),
        "--prompt",
        prompt,
        "--model_path",
        args.model_path,
        "--local_files_only",
        "--device",
        args.device,
        "--torch_dtype",
        (
            "float16"
            if getattr(args, "torch_dtype", "auto") == "auto"
            and args.device.startswith("cuda")
            else "float32"
            if getattr(args, "torch_dtype", "auto") == "auto"
            else args.torch_dtype
        ),
        "--batch_size",
        "1",
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--blending_start_percentage",
        str(args.blending_start_percentage),
        "--seed",
        str(seed),
        "--generation_mask_dilation",
        str(args.generation_mask_dilation),
        "--generation_mask_feather",
        str(args.generation_mask_feather),
        "--cci_hook",
        "clean_constraint",
        "--cci_graph",
        str(graph_path),
        "--cci_sample_bindings",
        str(binding_path),
        "--classifier_path",
        args.classifier_path,
        "--identity_model_path",
        args.identity_model_path,
        "--cci_controller_mode",
        "feedback",
    ]


def load_completed_observation(
    run_dir: str | Path,
    *,
    target: str,
    label_index: int,
    desired_value: int,
    sample_id: int,
    seed: int,
    regions: tuple[str, ...],
) -> InterventionObservation:
    """Validate completed artifacts and convert their audit to one row."""

    directory = Path(run_dir)
    audit_path = directory / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit not found: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cci = audit.get("cci") or {}
    metrics = cci.get("metrics") or {}
    attributes = metrics.get("attributes") or {}
    source_values = attributes.get("source_probabilities") or []
    output_values = attributes.get("output_probabilities") or []
    if len(source_values) <= label_index or len(output_values) <= label_index:
        raise ValueError("Audit does not contain the requested target probability")

    source_path = _artifact_path(
        cci.get("source_image"), directory, "source image"
    )
    output_path = _artifact_path(
        audit.get("image_path") or directory / "sd2_bld_grid.png",
        directory,
        "output image",
    )
    artifacts = cci.get("mask_artifacts") or {}
    semantic_path = _artifact_path(
        artifacts.get("semantic_path") or directory / "semantic_mask.png",
        directory,
        "semantic mask",
    )
    generation_path = _artifact_path(
        artifacts.get("generation_path") or directory / "generation_mask.png",
        directory,
        "generation mask",
    )
    spatial = measure_spatial_change(
        source_path,
        output_path,
        semantic_path,
        generation_path,
    )
    semantic = ((metrics.get("locality") or {}).get("semantic_union") or {})
    return InterventionObservation(
        target=target,
        desired_value=desired_value,
        sample_id=sample_id,
        seed=seed,
        regions=regions,
        source_probability=float(source_values[label_index]),
        output_probability=float(output_values[label_index]),
        mask_fraction=float(
            semantic.get(
                "mask_fraction",
                artifacts.get(
                    "semantic_fraction",
                    spatial["semantic_mask_fraction"],
                ),
            )
        ),
        identity_cosine=_optional_float(metrics.get("identity_cosine")),
        non_target_drift=_optional_float(
            attributes.get("mean_non_target_drift")
        ),
        outside_l1=_optional_float(
            semantic.get("outside_mae", spatial["outside_semantic_l1"])
        ),
        changed_fraction=float(spatial["changed_fraction_5"]),
        output_path=str(output_path),
        audit_path=str(audit_path),
    )


def summarize_cardinality(
    rows: Sequence[InterventionObservation],
    region_sets: Sequence[tuple[str, ...]],
    *,
    expected_rows_per_set: int,
    stop_flip_rate: float,
) -> dict[str, Any]:
    """Summarize one completed cardinality with target effect as priority."""

    if expected_rows_per_set <= 0:
        raise ValueError("expected_rows_per_set must be positive")
    if not 0.0 <= stop_flip_rate <= 1.0:
        raise ValueError("stop_flip_rate must be in [0, 1]")
    metrics = []
    for regions in region_sets:
        selected = [row for row in rows if row.regions == regions]
        complete = len(selected) == expected_rows_per_set
        mean_effect = (
            sum(row.target_effect for row in selected) / len(selected)
            if selected
            else None
        )
        flip_rate = (
            sum(row.target_pass for row in selected) / len(selected)
            if selected
            else None
        )
        metrics.append(
            {
                "regions": list(regions),
                "completed_rows": len(selected),
                "expected_rows": expected_rows_per_set,
                "complete": complete,
                "mean_target_effect": mean_effect,
                "flip_rate": flip_rate,
            }
        )
    complete = all(item["complete"] for item in metrics)
    ranked = sorted(
        (item for item in metrics if item["complete"]),
        key=lambda item: (
            -float(item["mean_target_effect"]),
            -float(item["flip_rate"]),
            item["regions"],
        ),
    )
    best = ranked[0] if ranked else None
    threshold_reached = complete and any(
        float(item["flip_rate"]) >= stop_flip_rate for item in metrics
    )
    return {
        "cardinality": len(region_sets[0]) if region_sets else 0,
        "complete": complete,
        "threshold_reached": threshold_reached,
        "best_regions": best["regions"] if best else None,
        "best_mean_target_effect": (
            best["mean_target_effect"] if best else None
        ),
        "best_flip_rate": best["flip_rate"] if best else None,
        "region_sets": metrics,
    }


def run_interventions(args: argparse.Namespace) -> dict[str, Any]:
    """Execute or resume the requested paired intervention grid."""

    validate_args(args)
    template = load_concept_graph(args.template_graph)
    target = template.intervention.concept
    desired_value = template.intervention.desired_value
    label_index = resolve_celeba_attribute_index(target)
    requested_region_sets = canonical_region_sets(
        args.candidate_regions,
        max_set_size=args.max_set_size,
    )
    region_sets, region_set_aliases, region_set_signatures = (
        deduplicate_region_sets(
            requested_region_sets,
            sample_ids=args.sample_ids,
            mask_root=args.mask_root,
        )
    )
    prompt = _prompt_for_graph(args.template_graph)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 2,
        "target": target,
        "label_index": label_index,
        "desired_value": desired_value,
        "sample_ids": args.sample_ids,
        "seeds": args.seeds,
        "candidate_regions": sorted(set(args.candidate_regions)),
        "requested_region_sets": [
            list(regions) for regions in requested_region_sets
        ],
        "region_sets": [list(regions) for regions in region_sets],
        "region_set_aliases": [
            {
                "regions": list(regions),
                "canonical_regions": list(canonical),
                "cohort_union_sha256": region_set_signatures[regions],
            }
            for regions, canonical in sorted(
                region_set_aliases.items(),
                key=lambda item: (len(item[0]), item[0]),
            )
        ],
        "region_set_signatures": [
            {
                "regions": list(regions),
                "cohort_union_sha256": region_set_signatures[regions],
            }
            for regions in region_sets
        ],
        "deduplication": {
            "rule": "exact_union_mask_equality_over_discovery_cohort",
            "requested_set_count": len(requested_region_sets),
            "canonical_set_count": len(region_sets),
            "skipped_alias_count": len(region_set_aliases),
        },
        "planned_expected_rows": (
            len(args.sample_ids) * len(args.seeds) * len(region_sets)
        ),
        "expected_rows": (
            len(args.sample_ids) * len(args.seeds) * len(region_sets)
        ),
        "stop_flip_rate": args.stop_flip_rate,
        "planned_region_sets": [list(regions) for regions in region_sets],
        "executed_region_sets": [],
        "cardinality_results": [],
        "stop_reason": "not_started",
        "execution_complete": False,
        "template_graph": args.template_graph,
        "template_graph_sha256": sha256_file(args.template_graph),
        "classifier_path": args.classifier_path,
        "identity_model_path": args.identity_model_path,
        "model_path": args.model_path,
        "generation": {
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "blending_start_percentage": args.blending_start_percentage,
            "generation_mask_dilation": args.generation_mask_dilation,
            "generation_mask_feather": args.generation_mask_feather,
        },
        "post_attack": "disabled",
    }
    (output_dir / "intervention_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    rows: list[InterventionObservation] = []
    failures = []
    executed_region_sets = []
    cardinality_results = []
    stopped = False
    expected_rows_per_set = len(args.sample_ids) * len(args.seeds)
    cardinalities = sorted({len(regions) for regions in region_sets})
    for cardinality in cardinalities:
        level_sets = tuple(
            regions for regions in region_sets if len(regions) == cardinality
        )
        for regions in level_sets:
            executed_region_sets.append(regions)
            slug = "__".join(regions)
            for sample_id in args.sample_ids:
                source = Path(args.image_root) / f"{sample_id}.jpg"
                all_components = {
                    region: celebamask_component_path(
                        args.mask_root, sample_id, region
                    )
                    for region in sorted(set(args.candidate_regions))
                }
                policy_dir = (
                    output_dir / "policies" / f"{sample_id:05d}" / slug
                )
                graph_path, binding_path, _ = materialize_region_policy(
                    args.template_graph,
                    source,
                    all_components,
                    regions,
                    policy_dir,
                )
                for seed in args.seeds:
                    run_dir = (
                        output_dir
                        / "runs"
                        / f"{sample_id:05d}"
                        / f"seed_{seed}"
                        / slug
                    )
                    try:
                        row = load_completed_observation(
                            run_dir,
                            target=target,
                            label_index=label_index,
                            desired_value=desired_value,
                            sample_id=sample_id,
                            seed=seed,
                            regions=regions,
                        )
                    except (
                        FileNotFoundError,
                        ValueError,
                        KeyError,
                        json.JSONDecodeError,
                    ):
                        if args.dry_run:
                            continue
                        command = build_intervention_command(
                            args,
                            graph_path=graph_path,
                            binding_path=binding_path,
                            output_dir=run_dir,
                            seed=seed,
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
                                "seed": seed,
                                "regions": list(regions),
                                "returncode": completed.returncode,
                            }
                            failures.append(failure)
                            _write_jsonl(output_dir / "failures.jsonl", failure)
                            _write_observations(
                                output_dir / "intervention_results.csv", rows
                            )
                            if args.continue_on_error:
                                continue
                            raise RuntimeError(
                                f"Intervention subprocess failed: {failure}"
                            )
                        row = load_completed_observation(
                            run_dir,
                            target=target,
                            label_index=label_index,
                            desired_value=desired_value,
                            sample_id=sample_id,
                            seed=seed,
                            regions=regions,
                        )
                    rows.append(row)
                    _write_observations(
                        output_dir / "intervention_results.csv", rows
                    )
        summary = summarize_cardinality(
            rows,
            level_sets,
            expected_rows_per_set=expected_rows_per_set,
            stop_flip_rate=args.stop_flip_rate,
        )
        cardinality_results.append(summary)
        manifest.update(
            {
                "completed_rows": len(rows),
                "failures": failures,
                "executed_region_sets": [
                    list(regions) for regions in executed_region_sets
                ],
                "cardinality_results": cardinality_results,
                "expected_rows": (
                    len(executed_region_sets) * expected_rows_per_set
                ),
            }
        )
        (output_dir / "intervention_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        if not args.dry_run and summary["threshold_reached"]:
            manifest["stop_reason"] = "raw_flip_threshold_reached"
            stopped = True
            break
    manifest["completed_rows"] = len(rows)
    manifest["failures"] = failures
    manifest["stop_reason"] = (
        manifest["stop_reason"]
        if stopped
        else "dry_run"
        if args.dry_run
        else "maximum_cardinality_reached"
    )
    manifest["execution_complete"] = (
        not args.dry_run
        and not failures
        and all(item["complete"] for item in cardinality_results)
    )
    (output_dir / "intervention_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


def validate_args(args: argparse.Namespace) -> None:
    if not args.sample_ids or len(args.sample_ids) != len(set(args.sample_ids)):
        raise ValueError("sample_ids must be non-empty and unique")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if any(sample_id < 0 for sample_id in args.sample_ids):
        raise ValueError("sample_ids must be non-negative")
    canonical_region_sets(
        args.candidate_regions,
        max_set_size=args.max_set_size,
    )
    for name in (
        "template_graph",
        "classifier_path",
        "identity_model_path",
    ):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if not 0.5 < args.stop_flip_rate <= 1.0:
        raise ValueError("stop_flip_rate must be in (0.5, 1]")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_graph", required=True)
    parser.add_argument("--sample_ids", nargs="+", type=int, required=True)
    parser.add_argument("--candidate_regions", nargs="+", required=True)
    parser.add_argument("--max_set_size", type=int, default=2)
    parser.add_argument("--stop_flip_rate", type=float, default=0.96)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
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
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument(
        "--blending_start_percentage", type=float, default=0.25
    )
    parser.add_argument("--generation_mask_dilation", type=int, default=0)
    parser.add_argument("--generation_mask_feather", type=float, default=3.0)
    parser.add_argument(
        "--python_executable", default=".venv-ml/bin/python"
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def _prompt_for_graph(path: str | Path) -> str:
    from cci_diff.prompts import build_concept_prompt
    from cci_diff.spec import ConceptIntervention

    graph = load_concept_graph(path)
    nodes = {node.id: node for node in graph.nodes}
    semantic_evaluators = {"celeba_attribute", "facenet_identity"}
    preserved = tuple(
        edge.target
        for edge in graph.edges
        if edge.relation == "must_preserve"
        and nodes[edge.target].evaluator in semantic_evaluators
    )
    return build_concept_prompt(
        ConceptIntervention(
            graph.intervention.concept,
            graph.intervention.desired_value,
            preserved,
            tuple(node.id for node in graph.nodes),
        )
    ).positive


def _artifact_path(
    value: str | Path | None,
    run_dir: Path,
    label: str,
) -> Path:
    if value is None:
        raise FileNotFoundError(f"{label} path is absent from audit")
    path = Path(value)
    alternatives = (path, run_dir / path, REPO_ROOT / path)
    for candidate in alternatives:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{label} not found: {path}")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _load_binary_component(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Component mask not found: {path}")
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    if array.ndim != 2:
        raise ValueError(f"Component mask must be two-dimensional: {path}")
    return array > 0


def _write_observations(
    path: Path,
    observations: list[InterventionObservation],
) -> None:
    fields = [
        "target",
        "desired_value",
        "sample_id",
        "seed",
        "regions",
        "source_probability",
        "output_probability",
        "target_pass",
        "target_effect",
        "mask_fraction",
        "identity_cosine",
        "non_target_drift",
        "outside_l1",
        "changed_fraction",
        "output_path",
        "audit_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in observations:
            payload = {
                field: getattr(row, field)
                for field in fields
                if field not in {"target_pass", "target_effect"}
            }
            payload["regions"] = json.dumps(list(row.regions))
            payload["target_pass"] = row.target_pass
            payload["target_effect"] = row.target_effect
            writer.writerow(payload)


def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, allow_nan=False) + "\n")


def main() -> int:
    run_interventions(build_arg_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
