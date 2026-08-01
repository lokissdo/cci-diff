#!/usr/bin/env python3
"""Run paired same-seed semantic-region interventions for graph discovery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
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
from cci_diff.intervention_cache import (  # noqa: E402
    InterventionCacheKey,
    cache_key_for,
    load_cached_intervention,
    store_cached_intervention,
)
from cci_diff.region_screening import (  # noqa: E402
    build_union_mask,
    canonical_region_sets,
    celebamask_component_path,
)
from cci_diff.runtime_environment import resolve_device  # noqa: E402
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
    """Build a legacy clean-CCI command or an exact frozen A11 command."""

    policy_path = getattr(args, "generation_policy", None)
    policy = _load_a11_generation_policy(policy_path) if policy_path else None
    model_path = str(policy.get("checkpoint", args.model_path)) if policy else args.model_path
    inference_steps = (
        policy["num_inference_steps"] if policy else args.num_inference_steps
    )
    guidance_scale = policy["guidance_scale"] if policy else args.guidance_scale
    blending_start = (
        policy["blending_start_percentage"]
        if policy
        else args.blending_start_percentage
    )
    mask_dilation = (
        policy.get("mask_dilation", policy.get("generation_mask_dilation", 0))
        if policy
        else args.generation_mask_dilation
    )
    mask_feather = (
        policy.get("mask_feather", policy.get("generation_mask_feather", 3.0))
        if policy
        else args.generation_mask_feather
    )
    torch_dtype = (
        policy["torch_dtype"]
        if policy
        else "float16"
        if getattr(args, "torch_dtype", "auto") == "auto"
        and args.device.startswith("cuda")
        else "float32"
        if getattr(args, "torch_dtype", "auto") == "auto"
        else args.torch_dtype
    )
    controller_mode = policy["controller_mode"] if policy else "feedback"
    hook = policy["hook"] if policy else "clean_constraint"
    effective_prompt = str(policy.get("prompt", prompt)) if policy else prompt

    command = [
        args.python_executable,
        "scripts/run_sd2_bld_cci.py",
        "--output_dir",
        str(output_dir),
        "--prompt",
        effective_prompt,
        "--model_path",
        model_path,
        "--local_files_only",
        "--device",
        args.device,
        "--torch_dtype",
        str(torch_dtype),
        "--batch_size",
        "1",
        "--num_inference_steps",
        str(inference_steps),
        "--guidance_scale",
        str(guidance_scale),
        "--blending_start_percentage",
        str(blending_start),
        "--seed",
        str(seed),
        "--generation_mask_dilation",
        str(mask_dilation),
        "--generation_mask_feather",
        str(mask_feather),
        "--cci_hook",
        hook,
        "--cci_graph",
        str(graph_path),
        "--cci_sample_bindings",
        str(binding_path),
        "--classifier_path",
        args.classifier_path,
        "--identity_model_path",
        args.identity_model_path,
        "--cci_controller_mode",
        controller_mode,
    ]
    if getattr(args, "allow_model_download", False):
        command.remove("--local_files_only")
    if policy:
        if not policy["projection"]:
            command.append("--cci_disable_target_projection")
        post_attack = policy["post_attack"]
        if post_attack["mode"] != "none":
            schedule = post_attack["epsilon_schedule"]
            if isinstance(schedule, str):
                schedule_value = schedule
            else:
                schedule_value = ",".join(str(value) for value in schedule)
            command.extend(
                [
                    "--cci_post_attack",
                    str(post_attack["mode"]),
                    "--cci_post_attack_epsilon_schedule",
                    schedule_value,
                    "--cci_post_attack_boundary_margin",
                    str(post_attack["boundary_margin"]),
                ]
            )
    return command


def _load_a11_generation_policy(path: str | Path) -> dict[str, Any]:
    policy_path = Path(path)
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("generation_policy must contain a JSON object")
    expected = {
        "variant": "A11",
        "hook": "clean_constraint",
        "controller_mode": "trust_region",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"A11 generation policy requires {field}={value!r}"
            )
    if payload.get("projection") is not True:
        raise ValueError("A11 generation policy requires target projection")
    required = (
        "num_inference_steps",
        "guidance_scale",
        "blending_start_percentage",
        "torch_dtype",
        "post_attack",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"A11 generation policy is missing fields: {missing}")
    post_attack = payload["post_attack"]
    if not isinstance(post_attack, dict):
        raise ValueError("post_attack must be a JSON object")
    for field in ("mode", "epsilon_schedule", "boundary_margin"):
        if field not in post_attack:
            raise ValueError(f"post_attack is missing {field}")
    return payload


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

    import torch

    args.device = resolve_device(getattr(args, "device", "auto"), torch)
    validate_args(args)
    template = load_concept_graph(args.template_graph)
    target = template.intervention.concept
    desired_value = template.intervention.desired_value
    label_index = resolve_celeba_attribute_index(target)
    requested_region_sets = _requested_region_sets(args)
    candidate_regions = sorted(
        {region for regions in requested_region_sets for region in regions}
    )
    args.candidate_regions = candidate_regions
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
    cache_root_value = getattr(args, "intervention_cache_dir", None)
    cache_root = Path(cache_root_value) if cache_root_value else None
    cache_static = (
        _cache_static_digests(args) if cache_root is not None else None
    )
    policy_path = getattr(args, "generation_policy", None)
    generation_policy = (
        _load_a11_generation_policy(policy_path) if policy_path else None
    )
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
        "disable_early_stop": getattr(args, "disable_early_stop", False),
        "planned_region_sets": [list(regions) for regions in region_sets],
        "executed_region_sets": [],
        "cardinality_results": [],
        "stop_reason": "not_started",
        "execution_complete": False,
        "template_graph": args.template_graph,
        "template_graph_sha256": sha256_file(args.template_graph),
        "classifier_path": args.classifier_path,
        "identity_model_path": args.identity_model_path,
        "model_path": (
            str(generation_policy.get("checkpoint", args.model_path))
            if generation_policy
            else args.model_path
        ),
        "device": args.device,
        "generation": {
            "variant": "A11" if generation_policy else "legacy",
            "controller_mode": (
                generation_policy["controller_mode"]
                if generation_policy
                else "feedback"
            ),
            "num_inference_steps": (
                generation_policy["num_inference_steps"]
                if generation_policy
                else args.num_inference_steps
            ),
            "guidance_scale": (
                generation_policy["guidance_scale"]
                if generation_policy
                else args.guidance_scale
            ),
            "blending_start_percentage": (
                generation_policy["blending_start_percentage"]
                if generation_policy
                else args.blending_start_percentage
            ),
            "generation_mask_dilation": (
                generation_policy.get(
                    "mask_dilation",
                    generation_policy.get("generation_mask_dilation", 0),
                )
                if generation_policy
                else args.generation_mask_dilation
            ),
            "generation_mask_feather": (
                generation_policy.get(
                    "mask_feather",
                    generation_policy.get("generation_mask_feather", 3.0),
                )
                if generation_policy
                else args.generation_mask_feather
            ),
        },
        "post_attack": (
            generation_policy["post_attack"]
            if generation_policy
            else {"mode": "none"}
        ),
        "generation_policy": getattr(args, "generation_policy", None),
        "intervention_cache_dir": (
            str(cache_root) if cache_root is not None else None
        ),
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
                graph_path, binding_path, union_path = materialize_region_policy(
                    args.template_graph,
                    source,
                    all_components,
                    regions,
                    policy_dir,
                )
                for seed in args.seeds:
                    cache_key = (
                        _intervention_cache_key(
                            cache_static,
                            source=source,
                            union_mask=union_path,
                            graph=graph_path,
                            sample_id=sample_id,
                            seed=seed,
                        )
                        if cache_static is not None
                        else None
                    )
                    run_dir = (
                        output_dir
                        / "runs"
                        / f"{sample_id:05d}"
                        / f"seed_{seed}"
                        / slug
                    )
                    if cache_key is not None:
                        run_dir = run_dir / cache_key.digest
                        cached = load_cached_intervention(cache_root, cache_key)
                    else:
                        cached = None
                    if cached is not None:
                        rows.append(_observation_from_cache(cached))
                        _write_observations(
                            output_dir / "intervention_results.csv", rows
                        )
                        continue
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
                    if cache_key is not None:
                        cached = store_cached_intervention(
                            cache_root,
                            cache_key,
                            row,
                            _cache_artifacts(row, run_dir),
                        )
                        row = _observation_from_cache(cached)
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
        if (
            not args.dry_run
            and not getattr(args, "disable_early_stop", False)
            and summary["threshold_reached"]
        ):
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
    _requested_region_sets(args)
    for name in (
        "template_graph",
        "classifier_path",
        "identity_model_path",
    ):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    model_path = Path(args.model_path)
    if not args.allow_model_download and not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if not 0.5 < args.stop_flip_rate <= 1.0:
        raise ValueError("stop_flip_rate must be in (0.5, 1]")
    if getattr(args, "intervention_cache_dir", None) and not getattr(
        args, "generation_policy", None
    ):
        raise ValueError(
            "intervention_cache_dir requires a frozen generation_policy"
        )
    if getattr(args, "generation_policy", None):
        _load_a11_generation_policy(args.generation_policy)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_graph", required=True)
    parser.add_argument("--sample_ids", nargs="+", type=int, required=True)
    parser.add_argument("--candidate_regions", nargs="+", default=None)
    parser.add_argument(
        "--region_set",
        dest="region_sets",
        action="append",
        default=None,
        metavar="REGION+REGION",
        help="Explicit beam candidate; repeat to evaluate multiple sets.",
    )
    parser.add_argument("--max_set_size", type=int, default=2)
    parser.add_argument("--stop_flip_rate", type=float, default=0.96)
    parser.add_argument(
        "--disable_early_stop",
        action="store_true",
        help="Evaluate every region-set cardinality even after the flip threshold.",
    )
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
    parser.add_argument(
        "--allow_model_download",
        action="store_true",
        help="Allow Diffusers to resolve model_path from Hugging Face.",
    )
    parser.add_argument("--classifier_path", required=True)
    parser.add_argument("--identity_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--generation_policy",
        default=None,
        help="Frozen A11 generation-policy JSON for development runs.",
    )
    parser.add_argument(
        "--intervention_cache_dir",
        default=None,
        help="Shared content-addressed A11 cache (requires generation_policy).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
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


def _requested_region_sets(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], ...]:
    explicit = getattr(args, "region_sets", None)
    candidate_regions = getattr(args, "candidate_regions", None)
    if explicit and candidate_regions:
        raise ValueError("Use either region_set or candidate_regions, not both")
    if explicit:
        region_sets = tuple(
            tuple(
                sorted(
                    {
                        region.strip()
                        for region in str(value).split("+")
                        if region.strip()
                    }
                )
            )
            for value in explicit
        )
        if any(not regions for regions in region_sets):
            raise ValueError("region_set must contain a semantic component")
        if len(region_sets) != len(set(region_sets)):
            raise ValueError("region_set values must be unique")
        if any(len(regions) > args.max_set_size for regions in region_sets):
            raise ValueError("region_set exceeds max_set_size")
        return tuple(
            sorted(region_sets, key=lambda regions: (len(regions), regions))
        )
    if not candidate_regions:
        raise ValueError("candidate_regions or region_set is required")
    return canonical_region_sets(
        candidate_regions,
        max_set_size=args.max_set_size,
    )


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


def _cache_static_digests(args: argparse.Namespace) -> dict[str, str]:
    policy_path = Path(args.generation_policy)
    policy = _load_a11_generation_policy(policy_path)
    checkpoint = Path(str(policy.get("checkpoint", args.model_path)))
    return {
        "checkpoint_sha256": _checkpoint_content_sha256(checkpoint, policy),
        "classifier_sha256": sha256_file(args.classifier_path),
        "identity_sha256": sha256_file(args.identity_model_path),
        "policy_sha256": sha256_file(policy_path),
    }


def _checkpoint_content_sha256(
    checkpoint: Path,
    policy: Mapping[str, Any],
) -> str:
    if checkpoint.is_file():
        return sha256_file(checkpoint)
    if checkpoint.is_dir():
        inventory = {
            str(path.relative_to(checkpoint)): sha256_file(path)
            for path in sorted(checkpoint.rglob("*"))
            if path.is_file()
            and not any(
                part.startswith(".")
                for part in path.relative_to(checkpoint).parts
            )
            and path.name != "README.md"
        }
    else:
        inventory = policy.get("checkpoint_files")
        if not isinstance(inventory, dict) or not inventory:
            raise FileNotFoundError(
                "Checkpoint is unavailable and generation_policy has no "
                "checkpoint_files inventory"
            )
    return hashlib.sha256(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _intervention_cache_key(
    static: Mapping[str, str],
    *,
    source: Path,
    union_mask: Path,
    graph: Path,
    sample_id: int,
    seed: int,
) -> InterventionCacheKey:
    return cache_key_for(
        source_sha256=sha256_file(source),
        mask_sha256=sha256_file(union_mask),
        checkpoint_sha256=static["checkpoint_sha256"],
        classifier_sha256=static["classifier_sha256"],
        identity_sha256=static["identity_sha256"],
        graph_sha256=sha256_file(graph),
        policy_sha256=static["policy_sha256"],
        sample_id=sample_id,
        seed=seed,
    )


def _cache_artifacts(
    observation: InterventionObservation,
    run_dir: Path,
) -> dict[str, Path]:
    artifacts = {
        "output": Path(str(observation.output_path)),
        "audit": Path(str(observation.audit_path)),
    }
    for name, filename in (
        ("semantic_mask", "semantic_mask.png"),
        ("generation_mask", "generation_mask.png"),
    ):
        candidate = run_dir / filename
        if candidate.is_file():
            artifacts[name] = candidate
    return artifacts


def _observation_from_cache(cached: Any) -> InterventionObservation:
    return replace(
        cached.observation,
        output_path=str(cached.artifacts["output"]),
        audit_path=str(cached.artifacts["audit"]),
    )


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
