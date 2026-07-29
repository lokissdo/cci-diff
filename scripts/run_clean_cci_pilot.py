#!/usr/bin/env python3
"""Run the paired clean-CCI A0-A4 pilot sequentially and summarize it."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cci_diff.comparison_artifacts import (
    create_paginated_pair_sheets,
    materialize_selected_artifacts,
)
from cci_diff.spatial_selection import (
    measure_spatial_change,
    select_spatial_candidate,
)


# Direct script execution places ``scripts/`` on sys.path, but the pilot imports
# one helper through the repository-level ``scripts`` package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VARIANTS = {
    "A0": {
        "hook": "clean_constraint",
        "controller_mode": "disabled",
        "projection": True,
    },
    "A1": {"hook": "latent_classifier", "robust": True},
    "A2": {
        "hook": "clean_constraint",
        "controller_mode": "fixed_equal",
        "projection": True,
    },
    "A3": {
        "hook": "clean_constraint",
        "controller_mode": "feedback",
        "projection": True,
    },
    "A4": {
        "hook": "clean_constraint",
        "controller_mode": "feedback",
        "projection": False,
    },
    "A5": {
        "hook": "clean_constraint",
        "controller_mode": "feedback",
        "projection": True,
        "flags": ("--cci_disable_target_guidance",),
    },
    "A6": {
        "hook": "clean_constraint",
        "controller_mode": "feedback",
        "projection": True,
        "flags": ("--cci_disable_gradient_normalization",),
    },
    "A7": {
        "hook": "clean_constraint",
        "controller_mode": "feedback",
        "projection": True,
        "flags": ("--cci_disable_target_budget",),
    },
    "A8": {
        "hook": "clean_constraint",
        "controller_mode": "feedback",
        "projection": True,
        "flags": ("--cci_disable_guidance_schedule",),
    },
    "A9": {
        "hook": "clean_constraint",
        "controller_mode": "feedback",
        "projection": True,
        "flags": ("--cci_disable_final_correction",),
    },
    "A10": {
        "hook": "clean_constraint",
        "controller_mode": "fixed_trust_matched",
        "projection": True,
    },
    "A11": {
        "hook": "clean_constraint",
        "controller_mode": "trust_region",
        "projection": True,
    },
}

CONTROLLER_VARIANTS = {
    "disabled": "A0",
    "fixed_equal": "A2",
    "feedback": "A3",
    "fixed_trust_matched": "A10",
    "trust_region": "A11",
}


FEATURES = {
    "smile": {
        "graph": "examples/graphs/remove_smile_clean_cci.json",
        "legacy_config": "examples/remove_smile_intervention.json",
        "components": ("mouth", "u_lip", "l_lip"),
        "binding_roles": {
            "mouth": "mouth",
            "upper_lip": "u_lip",
            "lower_lip": "l_lip",
        },
        "audit_component": "mouth",
        "label_index": 31,
        "desired_value": 0,
        "source_threshold": 0.5,
        "minimum_a3_flips": 8,
    },
    "hair": {
        "graph": "examples/graphs/blond_hair_clean_cci.json",
        "legacy_config": "examples/hair_intervention.json",
        "components": ("hair",),
        "binding_roles": {"hair": "hair"},
        "audit_component": "hair",
        "label_index": 9,
        "desired_value": 1,
        "source_threshold": 0.5,
        "minimum_a3_flips": 11,
    },
}


@dataclass(frozen=True)
class MaskCandidate:
    label: str
    dilation: int
    dilation_x: int | None = None
    dilation_y: int | None = None
    feather_radius: float | None = None


def parse_mask_shape(value: str) -> MaskCandidate:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("mask shape must be x,y,feather")
    try:
        dilation_x = int(parts[0])
        dilation_y = int(parts[1])
        feather = float(parts[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "mask shape x and y must be integers and feather must be numeric"
        ) from error
    if dilation_x < 0 or dilation_y < 0 or feather < 0:
        raise argparse.ArgumentTypeError("mask shape values must be non-negative")
    label = f"x{dilation_x}_y{dilation_y}_f{feather:g}"
    return MaskCandidate(label, 0, dilation_x, dilation_y, feather)


def resolve_requested_variants(args: argparse.Namespace) -> list[str]:
    """Resolve readable controller modes to the existing variant definitions."""

    modes = getattr(args, "controller_modes", None)
    if modes:
        return [CONTROLLER_VARIANTS[mode] for mode in modes]
    return list(args.variants)


def load_excluded_ids(path: str | Path | None) -> dict[str, set[int]]:
    """Load task-specific discovery IDs that evaluation must not reuse."""

    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for feature in FEATURES:
        values = payload.get(feature)
        if values is None:
            values = (
                (payload.get("features") or {}).get(feature, {}).get(
                    "selected_ids"
                )
            )
        if values is not None:
            result[feature] = {int(value) for value in values}
    return result


def resolve_mask_candidates(args: argparse.Namespace) -> list[MaskCandidate]:
    if args.mask_shapes:
        return list(args.mask_shapes)
    return [MaskCandidate(f"d{radius}", radius) for radius in args.mask_dilations]


def annotation_paths(
    mask_root: Path,
    image_id: int,
    components: tuple[str, ...],
) -> dict[str, Path]:
    stem = f"{image_id:05d}"
    directory = mask_root / str(image_id // 2000)
    return {
        component: directory / f"{stem}_{component}.png"
        for component in components
    }


def discover_samples(
    image_root,
    mask_root,
    components,
    *,
    limit,
    max_image_id=30000,
):
    selected = []
    for image_id in range(max_image_id):
        source = Path(image_root) / f"{image_id}.jpg"
        masks = annotation_paths(Path(mask_root), image_id, tuple(components))
        if source.is_file() and all(path.is_file() for path in masks.values()):
            selected.append((image_id, source, masks))
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise ValueError(
            f"Found only {len(selected)} complete samples; required {limit}"
        )
    return selected


def _mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def _median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.median(finite) if finite else None


def _variant_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    flips = sum(bool(row.get("target_pass")) for row in rows)
    feasible = sum(bool(row.get("feasible")) for row in rows)
    return {
        "count": count,
        "flips": flips,
        "flip_rate": flips / count if count else 0.0,
        "feasibility_rate": feasible / count if count else 0.0,
        "mean_identity_cosine": _mean(
            [row.get("identity_cosine") for row in rows]
        ),
        "mean_outside_mae": _mean(
            [row.get("semantic_outside_mae") for row in rows]
        ),
        "mean_tv": _mean([row.get("residual_tv") for row in rows]),
        "median_runtime": _median(
            [row.get("runtime_seconds") for row in rows]
        ),
    }


def _better_or_lower(a: float | None, b: float | None, *, higher: bool) -> bool:
    if a is None or b is None:
        return False
    return a > b if higher else a < b


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    features = {}
    support_values = []
    for feature in sorted({row["feature"] for row in rows}):
        feature_rows = [row for row in rows if row["feature"] == feature]
        variants = {
            variant: _variant_summary(
                [row for row in feature_rows if row["variant"] == variant]
            )
            for variant in sorted({row["variant"] for row in feature_rows})
        }
        a2 = variants.get("A2")
        a3 = variants.get("A3")
        threshold = FEATURES.get(feature, {}).get("minimum_a3_flips")
        required_flips = (
            min(threshold, a3["count"])
            if a3 and threshold is not None
            else 1
        )
        adaptive_supported = False
        if a2 and a3 and a2["count"] and a3["count"]:
            target_gate = a3["flips"] >= required_flips
            adaptive_supported = target_gate and (
                a3["flip_rate"] > a2["flip_rate"]
                or (
                    a3["flip_rate"] == a2["flip_rate"]
                    and (
                        _better_or_lower(
                            a3["mean_identity_cosine"],
                            a2["mean_identity_cosine"],
                            higher=True,
                        )
                        or _better_or_lower(
                            a3["mean_outside_mae"],
                            a2["mean_outside_mae"],
                            higher=False,
                        )
                    )
                )
            )
            support_values.append(adaptive_supported)
        a0 = variants.get("A0")
        thresholds = {
            "a3_minimum_flips": (
                None if not a3 else a3["flips"] >= required_flips
            ),
            "a3_required_flips": required_flips if a3 else None,
            "outside_mae_within_10_percent_of_a0": (
                None
                if not a0
                or not a3
                or a0["mean_outside_mae"] is None
                or a3["mean_outside_mae"] is None
                else a3["mean_outside_mae"] <= 1.1 * a0["mean_outside_mae"]
            ),
            "identity_at_least_0_90": (
                None
                if not a3 or a3["mean_identity_cosine"] is None
                else a3["mean_identity_cosine"] >= 0.90
            ),
            "identity_no_more_than_0_02_below_a0": (
                None
                if not a0
                or not a3
                or a0["mean_identity_cosine"] is None
                or a3["mean_identity_cosine"] is None
                else a3["mean_identity_cosine"]
                >= a0["mean_identity_cosine"] - 0.02
            ),
            "runtime_no_more_than_3x_a0": (
                None
                if not a0
                or not a3
                or a0["median_runtime"] is None
                or a3["median_runtime"] is None
                else a3["median_runtime"] <= 3.0 * a0["median_runtime"]
            ),
            "artifact_review": "pending",
        }
        deltas = {}
        if a0:
            for variant, summary in variants.items():
                if variant == "A0":
                    continue
                deltas[variant] = {
                    "flip_rate": summary["flip_rate"] - a0["flip_rate"],
                    "identity_cosine": _difference(
                        summary["mean_identity_cosine"],
                        a0["mean_identity_cosine"],
                    ),
                    "outside_mae": _difference(
                        summary["mean_outside_mae"],
                        a0["mean_outside_mae"],
                    ),
                }
        conclusion = (
            "adaptive CCI supported"
            if adaptive_supported
            else "adaptive CCI not supported"
            if a2 and a3
            else "pilot incomplete"
        )
        features[feature] = {
            "variants": variants,
            "mean_delta_from_a0": deltas,
            "adaptive_supported": adaptive_supported,
            "thresholds": thresholds,
            "conclusion": conclusion,
        }
    return {
        "features": features,
        "adaptive_supported": bool(support_values) and all(support_values),
    }


def _difference(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def write_binding(
    path: Path,
    source: Path,
    masks: dict[str, Path],
    binding_roles: dict[str, str],
) -> None:
    payload = {
        "source_image": str(source),
        "masks": {
            role: str(masks[component])
            for role, component in binding_roles.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prompt_for_graph(path: str) -> str:
    from cci_diff.concept_graph import load_concept_graph
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


def build_variant_command(
    args: argparse.Namespace,
    *,
    feature: str,
    variant: str,
    sample_id: int,
    source: Path,
    masks: dict[str, Path],
    binding_path: Path,
    output_path: Path,
    dilation: int = 0,
    mask_candidate: MaskCandidate | None = None,
) -> list[str]:
    config = FEATURES[feature]
    candidate = mask_candidate or MaskCandidate(f"d{dilation}", dilation)
    common = [
        getattr(args, "python_executable", ".venv-ml/bin/python"),
        "scripts/run_sd2_bld_cci.py",
        "--output_dir",
        str(output_path),
        "--prompt",
        _prompt_for_graph(config["graph"]),
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
        "5.0",
        "--blending_start_percentage",
        "0.25",
        "--seed",
        str(args.seed),
        "--generation_mask_dilation",
        str(candidate.dilation),
    ]
    if getattr(args, "allow_model_download", False):
        common.remove("--local_files_only")
    if candidate.dilation_x is not None:
        common.extend(
            ["--generation_mask_dilation_x", str(candidate.dilation_x)]
        )
    if candidate.dilation_y is not None:
        common.extend(
            ["--generation_mask_dilation_y", str(candidate.dilation_y)]
        )
    if candidate.feather_radius is not None:
        common.extend(
            ["--generation_mask_feather", str(candidate.feather_radius)]
        )
    definition = VARIANTS[variant]
    if definition["hook"] == "clean_constraint":
        common.extend(
            [
                "--cci_hook",
                "clean_constraint",
                "--cci_graph",
                config["graph"],
                "--cci_sample_bindings",
                str(binding_path),
                "--classifier_path",
                args.classifier_path,
                "--identity_model_path",
                args.identity_model_path,
                "--cci_controller_mode",
                definition["controller_mode"],
            ]
        )
        if not definition["projection"]:
            common.append("--cci_disable_target_projection")
        common.extend(definition.get("flags", ()))
        if args.cci_post_attack != "none":
            common.extend(
                [
                    "--cci_post_attack",
                    args.cci_post_attack,
                    "--cci_post_attack_epsilon_schedule",
                    args.cci_post_attack_epsilon_schedule,
                    "--cci_post_attack_boundary_margin",
                    str(args.cci_post_attack_boundary_margin),
                ]
            )
        return common

    common.extend(
        [
            "--cci_config",
            config["legacy_config"],
            "--init_image",
            str(source),
            "--mask",
            str(masks[config["audit_component"]]),
            "--cci_hook",
            "latent_classifier",
            "--classifier_path",
            args.classifier_path,
            "--classifier_label_index",
            str(config["label_index"]),
            "--robust_classifier_guidance",
            "--cci_step_size",
            "0.08",
            "--cci_every_n_steps",
            "4",
            "--cci_start_step",
            "4",
            "--cci_end_step",
            "16",
            "--cci_normalize_grad",
        ]
    )
    if candidate.feather_radius is None:
        common.extend(["--generation_mask_feather", "3"])
    for component in config["components"]:
        common.extend(["--generation_mask_component", str(masks[component])])
    return common


def _probability_margin(probability: float, desired_value: int) -> float:
    clipped = min(max(probability, 1e-7), 1.0 - 1e-7)
    logit = math.log(clipped / (1.0 - clipped))
    return (2 * desired_value - 1) * logit - math.log(4.0)


def extract_audit_row(
    audit: dict[str, Any],
    *,
    feature: str,
    sample_id: int,
    variant: str,
    output_path: Path,
) -> dict[str, Any]:
    cci = audit.get("cci", {})
    metrics = cci.get("metrics") or {}
    final = metrics.get("final_feasibility") or {}
    attributes = metrics.get("attributes") or {}
    target_index = FEATURES[feature]["label_index"]
    desired_value = FEATURES[feature]["desired_value"]
    source_values = attributes.get("source_probabilities") or []
    output_values = attributes.get("output_probabilities") or []
    if source_values and output_values:
        source_probability = source_values[target_index]
        output_probability = output_values[target_index]
    else:
        classifier = cci.get("classifier") or {}
        source_probability = classifier.get("source_probability")
        outputs = classifier.get("output_probabilities") or []
        output_probability = outputs[0] if outputs else None
    post_attack = cci.get("post_attack") or {}
    post_candidates = post_attack.get("candidates") or []
    post_candidate = post_candidates[0] if post_candidates else None
    if post_candidate is not None:
        output_probability = post_candidate.get(
            "after_probability",
            output_probability,
        )
    desired_probability = (
        None
        if output_probability is None
        else output_probability
        if desired_value == 1
        else 1.0 - output_probability
    )
    signed_margin = None
    if post_candidate is not None and output_probability is not None:
        signed_margin = (
            output_probability - 0.5
            if desired_value == 1
            else 0.5 - output_probability
        )
    elif (final.get("target") or {}).get("signed_margin") is not None:
        signed_margin = (final.get("target") or {})["signed_margin"]
    elif output_probability is not None:
        signed_margin = _probability_margin(output_probability, desired_value)
    target_pass = (
        bool(post_candidate["target_pass"])
        if post_candidate is not None
        else signed_margin is not None and signed_margin >= 0
    )
    locality = metrics.get("locality") or {}
    strict = locality.get("strict_audit_mask") or {}
    semantic = locality.get("semantic_union") or {}
    if not locality:
        changes = ((cci.get("robust") or {}).get("change_metrics") or {})
        strict = changes.get("strict_mouth") or {}
        semantic = changes.get("semantic_union") or {}
    constraints = final.get("constraints") or {}
    identity = constraints.get("identity") or {}
    outside = constraints.get("outside_locality") or {}
    tv_constraint = constraints.get("residual_tv") or {}
    quality = metrics.get("quality") or {}
    return {
        "feature": feature,
        "sample_id": sample_id,
        "variant": variant,
        "source_probability": source_probability,
        "desired_probability": desired_probability,
        "signed_margin": signed_margin,
        "target_pass": target_pass,
        "feasible": (
            target_pass
            if post_candidate is not None
            else bool(final.get("feasible", target_pass))
        ),
        "identity_cosine": (
            post_candidate.get("identity_after")
            if (
                post_candidate is not None
                and post_candidate.get("identity_after") is not None
            )
            else metrics.get("identity_cosine")
        ),
        "strict_inside_mae": strict.get("inside_mae"),
        "strict_outside_mae": strict.get("outside_mae"),
        "semantic_inside_mae": semantic.get("inside_mae"),
        "semantic_outside_mae": semantic.get("outside_mae"),
        "non_target_drift": attributes.get("mean_non_target_drift"),
        "residual_tv": quality.get("residual_tv"),
        "boundary_discontinuity": quality.get("boundary_discontinuity"),
        "independent_semantic_agreement": (
            metrics.get("independent_semantic_agreement") or {}
        ).get("value"),
        "outside_perceptual_distance": (
            metrics.get("outside_perceptual_distance") or {}
        ).get("value"),
        "identity_pass": identity.get("passed"),
        "locality_pass": outside.get("passed"),
        "tv_pass": tv_constraint.get("passed"),
        "runtime_seconds": cci.get("wall_seconds"),
        "peak_mps_bytes": cci.get("peak_mps_bytes"),
        "graph_sha256": cci.get("graph_sha256"),
        "trace_path": cci.get("trace_path"),
        "output_path": str(output_path),
        "failed_constraints": final.get("failed_constraints", []),
        "post_attack_selected_epsilon": (
            post_candidate.get("selected_epsilon")
            if post_candidate is not None
            else None
        ),
        "post_attack_escalated": (
            post_candidate.get("escalated")
            if post_candidate is not None
            else None
        ),
        "post_attack_mean_abs_change": (
            post_candidate.get("mean_abs_change")
            if post_candidate is not None
            else None
        ),
        "post_attack_linf": (
            post_candidate.get("linf")
            if post_candidate is not None
            else None
        ),
        "post_attack_changed_fraction": (
            post_candidate.get("changed_fraction")
            if post_candidate is not None
            else None
        ),
    }


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def number(value, default):
        return default if value is None else float(value)

    return sorted(
        rows,
        key=lambda row: (
            not bool(row.get("feasible")),
            -number(row.get("signed_margin"), -math.inf),
            -number(row.get("identity_cosine"), -math.inf),
            number(row.get("semantic_outside_mae"), math.inf),
        ),
    )


def create_contact_sheets(rows: list[dict[str, Any]], output_dir: Path) -> None:
    from PIL import Image, ImageDraw

    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        images = []
        for row in selected:
            path = Path(row["output_path"])
            if not path.is_file():
                continue
            image = Image.open(path).convert("RGB")
            image.thumbnail((256, 256))
            tile = Image.new("RGB", (256, 286), "white")
            tile.paste(image, ((256 - image.width) // 2, 0))
            ImageDraw.Draw(tile).text(
                (4, 262),
                f"{row['feature']} {row['sample_id']} {variant}",
                fill="black",
            )
            images.append(tile)
        if not images:
            continue
        columns = min(5, len(images))
        rows_count = math.ceil(len(images) / columns)
        sheet = Image.new("RGB", (columns * 256, rows_count * 286), "white")
        for index, image in enumerate(images):
            sheet.paste(image, ((index % columns) * 256, (index // columns) * 286))
        sheet.save(output_dir / f"contact_sheet_{variant}.jpg")


def select_eligible_samples(
    args: argparse.Namespace,
    *,
    feature: str,
    classifier: Any,
    detector: Any,
) -> tuple[list[tuple[int, Path, dict[str, Path]]], list[dict[str, Any]]]:
    from cci_diff.identity.facenet import detect_largest_face_box
    from scripts.run_sd2_bld_cci import load_rgb_image_tensor, score_classifier_image_grid

    config = FEATURES[feature]
    excluded = getattr(args, "excluded_ids_by_feature", {}).get(feature, set())
    explicit_ids = getattr(args, "sample_ids", None)
    if explicit_ids is not None:
        image_ids = sorted(int(image_id) for image_id in explicit_ids)
    else:
        image_ids = list(range(args.max_image_id))
        random_seed = getattr(args, "random_sample_seed", None)
        if random_seed is not None:
            random.Random(random_seed).shuffle(image_ids)
    required_count = len(image_ids) if explicit_ids is not None else args.limit
    selected = []
    decisions = []
    for image_id in image_ids:
        if image_id in excluded:
            continue
        source = Path(args.image_root) / f"{image_id}.jpg"
        masks = annotation_paths(
            Path(args.mask_root),
            image_id,
            config["components"],
        )
        complete = source.is_file() and all(path.is_file() for path in masks.values())
        decision = {"image_id": image_id, "complete_files": complete}
        if not complete:
            decisions.append(decision)
            continue
        probability = score_classifier_image_grid(
            source,
            classifier=classifier,
            label_index=config["label_index"],
            input_size=args.classifier_input_size,
            device=args.device,
            batch_size=1,
        )[0]
        try:
            box = detect_largest_face_box(
                detector,
                load_rgb_image_tensor(source, device="cpu"),
            )
            face_ok = True
        except ValueError:
            box = None
            face_ok = False
        source_ok = (
            probability >= config["source_threshold"]
            if config["desired_value"] == 0
            else probability < config["source_threshold"]
        )
        eligible = source_ok and face_ok
        decision.update(
            {
                "source_probability": probability,
                "source_threshold": config["source_threshold"],
                "face_detected": face_ok,
                "face_box": list(box) if box is not None else None,
                "eligible": eligible,
            }
        )
        decisions.append(decision)
        if eligible:
            selected.append((image_id, source, masks))
        if len(selected) == required_count:
            break
    if len(selected) < required_count:
        raise ValueError(
            f"Found only {len(selected)} eligible {feature} samples; "
            f"required {required_count}"
        )
    return selected, decisions


def validate_pilot_args(args: argparse.Namespace) -> None:
    """Reject ambiguous candidate grids before loading model checkpoints."""

    from cci_diff.post_attack import parse_epsilon_schedule

    if len(args.variants) != len(set(args.variants)):
        raise ValueError("variants must be unique")
    if len(args.mask_dilations) != len(set(args.mask_dilations)):
        raise ValueError("mask_dilations must be unique")
    if any(radius < 0 for radius in args.mask_dilations):
        raise ValueError("mask_dilations must be non-negative")
    candidates = resolve_mask_candidates(args)
    labels = [candidate.label for candidate in candidates]
    if len(labels) != len(set(labels)):
        raise ValueError("mask candidates must be unique")
    if args.limit <= 0:
        raise ValueError("limit must be positive")
    if (
        getattr(args, "sample_ids", None) is not None
        and getattr(args, "random_sample_seed", None) is not None
    ):
        raise ValueError(
            "sample_ids and random_sample_seed are mutually exclusive"
        )
    parse_epsilon_schedule(args.cci_post_attack_epsilon_schedule)
    if not 0 <= args.cci_post_attack_boundary_margin < 0.5:
        raise ValueError(
            "cci_post_attack_boundary_margin must be in [0, 0.5)"
        )
    if args.cci_post_attack != "none" and any(
        VARIANTS[variant]["hook"] != "clean_constraint"
        for variant in args.variants
    ):
        raise ValueError("post-attack requires clean-constraint pilot variants")


def _candidate_row(
    run_dir: Path,
    *,
    feature: str,
    sample_id: int,
    variant: str,
    candidate: MaskCandidate,
    source: Path,
) -> dict[str, Any]:
    """Load and validate one completed candidate, including spatial metrics."""

    audit_path = run_dir / "audit.json"
    raw_output_path = run_dir / "sd2_bld_grid.png"
    semantic_path = run_dir / "semantic_mask.png"
    generation_path = run_dir / "generation_mask.png"
    required = (audit_path, raw_output_path, semantic_path, generation_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Candidate artifacts are incomplete: {missing}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    post_attack = (audit.get("cci") or {}).get("post_attack") or {}
    corrected_path = run_dir / "sd2_bld_grid_corrected.png"
    output_path = (
        corrected_path
        if post_attack and corrected_path.is_file()
        else raw_output_path
    )
    row = extract_audit_row(
        audit,
        feature=feature,
        sample_id=sample_id,
        variant=variant,
        output_path=output_path,
    )
    if row["desired_probability"] is None:
        raise ValueError("Candidate audit has no target probability")
    row.update(
        measure_spatial_change(
            source,
            output_path,
            semantic_path,
            generation_path,
        )
    )
    row.update(
        {
            "candidate": candidate.label,
            "dilation": candidate.dilation,
            "dilation_x": (
                candidate.dilation
                if candidate.dilation_x is None
                else candidate.dilation_x
            ),
            "dilation_y": (
                candidate.dilation
                if candidate.dilation_y is None
                else candidate.dilation_y
            ),
            "feather_radius": candidate.feather_radius,
            "source_path": str(source),
            "candidate_dir": str(run_dir),
            "audit_path": str(audit_path),
            "selected": False,
        }
    )
    return row


def _append_failure(path: Path, failure: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(failure, allow_nan=False) + "\n")


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from cci_diff.classifiers.celeba_resnet50 import load_celeba_resnet50
    from cci_diff.concept_graph import sha256_file
    from cci_diff.identity.facenet import build_face_detector

    args.variants = resolve_requested_variants(args)
    args.excluded_ids_by_feature = load_excluded_ids(
        getattr(args, "exclude_ids_json", None)
    )
    validate_pilot_args(args)
    mask_candidates = resolve_mask_candidates(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier = load_celeba_resnet50(
        args.classifier_path,
        device=args.device,
        dtype=torch.float32,
    )
    detector = build_face_detector()
    selected_by_feature = {}
    manifest = {
        "classifier_path": args.classifier_path,
        "classifier_sha256": sha256_file(args.classifier_path),
        "threshold": 0.5,
        "features": {},
        "variants": {name: VARIANTS[name] for name in args.variants},
        "controller_modes": [
            mode
            for mode, variant in CONTROLLER_VARIANTS.items()
            if variant in args.variants
        ],
        "excluded_ids_json": getattr(args, "exclude_ids_json", None),
        "random_sample_seed": getattr(args, "random_sample_seed", None),
        "mask_dilations": args.mask_dilations,
        "mask_shapes": [asdict(candidate) for candidate in mask_candidates],
        "post_attack": {
            "mode": args.cci_post_attack,
            "epsilon_schedule": args.cci_post_attack_epsilon_schedule,
            "boundary_margin": args.cci_post_attack_boundary_margin,
        },
        "historical_a1": {
            "boundary_weight": 0.3,
            "tv_weight": 0.05,
            "step_size": 0.08,
        },
    }
    for feature in args.features:
        selected, decisions = select_eligible_samples(
            args,
            feature=feature,
            classifier=classifier,
            detector=detector,
        )
        selected_by_feature[feature] = selected
        manifest["features"][feature] = {
            "selected_ids": [sample[0] for sample in selected],
            "scanned": decisions,
        }
    manifest_path = output_dir / "pilot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    rows = []
    candidate_rows = []
    unresolved = []
    failed_path = output_dir / "failures.jsonl"
    for feature in args.features:
        config = FEATURES[feature]
        for sample_id, source, masks in selected_by_feature[feature]:
            binding_path = output_dir / "bindings" / f"{feature}_{sample_id:05d}.json"
            write_binding(
                binding_path,
                source,
                masks,
                config["binding_roles"],
            )
            for variant in args.variants:
                result_dir = output_dir / feature / f"{sample_id:05d}" / variant
                group_rows = []
                for candidate in mask_candidates:
                    run_dir = result_dir / "candidates" / candidate.label
                    audit_path = run_dir / "audit.json"
                    try:
                        row = _candidate_row(
                            run_dir,
                            feature=feature,
                            sample_id=sample_id,
                            variant=variant,
                            candidate=candidate,
                            source=source,
                        )
                        candidate_rows.append(row)
                        group_rows.append(row)
                        write_rows(
                            candidate_rows,
                            output_dir / "candidate_results.csv",
                        )
                        continue
                    except (
                        FileNotFoundError,
                        KeyError,
                        ValueError,
                        json.JSONDecodeError,
                    ):
                        pass
                    command = build_variant_command(
                        args,
                        feature=feature,
                        variant=variant,
                        sample_id=sample_id,
                        source=source,
                        masks=masks,
                        binding_path=binding_path,
                        output_path=run_dir,
                        mask_candidate=candidate,
                    )
                    completed = subprocess.run(command, check=False)
                    if completed.returncode != 0:
                        failure = {
                            "feature": feature,
                            "variant": variant,
                            "sample_id": sample_id,
                            "candidate": candidate.label,
                            "dilation": candidate.dilation,
                            "dilation_x": candidate.dilation_x,
                            "dilation_y": candidate.dilation_y,
                            "feather_radius": candidate.feather_radius,
                            "exit_code": completed.returncode,
                            "audit_path": str(audit_path),
                        }
                        _append_failure(failed_path, failure)
                        unresolved.append(failure)
                        write_rows(
                            candidate_rows,
                            output_dir / "candidate_results.csv",
                        )
                        if args.continue_on_error:
                            continue
                        raise RuntimeError(f"Pilot subprocess failed: {failure}")
                    try:
                        row = _candidate_row(
                            run_dir,
                            feature=feature,
                            sample_id=sample_id,
                            variant=variant,
                            candidate=candidate,
                            source=source,
                        )
                    except (
                        FileNotFoundError,
                        KeyError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as error:
                        failure = {
                            "feature": feature,
                            "variant": variant,
                            "sample_id": sample_id,
                            "candidate": candidate.label,
                            "dilation": candidate.dilation,
                            "dilation_x": candidate.dilation_x,
                            "dilation_y": candidate.dilation_y,
                            "feather_radius": candidate.feather_radius,
                            "exit_code": completed.returncode,
                            "audit_path": str(audit_path),
                            "validation_error": str(error),
                        }
                        _append_failure(failed_path, failure)
                        unresolved.append(failure)
                        if args.continue_on_error:
                            continue
                        raise RuntimeError(
                            f"Pilot candidate validation failed: {failure}"
                        ) from error
                    candidate_rows.append(row)
                    group_rows.append(row)
                    write_rows(
                        candidate_rows,
                        output_dir / "candidate_results.csv",
                    )
                if len(group_rows) != len(mask_candidates):
                    continue
                selected = select_spatial_candidate(group_rows)
                for row in group_rows:
                    row["selected"] = row is selected
                artifacts = materialize_selected_artifacts(
                    source,
                    selected["candidate_dir"],
                    result_dir,
                    {
                        **selected,
                        "selected_dilation": selected["dilation"],
                        "label": f"{feature} {sample_id:05d} {variant} {selected['candidate']}",
                    },
                )
                selected_row = dict(selected)
                selected_row.update(artifacts)
                selected_row["candidate_output_path"] = selected_row["output_path"]
                selected_row["output_path"] = artifacts["output_path"]
                selected_row["selected_dilation"] = selected["dilation"]
                selected_row["label"] = (
                    f"{feature} {sample_id:05d} {variant} {selected['candidate']}"
                )
                rows.append(selected_row)
                write_rows(candidate_rows, output_dir / "candidate_results.csv")
                write_rows(rows, output_dir / "pilot_results.csv")

    write_rows(candidate_rows, output_dir / "candidate_results.csv")
    write_rows(rows, output_dir / "pilot_results.csv")
    ranked = rank_rows(rows)
    write_rows(ranked, output_dir / "pilot_ranked.csv")
    summary = summarize_results(rows)
    summary["ranked"] = ranked
    summary["candidate_count"] = len(candidate_rows)
    summary["selected_count"] = len(rows)
    summary["unresolved_candidates"] = unresolved
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    create_contact_sheets(rows, output_dir)
    for feature in args.features:
        create_paginated_pair_sheets(
            [row for row in rows if row["feature"] == feature],
            output_dir / "comparisons" / feature,
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", choices=sorted(FEATURES), required=True)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "float16", "float32"],
        default="auto",
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
        "--image_root",
        default="data/CelebAMask-HQ/CelebA-HQ-img",
    )
    parser.add_argument(
        "--mask_root",
        default="data/CelebAMask-HQ/CelebAMask-HQ-mask-anno",
    )
    parser.add_argument("--classifier_input_size", type=int, default=512)
    parser.add_argument("--max_image_id", type=int, default=30000)
    parser.add_argument(
        "--sample_ids",
        nargs="+",
        type=int,
        default=None,
        help="Evaluate exactly these preselected sample IDs.",
    )
    parser.add_argument(
        "--random_sample_seed",
        type=int,
        default=None,
        help=(
            "Shuffle the candidate ID scan deterministically before selecting "
            "the requested number of eligible samples."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument(
        "--controller_modes",
        nargs="+",
        choices=list(CONTROLLER_VARIANTS),
        default=None,
        help=(
            "Readable end-to-end CCI modes. Use fixed_equal feedback to "
            "compare CCI without and with adaptive weighting."
        ),
    )
    parser.add_argument(
        "--exclude_ids_json",
        default=None,
        help="Task-to-ID JSON produced by graph discovery.",
    )
    parser.add_argument("--mask_dilations", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--mask_shapes",
        nargs="+",
        type=parse_mask_shape,
        default=None,
        help="Explicit generation-mask candidates as x,y,feather.",
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument(
        "--cci_post_attack",
        choices=["none", "smooth_boundary"],
        default="none",
    )
    parser.add_argument(
        "--cci_post_attack_epsilon_schedule",
        default="0.05,0.08,0.10,0.30,0.50",
    )
    parser.add_argument(
        "--cci_post_attack_boundary_margin",
        type=float,
        default=0.03,
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_pilot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
