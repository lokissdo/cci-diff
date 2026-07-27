#!/usr/bin/env python3
"""Generate and evaluate a four-seed distribution-aware FID smoke test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cci_diff.comparison_artifacts import create_pair_image
from cci_diff.fid_reranking import (
    fit_reference_projection,
    frechet_distance,
    project_features,
    select_global_fid_candidates,
    select_independent_candidates,
    select_random_candidates,
    select_reference_ids,
    select_single_seed,
)
if __package__:
    from scripts.evaluate_fid_sfid import extract_or_load_activations
else:
    from evaluate_fid_sfid import extract_or_load_activations


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_path", required=True)
    parser.add_argument("--identity_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="checkpoints/sd2-1-base")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--image_root",
        default="data/CelebAMask-HQ/CelebA-HQ-img",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--reference_count", type=int, default=1000)
    parser.add_argument("--proxy_dims", type=int, default=64)
    parser.add_argument("--inception_dims", type=int, default=2048)
    parser.add_argument("--minimum_passes", type=int, default=10)
    parser.add_argument("--selector_seed", type=int, default=20260725)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--selection_only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique")
    if 42 not in args.seeds:
        raise ValueError("--seeds must include 42 for the S0 baseline")
    for name in (
        "limit",
        "num_inference_steps",
        "reference_count",
        "proxy_dims",
        "inception_dims",
        "minimum_passes",
        "batch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.minimum_passes > args.limit:
        raise ValueError("--minimum_passes cannot exceed --limit")


def build_pilot_commands(args: argparse.Namespace) -> list[list[str]]:
    """Build one exact resumable pilot command for every requested seed."""

    output_dir = Path(args.output_dir)
    commands = []
    for seed in args.seeds:
        commands.append(
            [
                ".venv-ml/bin/python",
                "scripts/run_clean_cci_pilot.py",
                "--features",
                "smile",
                "--limit",
                str(args.limit),
                "--seed",
                str(seed),
                "--num_inference_steps",
                str(args.num_inference_steps),
                "--device",
                args.device,
                "--model_path",
                args.model_path,
                "--classifier_path",
                args.classifier_path,
                "--identity_model_path",
                args.identity_model_path,
                "--output_dir",
                str(output_dir / "seeds" / f"seed_{seed}"),
                "--variants",
                "A3",
                "--mask_shapes",
                "4,4,3",
                "--cci_post_attack",
                "smooth_boundary",
                "--cci_post_attack_epsilon_schedule",
                "0.05,0.08,0.10,0.30,0.50",
                "--cci_post_attack_boundary_margin",
                "0.03",
                "--continue_on_error",
            ]
        )
    return commands


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing candidate table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resolved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _normalize_candidate_row(row: Mapping[str, Any], seed: int) -> dict[str, Any]:
    candidate_dir = _resolved_path(row["candidate_dir"])
    audit_path = candidate_dir / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing candidate audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    post_attack = (audit.get("cci") or {}).get("post_attack") or {}
    attack_candidates = post_attack.get("candidates") or []
    if len(attack_candidates) != 1:
        raise ValueError(f"candidate audit requires one post-attack row: {audit_path}")
    attack = attack_candidates[0]
    source_path = _resolved_path(row["source_path"])
    output_path = _resolved_path(row["output_path"])
    raw_output_path = _resolved_path(post_attack["raw_output_path"])
    corrected_output_path = _resolved_path(post_attack["corrected_output_path"])
    for path in (
        source_path,
        output_path,
        raw_output_path,
        corrected_output_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing candidate artifact: {path}")
    before_probability = float(attack["before_probability"])
    identity_cosine = _float_or_none(row.get("identity_cosine"))
    if identity_cosine is None:
        identity_cosine = _float_or_none(
            ((audit.get("cci") or {}).get("metrics") or {}).get(
                "identity_cosine"
            )
        )
    if identity_cosine is None:
        raise ValueError(f"candidate identity is unavailable: {audit_path}")
    return {
        **dict(row),
        "sample_id": int(float(row["sample_id"])),
        "seed": int(seed),
        "source_path": str(source_path),
        "output_path": str(output_path),
        "candidate_dir": str(candidate_dir),
        "audit_path": str(audit_path),
        "raw_output_path": str(raw_output_path),
        "corrected_output_path": str(corrected_output_path),
        "desired_probability": float(row["desired_probability"]),
        "identity_cosine": identity_cosine,
        "outside_semantic_l1": float(row["outside_semantic_l1"]),
        "post_attack_selected_epsilon": _float_or_none(
            attack.get("selected_epsilon")
        ),
        "post_attack_linf": float(attack.get("linf", 0.0)),
        "raw_smile_probability": before_probability,
        "raw_target_pass": before_probability < 0.5,
    }


def load_seed_candidate_pool(
    output_dir: str | Path,
    seeds: Sequence[int],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Load one aligned corrected candidate per source and seed."""

    output_dir = Path(output_dir)
    cohorts: dict[int, list[dict[str, Any]]] = {}
    expected_ids: tuple[int, ...] | None = None
    for seed in seeds:
        rows = _read_csv(
            output_dir / "seeds" / f"seed_{seed}" / "pilot_results.csv"
        )
        normalized = [
            _normalize_candidate_row(row, seed)
            for row in rows
            if row.get("feature") == "smile" and row.get("variant") == "A3"
        ]
        normalized.sort(key=lambda row: row["sample_id"])
        ids = tuple(row["sample_id"] for row in normalized)
        if len(ids) != expected_count or len(set(ids)) != expected_count:
            raise ValueError(
                f"seed {seed} cohort requires {expected_count} unique rows; "
                f"found {len(set(ids))}"
            )
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"seed {seed} cohort does not match the first seed cohort")
        cohorts[int(seed)] = normalized

    assert expected_ids is not None
    by_seed = {
        seed: {row["sample_id"]: row for row in rows}
        for seed, rows in cohorts.items()
    }
    return [
        by_seed[int(seed)][sample_id]
        for sample_id in expected_ids
        for seed in seeds
    ]


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_reference_manifest(
    path: str | Path,
    references: Sequence[tuple[int, Path]],
    *,
    excluded_ids: set[int],
    dimensions: int,
    cache_path: Path,
) -> dict[str, Any]:
    """Persist the exact disjoint reference cohort and feature provenance."""

    payload = {
        "reference_ids": [int(image_id) for image_id, _ in references],
        "reference_paths": [str(Path(value).resolve()) for _, value in references],
        "excluded_ids": sorted(int(value) for value in excluded_ids),
        "dimensions": int(dimensions),
        "cache_path": str(Path(cache_path)),
        "cache_sha256": _sha256(Path(cache_path)),
        "extractor": "pytorch_fid.InceptionV3",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def _attack_used(row: Mapping[str, Any]) -> bool:
    return row.get("post_attack_selected_epsilon") not in (None, "")


def _selector_metrics(
    name: str,
    result,
    *,
    source_activations: np.ndarray,
    candidate_activations: np.ndarray,
    projected_reference: np.ndarray,
    projected_candidates: np.ndarray,
) -> dict[str, Any]:
    selected_rows = list(result.rows)
    selected_indices = list(result.indices)
    seed_counts = {
        str(seed): sum(int(row["seed"]) == seed for row in selected_rows)
        for seed in sorted({int(row["seed"]) for row in selected_rows})
    }
    return {
        "selector": name,
        "count": len(selected_rows),
        "generation_fr": float(
            np.mean(
                [
                    float(row["desired_probability"]) >= 0.5
                    for row in selected_rows
                ]
            )
        ),
        "mean_desired_probability": float(
            np.mean([float(row["desired_probability"]) for row in selected_rows])
        ),
        "mean_identity_cosine": float(
            np.mean([float(row["identity_cosine"]) for row in selected_rows])
        ),
        "mean_outside_semantic_l1": float(
            np.mean([float(row["outside_semantic_l1"]) for row in selected_rows])
        ),
        "attack_rate": float(np.mean([_attack_used(row) for row in selected_rows])),
        "mean_post_attack_linf": float(
            np.mean([float(row["post_attack_linf"]) for row in selected_rows])
        ),
        "proxy_fid": frechet_distance(
            projected_reference,
            projected_candidates[selected_indices],
        ),
        "report_fid": frechet_distance(
            source_activations,
            candidate_activations[selected_indices],
        ),
        "initial_proxy_fid": float(result.initial_fid),
        "accepted_swaps": int(result.accepted_swaps),
        "optimization_passes": int(result.passes),
        "seed_counts": json.dumps(seed_counts, sort_keys=True),
    }


def _materialize_selector(
    output_dir: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selector = name.lower()
    selected_dir = output_dir / "selected" / selector
    selected_dir.mkdir(parents=True, exist_ok=True)
    materialized = []
    for row in rows:
        destination = selected_dir / f"{int(row['sample_id']):05d}.png"
        shutil.copyfile(row["output_path"], destination)
        record = {**dict(row), "selected_output_path": str(destination)}
        materialized.append(record)
        if name == "S3":
            create_pair_image(
                row["source_path"],
                destination,
                output_dir
                / "comparisons"
                / f"{int(row['sample_id']):05d}.jpg",
                f"S3 seed {int(row['seed'])}",
            )
    _write_csv(materialized, output_dir / f"selection_{selector}.csv")
    return materialized


def _write_report(
    output_dir: Path,
    metrics: Sequence[Mapping[str, Any]],
    *,
    acceptance: Mapping[str, bool],
) -> None:
    sample_count = int(metrics[0]["count"])
    lines = [
        "# Distribution-Aware FID Reranking",
        "",
        f"The {sample_count}-image FID values are exploratory.",
        "",
        "| Selector | N | Generation FR | Proxy FID | Report FID | Identity | Outside L1 | Attack rate | Seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['selector']} | {row['count']} | "
            f"{100 * float(row['generation_fr']):.1f} | "
            f"{float(row['proxy_fid']):.4f} | "
            f"{float(row['report_fid']):.4f} | "
            f"{float(row['mean_identity_cosine']):.4f} | "
            f"{float(row['mean_outside_semantic_l1']):.4f} | "
            f"{100 * float(row['attack_rate']):.1f} | "
            f"`{row['seed_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
        ]
    )
    for name, passed in acceptance.items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "S0 is the seed-42 baseline, S1 is deterministic random selection, "
            "S2 is independent reference-distance selection, and S3 is the "
            "constrained global proxy-FID selector.",
        ]
    )
    (output_dir / "fid_reranking_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def select_and_materialize(
    *,
    output_dir: str | Path,
    candidates: Sequence[Mapping[str, Any]],
    candidate_activations: np.ndarray,
    reference_activations: np.ndarray,
    source_activations: np.ndarray,
    proxy_dims: int,
    minimum_passes: int,
    selector_seed: int,
) -> dict[str, Any]:
    """Run S0-S3, materialize their images, and write comparison metrics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_activations = np.asarray(candidate_activations, dtype=np.float64)
    reference_activations = np.asarray(reference_activations, dtype=np.float64)
    source_activations = np.asarray(source_activations, dtype=np.float64)
    if len(candidates) != len(candidate_activations):
        raise ValueError("candidate rows and activations must align")
    source_count = len({int(row["sample_id"]) for row in candidates})
    if len(source_activations) != source_count:
        raise ValueError("source activations must align with unique source IDs")

    projection = fit_reference_projection(
        reference_activations,
        dimensions=proxy_dims,
    )
    projected_reference = projection.projected_reference
    projected_candidates = project_features(candidate_activations, projection)
    selectors = {
        "S0": select_single_seed(
            candidates,
            projected_candidates,
            projected_reference,
            seed=42,
        ),
        "S1": select_random_candidates(
            candidates,
            projected_candidates,
            projected_reference,
            selector_seed=selector_seed,
        ),
        "S2": select_independent_candidates(
            candidates,
            projected_candidates,
            projected_reference,
        ),
        "S3": select_global_fid_candidates(
            candidates,
            projected_candidates,
            projected_reference,
            minimum_passes=minimum_passes,
            maximum_passes=8,
        ),
    }

    metrics = []
    selector_payload = {}
    for name, result in selectors.items():
        materialized = _materialize_selector(output_dir, name, result.rows)
        metric = _selector_metrics(
            name,
            result,
            source_activations=source_activations,
            candidate_activations=candidate_activations,
            projected_reference=projected_reference,
            projected_candidates=projected_candidates,
        )
        metrics.append(metric)
        selector_payload[name] = {
            **metric,
            "selected": [
                {
                    "sample_id": int(row["sample_id"]),
                    "seed": int(row["seed"]),
                    "output_path": row["selected_output_path"],
                }
                for row in materialized
            ],
        }

    by_name = {row["selector"]: row for row in metrics}
    acceptance = {
        "candidate_count": len(candidates) == source_count * 4,
        "selector_counts": all(
            int(row["count"]) == source_count for row in metrics
        ),
        "s3_generation_fr": (
            int(by_name["S3"]["count"]) * float(by_name["S3"]["generation_fr"])
            >= minimum_passes
        ),
        "s3_fid_no_worse_than_s0": (
            float(by_name["S3"]["report_fid"])
            <= float(by_name["S0"]["report_fid"]) + 1e-8
        ),
        "s3_fid_no_worse_than_s1": (
            float(by_name["S3"]["report_fid"])
            <= float(by_name["S1"]["report_fid"]) + 1e-8
        ),
    }
    _write_csv(metrics, output_dir / "selector_metrics.csv")
    _write_report(output_dir, metrics, acceptance=acceptance)
    payload = {
        "selectors": selector_payload,
        "acceptance": acceptance,
        "acceptance_passed": all(acceptance.values()),
    }
    (output_dir / "fid_reranking_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def _build_inception_extractor(args: argparse.Namespace):
    import torch
    from pytorch_fid.fid_score import get_activations
    from pytorch_fid.inception import InceptionV3

    device = torch.device(args.device)
    block = InceptionV3.BLOCK_INDEX_BY_DIM[args.inception_dims]
    model = InceptionV3([block]).to(device).eval()

    def extractor(paths: Sequence[Path]) -> np.ndarray:
        return get_activations(
            [str(path) for path in paths],
            model,
            batch_size=args.batch_size,
            dims=args.inception_dims,
            device=device,
            num_workers=args.num_workers,
        )

    return extractor


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.selection_only:
        for command in build_pilot_commands(args):
            subprocess.run(command, check=True)

    candidates = load_seed_candidate_pool(
        output_dir,
        args.seeds,
        expected_count=args.limit,
    )
    _write_csv(candidates, output_dir / "candidate_metrics.csv")
    evaluation_ids = {int(row["sample_id"]) for row in candidates}
    references = select_reference_ids(
        args.image_root,
        count=args.reference_count,
        excluded_ids=evaluation_ids,
    )
    extractor = _build_inception_extractor(args)
    reference_cache = output_dir / "reference_features.npz"
    reference_activations = extract_or_load_activations(
        [path for _, path in references],
        reference_cache,
        extractor,
    )
    candidate_activations = extract_or_load_activations(
        [Path(row["output_path"]) for row in candidates],
        output_dir / "candidate_features.npz",
        extractor,
    )
    unique_sources = []
    seen = set()
    for row in candidates:
        sample_id = int(row["sample_id"])
        if sample_id not in seen:
            seen.add(sample_id)
            unique_sources.append(Path(row["source_path"]))
    source_activations = extract_or_load_activations(
        unique_sources,
        output_dir / "source_features.npz",
        extractor,
    )
    write_reference_manifest(
        output_dir / "reference_manifest.json",
        references,
        excluded_ids=evaluation_ids,
        dimensions=args.inception_dims,
        cache_path=reference_cache,
    )
    return select_and_materialize(
        output_dir=output_dir,
        candidates=candidates,
        candidate_activations=candidate_activations,
        reference_activations=reference_activations,
        source_activations=source_activations,
        proxy_dims=args.proxy_dims,
        minimum_passes=args.minimum_passes,
        selector_seed=args.selector_seed,
    )


def main() -> int:
    payload = run(build_arg_parser().parse_args())
    print(json.dumps(payload["acceptance"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
