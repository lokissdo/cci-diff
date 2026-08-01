#!/usr/bin/env python3
"""Prepare disjoint discovery, selector-development, and replay ID artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cci_diff.concept_graph import sha256_file  # noqa: E402


RegionTuple = tuple[str, ...]
DISCOVERY_FIELDS = (
    "target",
    "desired_value",
    "sample_id",
    "seed",
    "regions",
    "source_probability",
    "output_probability",
    "mask_fraction",
    "identity_cosine",
    "non_target_drift",
    "outside_l1",
    "changed_fraction",
    "output_path",
    "audit_path",
)
DEVELOPMENT_FIELDS = (
    "cohort",
    "sample_id",
    "regions",
    "desired_probability",
    "identity_distance",
    "outside_locality",
)


def _canonical_regions(regions: Iterable[str]) -> RegionTuple:
    return tuple(sorted({str(region).strip() for region in regions if str(region).strip()}))


def _parse_regions(value: str | Sequence[str]) -> RegionTuple:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = value.split("+")
    regions = _canonical_regions(value)
    if not regions:
        raise ValueError("candidate regions must be non-empty")
    return regions


def _deterministic_order(sample_ids: Iterable[int], seed: int) -> list[int]:
    unique = sorted({int(value) for value in sample_ids})
    return sorted(
        unique,
        key=lambda sample_id: (
            hashlib.sha256(f"{seed}:{sample_id}".encode()).digest(),
            sample_id,
        ),
    )


def _read_candidate_index(
    path: Path,
    *,
    variant: str,
    relevant_ids: set[int],
) -> dict[int, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"candidate results not found: {path}")
    index = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = int(row["sample_id"])
            if sample_id not in relevant_ids or row["variant"] != variant:
                continue
            if sample_id in index:
                raise ValueError(
                    f"duplicate {variant} candidate row for sample {sample_id}: {path}"
                )
            index[sample_id] = row
    missing = sorted(relevant_ids - set(index))
    if missing:
        raise ValueError(
            f"candidate results lack {variant} rows for IDs: {missing[:10]}"
        )
    return index


def _finite(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"candidate row lacks finite {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"candidate row lacks finite {field}")
    return value


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_adaptive_replay_data(
    candidate_results: Mapping[RegionTuple | str, str | Path],
    output_dir: str | Path,
    *,
    sample_ids: Iterable[int],
    discovery_count: int,
    fit_count: int,
    calibration_count: int,
    evaluation_count: int,
    random_seed: int,
    variant: str = "A11",
    target: str = "Smiling",
    desired_value: int = 0,
    generation_seed: int = 42,
    outside_field: str = "outside_semantic_l1",
) -> dict[str, Any]:
    """Prepare role-separated rows from deterministic fixed candidate runs."""

    if desired_value not in (0, 1) or isinstance(desired_value, bool):
        raise ValueError("desired_value must be zero or one")
    counts = (
        discovery_count,
        fit_count,
        calibration_count,
        evaluation_count,
    )
    if any(value <= 0 for value in counts):
        raise ValueError("all four cohort counts must be positive")
    ordered = _deterministic_order(sample_ids, random_seed)
    if sum(counts) != len(ordered):
        raise ValueError(
            "cohort counts must sum to the number of unique sample IDs"
        )
    cut1 = discovery_count
    cut2 = cut1 + fit_count
    cut3 = cut2 + calibration_count
    cohorts = {
        "discovery": ordered[:cut1],
        "fit": ordered[cut1:cut2],
        "calibration": ordered[cut2:cut3],
        "evaluation": ordered[cut3:],
    }
    development_ids = (
        set(cohorts["discovery"])
        | set(cohorts["fit"])
        | set(cohorts["calibration"])
    )
    paths = {
        _parse_regions(regions): Path(path)
        for regions, path in candidate_results.items()
    }
    if len(paths) < 2:
        raise ValueError("adaptive replay preparation requires at least two candidates")
    indexes = {
        regions: _read_candidate_index(
            path, variant=variant, relevant_ids=development_ids
        )
        for regions, path in paths.items()
    }

    discovery_rows = []
    development_rows = []
    for regions, index in sorted(indexes.items()):
        for sample_id in cohorts["discovery"]:
            row = index[sample_id]
            source_probability = _finite(row, "source_probability")
            desired_probability = _finite(row, "desired_probability")
            output_probability = (
                desired_probability
                if desired_value == 1
                else 1.0 - desired_probability
            )
            discovery_rows.append(
                {
                    "target": target,
                    "desired_value": desired_value,
                    "sample_id": sample_id,
                    "seed": generation_seed,
                    "regions": json.dumps(list(regions)),
                    "source_probability": source_probability,
                    "output_probability": output_probability,
                    "mask_fraction": _finite(row, "semantic_mask_fraction"),
                    "identity_cosine": _finite(row, "identity_cosine"),
                    "non_target_drift": _finite(row, "non_target_drift"),
                    "outside_l1": _finite(row, outside_field),
                    "changed_fraction": _finite(row, "changed_fraction_1"),
                    "output_path": row.get("output_path", ""),
                    "audit_path": row.get("audit_path", ""),
                }
            )
        for cohort in ("fit", "calibration"):
            for sample_id in cohorts[cohort]:
                row = index[sample_id]
                development_rows.append(
                    {
                        "cohort": cohort,
                        "sample_id": sample_id,
                        "regions": json.dumps(list(regions)),
                        "desired_probability": _finite(
                            row, "desired_probability"
                        ),
                        "identity_distance": 1.0
                        - _finite(row, "identity_cosine"),
                        "outside_locality": _finite(row, outside_field),
                    }
                )

    destination = Path(output_dir)
    _write_csv(
        destination / "discovery_interventions.csv",
        DISCOVERY_FIELDS,
        sorted(
            discovery_rows,
            key=lambda row: (row["sample_id"], row["regions"]),
        ),
    )
    _write_csv(
        destination / "development_outcomes.csv",
        DEVELOPMENT_FIELDS,
        sorted(
            development_rows,
            key=lambda row: (
                row["cohort"],
                row["sample_id"],
                row["regions"],
            ),
        ),
    )
    for cohort, ids in cohorts.items():
        (destination / f"{cohort}_ids.json").write_text(
            json.dumps({"sample_ids": ids}, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "version": 1,
        "random_seed": random_seed,
        "generation_variant": variant,
        "generation_seed": generation_seed,
        "target": target,
        "desired_value": desired_value,
        "outside_field": outside_field,
        "candidate_results": {
            "+".join(regions): {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for regions, path in sorted(paths.items())
        },
        "cohorts": cohorts,
        "evaluation_outputs_exported_to_selector_data": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_ids(path: str | Path) -> tuple[int, ...]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return tuple(int(value) for value in payload)
    for key in ("sample_ids", "selected_sample_ids"):
        if key in payload:
            return tuple(int(value) for value in payload[key])
    if "features" in payload:
        values = []
        for feature in payload["features"].values():
            values.extend(
                feature.get("sample_ids", feature.get("selected_ids", ()))
            )
        return tuple(int(value) for value in values)
    raise ValueError("sample ID manifest has no recognized ID list")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate_results", action="append", required=True, metavar="REGIONS=CSV"
    )
    parser.add_argument("--sample_ids_manifest", required=True)
    parser.add_argument("--discovery_count", type=int, required=True)
    parser.add_argument("--fit_count", type=int, required=True)
    parser.add_argument("--calibration_count", type=int, required=True)
    parser.add_argument("--evaluation_count", type=int, required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--variant", default="A11")
    parser.add_argument("--target", default="Smiling")
    parser.add_argument("--desired_value", type=int, default=0)
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument("--outside_field", default="outside_semantic_l1")
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    mappings = {}
    for item in args.candidate_results:
        if "=" not in item:
            raise ValueError("candidate_results must use REGIONS=CSV")
        regions, path = item.split("=", 1)
        canonical = _parse_regions(regions)
        if canonical in mappings:
            raise ValueError(f"duplicate candidate mapping: {canonical}")
        mappings[canonical] = path
    prepare_adaptive_replay_data(
        mappings,
        args.output_dir,
        sample_ids=_read_ids(args.sample_ids_manifest),
        discovery_count=args.discovery_count,
        fit_count=args.fit_count,
        calibration_count=args.calibration_count,
        evaluation_count=args.evaluation_count,
        random_seed=args.random_seed,
        variant=args.variant,
        target=args.target,
        desired_value=args.desired_value,
        generation_seed=args.generation_seed,
        outside_field=args.outside_field,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
