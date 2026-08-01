#!/usr/bin/env python3
"""Join frozen source-only mask decisions to existing fixed-region outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cci_diff.concept_graph import sha256_file  # noqa: E402
from cci_diff.risk_controlled_selection import (  # noqa: E402
    FrozenSelectorArtifact,
)


RegionTuple = tuple[str, ...]
FORBIDDEN_DECISION_FIELDS = {
    "output_path",
    "output_probability",
    "desired_probability",
    "target_pass",
    "identity_cosine",
    "identity_distance",
    "outside_l1",
    "outside_locality",
    "oracle_probability",
    "post_attack_selected_epsilon",
}


def _canonical_regions(regions: Iterable[str]) -> RegionTuple:
    return tuple(sorted({str(region).strip() for region in regions if str(region).strip()}))


def _parse_regions(value: Any) -> RegionTuple:
    if isinstance(value, str):
        stripped = value.strip()
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            value = stripped.split("+")
    if not isinstance(value, (list, tuple)):
        raise ValueError("regions must be a list or plus-separated string")
    regions = _canonical_regions(value)
    if not regions:
        raise ValueError("regions must be non-empty")
    return regions


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _load_selection_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"selection manifest not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    recorded_digest = str(payload.get("manifest_sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    actual_digest = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if recorded_digest != actual_digest:
        raise ValueError(
            "selection manifest SHA-256 changed after source-only selection"
        )
    if payload.get("version") != 1:
        raise ValueError("unsupported selection manifest version")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("selection manifest contains no decisions")
    normalized = []
    seen_ids = set()
    for decision in decisions:
        forbidden = FORBIDDEN_DECISION_FIELDS.intersection(decision)
        if forbidden:
            raise ValueError(
                "source-only decision contains generated-output fields: "
                f"{sorted(forbidden)}"
            )
        if decision.get("selection_uses_generated_output") is not False:
            raise ValueError("selection decision must deny generated-output use")
        sample_id = int(decision["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(f"duplicate selection decision for sample {sample_id}")
        seen_ids.add(sample_id)
        normalized.append(
            {
                **decision,
                "sample_id": sample_id,
                "selected_regions": _parse_regions(
                    decision["selected_regions"]
                ),
            }
        )
    return payload, sorted(normalized, key=lambda row: row["sample_id"]), actual_digest


def _read_candidate_rows(
    path: str | Path,
    regions: RegionTuple,
    expected_variants: tuple[str, ...],
) -> tuple[dict[tuple[int, str], dict[str, str]], tuple[str, ...]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"candidate results not found: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    required = {"sample_id", "variant", "source_path", "output_path"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"candidate results lack fields: {sorted(missing)}")
    index = {}
    for row in rows:
        key = (int(row["sample_id"]), row["variant"])
        if key in index:
            raise ValueError(
                f"duplicate candidate row for {regions}/{key[0]}/{key[1]}"
            )
        if row["variant"] not in expected_variants:
            raise ValueError(
                f"unexpected candidate variant {row['variant']!r} for {regions}"
            )
        index[key] = row
    return index, fields


def _write_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_selector_cohorts(
    selector_model: str | Path | None,
    manifest: Mapping[str, Any],
    sample_ids: set[int],
    *,
    exploratory: bool,
) -> str:
    if selector_model is None:
        return "selector_cohorts_not_verified"
    path = Path(selector_model)
    artifact = FrozenSelectorArtifact.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if manifest.get("selector_sha256") != sha256_file(path):
        raise ValueError("selection manifest selector SHA-256 mismatch")
    development = (
        set(artifact.discovery_sample_ids)
        | set(artifact.fit_sample_ids)
        | set(artifact.calibration_sample_ids)
    )
    overlap = sorted(development & sample_ids)
    if overlap and not exploratory:
        raise ValueError(
            "materialized IDs overlap selector development cohorts: "
            f"{overlap}"
        )
    if artifact.evaluation_sample_ids and not sample_ids.issubset(
        artifact.evaluation_sample_ids
    ):
        raise ValueError("materialized IDs are absent from selector evaluation cohort")
    return "exploratory_not_held_out" if overlap else "held_out_verified"


def materialize_adaptive_cohort(
    selection_manifest: str | Path,
    candidate_results: Mapping[RegionTuple | str, str | Path],
    output_dir: str | Path,
    *,
    expected_variants: Sequence[str] = ("A0", "A11"),
    expected_count: int | None = None,
    selector_model: str | Path | None = None,
    evaluation_ids: Iterable[int] | None = None,
    exploratory: bool = False,
) -> list[dict[str, str]]:
    """Materialize fixed outputs selected by a finalized source-only manifest."""

    variants = tuple(str(value) for value in expected_variants)
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("expected variants must be non-empty and unique")
    manifest, decisions, manifest_sha = _load_selection_manifest(
        selection_manifest
    )
    decision_ids = {row["sample_id"] for row in decisions}
    if expected_count is not None and len(decision_ids) != expected_count:
        raise ValueError(
            f"selection manifest must contain {expected_count} IDs; "
            f"found {len(decision_ids)}"
        )
    if evaluation_ids is not None and decision_ids != {
        int(value) for value in evaluation_ids
    }:
        raise ValueError("selection decisions do not match evaluation IDs")
    cohort_status = _validate_selector_cohorts(
        selector_model,
        manifest,
        decision_ids,
        exploratory=exploratory,
    )

    normalized_paths = {
        _parse_regions(regions): Path(path)
        for regions, path in candidate_results.items()
    }
    selected_families = {row["selected_regions"] for row in decisions}
    missing_families = selected_families - set(normalized_paths)
    if missing_families:
        raise ValueError(
            f"candidate results missing selected masks: {sorted(missing_families)}"
        )

    indexed = {}
    common_fields = None
    expected_keys = {
        (sample_id, variant)
        for sample_id in decision_ids
        for variant in variants
    }
    for regions, path in normalized_paths.items():
        index, fields = _read_candidate_rows(path, regions, variants)
        if set(index) != expected_keys:
            missing = sorted(expected_keys - set(index))
            extra = sorted(set(index) - expected_keys)
            missing_variants = sorted({variant for _, variant in missing})
            raise ValueError(
                f"candidate results for {regions} do not match IDs/variants; "
                f"missing variants={missing_variants}, missing={missing[:5]}, "
                f"extra={extra[:5]}"
            )
        if common_fields is None:
            common_fields = fields
        elif fields != common_fields:
            raise ValueError("candidate result schemas must be identical")
        indexed[regions] = index

    output_rows = []
    for decision in decisions:
        sample_id = decision["sample_id"]
        regions = decision["selected_regions"]
        for variant in variants:
            source = dict(indexed[regions][(sample_id, variant)])
            if source["source_path"] != str(decision["source_path"]):
                raise ValueError(
                    f"source path mismatch for sample {sample_id}/{variant}"
                )
            output_path = Path(source["output_path"])
            if not output_path.is_file():
                raise FileNotFoundError(
                    f"selected output not found: {output_path}"
                )
            source.update(
                {
                    "selected_regions": json.dumps(list(regions)),
                    "selection_manifest_sha256": manifest_sha,
                    "source_candidate_results": str(
                        normalized_paths[regions]
                    ),
                    "selection_policy": manifest["policy_type"],
                    "selection_fallback_used": str(
                        bool(decision.get("fallback_used", False))
                    ),
                }
            )
            output_rows.append(source)

    destination = Path(output_dir)
    additions = (
        "selected_regions",
        "selection_manifest_sha256",
        "source_candidate_results",
        "selection_policy",
        "selection_fallback_used",
    )
    output_fields = tuple(common_fields or ()) + additions
    _write_rows(destination / "adaptive_results.csv", output_rows, output_fields)
    _write_rows(destination / "pilot_results.csv", output_rows, output_fields)
    materialization = {
        "version": 1,
        "selection_manifest": str(selection_manifest),
        "selection_manifest_sha256": manifest_sha,
        "candidate_results": {
            "+".join(regions): {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for regions, path in sorted(normalized_paths.items())
        },
        "sample_count": len(decision_ids),
        "row_count": len(output_rows),
        "variants": list(variants),
        "cohort_status": cohort_status,
        "exploratory": exploratory,
        "generation_invoked": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "materialization_manifest.json").write_text(
        json.dumps(materialization, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    title = (
        "# Exploratory Adaptive Replay (NOT HELD-OUT)"
        if cohort_status == "exploratory_not_held_out"
        else "# Adaptive Source-Only Replay"
    )
    counts = {
        "+".join(regions): sum(
            decision["selected_regions"] == regions for decision in decisions
        )
        for regions in sorted(selected_families)
    }
    (destination / "materialization_report.md").write_text(
        "\n".join(
            [
                title,
                "",
                f"- Cohort status: `{cohort_status}`",
                f"- Sources: `{len(decision_ids)}`",
                f"- Materialized rows: `{len(output_rows)}`",
                f"- Variants: `{', '.join(variants)}`",
                f"- Mask counts: `{json.dumps(counts, sort_keys=True)}`",
                "- Diffusion invoked: `False`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_rows


def _read_ids(path: str | Path | None) -> tuple[int, ...] | None:
    if path is None:
        return None
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        values = payload.get("sample_ids", payload) if isinstance(payload, dict) else payload
        return tuple(int(value) for value in values)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if rows and "sample_id" in rows[0]:
        return tuple(int(row["sample_id"]) for row in rows)
    return tuple(
        int(line.strip())
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument(
        "--candidate_results",
        action="append",
        required=True,
        metavar="REGIONS=CSV",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_variants", nargs="+", default=["A0", "A11"])
    parser.add_argument("--expected_count", type=int, default=None)
    parser.add_argument("--selector_model", default=None)
    parser.add_argument("--evaluation_ids", default=None)
    parser.add_argument("--exploratory", action="store_true")
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
            raise ValueError(f"duplicate candidate_results mapping: {canonical}")
        mappings[canonical] = path
    materialize_adaptive_cohort(
        args.selection_manifest,
        mappings,
        args.output_dir,
        expected_variants=args.expected_variants,
        expected_count=args.expected_count,
        selector_model=args.selector_model,
        evaluation_ids=_read_ids(args.evaluation_ids),
        exploratory=args.exploratory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
