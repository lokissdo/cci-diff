#!/usr/bin/env python3
"""Fit and freeze a calibrated source-only semantic-region selector."""

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

import numpy as np  # noqa: E402

from cci_diff.concept_graph import sha256_file  # noqa: E402
from cci_diff.individual_region_selection import (  # noqa: E402
    FrozenInfluencePolicy,
    load_frozen_influence_policy,
)
from cci_diff.risk_controlled_selection import (  # noqa: E402
    FEATURE_NAMES,
    FrozenSelectorArtifact,
    SafeSuccessThresholds,
    choose_grouped_l2,
    choose_risk_threshold,
    fit_platt_calibrator,
    safe_success_label,
)


RegionTuple = tuple[str, ...]
OUTCOME_FIELDS = (
    "desired_probability",
    "identity_distance",
    "outside_locality",
)
FORBIDDEN_PROVENANCE_KEYS = {
    "oracle",
    "oracle_model",
    "fid",
    "sfid",
    "fva",
    "fs",
    "mnac",
    "cd",
    "cout",
}


def _canonical_regions(regions: Iterable[str]) -> RegionTuple:
    return tuple(sorted({str(region).strip() for region in regions if str(region).strip()}))


def _parse_regions(value: Any) -> RegionTuple:
    if isinstance(value, str):
        stripped = value.strip()
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = stripped.split("+")
        value = decoded
    if not isinstance(value, (list, tuple)):
        raise ValueError("regions must be a JSON list, tuple, or plus-separated string")
    regions = _canonical_regions(value)
    if not regions:
        raise ValueError("regions must be non-empty")
    return regions


def _validate_provenance(value: Any, *, path: str = "provenance") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            prefix = normalized.split("_", 1)[0]
            if normalized in FORBIDDEN_PROVENANCE_KEYS or prefix in {
                "oracle",
                "fid",
                "sfid",
                "fva",
                "mnac",
                "cout",
            }:
                raise ValueError(
                    f"{path}.{key} is evaluation-only and forbidden in selector provenance"
                )
            _validate_provenance(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_provenance(nested, path=f"{path}[{index}]")


def validate_candidate_family(
    candidate_family: Sequence[Iterable[str]] | None,
    policy: FrozenInfluencePolicy,
) -> tuple[RegionTuple, ...]:
    candidates = (
        policy.candidate_region_sets
        if candidate_family is None
        else tuple(sorted({_canonical_regions(item) for item in candidate_family}))
    )
    unknown = set(candidates) - set(policy.candidate_region_sets)
    if unknown:
        raise ValueError(f"candidate family contains a non-graph candidate: {sorted(unknown)}")
    if policy.fallback_regions not in candidates:
        raise ValueError("candidate family must include the graph fallback")
    return candidates


def _pairwise_cohort_overlap(cohorts: Mapping[str, set[int]]) -> dict[str, list[int]]:
    names = tuple(cohorts)
    return {
        f"{names[left]}:{names[right]}": sorted(
            cohorts[names[left]] & cohorts[names[right]]
        )
        for left in range(len(names))
        for right in range(left + 1, len(names))
        if cohorts[names[left]] & cohorts[names[right]]
    }


def _normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    policy: FrozenInfluencePolicy,
    candidates: tuple[RegionTuple, ...],
    thresholds: SafeSuccessThresholds,
) -> list[dict[str, Any]]:
    normalized = []
    seen = set()
    expected = set(candidates)
    for row_number, row in enumerate(rows, start=1):
        try:
            cohort = str(row["cohort"]).strip()
            sample_id = int(row["sample_id"])
            regions = _parse_regions(row["regions"])
            values = tuple(float(row[name]) for name in FEATURE_NAMES)
            outcomes = tuple(float(row[name]) for name in OUTCOME_FIELDS)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid selector row {row_number}: {error}") from error
        if cohort not in {"fit", "calibration"}:
            raise ValueError("selector row cohort must be fit or calibration")
        if regions not in expected:
            raise ValueError(f"selector row uses undeclared candidate: {regions}")
        if not all(math.isfinite(value) for value in values + outcomes):
            raise ValueError("selector rows must contain only finite values")
        key = (cohort, sample_id, regions)
        if key in seen:
            raise ValueError(f"duplicate selector row: {key}")
        seen.add(key)
        evidence = policy.region_set_evidence.get(regions)
        if evidence is None:
            raise ValueError(f"graph candidate lacks complete evidence: {regions}")
        expected_globals = (
            evidence.mean_effect,
            evidence.flip_rate,
            evidence.effect_ci_low,
        )
        actual_globals = tuple(
            values[FEATURE_NAMES.index(name)]
            for name in (
                "global_mean_effect",
                "global_flip_rate",
                "global_effect_ci_low",
            )
        )
        if not np.allclose(actual_globals, expected_globals, rtol=0.0, atol=1e-12):
            raise ValueError("selector row global evidence disagrees with graph")
        if values[FEATURE_NAMES.index("component_count")] != len(regions):
            raise ValueError("selector row component_count disagrees with regions")
        label = safe_success_label(*outcomes, thresholds=thresholds)
        normalized.append(
            {
                "cohort": cohort,
                "sample_id": sample_id,
                "regions": regions,
                **{name: value for name, value in zip(FEATURE_NAMES, values)},
                **{name: value for name, value in zip(OUTCOME_FIELDS, outcomes)},
                "target_pass": int(outcomes[0] >= thresholds.desired_probability),
                "identity_pass": int(outcomes[1] <= thresholds.identity_distance),
                "locality_pass": int(outcomes[2] <= thresholds.outside_locality),
                "safe_success": label,
            }
        )

    for cohort in ("fit", "calibration"):
        cohort_rows = [row for row in normalized if row["cohort"] == cohort]
        ids = sorted({row["sample_id"] for row in cohort_rows})
        for sample_id in ids:
            present = {
                row["regions"]
                for row in cohort_rows
                if row["sample_id"] == sample_id
            }
            if present != expected:
                raise ValueError(
                    f"{cohort} sample {sample_id} lacks the complete candidate family"
                )
    return sorted(
        normalized,
        key=lambda row: (row["cohort"], row["sample_id"], row["regions"]),
    )


def _matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row[name]) for name in FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )


def _labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["safe_success"]) for row in rows], dtype=np.float64)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _ids_digest(values: Iterable[int]) -> str:
    encoded = json.dumps(sorted(set(int(value) for value in values)), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "cohort",
        "sample_id",
        "regions",
        *FEATURE_NAMES,
        *OUTCOME_FIELDS,
        "target_pass",
        "identity_pass",
        "locality_pass",
        "safe_success",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["regions"] = json.dumps(payload["regions"])
            writer.writerow(payload)


def fit_region_selector(
    graph_path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    provenance: Mapping[str, Any],
    discovery_ids: Iterable[int] = (),
    evaluation_ids: Iterable[int] = (),
    candidate_family: Sequence[Iterable[str]] | None = None,
    thresholds: SafeSuccessThresholds = SafeSuccessThresholds(),
) -> FrozenSelectorArtifact:
    """Validate disjoint development data and freeze the selector artifact."""

    _validate_provenance(provenance)
    required_provenance = {
        "feature_signature",
        "classifier_sha256",
        "generation_policy_signature",
    }
    missing_provenance = required_provenance - set(provenance)
    if missing_provenance:
        raise ValueError(f"missing selector provenance: {sorted(missing_provenance)}")
    policy = load_frozen_influence_policy(graph_path)
    candidates = validate_candidate_family(candidate_family, policy)
    declared_graph = provenance.get("influence_graph_sha256")
    if declared_graph is not None and declared_graph != policy.graph_sha256:
        raise ValueError("source feature manifest graph SHA-256 mismatch")
    declared_candidates = provenance.get("candidate_region_sets")
    if declared_candidates is not None and tuple(
        sorted(_parse_regions(item) for item in declared_candidates)
    ) != candidates:
        raise ValueError("source feature manifest candidate family mismatch")
    normalized = _normalize_rows(rows, policy, candidates, thresholds)
    fit_rows = [row for row in normalized if row["cohort"] == "fit"]
    calibration_rows = [row for row in normalized if row["cohort"] == "calibration"]
    fit_ids = {row["sample_id"] for row in fit_rows}
    calibration_ids = {row["sample_id"] for row in calibration_rows}
    if not fit_ids or not calibration_ids:
        raise ValueError("fit and calibration cohorts must both be non-empty")
    cohorts = {
        "discovery": {int(value) for value in discovery_ids},
        "fit": fit_ids,
        "calibration": calibration_ids,
        "evaluation": {int(value) for value in evaluation_ids},
    }
    empty_cohorts = sorted(name for name, values in cohorts.items() if not values)
    if empty_cohorts:
        raise ValueError(
            "all four selector cohorts must be non-empty: "
            f"{empty_cohorts}"
        )
    overlap = _pairwise_cohort_overlap(cohorts)
    if overlap:
        raise ValueError(f"cohort sample IDs must be pairwise disjoint: {overlap}")
    declared_cohorts = provenance.get("declared_cohorts")
    if declared_cohorts is not None:
        frozen_declared = {
            name: {int(value) for value in declared_cohorts.get(name, ())}
            for name in cohorts
        }
        if frozen_declared != cohorts:
            raise ValueError("fit inputs disagree with the frozen split manifest")

    model, cv_audit = choose_grouped_l2(
        _matrix(fit_rows),
        _labels(fit_rows),
        [row["sample_id"] for row in fit_rows],
    )
    calibration_logits = model.predict_logit(_matrix(calibration_rows))
    calibrator = fit_platt_calibrator(calibration_logits, _labels(calibration_rows))
    calibration_scores = calibrator.predict_probability(calibration_logits)
    nonfallback = np.asarray(
        [row["regions"] != policy.fallback_regions for row in calibration_rows]
    )
    risk = choose_risk_threshold(
        calibration_scores[nonfallback], _labels(calibration_rows)[nonfallback]
    )
    artifact = FrozenSelectorArtifact(
        protocol_version=1,
        target=policy.target,
        desired_value=policy.desired_value,
        graph_sha256=policy.graph_sha256,
        candidate_region_sets=candidates,
        fallback_regions=policy.fallback_regions,
        feature_names=FEATURE_NAMES,
        feature_signature=str(provenance["feature_signature"]),
        classifier_sha256=str(provenance["classifier_sha256"]),
        generation_policy_signature=str(
            provenance["generation_policy_signature"]
        ),
        model=model,
        calibrator=calibrator,
        risk_calibration=risk,
        coverage_threshold=0.80,
        safe_success_thresholds=thresholds,
        fit_sample_ids=tuple(fit_ids),
        calibration_sample_ids=tuple(calibration_ids),
        discovery_sample_ids=tuple(cohorts["discovery"]),
        evaluation_sample_ids=tuple(cohorts["evaluation"]),
        provenance=dict(provenance),
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fit_payload = artifact.to_dict()
    model_bytes = _canonical_json_bytes(fit_payload)
    (destination / "selector_model.json").write_bytes(model_bytes)
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    _write_rows(destination / "selector_fit_rows.csv", fit_rows)
    _write_rows(
        destination / "selector_calibration_rows.csv", calibration_rows
    )
    data_manifest = {
        "graph_sha256": policy.graph_sha256,
        "candidate_region_sets": [list(item) for item in candidates],
        "cohorts": {
            name: {
                "sample_ids": sorted(values),
                "sample_ids_sha256": _ids_digest(values),
            }
            for name, values in cohorts.items()
        },
    }
    (destination / "selector_data_manifest.json").write_bytes(
        _canonical_json_bytes(data_manifest)
    )
    calibration_report = {
        "selector_model_sha256": model_sha,
        "risk_calibration": artifact.to_dict()["risk_calibration"],
        "coverage_threshold": artifact.coverage_threshold,
        "safe_success_thresholds": asdict_thresholds(thresholds),
        "cv": {
            "selected_l2": cv_audit.l2,
            "mean_log_loss": cv_audit.mean_log_loss,
            "losses_by_l2": [list(item) for item in cv_audit.losses_by_l2],
        },
    }
    (destination / "selector_calibration_report.json").write_bytes(
        _canonical_json_bytes(calibration_report)
    )
    report = (
        "# Frozen Region Selector\n\n"
        f"- Target: `{artifact.target}` -> `{artifact.desired_value}`\n"
        f"- Fit IDs: `{len(fit_ids)}`\n"
        f"- Calibration IDs: `{len(calibration_ids)}`\n"
        f"- Selected L2: `{model.l2}`\n"
        f"- Risk threshold: `{risk.threshold:.12g}`\n"
        f"- Accepted non-fallback rows: `{risk.accepted}`\n"
        f"- Wilson failure UCB: `{risk.failure_upper_bound:.12g}`\n"
        f"- Fallback only: `{risk.fallback_only}`\n"
        f"- Selector SHA-256: `{model_sha}`\n"
    )
    (destination / "selector_fit_report.md").write_text(
        report, encoding="utf-8"
    )
    return artifact


def asdict_thresholds(thresholds: SafeSuccessThresholds) -> dict[str, float]:
    return {
        "desired_probability": thresholds.desired_probability,
        "identity_distance": thresholds.identity_distance,
        "outside_locality": thresholds.outside_locality,
    }


def provenance_from_manifests(
    source_feature_manifest: str | Path,
    split_manifest: str | Path,
    source_features: str | Path | None = None,
) -> dict[str, Any]:
    """Build fitting provenance from already-frozen source/split artifacts."""

    source_path = Path(source_feature_manifest)
    split_path = Path(split_manifest)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    required = (
        "feature_signature",
        "classifier_sha256",
        "generation_policy_signature",
    )
    missing = [name for name in required if name not in source]
    if missing:
        raise ValueError(
            f"source feature manifest lacks provenance: {missing}"
        )
    raw_cohorts = split.get("cohorts")
    cohort_names = ("discovery", "fit", "calibration", "evaluation")
    if not isinstance(raw_cohorts, dict) or set(raw_cohorts) != set(cohort_names):
        raise ValueError("split manifest must declare exactly four cohorts")
    cohorts = {
        name: tuple(int(value) for value in raw_cohorts[name])
        for name in cohort_names
    }
    if any(not values for values in cohorts.values()):
        raise ValueError("split manifest cohorts must all be non-empty")
    if _pairwise_cohort_overlap({name: set(values) for name, values in cohorts.items()}):
        raise ValueError("split manifest cohorts must be pairwise disjoint")
    declared_source_ids = {int(value) for value in source.get("sample_ids", ())}
    split_ids = {value for values in cohorts.values() for value in values}
    if declared_source_ids != split_ids:
        raise ValueError("source feature IDs disagree with split manifest")
    if source_features is not None:
        expected_digest = source.get("source_features_sha256")
        if expected_digest != sha256_file(source_features):
            raise ValueError("source feature CSV SHA-256 mismatch")
    return {
        name: str(source[name]) for name in required
    } | {
        "source_feature_manifest_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(
            split_path.read_bytes()
        ).hexdigest(),
        "software_versions": {"numpy": np.__version__},
        "declared_cohorts": {name: list(values) for name, values in cohorts.items()},
        "influence_graph_sha256": source.get("influence_graph_sha256"),
        "candidate_region_sets": source.get("candidate_region_sets"),
    }


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def join_source_features_and_outcomes(
    source_features: Iterable[Mapping[str, Any]],
    outcomes: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    feature_index = {}
    for row in source_features:
        key = (int(row["sample_id"]), _parse_regions(row["regions"]))
        if key in feature_index:
            raise ValueError(f"duplicate source feature row: {key}")
        feature_index[key] = row
    joined = []
    for outcome in outcomes:
        key = (int(outcome["sample_id"]), _parse_regions(outcome["regions"]))
        if key not in feature_index:
            raise ValueError(f"development outcome lacks source features: {key}")
        joined.append({**feature_index[key], **outcome, "regions": key[1]})
    return joined


def _read_ids(path: str | Path | None) -> tuple[int, ...]:
    if path is None:
        return ()
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        values = payload.get("sample_ids", payload) if isinstance(payload, dict) else payload
        return tuple(int(value) for value in values)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "sample_id" in rows[0]:
        return tuple(int(row["sample_id"]) for row in rows)
    return tuple(
        int(line.strip())
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--influence_graph", required=True)
    parser.add_argument("--source_features", required=True)
    parser.add_argument("--development_outcomes", required=True)
    parser.add_argument("--source_feature_manifest", default=None)
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--discovery_ids", default=None)
    parser.add_argument("--evaluation_ids", default=None)
    parser.add_argument("--candidate_family", default=None)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.source_feature_manifest is None or args.split_manifest is None:
        raise ValueError(
            "source_feature_manifest and split_manifest are required"
        )
    provenance_payload = provenance_from_manifests(
        args.source_feature_manifest,
        args.split_manifest,
        args.source_features,
    )
    candidate_family = None
    if args.candidate_family:
        candidate_family = json.loads(
            Path(args.candidate_family).read_text(encoding="utf-8")
        )
    rows = join_source_features_and_outcomes(
        _read_csv(args.source_features), _read_csv(args.development_outcomes)
    )
    fit_region_selector(
        args.influence_graph,
        rows,
        args.output_dir,
        provenance=provenance_payload,
        discovery_ids=(
            _read_ids(args.discovery_ids)
            or tuple(provenance_payload.get("declared_cohorts", {}).get("discovery", ()))
        ),
        evaluation_ids=(
            _read_ids(args.evaluation_ids)
            or tuple(provenance_payload.get("declared_cohorts", {}).get("evaluation", ()))
        ),
        candidate_family=candidate_family,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
