#!/usr/bin/env python3
"""Run one target-generic development workflow for any valid data size."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cci_diff.concept_graph import sha256_file  # noqa: E402
from cci_diff.counterfactual_graph import (  # noqa: E402
    InterventionObservation,
    aggregate_region_sets,
    build_influence_graph,
)
from cci_diff.development_cohort import (  # noqa: E402
    DevelopmentCohort,
    assign_development_cohort,
)
from cci_diff.generic_region_discovery import (  # noqa: E402
    BeamSearchConfig,
    advance_beam,
    propose_region_sets,
    shortlist_atomic_components,
)
from cci_diff.region_screening import (  # noqa: E402
    CELEBAMASK_COMPONENT_SUFFIXES,
)


RegionTuple = tuple[str, ...]


class DevelopmentBackend(Protocol):
    """Expensive operations injected behind a deterministic orchestrator."""

    def screen(
        self, *, sample_ids: tuple[int, ...], regions: tuple[str, ...]
    ) -> list[dict[str, Any]]: ...

    def intervene(
        self,
        *,
        sample_ids: tuple[int, ...],
        region_sets: tuple[RegionTuple, ...],
    ) -> tuple[InterventionObservation, ...]: ...

    def extract_features(
        self, *, sample_ids: tuple[int, ...], graph_path: Path
    ) -> Path: ...

    def fit(
        self,
        *,
        graph_path: Path,
        source_features: Path,
        development_outcomes: Path,
        split_manifest: Path,
    ) -> Path: ...


class LocalDevelopmentBackend:
    """Compose the existing tested scripts for local or Kaggle execution."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(args.output_dir) / "backend"
        self.root.mkdir(parents=True, exist_ok=True)
        self.semantic_mask_manifest = (
            Path(args.output_dir) / "semantic_mask_manifest.json"
        )

    def select_eligible_ids(
        self,
        *,
        candidate_ids: tuple[int, ...],
        evaluation_ids: tuple[int, ...],
        data_size: int,
        seed: int,
    ) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
        """Apply target direction, face, source, and semantic-mask checks."""

        import torch
        from PIL import Image

        from cci_diff.classifiers.celeba_resnet50 import (
            load_celeba_resnet50,
            resolve_celeba_attribute_index,
        )
        from cci_diff.concept_graph import load_concept_graph
        from cci_diff.identity.facenet import (
            build_face_detector,
            detect_largest_face_box,
        )
        from cci_diff.region_screening import celebamask_component_path
        from cci_diff.runtime_environment import resolve_device
        from scripts.run_individual_region_cci import source_requires_flip
        from scripts.run_sd2_bld_cci import (
            load_rgb_image_tensor,
            score_classifier_image_grid,
        )

        self.args.device = resolve_device(self.args.device, torch)
        graph = load_concept_graph(self.args.template_graph)
        label_index = resolve_celeba_attribute_index(
            graph.intervention.concept
        )
        classifier = load_celeba_resnet50(
            self.args.classifier_path,
            device=self.args.device,
            dtype=torch.float32,
        )
        detector = build_face_detector()
        excluded = set(evaluation_ids)
        accepted = []
        decisions = []
        for sample_id in candidate_ids:
            decision: dict[str, Any] = {"sample_id": sample_id}
            if sample_id in excluded:
                decision.update(eligible=False, reason="evaluation_exclusion")
                decisions.append(decision)
                continue
            source = Path(self.args.image_root) / f"{sample_id}.jpg"
            if not source.is_file():
                decision.update(eligible=False, reason="missing_source")
                decisions.append(decision)
                continue
            available_regions = []
            for region in CELEBAMASK_COMPONENT_SUFFIXES:
                component = celebamask_component_path(
                    self.args.mask_root, sample_id, region
                )
                if not component.is_file():
                    continue
                with Image.open(component) as image:
                    if image.convert("L").getbbox() is not None:
                        available_regions.append(region)
            if not available_regions:
                decision.update(eligible=False, reason="no_semantic_mask")
                decisions.append(decision)
                continue
            probability = score_classifier_image_grid(
                source,
                classifier=classifier,
                label_index=label_index,
                input_size=self.args.classifier_input_size,
                device=self.args.device,
                batch_size=1,
            )[0]
            direction_ok = source_requires_flip(
                probability, graph.intervention.desired_value
            )
            try:
                face_box = detect_largest_face_box(
                    detector,
                    load_rgb_image_tensor(source, device="cpu"),
                )
                face_ok = True
            except ValueError:
                face_box = None
                face_ok = False
            eligible = direction_ok and face_ok
            decision.update(
                {
                    "eligible": eligible,
                    "reason": "accepted" if eligible else "source_or_face",
                    "source_probability": probability,
                    "face_detected": face_ok,
                    "face_box": list(face_box) if face_box is not None else None,
                    "available_regions": available_regions,
                }
            )
            decisions.append(decision)
            if eligible:
                accepted.append(sample_id)
                try:
                    assign_development_cohort(
                        accepted, evaluation_ids, data_size, seed
                    )
                except ValueError:
                    pass
                else:
                    break
        assign_development_cohort(accepted, evaluation_ids, data_size, seed)
        return tuple(accepted), decisions

    def screen(
        self, *, sample_ids: tuple[int, ...], regions: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        from scripts.screen_counterfactual_regions import screen_regions

        phase_args = _namespace_with(
            self.args,
            sample_ids=list(sample_ids),
            candidate_regions=list(regions),
            max_selected_regions=6,
            output_dir=str(self.root / "screen"),
        )
        screen_regions(phase_args)
        with (self.root / "screen/screening_rows.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            return list(csv.DictReader(handle))

    def intervene(
        self,
        *,
        sample_ids: tuple[int, ...],
        region_sets: tuple[RegionTuple, ...],
    ) -> tuple[InterventionObservation, ...]:
        from scripts.discover_counterfactual_graph import read_observations
        from scripts.run_counterfactual_region_interventions import (
            run_interventions,
        )

        identity = _payload_sha256(
            {
                "sample_ids": sample_ids,
                "region_sets": region_sets,
            }
        )
        phase_dir = self.root / "interventions" / identity
        phase_args = _namespace_with(
            self.args,
            sample_ids=list(sample_ids),
            candidate_regions=None,
            region_sets=["+".join(item) for item in region_sets],
            max_set_size=3,
            seeds=[int(self.args.seed)],
            output_dir=str(phase_dir),
            stop_flip_rate=float(self.args.required_flip_rate),
            disable_early_stop=True,
            continue_on_error=False,
            dry_run=False,
            intervention_cache_dir=self.args.cache_dir,
            generation_policy=self.args.generation_policy,
        )
        run_interventions(phase_args)
        return tuple(
            read_observations(phase_dir / "intervention_results.csv")
        )

    def extract_features(
        self, *, sample_ids: tuple[int, ...], graph_path: Path
    ) -> Path:
        from scripts.run_individual_region_cci import run_individual_cci

        phase_dir = self.root / "source_features"
        phase_args = _namespace_with(
            self.args,
            influence_graph=str(graph_path),
            sample_ids=list(sample_ids),
            coverage_threshold=0.80,
            selector_model=None,
            generation_policy_manifest=self.args.generation_policy,
            semantic_mask_manifest=str(self.semantic_mask_manifest),
            output_dir=str(phase_dir),
            discovery_manifest=None,
            selection_manifest=None,
            selection_only=False,
            source_features_only=True,
            exploratory=False,
            continue_on_error=False,
            dry_run=False,
        )
        run_individual_cci(phase_args)
        return phase_dir / "selector_source_features.csv"

    def fit(
        self,
        *,
        graph_path: Path,
        source_features: Path,
        development_outcomes: Path,
        split_manifest: Path,
    ) -> Path:
        from scripts.fit_region_selector import (
            _read_csv,
            fit_region_selector,
            join_source_features_and_outcomes,
            provenance_from_manifests,
        )

        split = json.loads(split_manifest.read_text(encoding="utf-8"))
        cohorts = split["cohorts"]
        provenance = provenance_from_manifests(
            source_features.parent / "source_feature_manifest.json",
            split_manifest,
            source_features,
        )
        rows = join_source_features_and_outcomes(
            _read_csv(source_features), _read_csv(development_outcomes)
        )
        destination = self.root / "selector"
        fit_region_selector(
            graph_path,
            rows,
            destination,
            provenance=provenance,
            discovery_ids=cohorts["discovery"],
            evaluation_ids=cohorts["evaluation"],
        )
        return destination / "selector_model.json"

def run_development(
    args: argparse.Namespace,
    *,
    backend: DevelopmentBackend,
) -> dict[str, Any]:
    """Run or resume the generic A11 discovery/fit/calibration workflow."""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "development_run.json"
    eligible_ids = tuple(int(value) for value in args.eligible_ids)
    evaluation_ids = tuple(int(value) for value in args.evaluation_ids)
    config = _configuration(args, eligible_ids, evaluation_ids)
    config_signature = _payload_sha256(config)
    existing: dict[str, Any] | None = None
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("configuration_sha256") != config_signature:
            raise ValueError(
                "Existing development run has a different configuration"
            )
        if existing.get("phase") == "complete":
            _validate_complete_manifest(existing, args)
            return existing

    eligibility_decisions = None
    eligibility_selector = getattr(backend, "select_eligible_ids", None)
    if callable(eligibility_selector):
        eligible_ids, eligibility_decisions = eligibility_selector(
            candidate_ids=eligible_ids,
            evaluation_ids=evaluation_ids,
            data_size=int(args.data_size),
            seed=int(args.seed),
        )

    cohort = assign_development_cohort(
        eligible_ids,
        evaluation_ids,
        int(args.data_size),
        int(args.seed),
    )
    evaluation_overlap = sorted(cohort.all_ids.intersection(evaluation_ids))
    if evaluation_overlap:
        raise ValueError("evaluation IDs overlap the development cohort")
    manifest: dict[str, Any] = {
        "version": 1,
        "workflow": "generic_region_development",
        "phase": "cohort",
        "configuration": config,
        "configuration_sha256": config_signature,
        "data_size": cohort.data_size,
        "counts": cohort.counts.to_dict(),
        "cohorts": {
            "discovery": list(cohort.discovery),
            "fit": list(cohort.fit),
            "calibration": list(cohort.calibration),
            "evaluation": list(evaluation_ids),
        },
        "evaluation_overlap": evaluation_overlap,
        "variant": "A11",
        "max_components": 3,
        "special_mode": None,
        "artifacts": {},
    }
    _write_json(manifest_path, manifest)
    split_path = output_dir / "split_manifest.json"
    _write_json(
        split_path,
        {
            "version": 1,
            "data_size": cohort.data_size,
            "seed": cohort.seed,
            "cohorts": manifest["cohorts"],
            "evaluation_outputs_exported_to_selector_data": False,
        },
    )
    _record_artifact(manifest, "split_manifest", split_path)
    if eligibility_decisions is not None:
        eligibility_path = output_dir / "eligibility_decisions.json"
        _write_json(eligibility_path, eligibility_decisions)
        _record_artifact(
            manifest, "eligibility_decisions", eligibility_path
        )
    if hasattr(args, "image_root") and hasattr(args, "mask_root"):
        source_inputs_path = output_dir / "source_inputs.json"
        _write_json(
            source_inputs_path,
            _source_input_payload(cohort, args),
        )
        _record_artifact(manifest, "source_inputs", source_inputs_path)

    semantic_universe = tuple(CELEBAMASK_COMPONENT_SUFFIXES)
    screening_path = output_dir / "screening_rows.json"
    if _valid_recorded_artifact(existing, "screening_rows"):
        screening_rows = json.loads(
            screening_path.read_text(encoding="utf-8")
        )
    else:
        screening_rows = backend.screen(
            sample_ids=cohort.discovery,
            regions=semantic_universe,
        )
        _write_json(screening_path, screening_rows)
    complete_regions = {
        str(row["region"])
        for row in screening_rows
        if len(
            {
                int(item["sample_id"])
                for item in screening_rows
                if str(item["region"]) == str(row["region"])
            }
        )
        == len(cohort.discovery)
    }
    complete_screening_rows = [
        row for row in screening_rows if str(row["region"]) in complete_regions
    ]
    shortlist = shortlist_atomic_components(
        complete_screening_rows, BeamSearchConfig()
    )
    manifest.update(
        {
            "phase": "screen",
            "semantic_universe": list(semantic_universe),
            "atomic_shortlist": list(shortlist),
            "incomplete_atomic_components": sorted(
                set(semantic_universe) - complete_regions
            ),
        }
    )
    _record_artifact(manifest, "screening_rows", screening_path)
    _write_json(manifest_path, manifest)

    search = BeamSearchConfig(
        atomic_shortlist_size=6,
        beam_width=4,
        level_evaluation_budget=6,
        max_components=3,
        minimum_samples=len(cohort.discovery),
    )
    observations_path = output_dir / "discovery_observations.json"
    resumed_levels = (
        list(existing.get("discovery_levels", ())) if existing else []
    )
    if resumed_levels and _valid_recorded_artifact(
        existing, "discovery_observations"
    ):
        discovery_observations = _read_observations_json(observations_path)
        level_audit = resumed_levels
        beam = tuple(
            tuple(item) for item in level_audit[-1]["beam"]
        )
    else:
        beam = ()
        discovery_observations = []
        level_audit = []
    for cardinality in range(
        len(level_audit) + 1, search.max_components + 1
    ):
        proposals = propose_region_sets(
            shortlist, beam, cardinality, search
        )
        level_rows = backend.intervene(
            sample_ids=cohort.discovery,
            region_sets=proposals,
        )
        discovery_observations.extend(level_rows)
        level_evidence = aggregate_region_sets(
            level_rows,
            bootstrap_samples=int(args.bootstrap_samples),
            confidence=float(args.confidence),
            random_seed=int(args.seed) + cardinality,
        )
        beam = advance_beam(level_evidence.values(), search)
        level_audit.append(
            {
                "cardinality": cardinality,
                "proposals": [list(item) for item in proposals],
                "beam": [list(item) for item in beam],
                "evidence": [
                    item.to_dict()
                    for _, item in sorted(level_evidence.items())
                ],
            }
        )
        _write_json(
            observations_path,
            [asdict(item) for item in discovery_observations],
        )
        _record_artifact(
            manifest, "discovery_observations", observations_path
        )
        manifest["phase"] = f"discover_{cardinality}"
        manifest["discovery_levels"] = level_audit
        _write_json(manifest_path, manifest)

    _write_json(
        observations_path,
        [asdict(item) for item in discovery_observations],
    )
    all_evidence = aggregate_region_sets(
        discovery_observations,
        bootstrap_samples=int(args.bootstrap_samples),
        confidence=float(args.confidence),
        random_seed=int(args.seed),
    )
    graph = build_influence_graph(
        target=discovery_observations[0].target,
        desired_value=discovery_observations[0].desired_value,
        evidence_by_regions=all_evidence,
        required_flip_rate=float(args.required_flip_rate),
        minimum_samples=len(cohort.discovery),
        provenance={
            "workflow": "generic_region_development",
            "data_size": cohort.data_size,
            "policy_signature": str(args.policy_signature),
        },
    )
    frozen_candidates = advance_beam(all_evidence.values(), search)
    supported = set(graph.candidate_region_sets)
    supported_candidates = tuple(
        item for item in frozen_candidates if item in supported
    )[: search.beam_width]
    frozen_candidates = tuple(
        dict.fromkeys((*supported_candidates, graph.fallback_regions))
    )
    graph = replace(
        graph,
        candidate_region_sets=frozen_candidates,
        selection_status=(
            "adaptive_candidates_ready"
            if len(supported_candidates) >= 2
            else "fallback_only_insufficient_supported_candidates"
        ),
    )
    graph_path = output_dir / "influence_graph.json"
    _write_json(graph_path, graph.to_dict())
    _record_artifact(manifest, "discovery_observations", observations_path)
    _record_artifact(manifest, "influence_graph", graph_path)
    manifest.update(
        {
            "phase": "freeze_graph",
            "candidate_region_sets": [
                list(item) for item in frozen_candidates
            ],
            "fallback_regions": list(graph.fallback_regions),
            "fallback_only": len(supported_candidates) < 2,
        }
    )
    _write_json(manifest_path, manifest)

    semantic_manifest_path = output_dir / "semantic_mask_manifest.json"
    if hasattr(args, "mask_root"):
        _write_semantic_mask_manifest(
            semantic_manifest_path,
            cohort,
            graph.verified_regions,
            Path(args.mask_root),
            influence_graph_sha256=sha256_file(graph_path),
        )
        _record_artifact(
            manifest, "semantic_mask_manifest", semantic_manifest_path
        )

    development_region_sets = tuple(
        dict.fromkeys((*frozen_candidates, graph.fallback_regions))
    )
    fitting_ids = (*cohort.fit, *cohort.calibration)
    outcomes_path = output_dir / "development_outcomes.csv"
    if not _valid_recorded_artifact(existing, "development_outcomes"):
        development_observations = backend.intervene(
            sample_ids=fitting_ids,
            region_sets=development_region_sets,
        )
        _write_development_outcomes(
            outcomes_path, development_observations, cohort
        )
    _record_artifact(manifest, "development_outcomes", outcomes_path)
    manifest["phase"] = "development_interventions"
    _write_json(manifest_path, manifest)

    if _valid_recorded_artifact(existing, "source_features"):
        source_features = Path(
            existing["artifacts"]["source_features"]["path"]
        )
    else:
        source_features = Path(
            backend.extract_features(
                sample_ids=tuple(sorted(cohort.all_ids)),
                graph_path=graph_path,
            )
        )
    _record_artifact(manifest, "source_features", source_features)
    manifest["phase"] = "source_features"
    _write_json(manifest_path, manifest)

    if _valid_recorded_artifact(existing, "selector_model"):
        selector_path = Path(
            existing["artifacts"]["selector_model"]["path"]
        )
    else:
        selector_path = Path(
            backend.fit(
                graph_path=graph_path,
                source_features=source_features,
                development_outcomes=outcomes_path,
                split_manifest=split_path,
            )
        )
    _record_artifact(manifest, "selector_model", selector_path)
    manifest["phase"] = "fit"
    _write_json(manifest_path, manifest)
    manifest["phase"] = "complete"
    _write_json(manifest_path, manifest)
    return manifest


def _configuration(
    args: argparse.Namespace,
    eligible_ids: tuple[int, ...],
    evaluation_ids: tuple[int, ...],
) -> dict[str, Any]:
    configuration = {
        "data_size": int(args.data_size),
        "seed": int(args.seed),
        "eligible_ids_sha256": _ids_sha256(eligible_ids),
        "evaluation_ids": list(evaluation_ids),
        "policy_signature": str(args.policy_signature),
        "beam": {
            "atomic_shortlist_size": 6,
            "beam_width": 4,
            "level_evaluation_budget": 6,
            "max_components": 3,
        },
        "required_flip_rate": float(args.required_flip_rate),
        "bootstrap_samples": int(args.bootstrap_samples),
        "confidence": float(args.confidence),
    }
    for name in (
        "template_graph",
        "classifier_path",
        "identity_model_path",
        "model_path",
        "image_root",
        "mask_root",
        "cache_dir",
        "device",
        "torch_dtype",
    ):
        if hasattr(args, name):
            configuration[name] = str(getattr(args, name))
    for name in ("template_graph", "classifier_path", "identity_model_path"):
        if hasattr(args, name) and Path(getattr(args, name)).is_file():
            configuration[f"{name}_sha256"] = sha256_file(
                getattr(args, name)
            )
    return configuration


def _write_development_outcomes(
    path: Path,
    observations: Iterable[InterventionObservation],
    cohort: DevelopmentCohort,
) -> None:
    fields = (
        "cohort",
        "sample_id",
        "regions",
        "desired_probability",
        "identity_distance",
        "outside_locality",
    )
    fit_ids = set(cohort.fit)
    calibration_ids = set(cohort.calibration)
    rows = []
    for item in observations:
        role = (
            "fit"
            if item.sample_id in fit_ids
            else "calibration"
            if item.sample_id in calibration_ids
            else None
        )
        if role is None:
            raise ValueError("development observation has an undeclared sample ID")
        rows.append(
            {
                "cohort": role,
                "sample_id": item.sample_id,
                "regions": json.dumps(list(item.regions)),
                "desired_probability": item.output_desired_probability,
                "identity_distance": (
                    ""
                    if item.identity_cosine is None
                    else 1.0 - item.identity_cosine
                ),
                "outside_locality": (
                    "" if item.outside_l1 is None else item.outside_l1
                ),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda row: (
                    row["cohort"], row["sample_id"], row["regions"]
                ),
            )
        )


def _write_semantic_mask_manifest(
    path: Path,
    cohort: DevelopmentCohort,
    verified_regions: tuple[str, ...],
    mask_root: Path,
    *,
    influence_graph_sha256: str,
) -> None:
    from cci_diff.region_screening import celebamask_component_path

    sample_masks = {}
    for sample_id in sorted(cohort.all_ids):
        available = {}
        for region in verified_regions:
            component = celebamask_component_path(
                mask_root, sample_id, region
            )
            if component.is_file():
                available[region] = sha256_file(component)
        if not available:
            raise FileNotFoundError(
                f"No verified semantic mask exists for sample {sample_id}"
            )
        sample_masks[str(sample_id)] = available
    _write_json(
        path,
        {
            "version": 1,
            "influence_graph_sha256": influence_graph_sha256,
            "verified_regions": list(verified_regions),
            "sample_masks": sample_masks,
        },
    )


def _record_artifact(
    manifest: dict[str, Any], name: str, path: Path
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} artifact not found: {path}")
    manifest["artifacts"][name] = {
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _valid_recorded_artifact(
    manifest: Mapping[str, Any] | None, name: str
) -> bool:
    if not manifest:
        return False
    item = (manifest.get("artifacts") or {}).get(name)
    if not isinstance(item, dict):
        return False
    path = Path(str(item.get("path", "")))
    return (
        path.is_file()
        and item.get("sha256") == sha256_file(path)
    )


def _read_observations_json(
    path: Path,
) -> list[InterventionObservation]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("discovery observations must contain a JSON list")
    return [InterventionObservation(**item) for item in payload]


def _validate_complete_manifest(
    manifest: Mapping[str, Any], args: argparse.Namespace
) -> None:
    for name, item in (manifest.get("artifacts") or {}).items():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Completed artifact failed validation: {name}")
    source_item = (manifest.get("artifacts") or {}).get("source_inputs")
    if source_item is not None:
        cohorts = manifest["cohorts"]
        cohort = DevelopmentCohort(
            discovery=tuple(cohorts["discovery"]),
            fit=tuple(cohorts["fit"]),
            calibration=tuple(cohorts["calibration"]),
            data_size=int(manifest["data_size"]),
            seed=int(manifest["configuration"]["seed"]),
        )
        recorded = json.loads(
            Path(source_item["path"]).read_text(encoding="utf-8")
        )
        if recorded != _source_input_payload(cohort, args):
            raise ValueError("Completed source or semantic-mask inputs changed")


def _source_input_payload(
    cohort: DevelopmentCohort, args: argparse.Namespace
) -> dict[str, Any]:
    from cci_diff.region_screening import celebamask_component_path

    samples = {}
    for sample_id in sorted(cohort.all_ids):
        source = Path(args.image_root) / f"{sample_id}.jpg"
        if not source.is_file():
            raise FileNotFoundError(f"Source image not found: {source}")
        components = {}
        for region in CELEBAMASK_COMPONENT_SUFFIXES:
            path = celebamask_component_path(
                args.mask_root, sample_id, region
            )
            components[region] = sha256_file(path) if path.is_file() else None
        samples[str(sample_id)] = {
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "components": components,
        }
    return {"version": 1, "samples": samples}


def _ids_sha256(values: Iterable[int]) -> str:
    return _payload_sha256(sorted(set(int(value) for value in values)))


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _namespace_with(
    args: argparse.Namespace, **updates: Any
) -> argparse.Namespace:
    payload = dict(vars(args))
    payload.update(updates)
    return argparse.Namespace(**payload)


def _read_ids_manifest(path: str | Path) -> tuple[int, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = None
        for key in (
            "sample_ids",
            "selected_sample_ids",
            "eligible_ids",
            "evaluation_ids",
        ):
            if key in payload:
                values = payload[key]
                break
        if values is None:
            raise ValueError(f"ID manifest has no recognized list: {path}")
    else:
        raise ValueError(f"ID manifest must contain a list or object: {path}")
    result = tuple(int(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"ID manifest must be non-empty and unique: {path}")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eligible_ids_manifest", required=True)
    parser.add_argument("--evaluation_ids_manifest", required=True)
    parser.add_argument("--template_graph", required=True)
    parser.add_argument("--generation_policy", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--mask_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--classifier_path", required=True)
    parser.add_argument("--identity_model_path", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=("auto", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--classifier_input_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument(
        "--blending_start_percentage", type=float, default=0.25
    )
    parser.add_argument("--generation_mask_dilation", type=int, default=0)
    parser.add_argument("--generation_mask_feather", type=float, default=3.0)
    parser.add_argument(
        "--python_executable", default=sys.executable
    )
    parser.add_argument(
        "--saliency_coverage_threshold", type=float, default=0.80
    )
    parser.add_argument(
        "--cohort_frequency_threshold", type=float, default=0.90
    )
    parser.add_argument(
        "--minimum_coverage_frequency", type=float, default=0.0
    )
    parser.add_argument(
        "--minimum_captured_saliency", type=float, default=0.02
    )
    parser.add_argument("--required_flip_rate", type=float, default=0.95)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.eligible_ids = _read_ids_manifest(args.eligible_ids_manifest)
    args.evaluation_ids = _read_ids_manifest(args.evaluation_ids_manifest)
    args.policy_signature = sha256_file(args.generation_policy)
    args.allow_model_download = False
    run_development(args, backend=LocalDevelopmentBackend(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
