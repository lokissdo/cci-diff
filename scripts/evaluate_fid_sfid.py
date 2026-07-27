#!/usr/bin/env python3
"""Compute deterministic FID and symmetric FID for paired experiments."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


FULL_METRIC_FIELDS = (
    "target_accuracy",
    "directional_fr",
    "same_classifier_fr_05",
    "strong_target_rate_08",
    "desired_probability",
    "fva_rate",
    "fva_cosine",
    "fs",
    "mnac",
    "cd",
    "changed_fraction_5",
    "outside_semantic_fraction_5",
    "outside_generation_fraction_5",
    "inside_semantic_l1",
    "outside_semantic_l1",
    "median_runtime_seconds",
)


def _resolved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def load_experiment_rows(
    root: Path,
    *,
    expected_count: int = 100,
    variant: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load a unique, complete smile/hair cohort, optionally for one variant."""

    table = Path(root) / "ace_pair_metrics.csv"
    if not table.is_file():
        raise FileNotFoundError(f"missing experiment table: {table}")
    with table.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, Any]]] = {"smile": [], "hair": []}
    seen: set[tuple[str, int]] = set()
    for row in raw_rows:
        if variant is not None and row.get("variant") != variant:
            continue
        feature = row.get("feature")
        if feature not in grouped:
            continue
        sample_id = int(float(row["sample_id"]))
        key = (feature, sample_id)
        if key in seen:
            raise ValueError(f"duplicate feature/sample pair: {key}")
        seen.add(key)
        source_path = _resolved_path(row["source_path"])
        output_path = _resolved_path(row["output_path"])
        if not source_path.is_file() or not output_path.is_file():
            raise ValueError(f"missing source/output image for {key}")
        normalized = dict(row)
        normalized.update(
            {
                "sample_id": sample_id,
                "source_path": source_path,
                "output_path": output_path,
            }
        )
        grouped[feature].append(normalized)
    for feature, rows in grouped.items():
        rows.sort(key=lambda row: row["sample_id"])
        if len(rows) != expected_count:
            raise ValueError(
                f"{feature} must contain exactly {expected_count} rows; found {len(rows)}"
            )
    return grouped


def validate_aligned_cohorts(
    experiments: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> None:
    """Require identical feature/sample cohorts across experiments."""

    expected: dict[str, tuple[int, ...]] | None = None
    for name, grouped in experiments.items():
        current = {
            feature: tuple(int(row["sample_id"]) for row in rows)
            for feature, rows in grouped.items()
        }
        if expected is None:
            expected = current
        elif current != expected:
            raise ValueError(f"experiment cohort mismatch for {name}")


def _path_fingerprint(paths: Sequence[Path]) -> str:
    records = []
    for value in paths:
        path = _resolved_path(value)
        stat = path.stat()
        records.append(
            {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        )
    return json.dumps(records, sort_keys=True, separators=(",", ":"))


def extract_or_load_activations(
    paths: Sequence[Path],
    cache_path: Path,
    extractor: Callable[[Sequence[Path]], np.ndarray],
) -> np.ndarray:
    """Reuse activations only when every source path fingerprint matches."""

    paths = [_resolved_path(path) for path in paths]
    fingerprint = _path_fingerprint(paths)
    cache_path = Path(cache_path)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["fingerprint"].item()) == fingerprint:
                activations = np.asarray(cached["activations"])
                if len(activations) == len(paths) and np.isfinite(activations).all():
                    return activations
    activations = np.asarray(extractor(paths))
    if activations.ndim != 2 or len(activations) != len(paths):
        raise ValueError("extractor returned an invalid activation shape")
    if not np.isfinite(activations).all():
        raise ValueError("extractor returned non-finite activations")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        activations=activations,
        fingerprint=np.asarray(fingerprint),
    )
    return activations


def split_indices(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, equal, disjoint split indices."""

    if count < 4 or count % 2:
        raise ValueError("sample count must be even and at least four")
    shuffled = np.random.default_rng(seed).permutation(count)
    midpoint = count // 2
    return shuffled[:midpoint], shuffled[midpoint:]


def _validate_activations(source: np.ndarray, output: np.ndarray) -> None:
    if source.ndim != 2 or output.ndim != 2 or source.shape != output.shape:
        raise ValueError("activation arrays must have equal two-dimensional shape")
    if len(source) < 2:
        raise ValueError("at least two activations are required")
    if not np.isfinite(source).all() or not np.isfinite(output).all():
        raise ValueError("activation arrays must contain only finite values")


def fid_from_activations(source: np.ndarray, output: np.ndarray) -> float:
    """Calculate Frechet distance from aligned activation matrices."""

    from pytorch_fid.fid_score import calculate_frechet_distance

    source = np.asarray(source, dtype=np.float64)
    output = np.asarray(output, dtype=np.float64)
    _validate_activations(source, output)
    return float(
        calculate_frechet_distance(
            source.mean(axis=0),
            np.cov(source, rowvar=False),
            output.mean(axis=0),
            np.cov(output, rowvar=False),
        )
    )


def fid_sfid_from_activations(
    source: np.ndarray,
    output: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Compute standard FID and deterministic cross-split symmetric FID."""

    source = np.asarray(source)
    output = np.asarray(output)
    _validate_activations(source, output)
    split_1, split_2 = split_indices(len(source), seed)
    sfid_1 = fid_from_activations(source[split_1], output[split_2])
    sfid_2 = fid_from_activations(source[split_2], output[split_1])
    return {
        "fid": fid_from_activations(source, output),
        "sfid_1": sfid_1,
        "sfid_2": sfid_2,
        "sfid": (sfid_1 + sfid_2) / 2.0,
        "seed": seed,
        "split_1_count": len(split_1),
        "split_2_count": len(split_2),
    }


def _truth(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def runtime_summary_value(
    pilot_summary: Mapping[str, Any], task: str, variant: str
) -> float | None:
    """Return controller runtime when the variant records one."""

    value = pilot_summary["features"][task]["variants"][variant].get(
        "median_runtime"
    )
    return None if value in (None, "") else float(value)


def summarize_pair_rows(
    ace_rows: Sequence[Mapping[str, Any]],
    pilot_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Summarize counterfactual metrics without conflating accuracy and FR."""

    if not ace_rows or len(ace_rows) != len(pilot_rows):
        raise ValueError("ACE and generation rows must be non-empty and aligned")
    generation_probabilities = [float(row["desired_probability"]) for row in pilot_rows]
    return {
        "target_accuracy": float(np.mean([_truth(row["target_success"]) for row in ace_rows])),
        "directional_fr": float(np.mean([_truth(row["directional_flip"]) for row in ace_rows])),
        "same_classifier_fr_05": float(np.mean([value >= 0.5 for value in generation_probabilities])),
        "strong_target_rate_08": float(np.mean([value >= 0.8 for value in generation_probabilities])),
        "desired_probability": _mean(ace_rows, "desired_probability"),
        "fva_cosine": _mean(ace_rows, "fva_cosine"),
        "fs": _mean(ace_rows, "fs_cosine"),
        "mnac": _mean(ace_rows, "mnac"),
        "changed_fraction_5": _mean(ace_rows, "changed_fraction_5"),
        "outside_semantic_fraction_5": _mean(ace_rows, "outside_semantic_fraction_5"),
        "outside_generation_fraction_5": _mean(ace_rows, "outside_generation_fraction_5"),
        "inside_semantic_l1": _mean(ace_rows, "inside_semantic_l1"),
        "outside_semantic_l1": _mean(ace_rows, "outside_semantic_l1"),
    }


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: Any) -> str:
    return f"{100 * float(value):.1f}"


def write_reports(
    fid_rows: Sequence[Mapping[str, Any]],
    full_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    metadata: Mapping[str, Any],
) -> None:
    """Write machine-readable results and readable complete metric tables."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(fid_rows, output_dir / "fid_sfid_metrics.csv")
    _write_csv(full_rows, output_dir / "full_metrics.csv")
    payload = {"metadata": dict(metadata), "metrics": list(fid_rows), "full_metrics": list(full_rows)}
    (output_dir / "fid_sfid_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    fid_lines = [
        "# Deterministic FID and sFID",
        "",
        "These 100-image estimates are exploratory and are not directly comparable to paper values computed with a different sample count or split.",
        "",
        "| Method | Steps | Task | N | FID down | sFID down | sFID 1 | sFID 2 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in fid_rows:
        fid_lines.append(
            f"| {row['method']} | {row['steps']} | {row['task']} | {row['n']} | "
            f"{float(row['fid']):.4f} | {float(row['sfid']):.4f} | "
            f"{float(row['sfid_1']):.4f} | {float(row['sfid_2']):.4f} |"
        )
    (output_dir / "fid_sfid_comparison.md").write_text(
        "\n".join(fid_lines) + "\n", encoding="utf-8"
    )

    lines = [
        "# Full Counterfactual Metrics",
        "",
        "FID and sFID use 100 images per task and are exploratory. Target accuracy and directional FR are intentionally separate.",
        "",
        "## Counterfactual Success",
        "",
        "| Method | Steps | Task | Target accuracy | Directional FR | Same-classifier FR | Strong target | Desired probability |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in full_rows:
        runtime = row["median_runtime_seconds"]
        runtime_text = "-" if runtime is None else f"{float(runtime):.2f}"
        lines.append(
            f"| {row['method']} | {row['steps']} | {row['task']} | "
            f"{_percent(row['target_accuracy'])} | {_percent(row['directional_fr'])} | "
            f"{_percent(row['same_classifier_fr_05'])} | {_percent(row['strong_target_rate_08'])} | "
            f"{float(row['desired_probability']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Preservation and Collateral Change",
            "",
            "| Method | Steps | Task | FVA | FVA cosine | FS | MNAC | CD |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in full_rows:
        lines.append(
            f"| {row['method']} | {row['steps']} | {row['task']} | {_percent(row['fva_rate'])} | "
            f"{float(row['fva_cosine']):.4f} | {float(row['fs']):.4f} | "
            f"{float(row['mnac']):.4f} | {float(row['cd']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Distribution, Locality, and Runtime",
            "",
            "| Method | Steps | Task | FID | sFID | Changed % | Outside semantic % | Outside generation % | Inside L1 | Outside L1 | Runtime s |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in full_rows:
        lines.append(
            f"| {row['method']} | {row['steps']} | {row['task']} | {float(row['fid']):.4f} | "
            f"{float(row['sfid']):.4f} | {_percent(row['changed_fraction_5'])} | "
            f"{_percent(row['outside_semantic_fraction_5'])} | {_percent(row['outside_generation_fraction_5'])} | "
            f"{float(row['inside_semantic_l1']):.4f} | {float(row['outside_semantic_l1']):.4f} | "
            f"{runtime_text} |"
        )
    (output_dir / "full_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _version(name: str) -> str:
    return importlib.metadata.version(name)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run cached Inception extraction and write all requested reports."""

    import torch
    from pytorch_fid.fid_score import get_activations
    from pytorch_fid.inception import InceptionV3

    configurations = []
    for method, steps, root in args.experiment:
        configurations.append((method, int(steps), Path(root)))
    if len(configurations) < 2 or len({(m, s) for m, s, _ in configurations}) != len(
        configurations
    ):
        raise ValueError("at least two unique method/step experiments are required")
    experiments = {
        f"{method}_{steps}": load_experiment_rows(
            root,
            expected_count=args.expected_count,
            variant=method.upper() if method.upper().startswith("A") else None,
        )
        for method, steps, root in configurations
    }
    validate_aligned_cohorts(experiments)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "features"
    device = torch.device(args.device)
    block = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block]).to(device).eval()

    def extractor(paths: Sequence[Path]) -> np.ndarray:
        return get_activations(
            [str(path) for path in paths],
            model,
            batch_size=args.batch_size,
            dims=args.dims,
            device=device,
            num_workers=args.num_workers,
        )

    first_grouped = next(iter(experiments.values()))
    source_activations = {}
    for task in ("smile", "hair"):
        source_activations[task] = extract_or_load_activations(
            [row["source_path"] for row in first_grouped[task]],
            cache_dir / f"source_{task}.npz",
            extractor,
        )

    fid_rows = []
    full_rows = []
    for method, steps, root in configurations:
        grouped = experiments[f"{method}_{steps}"]
        pilot_rows = _read_csv(root / "pilot_results.csv")
        requested_variant = method.upper() if method.upper().startswith("A") else None
        if requested_variant is not None:
            pilot_rows = [
                row for row in pilot_rows if row.get("variant") == requested_variant
            ]
        pilot_by_task = {
            task: sorted(
                [row for row in pilot_rows if row["feature"] == task],
                key=lambda row: int(float(row["sample_id"])),
            )
            for task in ("smile", "hair")
        }
        ace_payload = json.loads((root / "ace_metrics.json").read_text(encoding="utf-8"))
        pilot_summary = json.loads((root / "pilot_summary.json").read_text(encoding="utf-8"))
        variant = requested_variant or ("A0" if method.upper() == "BLD" else "A3")
        for task in ("smile", "hair"):
            output_activations = extract_or_load_activations(
                [row["output_path"] for row in grouped[task]],
                cache_dir / f"output_{method.lower()}_{steps}_{task}.npz",
                extractor,
            )
            metric = fid_sfid_from_activations(
                source_activations[task], output_activations, seed=args.seed
            )
            fid_row = {
                "method": method.upper(),
                "steps": steps,
                "task": task,
                "n": len(grouped[task]),
                **metric,
                "dims": args.dims,
                "device": args.device,
                "experiment_root": str(root),
            }
            fid_rows.append(fid_row)
            summary = summarize_pair_rows(grouped[task], pilot_by_task[task])
            task_payload = (
                ace_payload["variants"][variant][task]
                if "variants" in ace_payload
                else ace_payload["tasks"][task]
            )
            summary.update(
                {
                    "fva_rate": float(task_payload["fva_rate"]),
                    "cd": float(task_payload["cd"]),
                    "median_runtime_seconds": runtime_summary_value(
                        pilot_summary, task, variant
                    ),
                }
            )
            full_rows.append({**fid_row, **summary})

    fid_rows.sort(key=lambda row: (row["method"], row["steps"], row["task"]))
    full_rows.sort(key=lambda row: (row["method"], row["steps"], row["task"]))
    metadata = {
        "seed": args.seed,
        "dims": args.dims,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "versions": {
            name: _version(name)
            for name in ("pytorch-fid", "torch", "torchvision", "numpy", "scipy", "Pillow")
        },
    }
    write_reports(fid_rows, full_rows, output_dir, metadata)
    return {"metadata": metadata, "metrics": fid_rows, "full_metrics": full_rows}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        action="append",
        nargs=3,
        metavar=("METHOD", "STEPS", "ROOT"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dims", type=int, default=2048, choices=(64, 192, 768, 2048))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-count", type=int, default=100)
    return parser


def main() -> int:
    evaluate(build_arg_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
