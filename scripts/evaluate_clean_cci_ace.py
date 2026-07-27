#!/usr/bin/env python3
"""Evaluate selected clean-CCI results with independent ACE checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


TARGETS = {
    "smile": {"index": 31, "desired_value": 0},
    "hair": {"index": 9, "desired_value": 1},
}
SUMMARY_FIELDS = (
    "desired_probability",
    "fva_cosine",
    "fs_cosine",
    "mnac",
    "changed_fraction_1",
    "changed_fraction_5",
    "changed_fraction_10",
    "outside_semantic_fraction_5",
    "outside_generation_fraction_5",
    "inside_semantic_l1",
    "outside_semantic_l1",
)


def directional_target_metrics(
    source_probabilities: np.ndarray,
    output_probabilities: np.ndarray,
    target_indices: np.ndarray,
    desired_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute directional target success and source-to-output target flips."""

    source_probabilities = np.asarray(source_probabilities, dtype=float)
    output_probabilities = np.asarray(output_probabilities, dtype=float)
    target_indices = np.asarray(target_indices, dtype=int)
    desired_values = np.asarray(desired_values, dtype=int)
    if source_probabilities.shape != output_probabilities.shape:
        raise ValueError("Source and output probability arrays must have equal shape")
    if source_probabilities.ndim != 2:
        raise ValueError("Probability arrays must be two-dimensional")
    if len(target_indices) != len(source_probabilities):
        raise ValueError("One target index is required per image pair")
    row_indices = np.arange(len(target_indices))
    source_target = source_probabilities[row_indices, target_indices]
    output_target = output_probabilities[row_indices, target_indices]
    desired_probability = np.where(
        desired_values == 1,
        output_target,
        1.0 - output_target,
    )
    source_binary = source_target >= 0.5
    output_binary = output_target >= 0.5
    target_success = output_binary == desired_values.astype(bool)
    return {
        "source_target_probability": source_target,
        "output_target_probability": output_target,
        "desired_probability": desired_probability,
        "target_success": target_success,
        "directional_flip": np.logical_and(source_binary != output_binary, target_success),
    }


def collateral_flips(
    source_binary: np.ndarray,
    output_binary: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Count changed binary attributes after excluding each intended target."""

    source_binary = np.asarray(source_binary, dtype=bool)
    output_binary = np.asarray(output_binary, dtype=bool)
    target_indices = np.asarray(target_indices, dtype=int)
    if source_binary.shape != output_binary.shape or source_binary.ndim != 2:
        raise ValueError("Binary attribute arrays must be aligned two-dimensional arrays")
    changed = source_binary != output_binary
    changed[np.arange(len(changed)), target_indices] = False
    return changed.sum(axis=1)


def paired_cosine_similarity(source: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for corresponding source/output embeddings."""

    source = np.asarray(source, dtype=float).reshape(len(source), -1)
    output = np.asarray(output, dtype=float).reshape(len(output), -1)
    if source.shape != output.shape:
        raise ValueError("Source and output embeddings must have equal shape")
    denominator = np.linalg.norm(source, axis=1) * np.linalg.norm(output, axis=1)
    numerator = np.sum(source * output, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def correlation_difference(
    source_binary: np.ndarray,
    output_binary: np.ndarray,
    target_index: int,
) -> float:
    """Compute ACE CD as absolute correlations with the target change vector."""

    source_binary = np.asarray(source_binary, dtype=float)
    output_binary = np.asarray(output_binary, dtype=float)
    if source_binary.shape != output_binary.shape or source_binary.ndim != 2:
        raise ValueError("Binary attribute arrays must be aligned two-dimensional arrays")
    deltas = output_binary - source_binary
    target_delta = deltas[:, target_index]
    correlations = []
    for index in range(deltas.shape[1]):
        if np.std(deltas[:, index]) == 0 or np.std(target_delta) == 0:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(deltas[:, index], target_delta)[0, 1]))
    return float(np.sum(np.abs(correlations)))


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    iterations: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for the mean."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def _metric_summary(
    rows: Sequence[dict[str, Any]],
    field: str,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    values = np.asarray(
        [float(row[field]) for row in rows if row.get(field) not in (None, "")],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "ci95": None}
    low, high = bootstrap_mean_interval(values, seed=seed, iterations=iterations)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [low, high],
    }


def summarize_task_rows(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_iterations: int = 10_000,
) -> dict[str, Any]:
    """Summarize all pairs and the target-success subset separately."""

    successful = [row for row in rows if bool(row.get("target_success"))]
    summary = {
        "count": len(rows),
        "target_success_count": len(successful),
        "fr": len(successful) / len(rows) if rows else 0.0,
        "unconditional": {},
        "target_success_conditioned": {},
    }
    for offset, field in enumerate(SUMMARY_FIELDS):
        summary["unconditional"][field] = _metric_summary(
            rows,
            field,
            seed=bootstrap_seed + offset,
            iterations=bootstrap_iterations,
        )
        summary["target_success_conditioned"][field] = _metric_summary(
            successful,
            field,
            seed=bootstrap_seed + 100 + offset,
            iterations=bootstrap_iterations,
        )
    if rows and all(row.get("fva_cosine") is not None for row in rows):
        summary["fva_rate"] = float(
            np.mean([float(row["fva_cosine"]) > 0.5 for row in rows])
        )
    else:
        summary["fva_rate"] = None
    return summary


def group_variant_task_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Group a mixed ablation table without pooling variants or tasks."""

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        variant = str(row.get("variant") or "unknown")
        feature = str(row["feature"])
        grouped.setdefault(variant, {}).setdefault(feature, []).append(row)
    return grouped


def _read_selected_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Selected result table not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric_fields = set(SUMMARY_FIELDS) | {
        "sample_id",
        "dilation",
        "selected_dilation",
        "identity_cosine",
    }
    for row in rows:
        for field in numeric_fields:
            if row.get(field) not in (None, ""):
                row[field] = float(row[field])
    return rows


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _image_batch(paths: Sequence[str], *, mode: str):
    import torch
    from PIL import Image

    tensors = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
        values = np.asarray(image, dtype=np.float32)
        if mode == "vggface":
            values = values[:, :, ::-1].copy()
            values -= np.array([91.4953, 103.8827, 131.0912], dtype=np.float32)
        else:
            values /= 255.0
            if mode == "oracle":
                values = (values - 0.5) / 0.5
            elif mode == "simsiam":
                values = (
                    values - np.array([0.485, 0.456, 0.406], dtype=np.float32)
                ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensors.append(torch.from_numpy(values.transpose(2, 0, 1)))
    return torch.stack(tensors)


def _paired_model_outputs(
    model,
    rows: Sequence[dict[str, Any]],
    *,
    mode: str,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    source_outputs = []
    output_outputs = []
    with torch.no_grad():
        for batch in _chunks(rows, batch_size):
            sources = _image_batch([row["source_path"] for row in batch], mode=mode).to(device)
            outputs = _image_batch([row["output_path"] for row in batch], mode=mode).to(device)
            source_value = model(sources)
            output_value = model(outputs)
            if isinstance(source_value, (tuple, list)):
                source_value = source_value[0]
                output_value = output_value[0]
            source_outputs.append(source_value.detach().cpu().numpy().reshape(len(batch), -1))
            output_outputs.append(output_value.detach().cpu().numpy().reshape(len(batch), -1))
    return np.concatenate(source_outputs), np.concatenate(output_outputs)


def _load_oracle(ace_root: Path, device: str):
    import torch

    sys.path.insert(0, str(ace_root))
    from eval_utils.oracle_celebahq_metrics import OracleResnet

    model = OracleResnet(weights_path=None, freeze_layers=True)
    checkpoint = torch.load(ace_root / "models" / "checkpoint.tar", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def _load_vggface(ace_root: Path, device: str):
    sys.path.insert(0, str(ace_root))
    from eval_utils.resnet50_facevgg2_FVA import load_state_dict, resnet50

    model = resnet50(num_classes=8631, include_top=False)
    load_state_dict(model, ace_root / "pretrained_models" / "resnet50_ft_weight.pkl")
    return model.to(device).eval()


def _load_simsiam(ace_root: Path, device: str):
    sys.path.insert(0, str(ace_root))
    from eval_utils.simsiam import get_simsiam_dist

    model = get_simsiam_dist(
        ace_root / "pretrained_models" / "checkpoint_0099.pth.tar"
    )
    return model.encoder.to(device).eval()


def _release_model(model, device: str) -> None:
    del model
    if device == "mps":
        import torch

        torch.mps.empty_cache()


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _exploratory_fid(rows, experiment_root: Path, device: str, batch_size: int):
    results = {}
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths

        for feature in sorted({row["feature"] for row in rows}):
            selected = [row for row in rows if row["feature"] == feature]
            with tempfile.TemporaryDirectory(dir=experiment_root) as temp:
                source_dir = Path(temp) / "source"
                output_dir = Path(temp) / "output"
                source_dir.mkdir()
                output_dir.mkdir()
                for index, row in enumerate(selected):
                    shutil.copyfile(row["source_path"], source_dir / f"{index:05d}.png")
                    shutil.copyfile(row["output_path"], output_dir / f"{index:05d}.png")
                results[feature] = float(
                    calculate_fid_given_paths(
                        [str(source_dir), str(output_dir)],
                        batch_size,
                        device,
                        2048,
                        0,
                    )
                )
    except Exception as error:
        results["error"] = f"{type(error).__name__}: {error}"
        results["retry"] = (
            "Install pytorch-fid and rerun scripts/evaluate_clean_cci_ace.py "
            "with network access if Inception weights are not cached."
        )
    return results


def _write_report(path: Path, summaries: dict[str, Any], fid: dict[str, Any]) -> None:
    lines = [
        "# A3 Clean-CCI ACE Evaluation",
        "",
        "## Prior Paper Results",
        "",
        "| Method | FID down | FVA up | FS up | MNAC down | CD down | COUT up | FR up |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| DiVE | 107.0 | 35.7 | - | 7.41 | - | - | - |",
        "| STEEX | 21.9 | 97.6 | - | 5.27 | - | - | - |",
        "| DiME | 18.1 | 96.7 | 0.67 | 2.63 | 1.82 | 0.65 | - |",
        "| ACE | 3.21 | 100.0 | 0.89 | 1.56 | 2.61 | 0.55 | - |",
        "| TIME | 10.98 | 96.6 | 0.79 | 2.97 | 2.32 | 0.63 | - |",
        "| ECED | 7.56 | 100.0 | 0.90 | 1.32 | 5.08 | 0.84 | 100.0 |",
        "| Previous method | 26.7 | 100.0 | 0.84 | 1.91 | 3.84 | 0.43 | 98.0 |",
        "",
        "## A3 Results",
        "",
        "| Task | N | FR | FVA | FS | MNAC | CD | FID |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for feature, summary in summaries.items():
        unconditional = summary["unconditional"]
        fid_value = fid.get(feature, "N/A")
        lines.append(
            f"| {feature} | {summary['count']} | {100 * summary['fr']:.1f} | "
            f"{100 * (summary['fva_rate'] or 0):.1f} | "
            f"{unconditional['fs_cosine']['mean']} | "
            f"{unconditional['mnac']['mean']} | {summary['cd']} | {fid_value} |"
        )
    lines.extend(
        [
            "",
            "## Protocol Notes",
            "",
            "- FVA and FS compare each source embedding with its paired counterfactual embedding.",
            "- MNAC excludes the intended target attribute and therefore measures collateral flips only.",
            "- FR is directional target success under the independent ACE oracle.",
            "- Spatial metrics are reported unconditionally and conditioned on oracle target success.",
            "- COUT is unavailable because the local binary-attribute mapping is not valid for this protocol.",
            "- FID is exploratory at 100 images per task and is not directly comparable to larger paper test sets.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ablation_report(
    path: Path,
    summaries: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# CCI Component Ablation: Independent ACE Evaluation",
        "",
        "| Variant | Task | N | FR | FVA | FS | MNAC | CD | Desired probability |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, tasks in sorted(summaries.items()):
        for feature, summary in sorted(tasks.items()):
            unconditional = summary["unconditional"]
            lines.append(
                f"| {variant} | {feature} | {summary['count']} | "
                f"{100 * summary['fr']:.1f} | {100 * (summary['fva_rate'] or 0):.1f} | "
                f"{unconditional['fs_cosine']['mean']:.4f} | "
                f"{unconditional['mnac']['mean']:.4f} | {summary['cd']:.4f} | "
                f"{unconditional['desired_probability']['mean']:.4f} |"
            )
    lines.extend(
        [
            "",
            "FR is directional target success under the independent ACE oracle. "
            "All rows remain separated by task and ablation variant.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    experiment_root = Path(args.experiment_root)
    ace_root = Path(args.ace_root)
    rows = _read_selected_rows(experiment_root / "pilot_results.csv")
    if not rows:
        raise ValueError("No selected rows found for ACE evaluation")

    target_indices = np.array([TARGETS[row["feature"]]["index"] for row in rows])
    desired_values = np.array([TARGETS[row["feature"]]["desired_value"] for row in rows])
    oracle = _load_oracle(ace_root, args.device)
    source_probs, output_probs = _paired_model_outputs(
        oracle,
        rows,
        mode="oracle",
        device=args.device,
        batch_size=args.batch_size,
    )
    target = directional_target_metrics(
        source_probs,
        output_probs,
        target_indices,
        desired_values,
    )
    mnac = collateral_flips(source_probs >= 0.5, output_probs >= 0.5, target_indices)
    _release_model(oracle, args.device)

    vggface = _load_vggface(ace_root, args.device)
    source_fva, output_fva = _paired_model_outputs(
        vggface,
        rows,
        mode="vggface",
        device=args.device,
        batch_size=args.batch_size,
    )
    fva = paired_cosine_similarity(source_fva, output_fva)
    _release_model(vggface, args.device)

    simsiam = _load_simsiam(ace_root, args.device)
    source_fs, output_fs = _paired_model_outputs(
        simsiam,
        rows,
        mode="simsiam",
        device=args.device,
        batch_size=args.batch_size,
    )
    fs = paired_cosine_similarity(source_fs, output_fs)
    _release_model(simsiam, args.device)

    for index, row in enumerate(rows):
        row.update(
            {
                "oracle_source_probability": float(target["source_target_probability"][index]),
                "oracle_output_probability": float(target["output_target_probability"][index]),
                "desired_probability": float(target["desired_probability"][index]),
                "target_success": bool(target["target_success"][index]),
                "directional_flip": bool(target["directional_flip"][index]),
                "mnac": int(mnac[index]),
                "fva_cosine": float(fva[index]),
                "fs_cosine": float(fs[index]),
            }
        )

    grouped = group_variant_task_rows(rows)
    variant_summaries = {}
    for variant, tasks in sorted(grouped.items()):
        variant_summaries[variant] = {}
        for feature, task_rows in sorted(tasks.items()):
            summary = summarize_task_rows(
                task_rows,
                bootstrap_seed=args.bootstrap_seed,
            )
            row_ids = {id(row) for row in task_rows}
            indices = [index for index, row in enumerate(rows) if id(row) in row_ids]
            summary["cd"] = correlation_difference(
                (source_probs[indices] >= 0.5),
                (output_probs[indices] >= 0.5),
                TARGETS[feature]["index"],
            )
            variant_summaries[variant][feature] = summary

    _write_csv(rows, experiment_root / "ace_pair_metrics.csv")
    summary_rows = []
    for variant, tasks in sorted(variant_summaries.items()):
        for feature, summary in sorted(tasks.items()):
            summary_rows.append(
                {
                    "variant": variant,
                    "feature": feature,
                    "count": summary["count"],
                    "fr": summary["fr"],
                    "fva_rate": summary["fva_rate"],
                    "fs": summary["unconditional"]["fs_cosine"]["mean"],
                    "mnac": summary["unconditional"]["mnac"]["mean"],
                    "cd": summary["cd"],
                    "desired_probability": summary["unconditional"][
                        "desired_probability"
                    ]["mean"],
                    "changed_fraction_5": summary["unconditional"][
                        "changed_fraction_5"
                    ]["mean"],
                }
            )
    _write_csv(summary_rows, experiment_root / "ace_task_summary.csv")
    if len(variant_summaries) == 1:
        summaries = next(iter(variant_summaries.values()))
        fid = _exploratory_fid(rows, experiment_root, args.device, args.batch_size)
        payload = {
            "tasks": summaries,
            "variants": variant_summaries,
            "fid": fid,
            "cout": None,
        }
    else:
        fid = {
            "note": "Mixed variants are evaluated separately by evaluate_fid_sfid.py."
        }
        payload = {"variants": variant_summaries, "fid": fid, "cout": None}
    (experiment_root / "ace_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if len(variant_summaries) == 1:
        _write_report(experiment_root / "ace_paper_comparison.md", summaries, fid)
    else:
        _write_ablation_report(
            experiment_root / "ace_component_ablation.md", variant_summaries
        )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--ace_root", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    return parser


def main() -> int:
    evaluate(build_arg_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
