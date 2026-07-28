"""Pure helpers for deterministic multi-device experiment sharding."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence


def resolve_worker_devices(
    device: str,
    max_workers: int,
    *,
    cuda_device_count: int,
) -> tuple[str, ...]:
    """Resolve a user device request to independent worker devices."""

    if max_workers < 0:
        raise ValueError("max_workers must be non-negative")
    if device in {"mps", "cpu"}:
        return (device,)
    if device.startswith("cuda:"):
        try:
            index = int(device.split(":", 1)[1])
        except ValueError as error:
            raise ValueError(f"Invalid CUDA device: {device}") from error
        if index < 0 or index >= cuda_device_count:
            raise RuntimeError(f"CUDA device {device} is not available")
        return (device,)
    if device != "cuda":
        raise ValueError("device must be cuda, cuda:N, mps, or cpu")
    if cuda_device_count <= 0:
        raise RuntimeError("CUDA was requested but no CUDA devices are available")
    worker_count = cuda_device_count
    if max_workers:
        worker_count = min(worker_count, max_workers)
    return tuple(f"cuda:{index}" for index in range(worker_count))


def partition_ids(
    sample_ids: Sequence[int],
    worker_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition sorted unique IDs round-robin without empty shards."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    ordered = tuple(sorted(int(sample_id) for sample_id in sample_ids))
    if len(ordered) != len(set(ordered)):
        raise ValueError("sample_ids must be unique")
    if not ordered:
        return ()
    shard_count = min(worker_count, len(ordered))
    shards = [[] for _ in range(shard_count)]
    for index, sample_id in enumerate(ordered):
        shards[index % shard_count].append(sample_id)
    return tuple(tuple(shard) for shard in shards)


def merge_csv_files(inputs: Iterable[str | Path], output: str | Path) -> int:
    """Merge compatible CSV files and reject exact duplicate rows."""

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    seen: set[str] = set()
    for input_path in inputs:
        path = Path(input_path)
        if not path.is_file() or path.stat().st_size == 0:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or ():
                if field not in fieldnames:
                    fieldnames.append(field)
            for row in reader:
                fingerprint = json.dumps(row, sort_keys=True)
                if fingerprint in seen:
                    raise ValueError(f"Duplicate CSV row while merging {path}")
                seen.add(fingerprint)
                rows.append(dict(row))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        destination.write_text("", encoding="utf-8")
        return 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return len(rows)


def merge_jsonl_files(inputs: Iterable[str | Path], output: str | Path) -> int:
    """Merge JSONL files and reject exact duplicate records."""

    records = []
    seen: set[str] = set()
    for input_path in inputs:
        path = Path(input_path)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            fingerprint = json.dumps(record, sort_keys=True, separators=(",", ":"))
            if fingerprint in seen:
                raise ValueError(f"Duplicate JSONL record while merging {path}")
            seen.add(fingerprint)
            records.append(record)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return len(records)
