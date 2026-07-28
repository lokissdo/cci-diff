import csv
import json

import pytest

from cci_diff.execution_shards import (
    merge_csv_files,
    merge_jsonl_files,
    partition_ids,
    resolve_worker_devices,
)


def test_cuda_device_expansion_respects_available_devices_and_limit():
    assert resolve_worker_devices("cuda", 0, cuda_device_count=2) == (
        "cuda:0",
        "cuda:1",
    )
    assert resolve_worker_devices("cuda", 1, cuda_device_count=2) == ("cuda:0",)
    assert resolve_worker_devices("cuda:1", 2, cuda_device_count=2) == ("cuda:1",)


def test_mps_and_cpu_always_use_one_worker():
    assert resolve_worker_devices("mps", 8, cuda_device_count=0) == ("mps",)
    assert resolve_worker_devices("cpu", 8, cuda_device_count=0) == ("cpu",)


def test_cuda_requires_an_available_device():
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_worker_devices("cuda", 2, cuda_device_count=0)


def test_round_robin_partitions_are_deterministic_complete_and_disjoint():
    shards = partition_ids([9, 3, 7, 1, 5], 2)

    assert shards == ((1, 5, 9), (3, 7))
    assert sorted(sample_id for shard in shards for sample_id in shard) == [
        1,
        3,
        5,
        7,
        9,
    ]


def test_partition_omits_empty_workers():
    assert partition_ids([4], 2) == ((4,),)


def test_merge_csv_files_preserves_rows_once(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "merged.csv"
    first.write_text("sample_id,variant\n1,A0\n3,A2\n", encoding="utf-8")
    second.write_text("sample_id,variant\n2,A0\n4,A3\n", encoding="utf-8")

    count = merge_csv_files((first, second), output)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert count == 4
    assert [(row["sample_id"], row["variant"]) for row in rows] == [
        ("1", "A0"),
        ("3", "A2"),
        ("2", "A0"),
        ("4", "A3"),
    ]


def test_merge_jsonl_files_preserves_records_once(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    first.write_text('{"sample_id": 1}\n', encoding="utf-8")
    second.write_text('{"sample_id": 2}\n', encoding="utf-8")

    count = merge_jsonl_files((first, second), output)

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert count == 2
    assert records == [{"sample_id": 1}, {"sample_id": 2}]


def test_merge_rejects_duplicate_records(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    first.write_text('{"sample_id": 1}\n', encoding="utf-8")
    second.write_text('{"sample_id": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate"):
        merge_jsonl_files((first, second), output)
