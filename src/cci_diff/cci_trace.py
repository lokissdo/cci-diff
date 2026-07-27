"""Authoritative JSONL trace I/O for clean CCI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_TRACE_FIELDS = frozenset(
    {"step", "timestep", "progress", "target", "constraints", "update"}
)


class JSONLTraceWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_cci_trace(path: str | Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_cci_trace(records)
    return records


def validate_cci_trace(records: Iterable[dict[str, Any]]) -> None:
    previous = -1
    for record in records:
        if not REQUIRED_TRACE_FIELDS.issubset(record):
            raise ValueError("CCI trace record is missing required fields")
        step = int(record["step"])
        if step <= previous:
            raise ValueError("CCI trace steps must be strictly increasing")
        previous = step
