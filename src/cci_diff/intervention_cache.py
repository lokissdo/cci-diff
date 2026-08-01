"""Immutable content-addressed storage for generated A11 interventions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from cci_diff.counterfactual_graph import InterventionObservation


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class InterventionCacheKey:
    """Canonical scientific identity of one intervention."""

    digest: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class CachedIntervention:
    """A validated complete cache entry."""

    key: InterventionCacheKey
    path: Path
    observation: InterventionObservation
    artifacts: Mapping[str, Path]


def cache_key_for(
    *,
    source_sha256: str,
    mask_sha256: str,
    checkpoint_sha256: str,
    classifier_sha256: str,
    graph_sha256: str,
    policy_sha256: str,
    sample_id: int,
    seed: int,
    identity_sha256: str | None = None,
) -> InterventionCacheKey:
    """Build a versioned key from every input that can change the result."""

    digests = {
        "source_sha256": source_sha256,
        "mask_sha256": mask_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "classifier_sha256": classifier_sha256,
        "graph_sha256": graph_sha256,
        "policy_sha256": policy_sha256,
    }
    if identity_sha256 is not None:
        digests["identity_sha256"] = identity_sha256
    for name, value in digests.items():
        if not isinstance(value, str) or not _DIGEST.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    for name, value in (("sample_id", sample_id), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    payload: dict[str, object] = {
        "version": 1,
        **digests,
        "sample_id": sample_id,
        "seed": seed,
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return InterventionCacheKey(digest=digest, payload=payload)


def load_cached_intervention(
    root: str | Path,
    key: InterventionCacheKey,
) -> CachedIntervention | None:
    """Load a complete entry, returning ``None`` for absent/partial entries."""

    entry_path = Path(root) / key.digest
    complete_path = entry_path / "complete.json"
    if not complete_path.is_file():
        return None
    try:
        metadata = _read_json(entry_path / "metadata.json")
        observation_payload = _read_json(entry_path / "observation.json")
        completion = _read_json(complete_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if metadata.get("cache_key") != key.digest or metadata.get("key") != dict(
        key.payload
    ):
        raise ValueError("Cache metadata conflicts with the requested key")
    if completion.get("cache_key") != key.digest:
        return None
    if completion.get("metadata_sha256") != _sha256_file(
        entry_path / "metadata.json"
    ) or completion.get("observation_sha256") != _sha256_file(
        entry_path / "observation.json"
    ):
        return None

    artifacts: dict[str, Path] = {}
    for name, item in sorted((metadata.get("artifacts") or {}).items()):
        artifact = entry_path / "artifacts" / str(item["filename"])
        if not artifact.is_file() or _sha256_file(artifact) != item["sha256"]:
            return None
        artifacts[name] = artifact
    try:
        observation = InterventionObservation(**observation_payload)
    except (TypeError, ValueError):
        return None
    return CachedIntervention(
        key=key,
        path=entry_path,
        observation=observation,
        artifacts=artifacts,
    )


def store_cached_intervention(
    root: str | Path,
    key: InterventionCacheKey,
    observation: InterventionObservation,
    artifacts: Mapping[str, str | Path],
) -> CachedIntervention:
    """Atomically publish an immutable entry, with completion written last."""

    cache_root = Path(root)
    cache_root.mkdir(parents=True, exist_ok=True)
    sources = _validate_artifacts(artifacts)
    existing = load_cached_intervention(cache_root, key)
    if existing is not None:
        requested_digests = {
            name: _sha256_file(path) for name, path in sources.items()
        }
        existing_digests = {
            name: _sha256_file(path)
            for name, path in existing.artifacts.items()
        }
        if (
            existing.observation == observation
            and requested_digests == existing_digests
        ):
            return existing
        raise ValueError("Complete cache entries are immutable")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{key.digest}.tmp-", dir=cache_root)
    )
    final_path = cache_root / key.digest
    try:
        artifact_dir = staging / "artifacts"
        artifact_dir.mkdir()
        artifact_metadata = {}
        for name, source in sorted(sources.items()):
            suffix = source.suffix if source.suffix else ".bin"
            filename = f"{name}{suffix}"
            destination = artifact_dir / filename
            shutil.copy2(source, destination)
            _fsync_file(destination)
            artifact_metadata[name] = {
                "filename": filename,
                "sha256": _sha256_file(destination),
            }
        metadata = {
            "version": 1,
            "cache_key": key.digest,
            "key": dict(key.payload),
            "artifacts": artifact_metadata,
        }
        _write_json(staging / "metadata.json", metadata)
        _write_json(staging / "observation.json", asdict(observation))
        _fsync_directory(staging)

        if final_path.exists():
            quarantine = cache_root / (
                f".{key.digest}.incomplete-{uuid.uuid4().hex}"
            )
            os.replace(final_path, quarantine)
        os.replace(staging, final_path)
        _fsync_directory(cache_root)
        completion = {
            "version": 1,
            "cache_key": key.digest,
            "metadata_sha256": _sha256_file(final_path / "metadata.json"),
            "observation_sha256": _sha256_file(
                final_path / "observation.json"
            ),
        }
        completion_tmp = final_path / ".complete.json.tmp"
        _write_json(completion_tmp, completion)
        os.replace(completion_tmp, final_path / "complete.json")
        _fsync_directory(final_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    loaded = load_cached_intervention(cache_root, key)
    if loaded is None:
        raise RuntimeError("Published cache entry did not validate")
    return loaded


def _validate_artifacts(
    artifacts: Mapping[str, str | Path],
) -> dict[str, Path]:
    if not artifacts:
        raise ValueError("At least one artifact is required")
    result = {}
    for raw_name, raw_path in artifacts.items():
        name = str(raw_name)
        if not _ARTIFACT_NAME.fullmatch(name) or name in {".", ".."}:
            raise ValueError(f"Invalid artifact name: {name!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Artifact not found: {path}")
        result[name] = path
    return result


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    with path.open("wb") as handle:
        handle.write(_canonical_json(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
