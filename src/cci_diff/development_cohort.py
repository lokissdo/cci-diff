"""Deterministic nested cohorts for selector development."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable


_ROLES = ("discovery", "fit", "calibration")
_WEIGHTS = (2, 5, 8)


@dataclass(frozen=True)
class DevelopmentCounts:
    discovery: int
    fit: int
    calibration: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.discovery, self.fit, self.calibration)
        ):
            raise ValueError("development counts must be positive integers")

    def to_dict(self) -> dict[str, int]:
        return {role: getattr(self, role) for role in _ROLES}


@dataclass(frozen=True)
class DevelopmentCohort:
    discovery: tuple[int, ...]
    fit: tuple[int, ...]
    calibration: tuple[int, ...]
    data_size: int
    seed: int

    def __post_init__(self) -> None:
        roles = tuple(tuple(int(value) for value in getattr(self, role)) for role in _ROLES)
        if any(not values or len(values) != len(set(values)) for values in roles):
            raise ValueError("development roles must be non-empty and unique")
        for left in range(len(roles)):
            for right in range(left + 1, len(roles)):
                if set(roles[left]).intersection(roles[right]):
                    raise ValueError("development roles must be pairwise disjoint")
        if sum(len(values) for values in roles) != self.data_size:
            raise ValueError("development role sizes must equal data_size")
        for role, values in zip(_ROLES, roles):
            object.__setattr__(self, role, values)

    @property
    def all_ids(self) -> frozenset[int]:
        return frozenset((*self.discovery, *self.fit, *self.calibration))

    @property
    def counts(self) -> DevelopmentCounts:
        return DevelopmentCounts(
            len(self.discovery), len(self.fit), len(self.calibration)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "data_size": self.data_size,
            "seed": self.seed,
            "counts": self.counts.to_dict(),
            "cohorts": {role: list(getattr(self, role)) for role in _ROLES},
        }


def allocate_development_counts(data_size: int) -> DevelopmentCounts:
    """Allocate a size using deterministic largest-remainder 2:5:8 ratios."""

    if isinstance(data_size, bool) or not isinstance(data_size, int) or data_size < 15:
        raise ValueError("data_size must be an integer at least 15")
    exact = tuple(data_size * weight / sum(_WEIGHTS) for weight in _WEIGHTS)
    counts = [math.floor(value) for value in exact]
    order = sorted(
        range(len(_ROLES)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for index in order[: data_size - sum(counts)]:
        counts[index] += 1
    return DevelopmentCounts(*counts)


def _digest(seed: int, purpose: str, sample_id: int) -> bytes:
    return hashlib.sha256(f"{seed}:{purpose}:{sample_id}".encode("ascii")).digest()


def _role_for(seed: int, sample_id: int) -> str:
    bucket = int.from_bytes(_digest(seed, "role", sample_id), "big") % 15
    if bucket < 2:
        return "discovery"
    if bucket < 7:
        return "fit"
    return "calibration"


def assign_development_cohort(
    eligible_ids: Iterable[int],
    evaluation_ids: Iterable[int],
    data_size: int,
    seed: int,
) -> DevelopmentCohort:
    """Select nested role-stable development IDs outside evaluation."""

    eligible = tuple(int(value) for value in eligible_ids)
    if not eligible or len(eligible) != len(set(eligible)):
        raise ValueError("eligible_ids must be non-empty and unique")
    excluded = {int(value) for value in evaluation_ids}
    required = allocate_development_counts(data_size)
    buckets: dict[str, list[int]] = {role: [] for role in _ROLES}
    for sample_id in eligible:
        if sample_id in excluded:
            continue
        buckets[_role_for(seed, sample_id)].append(sample_id)
    for role in _ROLES:
        buckets[role].sort(
            key=lambda sample_id: (_digest(seed, f"order:{role}", sample_id), sample_id)
        )
        count = getattr(required, role)
        if len(buckets[role]) < count:
            raise ValueError(
                f"insufficient {role} IDs: required={count}, "
                f"available={len(buckets[role])}"
            )
    return DevelopmentCohort(
        discovery=tuple(buckets["discovery"][: required.discovery]),
        fit=tuple(buckets["fit"][: required.fit]),
        calibration=tuple(buckets["calibration"][: required.calibration]),
        data_size=data_size,
        seed=int(seed),
    )
