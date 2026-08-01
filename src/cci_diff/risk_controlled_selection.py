"""Risk-controlled source-only semantic mask selection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cci_diff.individual_region_selection import (
    FrozenInfluencePolicy,
    RegionTuple,
)


FEATURE_NAMES = (
    "difficulty",
    "coverage",
    "saliency_density",
    "mask_fraction",
    "component_count",
    "global_mean_effect",
    "global_flip_rate",
    "global_effect_ci_low",
)
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
WILSON_Z_95_ONE_SIDED = 1.6448536269514722


def _canonical_regions(regions: Iterable[str]) -> RegionTuple:
    return tuple(sorted({str(region).strip() for region in regions if str(region).strip()}))


@dataclass(frozen=True)
class SafeSuccessThresholds:
    desired_probability: float = 0.53
    identity_distance: float = 0.08
    outside_locality: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 <= self.desired_probability <= 1.0:
            raise ValueError("desired_probability must be between zero and one")
        if self.identity_distance < 0.0 or self.outside_locality < 0.0:
            raise ValueError("preservation thresholds must be non-negative")
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("safe-success thresholds must be finite")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SafeSuccessThresholds":
        return cls(**{name: float(payload[name]) for name in asdict(cls()).keys()})


@dataclass(frozen=True)
class CandidateFeatureRow:
    regions: RegionTuple
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        regions = _canonical_regions(self.regions)
        values = tuple(float(value) for value in self.values)
        if not regions:
            raise ValueError("candidate regions must be non-empty")
        if len(values) != len(FEATURE_NAMES):
            raise ValueError("candidate feature row has an unknown schema")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("candidate features must be finite")
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "values", values)

    def _value(self, name: str) -> float:
        return self.values[FEATURE_NAMES.index(name)]

    @property
    def difficulty(self) -> float:
        return self._value("difficulty")

    @property
    def coverage(self) -> float:
        return self._value("coverage")

    @property
    def saliency_density(self) -> float:
        return self._value("saliency_density")

    @property
    def mask_fraction(self) -> float:
        return self._value("mask_fraction")

    @property
    def component_count(self) -> float:
        return self._value("component_count")

    @property
    def global_mean_effect(self) -> float:
        return self._value("global_mean_effect")


def safe_success_label(
    desired_probability: float,
    identity_distance: float,
    outside_locality: float,
    thresholds: SafeSuccessThresholds = SafeSuccessThresholds(),
) -> int:
    """Return the declared joint target, identity, and locality label."""

    values = (desired_probability, identity_distance, outside_locality)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("safe-success inputs must be finite")
    return int(
        desired_probability >= thresholds.desired_probability
        and identity_distance <= thresholds.identity_distance
        and outside_locality <= thresholds.outside_locality
    )


def source_feature_signature(provenance: Mapping[str, Any]) -> str:
    """Hash the canonical source feature and model provenance contract."""

    payload = {
        "feature_names": list(FEATURE_NAMES),
        "provenance": dict(provenance),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_candidate_feature_rows(
    source_probability: float,
    saliency: np.ndarray,
    component_masks: Mapping[str, np.ndarray],
    policy: FrozenInfluencePolicy,
    *,
    candidate_region_sets: Sequence[RegionTuple] | None = None,
    eps: float = 1e-12,
) -> tuple[CandidateFeatureRow, ...]:
    """Build exact source-only features for the frozen candidate family."""

    if not math.isfinite(source_probability) or not 0.0 <= source_probability <= 1.0:
        raise ValueError("source_probability must be finite and between zero and one")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    saliency_array = np.asarray(saliency, dtype=np.float64)
    if saliency_array.ndim != 2:
        raise ValueError("saliency must be two-dimensional")
    if not np.all(np.isfinite(saliency_array)) or np.any(saliency_array < 0.0):
        raise ValueError("saliency must be finite and non-negative")

    masks: dict[str, np.ndarray] = {}
    for region in policy.verified_regions:
        if region not in component_masks:
            raise ValueError(f"Missing required semantic mask: {region}")
        mask = np.asarray(component_masks[region])
        if mask.shape != saliency_array.shape:
            raise ValueError("semantic masks and saliency must have identical shapes")
        binary = mask > 0
        if not np.any(binary):
            raise ValueError(f"Required semantic mask is empty: {region}")
        masks[region] = binary

    candidates = tuple(
        _canonical_regions(regions)
        for regions in (
            candidate_region_sets
            if candidate_region_sets is not None
            else policy.candidate_region_sets
        )
    )
    if not candidates:
        raise ValueError("frozen policy has no selector candidate region sets")
    verified_union = np.logical_or.reduce(tuple(masks.values()))
    support = float(np.sum(saliency_array[verified_union]))
    probability = float(np.clip(source_probability, eps, 1.0 - eps))
    difficulty = -(2 * policy.desired_value - 1) * math.log(
        probability / (1.0 - probability)
    )

    rows = []
    for regions in candidates:
        if not set(regions).issubset(masks):
            raise ValueError(f"Candidate uses unavailable regions: {regions}")
        evidence = policy.region_set_evidence.get(regions)
        if evidence is None:
            raise ValueError(f"Candidate lacks complete frozen evidence: {regions}")
        union = np.logical_or.reduce(tuple(masks[region] for region in regions))
        saliency_mass = float(np.sum(saliency_array[union]))
        pixel_count = int(np.sum(union))
        rows.append(
            CandidateFeatureRow(
                regions=regions,
                values=(
                    difficulty,
                    saliency_mass / support if support > eps else 0.0,
                    saliency_mass / max(pixel_count, 1),
                    float(np.mean(union)),
                    float(len(regions)),
                    evidence.mean_effect,
                    evidence.flip_rate,
                    evidence.effect_ci_low,
                ),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class LogisticModel:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    l2: float
    iterations: int

    def __post_init__(self) -> None:
        mean = tuple(float(value) for value in self.mean)
        scale = tuple(float(value) for value in self.scale)
        coefficients = tuple(float(value) for value in self.coefficients)
        if not mean or len(mean) != len(scale) or len(mean) != len(coefficients):
            raise ValueError("logistic model dimensions must be non-empty and equal")
        values = mean + scale + (float(self.intercept),) + coefficients
        if not all(math.isfinite(value) for value in values):
            raise ValueError("logistic model values must be finite")
        if any(value <= 0.0 for value in scale):
            raise ValueError("logistic scales must be positive")
        if not math.isfinite(self.l2) or self.l2 < 0.0:
            raise ValueError("l2 must be finite and non-negative")
        if self.iterations <= 0 or self.iterations > 200:
            raise ValueError("iterations must be between one and 200")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "coefficients", coefficients)

    def predict_logit(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != len(self.coefficients):
            raise ValueError("prediction matrix has the wrong feature count")
        standardized = (
            array - np.asarray(self.mean)
        ) / np.asarray(self.scale)
        return self.intercept + standardized @ np.asarray(self.coefficients)

    def predict_probability(self, values: np.ndarray) -> np.ndarray:
        logits = np.clip(self.predict_logit(values), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": list(self.mean),
            "scale": list(self.scale),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "l2": self.l2,
            "iterations": self.iterations,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogisticModel":
        return cls(
            mean=tuple(payload["mean"]),
            scale=tuple(payload["scale"]),
            intercept=float(payload["intercept"]),
            coefficients=tuple(payload["coefficients"]),
            l2=float(payload["l2"]),
            iterations=int(payload["iterations"]),
        )


def _validate_fit_arrays(
    values: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("fit matrix must contain at least two rows and one column")
    if y.shape != (x.shape[0],):
        raise ValueError("labels must have one value per fit row")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("fit values must be finite")
    if not set(np.unique(y)).issubset({0.0, 1.0}) or len(np.unique(y)) != 2:
        raise ValueError("fit labels must contain both binary classes")
    return x, y


def _penalized_log_loss(
    design: np.ndarray, labels: np.ndarray, beta: np.ndarray, l2: float
) -> float:
    logits = design @ beta
    return float(
        np.sum(np.logaddexp(0.0, logits) - labels * logits)
        + 0.5 * l2 * np.dot(beta[1:], beta[1:])
    )


def fit_logistic_newton(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> LogisticModel:
    """Fit a deterministic standardized L2 logistic model on CPU."""

    x, y = _validate_fit_arrays(values, labels)
    if not math.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be finite and non-negative")
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(x.shape[0]), standardized])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.diag(np.concatenate([[0.0], np.full(x.shape[1], l2)]))
    damping = 1e-10

    for iteration in range(1, max_iterations + 1):
        logits = np.clip(design @ beta, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probability - y) + penalty @ beta
        weights = np.maximum(probability * (1.0 - probability), 1e-12)
        hessian = design.T @ (design * weights[:, None]) + penalty
        current_loss = _penalized_log_loss(design, y, beta, l2)
        accepted = False
        for _ in range(20):
            try:
                delta = np.linalg.solve(
                    hessian + damping * np.eye(hessian.shape[0]), gradient
                )
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = beta - delta
            if not np.all(np.isfinite(candidate)):
                raise ValueError("logistic fit produced a non-finite update")
            candidate_loss = _penalized_log_loss(design, y, candidate, l2)
            if candidate_loss <= current_loss + 1e-14:
                beta = candidate
                damping = max(damping / 10.0, 1e-12)
                accepted = True
                break
            damping *= 10.0
        if not accepted:
            raise ValueError("logistic fit could not find a finite descent update")
        if float(np.max(np.abs(delta))) <= tolerance:
            return LogisticModel(
                mean=tuple(mean),
                scale=tuple(scale),
                intercept=float(beta[0]),
                coefficients=tuple(beta[1:]),
                l2=float(l2),
                iterations=iteration,
            )
    raise ValueError("logistic fit did not converge within 200 iterations")


@dataclass(frozen=True)
class GroupedFold:
    train_sample_ids: tuple[int, ...]
    validation_sample_ids: tuple[int, ...]


@dataclass(frozen=True)
class GroupedL2Audit:
    l2: float
    mean_log_loss: float
    folds: tuple[GroupedFold, ...]
    losses_by_l2: tuple[tuple[float, float], ...]


def choose_grouped_l2(
    values: np.ndarray,
    labels: np.ndarray,
    sample_ids: Sequence[int],
    *,
    folds: int = 5,
    l2_grid: Sequence[float] = L2_GRID,
) -> tuple[LogisticModel, GroupedL2Audit]:
    """Select L2 by deterministic source-grouped validation log loss."""

    x, y = _validate_fit_arrays(values, labels)
    groups = np.asarray(sample_ids)
    if groups.shape != (x.shape[0],):
        raise ValueError("sample_ids must have one value per fit row")
    unique_ids = tuple(sorted({int(value) for value in groups}))
    if folds < 2 or len(unique_ids) < folds:
        raise ValueError("grouped cross-validation needs at least one group per fold")
    fold_ids = tuple(tuple(unique_ids[index::folds]) for index in range(folds))
    fold_audit = tuple(
        GroupedFold(
            train_sample_ids=tuple(
                sample_id for sample_id in unique_ids if sample_id not in validation
            ),
            validation_sample_ids=validation,
        )
        for validation in fold_ids
    )
    results = []
    for raw_l2 in l2_grid:
        l2 = float(raw_l2)
        losses = []
        for fold in fold_audit:
            validation_mask = np.isin(groups, fold.validation_sample_ids)
            model = fit_logistic_newton(x[~validation_mask], y[~validation_mask], l2=l2)
            probability = np.clip(
                model.predict_probability(x[validation_mask]), 1e-12, 1.0 - 1e-12
            )
            validation_y = y[validation_mask]
            losses.append(
                float(
                    np.mean(
                        -validation_y * np.log(probability)
                        - (1.0 - validation_y) * np.log(1.0 - probability)
                    )
                )
            )
        results.append((l2, float(np.mean(losses))))
    chosen_l2, chosen_loss = min(results, key=lambda item: (item[1], item[0]))
    model = fit_logistic_newton(x, y, l2=chosen_l2)
    return model, GroupedL2Audit(
        l2=chosen_l2,
        mean_log_loss=chosen_loss,
        folds=fold_audit,
        losses_by_l2=tuple(results),
    )


@dataclass(frozen=True)
class PlattCalibrator:
    intercept: float
    slope: float
    iterations: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.intercept) or not math.isfinite(self.slope):
            raise ValueError("Platt coefficients must be finite")
        if self.iterations <= 0 or self.iterations > 200:
            raise ValueError("Platt iterations must be between one and 200")

    def predict_probability(self, raw_logits: np.ndarray) -> np.ndarray:
        logits = np.clip(
            self.intercept + self.slope * np.asarray(raw_logits, dtype=np.float64),
            -40.0,
            40.0,
        )
        return 1.0 / (1.0 + np.exp(-logits))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlattCalibrator":
        return cls(
            intercept=float(payload["intercept"]),
            slope=float(payload["slope"]),
            iterations=int(payload["iterations"]),
        )


def fit_platt_calibrator(
    raw_logits: np.ndarray, labels: np.ndarray
) -> PlattCalibrator:
    values = np.asarray(raw_logits, dtype=np.float64).reshape(-1, 1)
    model = fit_logistic_newton(values, labels, l2=1e-6)
    slope = model.coefficients[0] / model.scale[0]
    intercept = model.intercept - slope * model.mean[0]
    return PlattCalibrator(intercept, slope, model.iterations)


@dataclass(frozen=True)
class RiskThreshold:
    threshold: float
    accepted: int
    failures: int
    failure_upper_bound: float
    fallback_only: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("risk threshold must be between zero and one")
        if self.accepted < 0 or self.failures < 0 or self.failures > self.accepted:
            raise ValueError("risk support counts are invalid")
        if not 0.0 <= self.failure_upper_bound <= 1.0:
            raise ValueError("failure upper bound must be between zero and one")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RiskThreshold":
        return cls(
            threshold=float(payload["threshold"]),
            accepted=int(payload["accepted"]),
            failures=int(payload["failures"]),
            failure_upper_bound=float(payload["failure_upper_bound"]),
            fallback_only=bool(payload.get("fallback_only", False)),
        )


def wilson_failure_upper_bound(
    failures: int,
    total: int,
    z: float = WILSON_Z_95_ONE_SIDED,
) -> float:
    if total <= 0:
        return 1.0
    if failures < 0 or failures > total:
        raise ValueError("failures must be between zero and total")
    p = failures / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    radius = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    )
    return (centre + radius) / denominator


def choose_risk_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    min_accepted: int = 60,
    max_failure_ucb: float = 0.05,
) -> RiskThreshold:
    probabilities = np.asarray(scores, dtype=np.float64)
    outcomes = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 1 or outcomes.shape != probabilities.shape:
        raise ValueError("scores and labels must be aligned vectors")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("scores must be finite probabilities")
    if not set(np.unique(outcomes)).issubset({0, 1}):
        raise ValueError("risk labels must be binary")
    if min_accepted <= 0:
        raise ValueError("min_accepted must be positive")

    valid = []
    for threshold in sorted(set(float(value) for value in probabilities)):
        accepted_mask = probabilities >= threshold
        accepted = int(np.sum(accepted_mask))
        if accepted < min_accepted:
            continue
        failures = int(np.sum(1 - outcomes[accepted_mask]))
        upper = wilson_failure_upper_bound(failures, accepted)
        if upper <= max_failure_ucb:
            valid.append((threshold, accepted, failures, upper))
    if not valid:
        return RiskThreshold(1.0, 0, 0, 1.0, fallback_only=True)
    threshold, accepted, failures, upper = min(valid, key=lambda row: row[0])
    return RiskThreshold(threshold, accepted, failures, upper)


@dataclass(frozen=True)
class FrozenSelectorArtifact:
    protocol_version: int
    target: str
    desired_value: int
    graph_sha256: str
    candidate_region_sets: tuple[RegionTuple, ...]
    fallback_regions: RegionTuple
    feature_names: tuple[str, ...]
    feature_signature: str
    classifier_sha256: str
    generation_policy_signature: str
    model: LogisticModel
    calibrator: PlattCalibrator
    risk_calibration: RiskThreshold
    coverage_threshold: float
    safe_success_thresholds: SafeSuccessThresholds
    fit_sample_ids: tuple[int, ...] = ()
    calibration_sample_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        candidates = tuple(sorted({_canonical_regions(item) for item in self.candidate_region_sets}))
        fallback = _canonical_regions(self.fallback_regions)
        if self.protocol_version != 1:
            raise ValueError("unsupported selector protocol version")
        if not self.target.strip() or self.desired_value not in (0, 1):
            raise ValueError("selector target or desired value is invalid")
        if not candidates or fallback not in candidates:
            raise ValueError("selector candidates must include the fallback")
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("selector artifact has an unknown feature schema")
        if len(self.model.coefficients) != len(FEATURE_NAMES):
            raise ValueError("selector model has the wrong feature count")
        for name in (
            "graph_sha256",
            "feature_signature",
            "classifier_sha256",
            "generation_policy_signature",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not 0.0 < self.coverage_threshold <= 1.0:
            raise ValueError("coverage_threshold must be in (0, 1]")
        object.__setattr__(self, "target", self.target.strip())
        object.__setattr__(self, "candidate_region_sets", candidates)
        object.__setattr__(self, "fallback_regions", fallback)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "fit_sample_ids", tuple(sorted(set(self.fit_sample_ids))))
        object.__setattr__(
            self,
            "calibration_sample_ids",
            tuple(sorted(set(self.calibration_sample_ids))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "target": self.target,
            "desired_value": self.desired_value,
            "graph_sha256": self.graph_sha256,
            "candidate_region_sets": [list(item) for item in self.candidate_region_sets],
            "fallback_regions": list(self.fallback_regions),
            "feature_names": list(self.feature_names),
            "feature_signature": self.feature_signature,
            "classifier_sha256": self.classifier_sha256,
            "generation_policy_signature": self.generation_policy_signature,
            "model": self.model.to_dict(),
            "calibrator": self.calibrator.to_dict(),
            "risk_calibration": asdict(self.risk_calibration),
            "coverage_threshold": self.coverage_threshold,
            "safe_success_thresholds": asdict(self.safe_success_thresholds),
            "fit_sample_ids": list(self.fit_sample_ids),
            "calibration_sample_ids": list(self.calibration_sample_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenSelectorArtifact":
        return cls(
            protocol_version=int(payload["protocol_version"]),
            target=str(payload["target"]),
            desired_value=int(payload["desired_value"]),
            graph_sha256=str(payload["graph_sha256"]),
            candidate_region_sets=tuple(tuple(item) for item in payload["candidate_region_sets"]),
            fallback_regions=tuple(payload["fallback_regions"]),
            feature_names=tuple(payload["feature_names"]),
            feature_signature=str(payload["feature_signature"]),
            classifier_sha256=str(payload["classifier_sha256"]),
            generation_policy_signature=str(payload["generation_policy_signature"]),
            model=LogisticModel.from_dict(payload["model"]),
            calibrator=PlattCalibrator.from_dict(payload["calibrator"]),
            risk_calibration=RiskThreshold.from_dict(payload["risk_calibration"]),
            coverage_threshold=float(payload["coverage_threshold"]),
            safe_success_thresholds=SafeSuccessThresholds.from_dict(
                payload["safe_success_thresholds"]
            ),
            fit_sample_ids=tuple(int(value) for value in payload.get("fit_sample_ids", ())),
            calibration_sample_ids=tuple(
                int(value) for value in payload.get("calibration_sample_ids", ())
            ),
        )


@dataclass(frozen=True)
class CandidateScore:
    regions: RegionTuple
    coverage: float
    mask_fraction: float
    raw_probability: float
    calibrated_probability: float
    globally_verified_effect: float
    feasible: bool


@dataclass(frozen=True)
class RiskControlledSelection:
    selected_regions: RegionTuple
    coverage: float
    mask_fraction: float
    safe_probability: float
    coverage_threshold: float
    risk_threshold: float
    fallback_used: bool
    fallback_reason: str | None
    candidate_scores: tuple[CandidateScore, ...]


def select_risk_controlled_regions(
    rows: Sequence[CandidateFeatureRow],
    policy: FrozenInfluencePolicy,
    artifact: FrozenSelectorArtifact,
) -> RiskControlledSelection:
    """Select the smallest source-only candidate that passes both gates."""

    if artifact.target != policy.target or artifact.desired_value != policy.desired_value:
        raise ValueError("selector target does not match the frozen graph")
    if artifact.graph_sha256 != policy.graph_sha256:
        raise ValueError("selector graph digest does not match the frozen graph")
    if artifact.candidate_region_sets != policy.candidate_region_sets:
        raise ValueError("selector candidate family does not match the frozen graph")
    if artifact.fallback_regions != policy.fallback_regions:
        raise ValueError("selector fallback does not match the frozen graph")

    by_regions = {row.regions: row for row in rows}
    if len(by_regions) != len(rows):
        raise ValueError("duplicate candidate feature rows")
    missing = set(artifact.candidate_region_sets) - set(by_regions)
    if missing:
        raise ValueError(f"missing candidate feature rows: {sorted(missing)}")
    ordered = tuple(by_regions[regions] for regions in artifact.candidate_region_sets)
    matrix = np.asarray([row.values for row in ordered], dtype=np.float64)
    raw_probability = artifact.model.predict_probability(matrix)
    calibrated = artifact.calibrator.predict_probability(
        artifact.model.predict_logit(matrix)
    )
    scores = []
    for row, raw, probability in zip(ordered, raw_probability, calibrated):
        feasible = bool(
            not artifact.risk_calibration.fallback_only
            and row.coverage >= artifact.coverage_threshold
            and probability >= artifact.risk_calibration.threshold
        )
        scores.append(
            CandidateScore(
                regions=row.regions,
                coverage=row.coverage,
                mask_fraction=row.mask_fraction,
                raw_probability=float(raw),
                calibrated_probability=float(probability),
                globally_verified_effect=row.global_mean_effect,
                feasible=feasible,
            )
        )
    feasible_scores = [score for score in scores if score.feasible]
    if feasible_scores:
        selected = min(
            feasible_scores,
            key=lambda score: (
                score.mask_fraction,
                -score.calibrated_probability,
                -score.globally_verified_effect,
                score.regions,
            ),
        )
        fallback_used = False
        fallback_reason = None
    else:
        selected = next(
            score for score in scores if score.regions == artifact.fallback_regions
        )
        fallback_used = True
        fallback_reason = (
            "risk_calibration_requires_fallback"
            if artifact.risk_calibration.fallback_only
            else "no_candidate_passed_coverage_and_risk"
        )
    return RiskControlledSelection(
        selected_regions=selected.regions,
        coverage=selected.coverage,
        mask_fraction=selected.mask_fraction,
        safe_probability=selected.calibrated_probability,
        coverage_threshold=artifact.coverage_threshold,
        risk_threshold=artifact.risk_calibration.threshold,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        candidate_scores=tuple(scores),
    )
