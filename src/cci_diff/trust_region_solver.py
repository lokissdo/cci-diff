"""Deterministic dependency-free solver for small trust-region subproblems."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ProjectionResult:
    step: tuple[float, ...]
    multipliers: tuple[float, ...]
    active_indices: tuple[int, ...]
    objective: float
    norm: float
    primal_violation: float
    dual_violation: float


def project_to_linear_constraints(
    nominal: Sequence[float],
    gram: Sequence[Sequence[float]],
    bounds: Sequence[float],
    *,
    radius: float,
    tolerance: float,
) -> ProjectionResult | None:
    """Project a nominal gradient combination into a linearized trust region."""

    nominal_values, gram_values, bound_values = _validate_problem(
        nominal,
        gram,
        bounds,
        radius,
        tolerance,
    )
    candidates: list[ProjectionResult] = []
    for active_indices in _powerset_indices(len(bound_values)):
        inactive = _build_candidate(
            nominal_values,
            gram_values,
            bound_values,
            active_indices,
            ball_multiplier=0.0,
            radius=radius,
            tolerance=tolerance,
        )
        if inactive is not None:
            candidates.append(inactive)
            continue

        active_ball = _build_active_ball_candidate(
            nominal_values,
            gram_values,
            bound_values,
            active_indices,
            radius=radius,
            tolerance=tolerance,
        )
        if active_ball is not None:
            candidates.append(active_ball)

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            candidate.objective,
            candidate.norm,
            candidate.active_indices,
        ),
    )


def solve_lexicographic_envelope(
    gram: Sequence[Sequence[float]],
    target_bound: float,
    safety_residuals: Sequence[float],
    *,
    radius: float,
    tolerance: float,
    bisection_steps: int = 40,
) -> tuple[float, ProjectionResult]:
    """Minimize the worst post-step safety residual after target feasibility."""

    gram_values = tuple(tuple(float(value) for value in row) for row in gram)
    residuals = tuple(float(value) for value in safety_residuals)
    if len(residuals) != 2:
        raise ValueError("safety_residuals must contain identity and locality")
    if len(gram_values) < 3:
        raise ValueError("gram must contain target and two safety gradients")
    if bisection_steps <= 0:
        raise ValueError("bisection_steps must be positive")

    nominal = (0.0,) * len(gram_values)

    def solve(tau: float) -> ProjectionResult | None:
        return project_to_linear_constraints(
            nominal,
            gram_values,
            (
                float(target_bound),
                tau - residuals[0],
                tau - residuals[1],
            ),
            radius=radius,
            tolerance=tolerance,
        )

    lower = max(
        residuals[index] - radius * math.sqrt(max(gram_values[index + 1][index + 1], 0.0))
        for index in range(2)
    )
    upper = max(residuals)
    feasible = solve(upper)
    expansion = max(1.0, abs(upper), abs(lower))
    for _ in range(60):
        if feasible is not None:
            break
        upper += expansion
        expansion *= 2.0
        feasible = solve(upper)
    if feasible is None:
        raise ValueError("target constraint is infeasible within trust radius")

    for _ in range(bisection_steps):
        middle = (lower + upper) / 2.0
        candidate = solve(middle)
        if candidate is None:
            lower = middle
        else:
            upper = middle
            feasible = candidate
    return upper, feasible


def _powerset_indices(count: int) -> Iterable[tuple[int, ...]]:
    for size in range(count + 1):
        yield from itertools.combinations(range(count), size)


def _build_active_ball_candidate(
    nominal: tuple[float, ...],
    gram: tuple[tuple[float, ...], ...],
    bounds: tuple[float, ...],
    active_indices: tuple[int, ...],
    *,
    radius: float,
    tolerance: float,
) -> ProjectionResult | None:
    zero = _candidate_values(
        nominal,
        gram,
        bounds,
        active_indices,
        ball_multiplier=0.0,
    )
    if zero is None or _metric_norm(zero[0], gram) <= radius + tolerance:
        return None

    low = 0.0
    high = 1.0
    high_values = None
    for _ in range(80):
        high_values = _candidate_values(
            nominal,
            gram,
            bounds,
            active_indices,
            ball_multiplier=high,
        )
        if (
            high_values is not None
            and _metric_norm(high_values[0], gram) <= radius
        ):
            break
        high *= 2.0
    else:
        return None

    for _ in range(60):
        middle = (low + high) / 2.0
        values = _candidate_values(
            nominal,
            gram,
            bounds,
            active_indices,
            ball_multiplier=middle,
        )
        if values is None or _metric_norm(values[0], gram) > radius:
            low = middle
        else:
            high = middle
            high_values = values

    return _build_candidate(
        nominal,
        gram,
        bounds,
        active_indices,
        ball_multiplier=high,
        radius=radius,
        tolerance=tolerance,
    )


def _build_candidate(
    nominal: tuple[float, ...],
    gram: tuple[tuple[float, ...], ...],
    bounds: tuple[float, ...],
    active_indices: tuple[int, ...],
    *,
    ball_multiplier: float,
    radius: float,
    tolerance: float,
) -> ProjectionResult | None:
    values = _candidate_values(
        nominal,
        gram,
        bounds,
        active_indices,
        ball_multiplier=ball_multiplier,
    )
    if values is None:
        return None
    step, active_multipliers = values
    products = _matrix_vector(gram, step)
    norm = _metric_norm(step, gram)
    multipliers = [0.0] * len(bounds)
    for index, multiplier in zip(active_indices, active_multipliers):
        multipliers[index] = multiplier

    inequality_violation = max(
        (products[index] - bound for index, bound in enumerate(bounds)),
        default=0.0,
    )
    equality_violation = max(
        (
            abs(products[index] - bounds[index])
            for index in active_indices
        ),
        default=0.0,
    )
    primal_violation = max(
        0.0,
        inequality_violation,
        equality_violation,
        norm - radius,
    )
    dual_violation = max(
        (max(0.0, -value) for value in active_multipliers),
        default=0.0,
    )
    if (
        primal_violation > tolerance
        or dual_violation > tolerance
        or not all(math.isfinite(value) for value in (*step, *multipliers))
    ):
        return None

    displacement = tuple(
        value - nominal[index] for index, value in enumerate(step)
    )
    objective = 0.5 * max(
        0.0,
        _dot(displacement, _matrix_vector(gram, displacement)),
    )
    return ProjectionResult(
        step=step,
        multipliers=tuple(multipliers),
        active_indices=active_indices,
        objective=objective,
        norm=norm,
        primal_violation=primal_violation,
        dual_violation=dual_violation,
    )


def _candidate_values(
    nominal: tuple[float, ...],
    gram: tuple[tuple[float, ...], ...],
    bounds: tuple[float, ...],
    active_indices: tuple[int, ...],
    *,
    ball_multiplier: float,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    scale = 1.0 + ball_multiplier
    if not active_indices:
        return tuple(value / scale for value in nominal), ()

    active_gram = tuple(
        tuple(gram[row][column] for column in active_indices)
        for row in active_indices
    )
    nominal_products = _matrix_vector(gram, nominal)
    rhs = tuple(
        nominal_products[index] - scale * bounds[index]
        for index in active_indices
    )
    multipliers = _solve_regularized(active_gram, rhs)
    if multipliers is None:
        return None
    step = list(nominal)
    for index, multiplier in zip(active_indices, multipliers):
        step[index] -= multiplier
    return tuple(value / scale for value in step), multipliers


def _solve_regularized(
    matrix: Sequence[Sequence[float]],
    rhs: Sequence[float],
    ridge: float = 1e-10,
) -> tuple[float, ...] | None:
    size = len(rhs)
    if size == 0:
        return ()
    augmented = [
        [
            *(
                float(matrix[row][column])
                + (ridge if row == column else 0.0)
                for column in range(size)
            ),
            float(rhs[row]),
        ]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: (abs(augmented[row][column]), -row),
        )
        if abs(augmented[pivot][column]) < ridge * 1e-6:
            return None
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        pivot_value = augmented[column][column]
        for entry in range(column, size + 1):
            augmented[column][entry] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for entry in range(column, size + 1):
                augmented[row][entry] -= factor * augmented[column][entry]
    solution = tuple(augmented[row][-1] for row in range(size))
    if not all(math.isfinite(value) for value in solution):
        return None
    return solution


def _validate_problem(
    nominal: Sequence[float],
    gram: Sequence[Sequence[float]],
    bounds: Sequence[float],
    radius: float,
    tolerance: float,
) -> tuple[
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
]:
    nominal_values = tuple(float(value) for value in nominal)
    gram_values = tuple(tuple(float(value) for value in row) for row in gram)
    bound_values = tuple(float(value) for value in bounds)
    size = len(nominal_values)
    if size == 0 or len(gram_values) != size:
        raise ValueError("gram dimensions must match a non-empty nominal")
    if any(len(row) != size for row in gram_values):
        raise ValueError("gram must be square")
    if len(bound_values) > size:
        raise ValueError("bounds cannot outnumber gradients")
    if radius <= 0 or tolerance <= 0:
        raise ValueError("radius and tolerance must be positive")
    values = (
        *nominal_values,
        *bound_values,
        *(value for row in gram_values for value in row),
        radius,
        tolerance,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("solver inputs must be finite")
    for row in range(size):
        for column in range(size):
            if abs(gram_values[row][column] - gram_values[column][row]) > tolerance:
                raise ValueError("gram must be symmetric")
    return nominal_values, gram_values, bound_values


def _matrix_vector(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, ...]:
    return tuple(_dot(row, vector) for row in matrix)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _metric_norm(
    vector: Sequence[float],
    gram: Sequence[Sequence[float]],
) -> float:
    squared = _dot(vector, _matrix_vector(gram, vector))
    return math.sqrt(max(0.0, squared))
