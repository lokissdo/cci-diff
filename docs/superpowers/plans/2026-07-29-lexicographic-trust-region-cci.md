# Lexicographic Trust-Region CCI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a target-progress-constrained trust-region CCI mode that minimizes non-target attribute drift while treating identity and outside-mask locality as safety constraints.

**Architecture:** Keep archived controllers unchanged. Add a differentiable non-target drift evaluator, a dependency-free small active-set trust-region solver, a new lexicographic controller, and separate SD2 adapter/final-restoration hooks. Wire the new adaptive and matched-fixed modes through the existing graph runner and pilot while preserving version-one graph compatibility and audit provenance.

**Tech Stack:** Python 3.10+, PyTorch, dataclasses, existing SD2/diffusers adapters, JSONL audit traces, pytest.

## Global Constraints

- Do not change the behavior or audit meaning of `disabled`, `fixed_equal`, `feedback`, or `FinalTargetLatentCorrectionHook`.
- Add no SciPy dependency and perform no network/model downloads in tests.
- Compute the primary surrogate from all 39 non-target CelebA attributes with exactly the target index excluded.
- Treat identity and outside-mask locality as hard safety constraints; keep residual TV audit-only.
- Measure trust radii in predicted-clean latent coordinates and explicitly map the step to epsilon prediction space.
- Apply effective mask support before gradient geometry is computed.
- Never silently fall back from the trust-region solver to a fixed weighted update.
- The old target-only final correction must be disabled for the new controller modes.
- Confirmatory evaluation must use independent ACE probabilities, paired cohorts, and matched target-success operating points.

---

## File Structure

- Create `src/cci_diff/trust_region_solver.py`: dependency-free active-set projection and lexicographic feasibility solve.
- Create `src/cci_diff/trust_region_controller.py`: PyTorch gradient extraction, masked geometry, target-progress policy, and trace records.
- Modify `src/cci_diff/constraints.py`: differentiable all-non-target drift evaluator.
- Modify `src/cci_diff/concept_graph.py`: backward-compatible trust-region settings and validation.
- Modify `src/cci_diff/adapters/sd2_clean_cci.py`: clean/noise coordinate conversion, trust-region denoising hook, and preservation-aware final hook.
- Modify `src/cci_diff/sd2_bld_backend.py`: optional post-scheduler/post-blend retention observations for the new hook.
- Modify `scripts/run_sd2_bld_cci.py`: mode selection, evaluator/controller construction, final hook, and audit fields.
- Modify `scripts/run_clean_cci_pilot.py`: `fixed_trust_matched` and `trust_region` variants.
- Modify `scripts/evaluate_clean_cci_ace.py`: independent continuous non-target drift.
- Create `src/cci_diff/matched_success.py`: calibrated frontier, interpolation, nAUC, and paired cluster bootstrap.
- Create `scripts/evaluate_matched_success.py`: frozen calibration/test matched-success report.
- Create `tests/test_trust_region_solver.py`: pure numerical solver coverage.
- Create `tests/test_trust_region_controller.py`: controller behavior and diagnostic coverage.
- Modify `tests/test_constraints.py`: non-target evaluator coverage.
- Modify `tests/test_concept_graph.py`: defaults, serialization, and invalid settings.
- Modify `tests/test_sd2_clean_cci.py`: coordinate conversion and hook behavior.
- Modify `tests/test_clean_cci_cli.py`: CLI construction and audit wiring.
- Modify `tests/test_clean_cci_pilot.py`: paired variant command construction.
- Modify `tests/test_evaluate_clean_cci_ace.py`: independent continuous drift metrics.
- Create `tests/test_matched_success.py`: frontier and paired inference coverage.

---

### Task 1: Backward-Compatible Trust-Region Configuration

**Files:**
- Modify: `src/cci_diff/concept_graph.py:48-60`
- Modify: `src/cci_diff/concept_graph.py:95-110`
- Modify: `src/cci_diff/concept_graph.py:135-149`
- Modify: `src/cci_diff/concept_graph.py:330-364`
- Test: `tests/test_concept_graph.py`

**Interfaces:**
- Consumes: existing `ControllerSpec` JSON fields.
- Produces: `TrustRegionSpec`, `DEFAULT_TRUST_REGION_SPEC`, and `ControllerSpec.trust_region: TrustRegionSpec`.

- [ ] **Step 1: Write failing default and validation tests**

Add:

```python
def test_trust_region_defaults_preserve_version_one_round_trip():
    from cci_diff.concept_graph import (
        DEFAULT_TRUST_REGION_SPEC,
        concept_graph_from_dict,
    )

    payload = valid_graph_payload()
    graph = concept_graph_from_dict(payload)

    assert graph.controller.trust_region == DEFAULT_TRUST_REGION_SPEC
    assert graph.to_dict() == payload


def test_explicit_trust_region_settings_round_trip_and_validate():
    from cci_diff.concept_graph import concept_graph_from_dict

    payload = valid_graph_payload()
    payload["controller"]["trust_region"] = {
        "initial_radius": 0.15,
        "minimum_radius": 0.01,
        "maximum_radius": 0.30,
        "target_progress_fraction": 0.5,
        "feasibility_tolerance": 0.0001,
        "reliability_alpha_min": 0.10,
        "huber_delta": 0.02,
        "support_floor": 0.05,
        "maximum_blend_compensation": 4.0,
        "final_cumulative_radius": 0.60,
        "final_iterations": 12,
    }

    graph = concept_graph_from_dict(payload)

    assert graph.controller.trust_region.initial_radius == 0.15
    assert graph.to_dict()["controller"]["trust_region"] == payload["controller"]["trust_region"]

    payload["controller"]["trust_region"]["minimum_radius"] = 0.31
    with pytest.raises(ValueError, match="minimum_radius"):
        concept_graph_from_dict(payload)
```

Import `pytest` in the test module.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
pytest -q tests/test_concept_graph.py
```

Expected: FAIL because `DEFAULT_TRUST_REGION_SPEC` and the nested settings do not exist.

- [ ] **Step 3: Implement typed defaults and canonical serialization**

Add:

```python
@dataclass(frozen=True)
class TrustRegionSpec:
    initial_radius: float = 0.15
    minimum_radius: float = 0.01
    maximum_radius: float = 0.30
    target_progress_fraction: float = 0.5
    feasibility_tolerance: float = 1e-4
    reliability_alpha_min: float = 0.10
    huber_delta: float = 0.02
    support_floor: float = 0.05
    maximum_blend_compensation: float = 4.0
    final_cumulative_radius: float = 0.60
    final_iterations: int = 12


DEFAULT_TRUST_REGION_SPEC = TrustRegionSpec()
```

Add `trust_region: TrustRegionSpec = DEFAULT_TRUST_REGION_SPEC` to
`ControllerSpec`. Parse `controller.trust_region` through a dedicated helper,
validate finite values and:

```python
0 < minimum_radius <= initial_radius <= maximum_radius
0 < target_progress_fraction < 1
feasibility_tolerance > 0
0 < reliability_alpha_min < 1
huber_delta > 0
0 < support_floor <= 1
maximum_blend_compensation >= 1
final_cumulative_radius > 0
final_iterations >= 0
```

Only emit `"trust_region"` from `ConceptGraph.to_dict()` when the value differs
from `DEFAULT_TRUST_REGION_SPEC`, preserving exact archived round trips.

- [ ] **Step 4: Run configuration tests**

Run:

```bash
pytest -q tests/test_concept_graph.py tests/test_json_graph_compiler.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cci_diff/concept_graph.py tests/test_concept_graph.py
git commit -m "feat: add trust-region controller configuration"
```

---

### Task 2: Differentiable Non-Target Attribute Drift

**Files:**
- Modify: `src/cci_diff/constraints.py`
- Test: `tests/test_constraints.py`

**Interfaces:**
- Consumes: frozen CelebA classifier, target attribute index, classifier input size, and `ConstraintContext.source_image`.
- Produces: `NonTargetDriftEvaluator.bind(context)`, `.measure(image)`, and `.audit(image)`.

- [ ] **Step 1: Write failing evaluator tests**

Add a fake 4-output logits model and tests:

```python
def test_non_target_drift_excludes_target_and_is_zero_for_source():
    import torch
    from cci_diff.constraints import ConstraintContext, NonTargetDriftEvaluator

    class Model:
        def forward_logits(self, images):
            means = images.mean(dim=(2, 3))
            return torch.stack(
                [means[:, 0], means[:, 1], means[:, 2], means.mean(dim=1)],
                dim=1,
            )

    source = torch.full((1, 3, 2, 2), 0.25)
    evaluator = NonTargetDriftEvaluator(
        Model(), target_index=1, input_size=2, huber_delta=0.02
    )
    evaluator.bind(ConstraintContext(source, torch.ones_like(source[:, :1]), torch.ones_like(source[:, :1])))

    identical = evaluator.measure(source)
    changed = source.clone()
    changed[:, 0] += 0.1
    loss = evaluator.measure(changed)
    loss.backward()

    assert identical.item() == pytest.approx(0.0)
    assert evaluator.non_target_indices == (0, 2, 3)
    assert changed.grad is not None


def test_non_target_drift_audit_reports_continuous_mean_and_values():
    import torch
    from cci_diff.constraints import ConstraintContext, NonTargetDriftEvaluator

    class Model:
        def forward_logits(self, images):
            means = images.mean(dim=(2, 3))
            return torch.stack(
                [means[:, 0], means[:, 1], means[:, 2], means.mean(dim=1)],
                dim=1,
            )

    source = torch.full((1, 3, 2, 2), 0.25)
    changed = source.clone()
    changed[:, 2] = 0.75
    evaluator = NonTargetDriftEvaluator(
        Model(), target_index=1, input_size=2, huber_delta=0.02
    )
    evaluator.bind(
        ConstraintContext(
            source,
            torch.ones_like(source[:, :1]),
            torch.ones_like(source[:, :1]),
        )
    )
    payload = evaluator.audit(changed)
    assert payload["excluded_index"] == 1
    assert len(payload["absolute_probability_drift"]) == 3
    assert payload["mean_absolute_probability_drift"] > 0
```

Make `changed = source.clone().requires_grad_(True)` in the first test.

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
pytest -q tests/test_constraints.py
```

Expected: FAIL because `NonTargetDriftEvaluator` is undefined.

- [ ] **Step 3: Implement the evaluator**

Use one 40-output forward per image:

```python
class NonTargetDriftEvaluator:
    def __init__(self, model, *, target_index: int, input_size: int, huber_delta: float):
        if target_index < 0:
            raise ValueError("target_index must be non-negative")
        if input_size <= 0 or huber_delta <= 0:
            raise ValueError("input_size and huber_delta must be positive")
        self.model = model
        self.target_index = target_index
        self.input_size = input_size
        self.huber_delta = huber_delta
        self._source_probabilities = None

    @property
    def non_target_indices(self) -> tuple[int, ...]:
        if self._source_probabilities is None:
            raise RuntimeError("non-target drift evaluator is not bound")
        return tuple(
            index
            for index in range(self._source_probabilities.shape[-1])
            if index != self.target_index
        )

    def bind(self, context: ConstraintContext) -> None:
        import torch
        with torch.no_grad():
            logits = classifier_logits(
                self.model, context.source_image, size=self.input_size
            )
            self._source_probabilities = torch.sigmoid(logits).mean(dim=0).detach()

    def measure(self, image):
        import torch
        import torch.nn.functional as functional
        current = torch.sigmoid(
            classifier_logits(self.model, image, size=self.input_size)
        ).mean(dim=0)
        indices = torch.as_tensor(
            self.non_target_indices, device=current.device, dtype=torch.long
        )
        delta = current.index_select(0, indices) - self._source_probabilities.to(current).index_select(0, indices)
        return functional.huber_loss(
            delta,
            torch.zeros_like(delta),
            delta=self.huber_delta,
            reduction="mean",
        )
```

Implement `audit()` under `torch.no_grad()` and return the excluded index,
included indices, per-attribute absolute drift, and their arithmetic mean.

- [ ] **Step 4: Run evaluator tests**

Run:

```bash
pytest -q tests/test_constraints.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cci_diff/constraints.py tests/test_constraints.py
git commit -m "feat: add differentiable non-target drift"
```

---

### Task 3: Small Active-Set Trust-Region Solver

**Files:**
- Create: `src/cci_diff/trust_region_solver.py`
- Create: `tests/test_trust_region_solver.py`

**Interfaces:**
- Produces immutable `ProjectionResult` fields `step`, `multipliers`,
  `active_indices`, `objective`, `norm`, `primal_violation`, and
  `dual_violation`.
- Produces `project_to_linear_constraints(nominal, gram, bounds, *, radius,
  tolerance) -> ProjectionResult | None`.
- Produces `solve_lexicographic_envelope(gram, target_bound,
  safety_residuals, *, radius, tolerance, bisection_steps=40) ->
  tuple[float, ProjectionResult]`.

The coefficient-space convention is `d = sum(step[i] * gradient[i])`, with
gradient order target, identity, locality, drift. The first three vectors are
the linear constraint normals; the fourth supplies the preservation nominal.
Linear constraint products are read from the Gram matrix.

- [ ] **Step 1: Write failing projection and envelope tests**

Cover:

```python
def test_projection_keeps_nominal_when_feasible():
    result = project_to_linear_constraints(
        nominal=(-0.1, 0.0),
        gram=((1.0, 0.0), (0.0, 1.0)),
        bounds=(-0.05,),
        radius=0.2,
        tolerance=1e-8,
    )
    assert result is not None
    assert result.step == pytest.approx((-0.1, 0.0))


def test_projection_finds_minimum_norm_target_step():
    result = project_to_linear_constraints(
        nominal=(0.0, 0.0),
        gram=((1.0, 0.0), (0.0, 1.0)),
        bounds=(-0.1,),
        radius=0.2,
        tolerance=1e-8,
    )
    assert result.step == pytest.approx((-0.1, 0.0))
    assert result.active_indices == (0,)


def test_projection_returns_none_when_target_exceeds_radius():
    assert project_to_linear_constraints(
        nominal=(0.0,),
        gram=((1.0,),),
        bounds=(-0.3,),
        radius=0.2,
        tolerance=1e-8,
    ) is None


def test_lexicographic_envelope_minimizes_worst_safety_residual():
    tau, result = solve_lexicographic_envelope(
        gram=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        target_bound=-0.1,
        safety_residuals=(0.2, 0.1),
        radius=0.3,
        tolerance=1e-7,
    )
    assert result is not None
    assert tau < 0.2
    assert result.norm <= 0.3 + 1e-7
```

Also test collinear gradients, deterministic active-set tie breaking, singular
regularization, and primal residual reporting.

- [ ] **Step 2: Run the solver tests and verify failure**

Run:

```bash
pytest -q tests/test_trust_region_solver.py
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement deterministic active-set enumeration**

Implement `_powerset_indices(count)` with
`itertools.combinations`, `_solve_regularized(matrix, rhs, ridge=1e-10)` with
deterministic partial-pivot Gaussian elimination, and separate inactive-ball
and active-ball candidate builders. The inactive-ball builder solves the KKT
equality system for each active subset. The active-ball builder bisects a
non-negative ball multiplier for 60 iterations and resolves the equality KKT
system at each point.

For every active subset, generate candidates with the ball inactive and
active. Reject candidates that violate any inactive inequality, have a
negative active multiplier below tolerance, exceed the radius, or are
non-finite. Select by `(objective, norm, active_indices)` for deterministic
ties.

Implement the feasibility envelope by bisection over `tau`. For a proposed
`tau`, set bounds to:

```python
(target_bound, tau - safety_residuals[0], tau - safety_residuals[1])
```

and project the zero nominal. Use a finite lower bound derived from each safety
gradient norm and the radius; use `max(safety_residuals)` as the feasible upper
bound.

- [ ] **Step 4: Run pure solver tests**

Run:

```bash
pytest -q tests/test_trust_region_solver.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cci_diff/trust_region_solver.py tests/test_trust_region_solver.py
git commit -m "feat: add small trust-region solver"
```

---

### Task 4: Lexicographic PyTorch Controller

**Files:**
- Create: `src/cci_diff/trust_region_controller.py`
- Create: `tests/test_trust_region_controller.py`

**Interfaces:**
- Consumes: `TrustRegionSpec`, `TargetMargin`, non-target drift scalar,
  identity/locality `ConstraintObservation`s, and effective latent support.
- Produces immutable `TrustRegionResult(clean_delta, record)`.
- Produces `LexicographicTrustRegionController.compute_update(*, latents,
  target, drift_loss, safety_constraints, effective_support,
  mode="trust_region") -> TrustRegionResult`.
- Produces `LexicographicTrustRegionController.observe_outcome(*,
  requested_progress, actual_progress, step_norm) -> None`.

- [ ] **Step 1: Write failing controller tests**

Use two-dimensional tensors to verify:

```python
def test_controller_makes_target_progress_and_reduces_drift():
    latents = torch.tensor([1.0, 1.0], requires_grad=True)
    target = target_margin(latents[0], desired_value=0, target_probability=0.8)
    drift = latents[1].square()
    safety = (
        ConstraintObservation("identity", latents[0].square(), 0.5),
        ConstraintObservation("outside_locality", latents[1].square(), 2.0),
    )
    controller = LexicographicTrustRegionController(trust_spec())

    result = controller.compute_update(
        latents=latents,
        target=target,
        drift_loss=drift,
        safety_constraints=safety,
        effective_support=torch.ones_like(latents),
    )

    assert torch.dot(torch.autograd.grad(target.loss, latents, retain_graph=True)[0], result.clean_delta) < 0
    assert result.record["solver"]["requested_target_progress"] > 0
    assert result.record["solver"]["primal_violation"] <= 1e-5


def test_controller_masks_gradients_before_gram_geometry():
    latents = torch.tensor([1.0, 1.0], requires_grad=True)
    target = target_margin(
        latents.sum(), desired_value=0, target_probability=0.8
    )
    safety = (
        ConstraintObservation("identity", latents[0].square(), 0.5),
        ConstraintObservation("outside_locality", latents[1].square(), 2.0),
    )
    controller = LexicographicTrustRegionController(trust_spec())
    result = controller.compute_update(
        latents=latents,
        target=target,
        drift_loss=latents[1].square(),
        safety_constraints=safety,
        effective_support=torch.tensor([1.0, 0.0]),
    )
    assert result.clean_delta[1].item() == 0.0
    assert result.record["gradients"]["target"]["masked_norm"] < result.record["gradients"]["target"]["raw_norm"]


def test_controller_uses_margin_guard_after_target_is_feasible():
    latents = torch.tensor([-2.0, 0.5], requires_grad=True)
    target = target_margin(
        latents[0], desired_value=0, target_probability=0.8
    )
    safety = (
        ConstraintObservation("identity", latents[0].square() * 0.0, 0.5),
        ConstraintObservation("outside_locality", latents[1].square(), 2.0),
    )
    result = LexicographicTrustRegionController(trust_spec()).compute_update(
        latents=latents,
        target=target,
        drift_loss=latents[1].square(),
        safety_constraints=safety,
        effective_support=torch.ones_like(latents),
    )
    assert result.record["target"]["guard_mode"] == "maintain"


def test_fixed_trust_matched_uses_same_constraints_but_fixed_nominal():
    latents = torch.tensor([1.0, 1.0], requires_grad=True)
    safety = (
        ConstraintObservation("identity", latents[0].square(), 0.5),
        ConstraintObservation("outside_locality", latents[1].square(), 2.0),
    )
    result = LexicographicTrustRegionController(trust_spec()).compute_update(
        latents=latents,
        target=target_margin(latents[0], 0, 0.8),
        drift_loss=latents[1].square(),
        safety_constraints=safety,
        effective_support=torch.ones_like(latents),
        mode="fixed_trust_matched",
    )
    assert result.record["solver"]["mode"] == "fixed_trust_matched"


def test_controller_skips_unreliable_zero_target_gradient():
    latents = torch.tensor([1.0, 1.0], requires_grad=True)
    safety = (
        ConstraintObservation("identity", latents[0].square(), 0.5),
        ConstraintObservation("outside_locality", latents[1].square(), 2.0),
    )
    result = LexicographicTrustRegionController(trust_spec()).compute_update(
        latents=latents,
        target=target_margin(latents[0] * 0.0, 0, 0.8),
        drift_loss=latents[1].square(),
        safety_constraints=safety,
        effective_support=torch.ones_like(latents),
    )
    assert torch.equal(result.clean_delta, torch.zeros_like(latents))
    assert result.record["update"]["skip_reason"] == "unreliable_target_gradient"
```

Also cover non-finite rollback, unreliable target gradient, exactly two safety
constraints, and deterministic Gram/cosine records.

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
pytest -q tests/test_trust_region_controller.py
```

Expected: FAIL because the controller module is missing.

- [ ] **Step 3: Implement masked gradient extraction and solver orchestration**

Implement:

```python
def _masked_gradient(loss, latents, support, *, retain_graph):
    raw = torch.autograd.grad(
        loss, latents, retain_graph=retain_graph, allow_unused=True
    )[0]
    if raw is None:
        raw = torch.zeros_like(latents)
    mask = support.to(device=latents.device, dtype=latents.dtype)
    return raw, raw * mask


def _gram_matrix(gradients):
    return [
        [
            float(torch.sum(left.detach().float() * right.detach().float()).item())
            for right in gradients
        ]
        for left in gradients
    ]
```

Gradient order is target, identity, locality, drift. Require exactly the
`identity` and `outside_locality` observations and reject duplicates/missing
names.

For `trust_region`, use zero nominal in the envelope solve, then nominal
coefficients `(0, 0, 0, -eta)` in the second projection. For
`fixed_trust_matched`, use the normalized fixed nominal
`(-target_activation, -1, -1, -1)` before projection. Both modes use identical
target/safety bounds and radius.

Convert coefficient results back to a tensor:

```python
clean_delta = sum(
    coefficient * gradient
    for coefficient, gradient in zip(result.step, gradients)
)
```

Record raw/masked norms, full Gram/cosine matrices, residuals, requested and
achieved target progress, envelope tau, active constraints, multipliers,
radius, primal/dual residuals, and skip reasons.

Implement trust adaptation in `observe_outcome`: shrink radius by `0.5` when
`actual_progress / requested_progress < 0.25`; grow it by `1.5` when the ratio
is above `0.75` and `step_norm` used at least 90% of the current radius;
otherwise retain it. Clamp to configured minimum and maximum radii. Ignore
outcomes with non-positive requested progress.

- [ ] **Step 4: Run controller and archived-controller tests**

Run:

```bash
pytest -q tests/test_trust_region_controller.py tests/test_constraint_controller.py
```

Expected: PASS, demonstrating no archived-controller regression.

- [ ] **Step 5: Commit**

```bash
git add src/cci_diff/trust_region_controller.py tests/test_trust_region_controller.py
git commit -m "feat: add lexicographic trust-region controller"
```

---

### Task 5: SD2 Coordinate Mapping and Trust-Region Hooks

**Files:**
- Modify: `src/cci_diff/adapters/sd2_clean_cci.py`
- Modify: `src/cci_diff/sd2_bld_backend.py`
- Test: `tests/test_sd2_clean_cci.py`
- Test: `tests/test_sd2_bld_backend.py`

**Interfaces:**
- Consumes: `LexicographicTrustRegionController`,
  `NonTargetDriftEvaluator`, existing target/safety evaluators, scheduler, VAE,
  and `SD2DenoisingStep`.
- Produces `clean_delta_to_epsilon_delta(clean_delta, alpha_prod_t)`.
- Produces `TrustRegionCleanCCIGuidanceHook` with the same callable contract as
  `CleanCCIGuidanceHook`.
- Produces `FinalPreservationTrustRegionHook` with
  `apply_after_blend = True`.

- [ ] **Step 1: Write failing coordinate and schedule tests**

Add:

```python
def test_clean_delta_to_epsilon_delta_round_trips_predicted_clean_shift():
    sample = torch.tensor([1.0])
    epsilon = torch.tensor([0.2])
    alpha = torch.tensor(0.25)
    requested = torch.tensor([-0.1])
    mapped = clean_delta_to_epsilon_delta(requested, alpha)

    before = predict_clean_latents(sample, epsilon, alpha, "epsilon")
    after = predict_clean_latents(sample, epsilon + mapped, alpha, "epsilon")

    assert after - before == pytest.approx(requested)
```

Using the existing `Scheduler`, `VAE`, `Target`, `SD2DenoisingStep`, temporary
trace, and fake-controller patterns in `tests/test_sd2_clean_cci.py`, add
`test_trust_hook_skips_below_reliability_alpha` with alpha `0.05` and assert
the hook returns `None` and writes no trace row. Add
`test_trust_hook_records_clean_and_noise_displacements` with alpha `0.5`, a
fake controller returning a finite nonzero clean delta, and assert the parsed
trace row contains positive `clean_delta_norm` and `epsilon_delta_norm`.

In `tests/test_sd2_bld_backend.py`, add a callable fake hook with an
`observe_retention(phase, latents)` method. Run the fake backend and assert it
receives exactly one `scheduler_step` and one `blend` observation for every
step where the trust hook returned a non-null update.

- [ ] **Step 2: Write failing final-restoration tests**

Add `test_final_trust_hook_rejects_target_gain_that_worsens_safety` using a
two-value fake latent, identity measurement equal to the second value, and a
fake controller step that improves the first-value target while increasing
the second-value identity residual. Assert the hook returns `None`, the first
attempt is rejected, and its reason is `safety_envelope`.

Add `test_final_trust_hook_restores_drift_while_maintaining_target` with an
already-feasible first-value target and a second-value drift objective. Make
the fake controller reduce only the second value. Assert the returned latent
is non-null, final target probability remains at least the required
probability, final non-target drift is smaller than initial drift, and
`cumulative_norm <= spec.final_cumulative_radius + 1e-6`.

- [ ] **Step 3: Run adapter tests and verify failure**

Run:

```bash
pytest -q tests/test_sd2_clean_cci.py
```

Expected: FAIL because the conversion and new hooks are absent.

- [ ] **Step 4: Implement clean-coordinate conversion and denoising hook**

Implement:

```python
def clean_delta_to_epsilon_delta(clean_delta, alpha_prod_t):
    import torch
    alpha = torch.as_tensor(
        alpha_prod_t, device=clean_delta.device, dtype=clean_delta.dtype
    )
    beta = (1.0 - alpha).clamp_min(torch.finfo(clean_delta.dtype).eps)
    return -torch.sqrt(alpha / beta) * clean_delta
```

Build `TrustRegionCleanCCIGuidanceHook` alongside the archived hook. Reuse
source binding and predicted-clean decoding helpers. Bind the drift evaluator
to the same `ConstraintContext`. Evaluate all constraints for audit but pass
only identity and outside locality to the new controller. Apply bounded
support compensation:

```python
support = torch.where(
    latent_mask >= spec.support_floor,
    torch.clamp(
        1.0 / latent_mask.clamp_min(spec.support_floor),
        max=spec.maximum_blend_compensation,
    ) * latent_mask,
    torch.zeros_like(latent_mask),
)
```

Do not multiply the composed update by the soft mask again. Map the returned
clean delta to epsilon space and return `noise_pred + epsilon_delta`.

Store the unguided predicted-clean target residual and requested progress for
the active step. After computing the immediate guided predicted-clean image,
measure actual residual reduction and call `controller.observe_outcome`.

In `sd2_bld_backend.py`, after the scheduler step and after the BLD blend,
call `cci_guidance_hook.observe_retention("scheduler_step", latents)` and
`cci_guidance_hook.observe_retention("blend", latents)` only when that method
exists and the hook marked the current step active. The archived hook has no
such method and follows its unchanged path. The trust hook compares each
observed latent with its stored pre-update reference. Unlike the archived hook,
the trust hook holds one pending trace dictionary in memory and writes it only
after the `blend` observation has added scheduler and blend retention norms.

- [ ] **Step 5: Implement preservation-aware final hook**

At progress `1.0`, decode the actual post-BLD latent, compute all four
objectives, request a trust-region step, and backtrack over
`(1.0, 0.5, 0.25, 0.125)`. Accept using actual decoded measurements:

```python
target_ok = (
    candidate_target > current_target + 1e-7
    if current_target < required_probability
    else candidate_target >= required_probability
)
safety_ok = candidate_envelope <= current_envelope + feasibility_tolerance
drift_ok = (
    True
    if current_target < required_probability or current_envelope > 0
    else candidate_drift < current_drift - 1e-8
)
radius_ok = candidate_cumulative_norm <= final_cumulative_radius
```

Record every measurement and reason. Return only the last explicitly accepted
latent.

- [ ] **Step 6: Run adapter and backend tests**

Run:

```bash
pytest -q tests/test_sd2_clean_cci.py tests/test_sd2_bld_backend.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cci_diff/adapters/sd2_clean_cci.py src/cci_diff/sd2_bld_backend.py tests/test_sd2_clean_cci.py tests/test_sd2_bld_backend.py
git commit -m "feat: integrate trust-region SD2 guidance"
```

---

### Task 6: CLI, Runner, Audit, and Matched Fixed Variant

**Files:**
- Modify: `scripts/run_sd2_bld_cci.py`
- Modify: `scripts/run_clean_cci_pilot.py`
- Modify: `tests/test_clean_cci_cli.py`
- Modify: `tests/test_clean_cci_pilot.py`

**Interfaces:**
- Consumes: new evaluator, controller, and hooks.
- Produces: CLI controller modes `fixed_trust_matched` and `trust_region`;
  pilot variants `A10` and `A11`.

- [ ] **Step 1: Write failing CLI and variant tests**

Add:

```python
def test_clean_mode_accepts_trust_region_modes():
    for mode in ("fixed_trust_matched", "trust_region"):
        args = self.parse("--cci_controller_mode", mode)
        validate_mode_args(args)
        self.assertEqual(args.cci_controller_mode, mode)


def test_trust_region_mode_disables_archived_target_only_final_hook():
    from scripts.run_sd2_bld_cci import (
        uses_archived_final_correction,
        uses_trust_region,
    )

    assert uses_trust_region("trust_region")
    assert uses_trust_region("fixed_trust_matched")
    assert not uses_trust_region("feedback")
    assert not uses_archived_final_correction("trust_region")
    assert not uses_archived_final_correction("fixed_trust_matched")
    assert uses_archived_final_correction("feedback")
```

In pilot tests:

```python
def test_trust_region_variants_are_explicit_and_matched():
    assert VARIANTS["A10"]["controller_mode"] == "fixed_trust_matched"
    assert VARIANTS["A11"]["controller_mode"] == "trust_region"
    assert CONTROLLER_VARIANTS["fixed_trust_matched"] == "A10"
    assert CONTROLLER_VARIANTS["trust_region"] == "A11"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
pytest -q tests/test_clean_cci_cli.py tests/test_clean_cci_pilot.py
```

Expected: FAIL because parser choices, variants, and runtime wiring are absent.

- [ ] **Step 3: Wire evaluator and controller construction**

Extend parser choices:

```python
choices=[
    "disabled",
    "fixed_equal",
    "feedback",
    "fixed_trust_matched",
    "trust_region",
]
```

In `run_clean`, branch on:

```python
trust_mode = args.cci_controller_mode in {
    "fixed_trust_matched",
    "trust_region",
}
```

For trust modes:

- construct `NonTargetDriftEvaluator` from the already-loaded classifier and
  target index;
- select identity and outside locality evaluators by name and fail if either
  is missing;
- construct `LexicographicTrustRegionController(plan.controller.trust_region)`;
- construct `TrustRegionCleanCCIGuidanceHook`;
- construct `FinalPreservationTrustRegionHook` only when final iterations are
  positive and `--cci_disable_final_correction` is not supplied.

For archived modes, retain the exact current construction.

- [ ] **Step 4: Extend audit fields**

Record:

```python
"controller_mode": args.cci_controller_mode,
"resolved_trust_region": (
    asdict(plan.controller.trust_region) if trust_mode else None
),
"non_target_drift_optimization": trust_mode,
"archived_final_correction": (
    final_hook.record if not trust_mode and final_hook is not None else None
),
"trust_region_final_restoration": (
    trust_final_hook.record if trust_mode and trust_final_hook is not None else None
),
```

Keep the existing `"final_correction"` field for archived modes so consumers
remain compatible.

- [ ] **Step 5: Add paired pilot modes**

Add:

```python
"A10": {
    "hook": "clean_constraint",
    "controller_mode": "fixed_trust_matched",
    "projection": True,
},
"A11": {
    "hook": "clean_constraint",
    "controller_mode": "trust_region",
    "projection": True,
},
```

Extend `CONTROLLER_VARIANTS` and readable CLI help. Ensure both commands
receive identical graph, mask, seed, scheduler, and correction flags.

- [ ] **Step 6: Run CLI and pilot tests**

Run:

```bash
pytest -q tests/test_clean_cci_cli.py tests/test_clean_cci_pilot.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_sd2_bld_cci.py scripts/run_clean_cci_pilot.py tests/test_clean_cci_cli.py tests/test_clean_cci_pilot.py
git commit -m "feat: expose matched trust-region CCI modes"
```

---

### Task 7: Independent Continuous Preservation Evaluation

**Files:**
- Modify: `scripts/evaluate_clean_cci_ace.py`
- Modify: `tests/test_evaluate_clean_cci_ace.py`

**Interfaces:**
- Produces `continuous_non_target_drift(source_probabilities,
  output_probabilities, target_indices) -> np.ndarray`.

- [ ] **Step 1: Write failing metric tests**

Add:

```python
def test_continuous_non_target_drift_excludes_each_target():
    source = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    output = np.array([[0.2, 0.9, 0.5], [0.1, 0.7, 0.9]])

    values = continuous_non_target_drift(
        source, output, np.array([1, 0])
    )

    assert values == pytest.approx([0.15, 0.25])
```

Add an integration assertion that every ACE row has
`independent_non_target_drift` and grouped summaries include its paired mean.

- [ ] **Step 2: Run evaluation tests and verify failure**

Run:

```bash
pytest -q tests/test_evaluate_clean_cci_ace.py
```

Expected: FAIL because the metric is undefined.

- [ ] **Step 3: Implement continuous independent drift**

Implement:

```python
def continuous_non_target_drift(
    source_probabilities,
    output_probabilities,
    target_indices,
):
    source = np.asarray(source_probabilities, dtype=float)
    output = np.asarray(output_probabilities, dtype=float)
    targets = np.asarray(target_indices, dtype=int)
    if source.shape != output.shape or source.ndim != 2:
        raise ValueError("probability arrays must be aligned and two-dimensional")
    if len(targets) != len(source):
        raise ValueError("one target index is required per row")
    absolute = np.abs(output - source)
    absolute[np.arange(len(absolute)), targets] = np.nan
    return np.nanmean(absolute, axis=1)
```

Compute it from the independent ACE probabilities already loaded in
`evaluate()`. Write the per-row value and summarize its paired bootstrap
surface without replacing MNAC.

- [ ] **Step 4: Run evaluation tests**

Run:

```bash
pytest -q tests/test_evaluate_clean_cci_ace.py tests/test_evaluate_fid_sfid.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_clean_cci_ace.py tests/test_evaluate_clean_cci_ace.py
git commit -m "feat: evaluate continuous independent drift"
```

---

### Task 8: Matched-Success Frontier and Paired Inference

**Files:**
- Create: `src/cci_diff/matched_success.py`
- Create: `scripts/evaluate_matched_success.py`
- Create: `tests/test_matched_success.py`

**Interfaces:**
- Consumes long-form calibration and test CSV rows with `identity_cluster`,
  `source_id`, `seed`, `variant`, `effort`, `target_success`,
  `independent_non_target_drift`, `identity_cosine`, and
  `outside_semantic_l1`.
- Produces frozen interpolation weights, common success grid, drift at the
  highest common operating point, normalized frontier AUC, paired cluster
  bootstrap intervals, and acceptance flags.

- [ ] **Step 1: Write failing frontier tests**

Add:

```python
def test_calibration_frontier_drops_dominated_and_unsafe_points():
    rows = [
        point("A10", "low", success=0.40, drift=0.08, identity=0.95, locality=0.01),
        point("A10", "mid", success=0.60, drift=0.09, identity=0.95, locality=0.01),
        point("A10", "bad", success=0.60, drift=0.11, identity=0.95, locality=0.01),
        point("A10", "unsafe", success=0.80, drift=0.07, identity=0.85, locality=0.01),
    ]

    frontier = calibration_frontier(rows, identity_floor=0.90, locality_ceiling=0.02)

    assert [item.effort for item in frontier] == ["low", "mid"]


def test_common_grid_and_interpolation_are_frozen_from_calibration():
    fixed = [
        FrontierPoint("low", 0.40, 0.08),
        FrontierPoint("high", 0.70, 0.11),
    ]
    adaptive = [
        FrontierPoint("low", 0.45, 0.06),
        FrontierPoint("high", 0.75, 0.08),
    ]

    frozen = freeze_common_operating_points(
        {"A10": fixed, "A11": adaptive}, step=0.05
    )

    assert frozen.grid[0] == pytest.approx(0.45)
    assert frozen.grid[-1] == pytest.approx(0.70)
    assert sum(frozen.weights["A11"][0.70].values()) == pytest.approx(1.0)


def test_normalized_auc_uses_trapezoids():
    assert normalized_auc(
        [(0.4, 0.10), (0.6, 0.06)]
    ) == pytest.approx(0.08)
```

- [ ] **Step 2: Write failing paired-bootstrap test**

Use four identity clusters with paired A10/A11 rows and a fixed bootstrap seed.
Assert the adaptive-minus-fixed drift estimate is negative, the interval keys
are `estimate`, `low`, and `high`, and resampling the same seed gives identical
values.

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
pytest -q tests/test_matched_success.py
```

Expected: FAIL because the module does not exist.

- [ ] **Step 4: Implement frontier and frozen interpolation**

Create immutable `FrontierPoint`, `Interpolation`, and
`FrozenOperatingPoints` dataclasses. Aggregate calibration rows by
variant/effort, reject points whose mean identity is below `0.90` or mean
outside locality exceeds `0.02`, sort by success, and remove points with
greater-or-equal drift at lower-or-equal success.

For every `0.05` grid point inside the intersection of variant success ranges,
find adjacent frontier points and compute linear interpolation weights:

```python
right_weight = (target_success - left.success) / (
    right.success - left.success
)
weights = {
    left.effort: 1.0 - right_weight,
    right.effort: right_weight,
}
```

Never extrapolate. Compute nAUC with trapezoids divided by common success-range
width.

- [ ] **Step 5: Implement paired cluster bootstrap and acceptance**

Average seeds within source, combine effort rows with frozen weights, then
resample identity clusters with replacement 10,000 times. Return percentile
intervals for adaptive-minus-fixed drift, target success, safety pass rate,
and nAUC. Implement acceptance flags for target equivalence ±0.05, at least
10% drift reduction with upper difference interval below zero, lower nAUC
with upper interval below zero, joint safety at least 95%, and safety
non-inferiority within 0.02.

- [ ] **Step 6: Add the report CLI**

`scripts/evaluate_matched_success.py` accepts:

```text
--calibration_csv
--test_csv
--fixed_variant A10
--adaptive_variant A11
--output_json
--bootstrap_seed 42
--bootstrap_samples 10000
```

It freezes the operating points from calibration only, applies them unchanged
to test rows, writes JSON with calibration frontiers, weights, common grid,
test estimates, paired intervals, acceptance flags, and an explicit
`supported` boolean.

- [ ] **Step 7: Run matched-success tests**

Run:

```bash
pytest -q tests/test_matched_success.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/cci_diff/matched_success.py scripts/evaluate_matched_success.py tests/test_matched_success.py
git commit -m "feat: evaluate matched-success preservation"
```

---

### Task 9: Regression Verification and Usage Documentation

**Files:**
- Modify: `README.md`
- Modify: `examples/graphs/remove_smile_clean_cci.json`
- Modify: `examples/graphs/blond_hair_clean_cci.json`

**Interfaces:**
- Documents exact invocation and audit interpretation.

- [ ] **Step 1: Add explicit trust-region settings to example graphs**

Add the validated `controller.trust_region` object from Task 1 to both example
graphs. Keep archived controller fields unchanged.

- [ ] **Step 2: Document matched invocation**

Add:

```bash
python scripts/run_clean_cci_pilot.py \
  --controller_modes fixed_trust_matched trust_region \
  --features smile \
  --sample_count 10 \
  --num_inference_steps 35 \
  --seed 42 \
  --cci_post_attack none
```

Document:

- `A10` is the clean-coordinate fixed comparator;
- `A11` is the lexicographic adaptive optimizer;
- archived A2/A3 meanings are unchanged;
- independent ACE continuous drift is the primary preservation metric;
- target-only archived final correction is not used by A10/A11.

- [ ] **Step 3: Run formatting and focused suites**

Run:

```bash
git diff --check
pytest -q \
  tests/test_concept_graph.py \
  tests/test_constraints.py \
  tests/test_trust_region_solver.py \
  tests/test_trust_region_controller.py \
  tests/test_sd2_clean_cci.py \
  tests/test_sd2_bld_backend.py \
  tests/test_clean_cci_cli.py \
  tests/test_clean_cci_pilot.py \
  tests/test_evaluate_clean_cci_ace.py \
  tests/test_matched_success.py
```

Expected: `git diff --check` exits 0 and all focused tests pass.

- [ ] **Step 4: Run the complete suite**

Run:

```bash
pytest -q
```

Expected: all tests pass with no network access.

- [ ] **Step 5: Inspect compatibility and final diff**

Run:

```bash
git status --short
git diff --stat HEAD~8..HEAD
git log -9 --oneline
```

Expected: only the planned source, test, example, and documentation files are
changed; commits show one independently testable deliverable per task.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md examples/graphs/remove_smile_clean_cci.json examples/graphs/blond_hair_clean_cci.json
git commit -m "docs: explain trust-region CCI comparison"
```

---

## Execution Checkpoints

After Tasks 3, 5, and 7:

1. inspect the complete diff since the previous checkpoint;
2. confirm archived mode tests still pass;
3. stop if the solver needs an unplanned external dependency;
4. stop if the clean-coordinate mapping cannot reproduce the requested
   predicted-clean displacement in the adapter test;
5. stop if fixed and adaptive commands differ in any non-controller setting.

The implementation is complete only after the full test suite passes and the
new trace demonstrates finite solver residuals, clean-coordinate radius
compliance, and explicit final-restoration acceptance reasons.
