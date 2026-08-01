# Risk-Controlled Source-Only Mask Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generic, source-only selector that chooses the smallest globally discovered semantic mask with calibrated safe-success evidence, generates once, and can replay the already-generated mouth versus mouth+lips 300-sample experiment without rerunning diffusion.

**Architecture:** Global discovery exports an eligible candidate pool plus a reliable fallback instead of hard-coding one inference mask. A deterministic NumPy logistic model scores each candidate from source classifier probability, source Grad-CAM++, source segmentation, and frozen graph statistics; a separately fitted Platt calibrator and Wilson risk threshold gate minimum-area selection. Selection manifests are immutable and variant-independent, while a separate materializer joins those decisions to existing A0/A11 candidate outputs so no generated output can influence selection.

**Tech Stack:** Python 3.10+, NumPy, Pillow, PyTorch/torchvision and `pytorch-grad-cam` through the existing optional ML environment, JSON/CSV artifacts, pytest.

## Global Constraints

- Selection may inspect only the source image, source classifier probability, source Grad-CAM++, source semantic masks, frozen discovery evidence, and the frozen selector artifact.
- The selector must never inspect a generated image, generated probability, oracle score, FVA, FS, FID, sFID, MNAC, CD, COUT, or post-attack result.
- The independent oracle remains evaluation-only for final FR/MNAC/CD reporting.
- Graph discovery, selector fit, selector calibration, and held-out evaluation sample IDs must be disjoint.
- Candidate safe success is `desired_probability >= 0.53`, `identity_distance <= 0.08`, and `outside_locality <= 0.02`.
- Candidate feasibility requires saliency coverage `>= 0.80` and calibrated safe-success probability at or above the frozen risk threshold.
- The selector minimizes the lexicographic key `(mask_fraction, -safe_probability, -global_mean_effect, regions)` and uses the reliable graph fallback when no candidate is feasible.
- Risk control uses the lowest calibration score threshold with at least 60 accepted non-fallback rows and a one-sided 95% Wilson upper failure bound `<= 0.05`, with `z = 1.6448536269514722`; if none exists, inference always falls back.
- Logistic fitting uses deterministic float64 damped Newton updates, maximum update `<= 1e-10`, at most 200 iterations, five grouped folds, and regularization candidates `[1e-4, 1e-3, 1e-2, 1e-1, 1.0]`.
- Platt calibration uses the same deterministic solver with fixed L2 `1e-6`.
- A0 and A11 for one sample must consume the same precomputed selection record.
- Replay may read generated candidate outputs only after the complete source-only selection manifest has been written and hashed.
- Replaying all 300 IDs with a selector fitted or calibrated on any of those IDs must be labeled exploratory; held-out metric claims may include evaluation IDs only.
- Keep legacy `selected_regions` and `generation_regions` fields as compatibility aliases for the fallback, not as the candidate pool.
- Add no new mandatory project dependency.

---

## File Structure

- Modify `src/cci_diff/counterfactual_graph.py`: export eligible region-set candidates and choose an operational flip-rate-aware fallback.
- Modify `scripts/discover_counterfactual_graph.py`: serialize and report candidate/fallback semantics.
- Modify `src/cci_diff/individual_region_selection.py`: load full frozen candidate evidence without collapsing to the legacy generation alias.
- Create `src/cci_diff/risk_controlled_selection.py`: feature extraction, deterministic logistic/Platt fitting, Wilson thresholding, artifact validation, and source-only selection.
- Create `scripts/fit_region_selector.py`: validate disjoint cohorts, build supervised candidate rows from source features plus development outcomes, and freeze the selector artifact.
- Modify `scripts/run_individual_region_cci.py`: consume a frozen selector, precompute source-only choices, and reuse each choice across A0/A11 before one generation per variant.
- Create `scripts/materialize_adaptive_region_cohort.py`: join an immutable selection manifest to already-generated fixed-region rows without running diffusion.
- Create `tests/test_risk_controlled_selection.py`, `tests/test_fit_region_selector.py`, and `tests/test_materialize_adaptive_region_cohort.py`; update graph, loader, and runner tests.
- Modify the approved design document only if an implementation-discovered invariant needs clarification; do not silently weaken it.

---

### Task 1: Export a Candidate Pool and Reliable Fallback from Discovery

**Files:**
- Modify: `src/cci_diff/counterfactual_graph.py`
- Modify: `tests/test_counterfactual_graph.py`

**Interfaces:**
- Consumes: existing `RegionSetEvidence`, `_is_selection_eligible`, and Pareto annotations.
- Produces: `eligible_candidate_region_sets(evidence, minimum_samples) -> tuple[RegionTuple, ...]`, `select_reliable_fallback(evidence, required_flip_rate, minimum_samples) -> tuple[RegionTuple, str]`, and new `InfluenceGraphResult.candidate_region_sets`, `.fallback_regions`, `.fallback_status` fields.

- [ ] **Step 1: Replace the legacy-threshold test with failing candidate/fallback tests**

```python
def test_graph_exports_all_positive_ci_supported_candidates():
    evidence = {
        ("mouth",): _evidence(("mouth",), effect=.20, low=.05, flip=.70, area=.02, n=40),
        ("mouth", "upper_lip", "lower_lip"): _evidence(
            ("mouth", "upper_lip", "lower_lip"), effect=.31, low=.11,
            flip=.97, area=.05, n=40,
        ),
        ("nose",): _evidence(("nose",), effect=.01, low=-.02, flip=.10, area=.01, n=40),
    }
    result = build_influence_graph(
        target="smile", desired_value=0, evidence_by_regions=evidence,
        required_flip_rate=.95, minimum_samples=20,
    )
    assert result.candidate_region_sets == (
        ("mouth",), ("lower_lip", "mouth", "upper_lip"),
    )
    assert result.fallback_regions == ("lower_lip", "mouth", "upper_lip")
    assert result.fallback_status == "meets_required_flip_rate"
    assert result.selected_regions == result.fallback_regions


def test_fallback_reports_below_threshold_and_maximizes_flip_rate():
    evidence = {
        ("mouth",): _evidence(("mouth",), effect=.20, low=.05, flip=.70, area=.02, n=40),
        ("mouth", "upper_lip"): _evidence(
            ("mouth", "upper_lip"), effect=.28, low=.08, flip=.89, area=.04, n=40,
        ),
    }
    result = build_influence_graph(
        target="smile", desired_value=0, evidence_by_regions=evidence,
        required_flip_rate=.95, minimum_samples=20,
    )
    assert result.fallback_regions == ("mouth", "upper_lip")
    assert result.fallback_status == "below_required_flip_rate"
```

- [ ] **Step 2: Run the focused tests and confirm the old behavior fails**

Run: `pytest -q tests/test_counterfactual_graph.py -k 'exports_all_positive or fallback_reports'`

Expected: FAIL because `InfluenceGraphResult` has no `candidate_region_sets` or `fallback_regions`.

- [ ] **Step 3: Implement eligibility, reliable fallback, and compatibility aliases**

```python
def eligible_candidate_region_sets(
    evidence: Iterable[RegionSetEvidence], minimum_samples: int,
) -> tuple[RegionTuple, ...]:
    candidates = {
        item.regions for item in evidence
        if _is_selection_eligible(item, minimum_samples)
        and item.pareto_optimal
        and item.effect_ci_low > 0.0
        and item.mean_mask_fraction is not None
        and math.isfinite(item.flip_rate)
    }
    return tuple(sorted(candidates, key=lambda regions: (len(regions), regions)))


def select_reliable_fallback(
    evidence: Iterable[RegionSetEvidence],
    required_flip_rate: float,
    minimum_samples: int,
) -> tuple[RegionTuple, str]:
    items = tuple(evidence)
    candidate_sets = set(eligible_candidate_region_sets(items, minimum_samples))
    eligible = [item for item in items if item.regions in candidate_sets]
    if not eligible:
        raise ValueError("No eligible region set has positive supported target effect")
    reliable = [item for item in eligible if item.flip_rate >= required_flip_rate]
    if reliable:
        chosen = min(
            reliable,
            key=lambda item: (
                item.mean_mask_fraction, -item.mean_effect, -item.flip_rate,
                item.regions,
            ),
        )
        return chosen.regions, "meets_required_flip_rate"
    chosen = min(
        eligible,
        key=lambda item: (-item.flip_rate, -item.mean_effect, item.mean_mask_fraction, item.regions),
    )
    return chosen.regions, "below_required_flip_rate"
```

Extend `InfluenceGraphResult` with the three fields, set `selected_regions=fallback_regions`, and emit:

```python
"candidate_region_sets": [list(regions) for regions in self.candidate_region_sets],
"fallback_regions": list(self.fallback_regions),
"fallback_status": self.fallback_status,
"selected_regions": list(self.fallback_regions),
"generation_regions": list(self.fallback_regions),
```

- [ ] **Step 4: Run all graph tests**

Run: `pytest -q tests/test_counterfactual_graph.py`

Expected: PASS.

- [ ] **Step 5: Commit the discovery core**

```bash
git add src/cci_diff/counterfactual_graph.py tests/test_counterfactual_graph.py
git commit -m "feat: export adaptive region candidates"
```

---

### Task 2: Update Discovery Artifacts and Reports

**Files:**
- Modify: `scripts/discover_counterfactual_graph.py`
- Modify: `tests/test_counterfactual_graph_cli.py`

**Interfaces:**
- Consumes: `InfluenceGraphResult.candidate_region_sets`, `.fallback_regions`, and `.fallback_status` from Task 1.
- Produces: `influence_graph.json`, `selected_execution_graph.json`, and `discovery_report.md` that clearly distinguish adaptive candidates from fallback aliases.

- [ ] **Step 1: Write a failing CLI artifact assertion**

```python
payload = json.loads((output_dir / "influence_graph.json").read_text())
assert payload["candidate_region_sets"] == [["mouth"], ["lower_lip", "mouth", "upper_lip"]]
assert payload["fallback_regions"] == ["lower_lip", "mouth", "upper_lip"]
assert payload["provenance"]["selection_rule"] == "risk_controlled_candidate_pool_v1"
execution = json.loads((output_dir / "selected_execution_graph.json").read_text())
assert execution["discovery"]["required_flip_rate_role"] == "fallback_reliability_threshold"
assert execution["discovery"]["candidate_region_sets"] == payload["candidate_region_sets"]
report = (output_dir / "discovery_report.md").read_text()
assert "Adaptive candidate sets" in report
assert "Reliable fallback" in report
```

- [ ] **Step 2: Verify the assertions fail**

Run: `pytest -q tests/test_counterfactual_graph_cli.py`

Expected: FAIL on the legacy `pareto_target_efficiency_v1` and `legacy_compatibility_only` values.

- [ ] **Step 3: Serialize the new semantics**

Use these exact discovery fields:

```python
payload["region"]["components"] = list(result.fallback_regions)
payload["discovery"] = {
    "graph_type": "classifier_counterfactual_influence",
    "selection_status": result.selection_status,
    "selection_rule": "risk_controlled_candidate_pool_v1",
    "required_flip_rate": result.required_flip_rate,
    "required_flip_rate_role": "fallback_reliability_threshold",
    "candidate_region_sets": [list(x) for x in result.candidate_region_sets],
    "fallback_regions": list(result.fallback_regions),
    "fallback_status": result.fallback_status,
    "selected_regions": list(result.fallback_regions),
}
```

Render report bullets for candidate count/list, fallback, fallback status, and threshold role; retain the evidence table unchanged.

- [ ] **Step 4: Run CLI and graph tests**

Run: `pytest -q tests/test_counterfactual_graph_cli.py tests/test_counterfactual_graph.py`

Expected: PASS.

- [ ] **Step 5: Commit artifact changes**

```bash
git add scripts/discover_counterfactual_graph.py tests/test_counterfactual_graph_cli.py
git commit -m "feat: report selector candidates and fallback"
```

---

### Task 3: Load Full Frozen Candidate Evidence

**Files:**
- Modify: `src/cci_diff/individual_region_selection.py`
- Modify: `tests/test_individual_region_selection.py`

**Interfaces:**
- Consumes: graph JSON from Task 2.
- Produces: `FrozenRegionSetEvidence` and a revised `FrozenInfluencePolicy` with exact `candidate_region_sets`, `fallback_regions`, and immutable `region_set_evidence` mappings.

- [ ] **Step 1: Write failing loader tests for new and legacy graphs**

```python
def test_new_graph_preserves_all_candidates_and_full_evidence(tmp_path):
    path = _write_graph(tmp_path, candidate_region_sets=[
        ["mouth"], ["mouth", "upper_lip", "lower_lip"],
    ], fallback_regions=["mouth", "upper_lip", "lower_lip"])
    policy = load_frozen_influence_policy(path)
    assert policy.candidate_region_sets == (
        ("mouth",), ("lower_lip", "mouth", "upper_lip"),
    )
    assert policy.fallback_regions == ("lower_lip", "mouth", "upper_lip")
    mouth = policy.region_set_evidence[("mouth",)]
    assert mouth.mean_effect == pytest.approx(.2)
    assert mouth.flip_rate == pytest.approx(.7)
    assert mouth.effect_ci_low == pytest.approx(.05)
    assert mouth.mean_mask_fraction == pytest.approx(.02)


def test_legacy_graph_migrates_generation_regions_to_single_candidate(tmp_path):
    path = _write_legacy_graph(tmp_path, generation_regions=["mouth"])
    policy = load_frozen_influence_policy(path)
    assert policy.candidate_region_sets == (("mouth",),)
    assert policy.fallback_regions == ("mouth",)
```

- [ ] **Step 2: Run loader tests and observe failure**

Run: `pytest -q tests/test_individual_region_selection.py -k 'preserves_all_candidates or migrates_generation'`

Expected: FAIL because the loader currently replaces verified regions with `generation_regions` and retains only mean effects.

- [ ] **Step 3: Add immutable evidence and migration logic**

```python
@dataclass(frozen=True)
class FrozenRegionSetEvidence:
    mean_effect: float
    flip_rate: float
    effect_ci_low: float
    mean_mask_fraction: float


@dataclass(frozen=True)
class FrozenInfluencePolicy:
    target: str
    desired_value: int
    verified_regions: RegionTuple
    candidate_region_sets: tuple[RegionTuple, ...]
    fallback_regions: RegionTuple
    region_set_evidence: Mapping[RegionTuple, FrozenRegionSetEvidence]
    graph_path: str
    graph_sha256: str
```

Load candidates with:

```python
raw_candidates = payload.get("candidate_region_sets")
if raw_candidates is None:
    legacy = payload.get("generation_regions") or payload.get("selected_regions")
    raw_candidates = [legacy]
candidate_region_sets = tuple(sorted(
    {_canonical_regions(item) for item in raw_candidates},
    key=lambda item: (len(item), item),
))
fallback_regions = _canonical_regions(
    payload.get("fallback_regions")
    or payload.get("generation_regions")
    or payload.get("selected_regions")
)
```

Validate every candidate/fallback is nonempty, uses verified audit regions, and has finite complete evidence. Keep `global_effect()` as a compatibility method backed by `region_set_evidence`.

- [ ] **Step 4: Run all loader/legacy selector tests**

Run: `pytest -q tests/test_individual_region_selection.py`

Expected: PASS after updating fixtures to construct the new policy fields.

- [ ] **Step 5: Commit frozen-policy changes**

```bash
git add src/cci_diff/individual_region_selection.py tests/test_individual_region_selection.py
git commit -m "refactor: load frozen adaptive region policy"
```

---

### Task 4: Implement Source-Only Candidate Features and Safe Labels

**Files:**
- Create: `src/cci_diff/risk_controlled_selection.py`
- Create: `tests/test_risk_controlled_selection.py`

**Interfaces:**
- Consumes: `FrozenInfluencePolicy`, source desired-class probability, nonnegative 2-D Grad-CAM++, and semantic component masks.
- Produces: `FEATURE_NAMES`, `SafeSuccessThresholds`, `CandidateFeatureRow`, `safe_success_label`, `extract_candidate_feature_rows`, and `source_feature_signature`.

- [ ] **Step 1: Write failing feature/label tests**

```python
def test_feature_rows_use_only_source_inputs_and_exact_candidate_sets(policy):
    saliency = np.array([[4., 1.], [0., 0.]])
    masks = {
        "mouth": np.array([[1, 0], [0, 0]]),
        "upper_lip": np.array([[0, 1], [0, 0]]),
        "lower_lip": np.array([[0, 0], [1, 0]]),
    }
    rows = extract_candidate_feature_rows(.90, saliency, masks, policy)
    assert [row.regions for row in rows] == list(policy.candidate_region_sets)
    assert rows[0].values == pytest.approx((
        math.log(.90 / .10), .80, 4.0, .25, 1.0, .20, .70, .05,
    ))
    assert rows[1].coverage == pytest.approx(1.0)


@pytest.mark.parametrize(
    "probability,identity,outside,expected",
    [(.53, .08, .02, 1), (.529, .08, .02, 0), (.80, .081, .02, 0), (.80, .08, .021, 0)],
)
def test_safe_success_label_uses_frozen_thresholds(probability, identity, outside, expected):
    assert safe_success_label(probability, identity, outside) == expected
```

- [ ] **Step 2: Run feature tests and confirm import failure**

Run: `pytest -q tests/test_risk_controlled_selection.py -k 'feature_rows or safe_success_label'`

Expected: FAIL with `ModuleNotFoundError: cci_diff.risk_controlled_selection`.

- [ ] **Step 3: Implement exact features and provenance signature**

```python
FEATURE_NAMES = (
    "difficulty", "coverage", "saliency_density", "mask_fraction",
    "component_count", "global_mean_effect", "global_flip_rate",
    "global_effect_ci_low",
)

@dataclass(frozen=True)
class SafeSuccessThresholds:
    desired_probability: float = .53
    identity_distance: float = .08
    outside_locality: float = .02

def safe_success_label(
    desired_probability: float, identity_distance: float,
    outside_locality: float,
    thresholds: SafeSuccessThresholds = SafeSuccessThresholds(),
) -> int:
    return int(
        desired_probability >= thresholds.desired_probability
        and identity_distance <= thresholds.identity_distance
        and outside_locality <= thresholds.outside_locality
    )
```

For each exact candidate union, compute:

```python
difficulty = -(2 * policy.desired_value - 1) * math.log(
    np.clip(source_probability, 1e-12, 1.0 - 1e-12)
    / (1.0 - np.clip(source_probability, 1e-12, 1.0 - 1e-12))
)
verified_union = np.logical_or.reduce([
    np.asarray(masks[region]) > 0 for region in policy.verified_regions
])
coverage = float(
    saliency[union].sum()
    / max(float(saliency[verified_union].sum()), 1e-12)
)
mask_fraction = float(union.mean())
density = float(saliency[union].mean()) if union.any() else 0.0
values = (
    difficulty, coverage, density, mask_fraction, float(len(regions)),
    evidence.mean_effect, evidence.flip_rate, evidence.effect_ci_low,
)
```

The signature is SHA-256 over canonical JSON containing feature names, desired value, graph SHA, the predeclared candidate family, classifier SHA, Grad-CAM method/config, segmentation provenance, source transform, and the complete generation-policy signature (diffusion checkpoint, prompt, seed policy, controller, attack, and mask preprocessing digests).

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_risk_controlled_selection.py -k 'feature_rows or safe_success_label or signature'`

Expected: PASS.

- [ ] **Step 5: Commit the feature boundary**

```bash
git add src/cci_diff/risk_controlled_selection.py tests/test_risk_controlled_selection.py
git commit -m "feat: extract source-only mask features"
```

---

### Task 5: Fit Deterministic Logistic and Platt Models

**Files:**
- Modify: `src/cci_diff/risk_controlled_selection.py`
- Modify: `tests/test_risk_controlled_selection.py`

**Interfaces:**
- Consumes: dense float64 feature matrices and binary labels from Task 4.
- Produces: `LogisticModel`, `fit_logistic_newton`, `choose_grouped_l2`, `PlattCalibrator`, and deterministic JSON round trips.

- [ ] **Step 1: Add failing numerical and determinism tests**

```python
def test_newton_logistic_converges_and_is_deterministic():
    x = np.array([[-2.], [-1.], [1.], [2.]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    first = fit_logistic_newton(x, y, l2=.01)
    second = fit_logistic_newton(x, y, l2=.01)
    assert first == second
    assert np.all(np.diff(first.predict_probability(x)) > 0)
    assert first.iterations <= 200


def test_grouped_l2_never_splits_one_sample_across_folds():
    model, audit = choose_grouped_l2(X, y, sample_ids, folds=5)
    assert audit.l2 in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    for fold in audit.folds:
        assert set(fold.train_sample_ids).isdisjoint(fold.validation_sample_ids)


def test_model_json_round_trip_preserves_scores():
    restored = LogisticModel.from_dict(model.to_dict())
    np.testing.assert_array_equal(restored.predict_probability(X), model.predict_probability(X))
```

- [ ] **Step 2: Run and verify missing-symbol failures**

Run: `pytest -q tests/test_risk_controlled_selection.py -k 'newton or grouped_l2 or json_round_trip'`

Expected: FAIL because fitting/model types are not defined.

- [ ] **Step 3: Implement the deterministic solver and grouped CV**

```python
@dataclass(frozen=True)
class LogisticModel:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    l2: float
    iterations: int

    def predict_logit(self, values: np.ndarray) -> np.ndarray:
        x = (np.asarray(values, dtype=np.float64) - self.mean) / self.scale
        return self.intercept + x @ np.asarray(self.coefficients)

    def predict_probability(self, values: np.ndarray) -> np.ndarray:
        logits = np.clip(self.predict_logit(values), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))
```

`fit_logistic_newton` must standardize columns (`scale=1` for zero variance), exclude the intercept from L2, solve `(X'WX + penalty + damping) delta = X'(p-y) + penalty*beta`, increase damping when loss rises, and stop when `max(abs(delta)) <= 1e-10` or raise after 200 iterations. `choose_grouped_l2` sorts unique sample IDs, assigns round-robin folds, minimizes mean validation log loss, then chooses the smaller L2 on exact ties. Fit `PlattCalibrator` to one-column raw logits with L2 `1e-6`.

- [ ] **Step 4: Run the numerical suite twice**

Run: `pytest -q tests/test_risk_controlled_selection.py && pytest -q tests/test_risk_controlled_selection.py`

Expected: both runs PASS with identical serialized coefficients in the determinism test.

- [ ] **Step 5: Commit model fitting**

```bash
git add src/cci_diff/risk_controlled_selection.py tests/test_risk_controlled_selection.py
git commit -m "feat: fit deterministic safe-success model"
```

---

### Task 6: Calibrate Risk and Select the Minimal Feasible Mask

**Files:**
- Modify: `src/cci_diff/risk_controlled_selection.py`
- Modify: `tests/test_risk_controlled_selection.py`

**Interfaces:**
- Consumes: calibrated candidate probabilities, source feature rows, and graph fallback.
- Produces: `wilson_failure_upper_bound`, `choose_risk_threshold`, `FrozenSelectorArtifact`, `RiskControlledSelection`, and `select_risk_controlled_regions`.

- [ ] **Step 1: Add failing threshold and ranking tests**

```python
def test_wilson_threshold_is_lowest_safe_score_with_minimum_support():
    scores = np.array([.95] * 60 + [.50] * 40)
    labels = np.array([1] * 60 + [0] * 40)
    result = choose_risk_threshold(scores, labels, min_accepted=60, max_failure_ucb=.05)
    assert result.threshold == pytest.approx(.95)
    assert result.accepted == 60
    assert result.failure_upper_bound <= .05


def test_selector_chooses_minimum_area_feasible_candidate(policy, artifact):
    rows = [
        _row(("mouth",), coverage=.85, area=.02, effect=.20),
        _row(("lower_lip", "mouth", "upper_lip"), coverage=.99, area=.05, effect=.31),
    ]
    selection = select_risk_controlled_regions(rows, policy, artifact)
    assert selection.selected_regions == ("mouth",)
    assert selection.fallback_used is False


def test_selector_falls_back_when_small_mask_is_risky(policy, artifact):
    artifact = replace(artifact, risk_threshold=.80)
    rows = [_row(("mouth",), coverage=.90, area=.02, forced_probability=.60)]
    selection = select_risk_controlled_regions(rows, policy, artifact)
    assert selection.selected_regions == policy.fallback_regions
    assert selection.fallback_reason == "no_candidate_passed_coverage_and_risk"
```

- [ ] **Step 2: Run and verify failures**

Run: `pytest -q tests/test_risk_controlled_selection.py -k 'wilson_threshold or minimum_area or falls_back'`

Expected: FAIL because risk calibration/selection symbols are absent.

- [ ] **Step 3: Implement risk control and lexicographic selection**

```python
def wilson_failure_upper_bound(failures: int, total: int, z: float = 1.6448536269514722) -> float:
    if total <= 0:
        return 1.0
    p = failures / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return (centre + radius) / denominator


def choose_risk_threshold(scores, labels, *, min_accepted=60, max_failure_ucb=.05):
    valid = []
    for threshold in sorted(set(float(x) for x in scores)):
        accepted = np.asarray(scores) >= threshold
        count = int(accepted.sum())
        if count < min_accepted:
            continue
        failures = int((1 - np.asarray(labels)[accepted]).sum())
        upper = wilson_failure_upper_bound(failures, count)
        if upper <= max_failure_ucb:
            valid.append((threshold, count, failures, upper))
    return RiskThreshold(*min(valid)) if valid else RiskThreshold(1.0, 0, 0, 1.0, fallback_only=True)
```

Validate the artifact target, desired value, graph SHA, feature signature, classifier SHA, candidate sets, fallback, thresholds, and finite coefficients before scoring. Filter candidates on `coverage >= .80` and `safe_probability >= risk_threshold`; choose `min(feasible, key=(mask_fraction, -safe_probability, -global_mean_effect, regions))`; otherwise emit the exact graph fallback with candidate score audit and reason.

- [ ] **Step 4: Run all selector unit tests**

Run: `pytest -q tests/test_risk_controlled_selection.py`

Expected: PASS.

- [ ] **Step 5: Commit risk-controlled selection**

```bash
git add src/cci_diff/risk_controlled_selection.py tests/test_risk_controlled_selection.py
git commit -m "feat: select minimum risk-controlled mask"
```

---

### Task 7: Freeze a Selector Artifact from Disjoint Development Cohorts

**Files:**
- Create: `scripts/fit_region_selector.py`
- Create: `tests/test_fit_region_selector.py`

**Interfaces:**
- Consumes: graph JSON; an optional predeclared candidate-family JSON that may remove but never add graph candidates; a source-feature CSV containing one source-only row per `(sample_id, regions)`; a separate development-outcome CSV containing `cohort`, `sample_id`, `regions`, `desired_probability`, `identity_distance`, and `outside_locality`; graph-discovery, fit, calibration, and optional held-out split manifests; classifier/Grad-CAM/segmentation/generation-policy provenance arguments.
- Produces: `fit_region_selector(...) -> FrozenSelectorArtifact`, `selector_model.json`, `selector_fit_rows.csv`, `selector_calibration_rows.csv`, and `selector_fit_report.md`.

- [ ] **Step 1: Write failing CLI/service tests for disjointness and freezing**

```python
def test_fit_rejects_overlapping_cohorts(tmp_path, graph_path):
    rows = _development_rows(fit_ids=range(80), calibration_ids=range(60, 140))
    with pytest.raises(ValueError, match="disjoint"):
        fit_region_selector(graph_path, rows, tmp_path / "out", provenance=_provenance())


def test_fit_rejects_graph_discovery_or_evaluation_overlap(tmp_path, graph_path):
    rows = _development_rows(fit_ids=range(100), calibration_ids=range(100, 200))
    with pytest.raises(ValueError, match="pairwise disjoint"):
        fit_region_selector(
            graph_path, rows, tmp_path / "out", provenance=_provenance(),
            discovery_ids={99}, evaluation_ids={200, 201},
        )


def test_fit_writes_reproducible_frozen_artifact(tmp_path, graph_path):
    rows = _development_rows(fit_ids=range(100), calibration_ids=range(100, 200))
    first = fit_region_selector(graph_path, rows, tmp_path / "first", provenance=_provenance())
    second = fit_region_selector(graph_path, rows, tmp_path / "second", provenance=_provenance())
    assert first.to_dict() == second.to_dict()
    assert (tmp_path / "first/selector_model.json").read_bytes() == (
        tmp_path / "second/selector_model.json"
    ).read_bytes()
    assert first.risk_calibration.accepted >= 60 or first.risk_calibration.fallback_only
```

- [ ] **Step 2: Run and verify module import failure**

Run: `pytest -q tests/test_fit_region_selector.py`

Expected: FAIL with `ModuleNotFoundError: scripts.fit_region_selector`.

- [ ] **Step 3: Implement strict CSV schema, cohort checks, fitting, and artifact writes**

```python
REQUIRED_FIELDS = {
    "cohort", "sample_id", "regions", *FEATURE_NAMES,
    "desired_probability", "identity_distance", "outside_locality",
}

def fit_region_selector(
    graph_path, rows, output_dir, *, provenance,
    discovery_ids=frozenset(), evaluation_ids=frozenset(), candidate_family=None,
):
    policy = load_frozen_influence_policy(graph_path)
    candidates = (
        policy.candidate_region_sets if candidate_family is None
        else validate_candidate_family(candidate_family, policy)
    )
    if policy.fallback_regions not in candidates:
        raise ValueError("predeclared candidate family must include the graph fallback")
    fit_rows = [row for row in rows if row["cohort"] == "fit"]
    calibration_rows = [row for row in rows if row["cohort"] == "calibration"]
    fit_ids = {int(row["sample_id"]) for row in fit_rows}
    calibration_ids = {int(row["sample_id"]) for row in calibration_rows}
    cohorts = {
        "discovery": set(discovery_ids), "fit": fit_ids,
        "calibration": calibration_ids, "evaluation": set(evaluation_ids),
    }
    overlaps = {
        (left, right): cohorts[left] & cohorts[right]
        for left in cohorts for right in cohorts if left < right
        if cohorts[left] & cohorts[right]
    }
    if overlaps:
        raise ValueError(f"cohort sample IDs must be pairwise disjoint: {overlaps}")
    _validate_complete_candidate_rows(fit_rows, fit_ids, candidates)
    _validate_complete_candidate_rows(calibration_rows, calibration_ids, candidates)
    model, cv = choose_grouped_l2(_matrix(fit_rows), _labels(fit_rows), _ids(fit_rows))
    calibrator = fit_platt_calibrator(model.predict_logit(_matrix(calibration_rows)), _labels(calibration_rows))
    scores = calibrator.predict_probability(model.predict_logit(_matrix(calibration_rows)))
    risk = choose_risk_threshold(scores, _labels(calibration_rows))
    artifact = FrozenSelectorArtifact.from_fit(policy, model, calibrator, risk, provenance, cv)
    _write_artifacts(output_dir, artifact, fit_rows, calibration_rows)
    return artifact
```

Join source features to outcomes only on `(sample_id, regions)` so no output metric can enter `X`. Reject a predeclared family that adds a graph-ineligible set or omits the fallback; duplicate `(cohort, sample_id, regions)`; missing candidate rows; nonfinite values; unknown candidates; nonbinary labels; any pairwise overlap among discovery/fit/calibration/evaluation manifests; graph/provenance signature mismatch; or an artifact/provenance key containing `oracle`, `fid`, `sfid`, `fva`, `fs`, `mnac`, `cd`, or `cout`. Write `selector_data_manifest.json`, canonical sorted model JSON plus SHA-256, `selector_calibration_report.json`, and the two audited joined row files.

- [ ] **Step 4: Run fitting and selector suites**

Run: `pytest -q tests/test_fit_region_selector.py tests/test_risk_controlled_selection.py`

Expected: PASS.

- [ ] **Step 5: Commit the fitting entry point**

```bash
git add scripts/fit_region_selector.py tests/test_fit_region_selector.py
git commit -m "feat: freeze calibrated region selector"
```

---

### Task 8: Integrate One-Generation Source-Only Selection

**Files:**
- Modify: `scripts/run_individual_region_cci.py`
- Modify: `tests/test_run_individual_region_cci.py`

**Interfaces:**
- Consumes: `FrozenSelectorArtifact`, source classifier/Grad-CAM outputs, semantic masks, and frozen policy.
- Produces: `prepare_risk_controlled_policy(...)`, `write_selection_manifest(...)`, `--selector_model`, and one immutable sample-level decision reused by all requested variants.

- [ ] **Step 1: Add failing integration tests for ordering and variant reuse**

```python
def test_selection_manifest_is_complete_before_any_generation(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(module, "write_selection_manifest", lambda *a, **k: events.append("manifest"))
    monkeypatch.setattr(module, "generate_one", lambda *a, **k: events.append("generate"))
    run_individual_cci(_args(tmp_path, variants=("A0", "A11")))
    assert events[0] == "manifest"
    assert events.count("generate") == 2


def test_a0_and_a11_reuse_same_precomputed_regions(monkeypatch, tmp_path):
    selections = []
    monkeypatch.setattr(module, "generate_one", lambda *, selection, variant, **k: selections.append((variant, selection.selected_regions)))
    run_individual_cci(_args(tmp_path, variants=("A0", "A11")))
    assert selections == [
        ("A0", ("mouth",)),
        ("A11", ("mouth",)),
    ]


def test_generated_outputs_are_not_passed_to_selector(monkeypatch, tmp_path):
    def selector(rows, policy, artifact):
        assert all(not hasattr(row, "output_path") for row in rows)
        return _selection(("mouth",))
    monkeypatch.setattr(module, "select_risk_controlled_regions", selector)
    run_individual_cci(_args(tmp_path, variants=("A0",)))
```

- [ ] **Step 2: Run runner tests and observe failures**

Run: `pytest -q tests/test_run_individual_region_cci.py -k 'manifest_is_complete or reuse_same or not_passed'`

Expected: FAIL because selection is currently prepared within the generation flow and no frozen selector is accepted.

- [ ] **Step 3: Separate source decision preparation from generation**

Add parser arguments:

```python
parser.add_argument("--selector_model", required=True)
parser.add_argument("--selection_manifest", default=None)
parser.add_argument("--selection_only", action="store_true")
```

Prepare all decisions first:

```python
decisions = []
for sample in samples:
    source_probability, saliency = source_classifier_probability_and_saliency(sample.source_path)
    masks = load_semantic_component_masks(sample.sample_id, policy.verified_regions)
    rows = extract_candidate_feature_rows(source_probability, saliency, masks, policy)
    selection = select_risk_controlled_regions(rows, policy, selector_artifact)
    decisions.append(SourceSelectionRecord.from_selection(sample, selection, source_probability))
manifest_path, manifest_sha256 = write_selection_manifest(decisions, args, policy, selector_artifact)
if args.selection_only:
    return manifest_path
for decision in decisions:
    for variant in args.variants:
        generate_one(selection=decision, variant=variant, selection_manifest_sha256=manifest_sha256, ...)
```

The manifest contains no output paths or generated metrics; each record includes sample ID, source path/SHA, selected regions, source probability, all candidate feature/score audits, fallback status, graph SHA, selector SHA, classifier SHA, feature signature, and generation-policy signature. Write both `adaptive_selections.csv` and canonical `adaptive_selection_manifest.json`; compute SHA-256 over the canonical JSON payload excluding the digest field, then store the digest in the binding/report. Preserve the existing assertion/test that generation is called at most once per sample/variant.

- [ ] **Step 4: Run runner, selector, and legacy loader tests**

Run: `pytest -q tests/test_run_individual_region_cci.py tests/test_risk_controlled_selection.py tests/test_individual_region_selection.py`

Expected: PASS.

- [ ] **Step 5: Commit inference integration**

```bash
git add scripts/run_individual_region_cci.py tests/test_run_individual_region_cci.py
git commit -m "feat: apply source-only mask selector at inference"
```

---

### Task 9: Materialize an Adaptive Cohort from Existing Fixed Outputs

**Files:**
- Create: `scripts/materialize_adaptive_region_cohort.py`
- Create: `tests/test_materialize_adaptive_region_cohort.py`

**Interfaces:**
- Consumes: a finalized selection manifest and mappings such as `mouth=.../mouth/pilot_results.csv` and `lower_lip+mouth+upper_lip=.../mouth_upper_lower_lip/pilot_results.csv`.
- Produces: `materialize_adaptive_cohort(...)`, `adaptive_results.csv`, compatibility alias `pilot_results.csv`, `materialization_manifest.json`, and `materialization_report.md`; never invokes generation.

- [ ] **Step 1: Write failing replay integrity tests**

```python
def test_materializer_chooses_selected_root_for_both_variants(tmp_path):
    manifest = _selection_manifest(tmp_path, {1: ("mouth",), 2: ("lower_lip", "mouth", "upper_lip")})
    roots = _candidate_csvs(tmp_path, sample_ids=(1, 2), variants=("A0", "A11"))
    rows = materialize_adaptive_cohort(manifest, roots, tmp_path / "adaptive", expected_variants=("A0", "A11"))
    assert [(r["sample_id"], r["variant"], r["selected_regions"]) for r in rows] == [
        ("1", "A0", '["mouth"]'), ("1", "A11", '["mouth"]'),
        ("2", "A0", '["lower_lip", "mouth", "upper_lip"]'),
        ("2", "A11", '["lower_lip", "mouth", "upper_lip"]'),
    ]


def test_materializer_rejects_incomplete_or_mismatched_outputs(tmp_path):
    manifest = _selection_manifest(tmp_path, {1: ("mouth",)})
    roots = _candidate_csvs(tmp_path, sample_ids=(1,), variants=("A0",))
    with pytest.raises(ValueError, match="A11"):
        materialize_adaptive_cohort(manifest, roots, tmp_path / "adaptive", expected_variants=("A0", "A11"))


def test_materializer_does_not_import_or_call_diffusion(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("generation called"))
    materialize_adaptive_cohort(_manifest(tmp_path), _roots(tmp_path), tmp_path / "adaptive")


def test_materializer_rejects_manifest_changed_after_hashing(tmp_path):
    manifest = _selection_manifest(tmp_path, {1: ("mouth",)})
    payload = json.loads(manifest.read_text())
    payload["decisions"][0]["selected_regions"] = ["lower_lip", "mouth", "upper_lip"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        materialize_adaptive_cohort(manifest, _roots(tmp_path), tmp_path / "adaptive")
```

- [ ] **Step 2: Run and verify import failure**

Run: `pytest -q tests/test_materialize_adaptive_region_cohort.py`

Expected: FAIL with `ModuleNotFoundError: scripts.materialize_adaptive_region_cohort`.

- [ ] **Step 3: Implement an auditable join, not file copying or generation**

```python
def materialize_adaptive_cohort(selection_manifest, candidate_results, output_dir, *, expected_variants=("A0", "A11")):
    manifest_bytes = Path(selection_manifest).read_bytes()
    selections = _load_and_validate_selection_manifest(manifest_bytes)
    indexed = {
        regions: _index_candidate_rows(path, regions, expected_variants)
        for regions, path in candidate_results.items()
    }
    output_rows = []
    for decision in sorted(selections, key=lambda row: int(row["sample_id"])):
        regions = tuple(decision["selected_regions"])
        for variant in expected_variants:
            key = (int(decision["sample_id"]), variant)
            source = dict(indexed[regions][key])
            if not Path(source["output_path"]).is_file():
                raise FileNotFoundError(source["output_path"])
            source["selected_regions"] = json.dumps(regions)
            source["selection_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            source["source_candidate_results"] = str(candidate_results[regions])
            output_rows.append(source)
    _write_materialized_artifacts(output_dir, output_rows, selections, candidate_results)
    return output_rows
```

Validate identical available sample-ID sets across roots before joining, exactly one row per `(sample_id, variant)`, source path consistency, output existence, the canonical selection-manifest hash before reading any candidate CSV, and no `output_path` field in decisions. Add `--evaluation_ids` and `--exploratory` flags: if the selector artifact declares any discovery/fit/calibration overlap with materialized IDs, reject unless `--exploratory`; the report must then state `NOT HELD-OUT` prominently. Add a synthetic direct-versus-replay integration assertion that both modes resolve the same deterministic selected output path.

- [ ] **Step 4: Run replay tests**

Run: `pytest -q tests/test_materialize_adaptive_region_cohort.py`

Expected: PASS.

- [ ] **Step 5: Commit replay support**

```bash
git add scripts/materialize_adaptive_region_cohort.py tests/test_materialize_adaptive_region_cohort.py
git commit -m "feat: replay adaptive masks from fixed outputs"
```

---

### Task 10: Validate the Existing 300-Sample Dataset and Produce the Replay Command

**Files:**
- Modify: `tests/test_materialize_adaptive_region_cohort.py`
- Create when artifacts exist: `outputs/attacked_a0_a11_smile300_seed42/adaptive_source_only/` (generated, ignored experiment artifacts; do not commit)

**Interfaces:**
- Consumes: both existing fixed-region `pilot_results.csv` files, a separately fitted selector artifact, and a complete source-only selection manifest.
- Produces: a validated adaptive results CSV over matching IDs and an explicit held-out/exploratory report.

- [ ] **Step 1: Add a regression test for real pilot-result headers and post-attack paths**

```python
def test_real_schema_preserves_post_attack_selected_output(tmp_path):
    row = _realistic_pilot_row(
        output_path="run/sd2_bld_grid.png",
        candidate_output_path="run/candidates/d8/sd2_bld_grid.png",
        post_attack_escalated="True",
    )
    materialized = _materialize_one(tmp_path, row)
    assert materialized[0]["output_path"].endswith("sd2_bld_grid.png")
    assert materialized[0]["post_attack_escalated"] == "True"
    assert materialized[0]["candidate_output_path"].endswith("candidates/d8/sd2_bld_grid.png")
```

- [ ] **Step 2: Run all implementation tests before touching experiment artifacts**

Run: `pytest -q tests/test_counterfactual_graph.py tests/test_counterfactual_graph_cli.py tests/test_individual_region_selection.py tests/test_risk_controlled_selection.py tests/test_fit_region_selector.py tests/test_run_individual_region_cci.py tests/test_materialize_adaptive_region_cohort.py`

Expected: PASS.

- [ ] **Step 3: Validate completion and ID/variant parity of the two fixed runs**

Run:

```bash
.venv-ml/bin/python - <<'PY'
import csv
from pathlib import Path
root = Path("outputs/attacked_a0_a11_smile300_seed42")
for name in ("mouth", "mouth_upper_lower_lip"):
    with (root / name / "pilot_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = {(int(row["sample_id"]), row["variant"]) for row in rows}
    ids = {sample_id for sample_id, _ in keys}
    print(name, len(rows), len(ids), sorted({variant for _, variant in keys}))
PY
```

Expected before replay:

```text
mouth 600 300 ['A0', 'A11']
mouth_upper_lower_lip 600 300 ['A0', 'A11']
```

If the second line is incomplete, stop replay and report that fixed generation is still unfinished; do not manufacture missing rows.

- [ ] **Step 4: Generate source-only decisions without diffusion**

Run after a disjointly trained selector exists:

```bash
.venv-ml/bin/python scripts/run_individual_region_cci.py \
  --influence_graph outputs/counterfactual_discovery_smile/influence_graph.json \
  --selector_model outputs/selector_smile/selector_model.json \
  --selection_only \
  --selection_manifest outputs/attacked_a0_a11_smile300_seed42/adaptive_source_only/selection_manifest.json \
  --sample_ids_file outputs/attacked_a0_a11_smile300_seed42/sample_ids.txt
```

Expected: command exits 0, writes/hash-verifies 300 source-only decisions, performs zero diffusion calls, and reports counts for mouth, mouth+lips, and fallback.

- [ ] **Step 5: Materialize both variants from the existing results**

Run:

```bash
.venv-ml/bin/python scripts/materialize_adaptive_region_cohort.py \
  --selection_manifest outputs/attacked_a0_a11_smile300_seed42/adaptive_source_only/selection_manifest.json \
  --candidate_results mouth=outputs/attacked_a0_a11_smile300_seed42/mouth/pilot_results.csv \
  --candidate_results lower_lip+mouth+upper_lip=outputs/attacked_a0_a11_smile300_seed42/mouth_upper_lower_lip/pilot_results.csv \
  --output_dir outputs/attacked_a0_a11_smile300_seed42/adaptive_source_only \
  --expected_variants A0 A11 \
  --expected_count 300
```

Expected: exits 0 with 600 rows, exactly 300 IDs, identical selected regions for each ID's A0/A11 pair, and no diffusion execution.

- [ ] **Step 6: Run the existing paper metric pipeline on the adaptive CSV**

Use the repository's current evaluation commands with `outputs/attacked_a0_a11_smile300_seed42/adaptive_source_only/pilot_results.csv` as the experiment input. Compare adaptive A0/A11 against each fixed region on the declared held-out IDs only; if `--exploratory` was required, title every table/report `Exploratory (Not Held-Out)` and do not use it as the paper's primary result.

- [ ] **Step 7: Run the full test suite**

Run: `pytest -q`

Expected: PASS with no regressions.

- [ ] **Step 8: Commit the real-schema regression test**

```bash
git add tests/test_materialize_adaptive_region_cohort.py
git commit -m "test: validate adaptive replay schema"
```

---

### Task 11: Final Documentation and Reproducibility Audit

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-01-risk-controlled-source-mask-selection-design.md` only if wording must reflect the implemented artifact schema.

**Interfaces:**
- Consumes: all commands and artifact names from Tasks 1-10.
- Produces: a reproducible paper-facing workflow and explicit leakage limitations.

- [ ] **Step 1: Add a failing documentation contract test**

Add to `tests/test_risk_controlled_selection.py`:

```python
def test_readme_documents_source_only_selector_and_oracle_boundary():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "Risk-controlled source-only mask selection" in text
    assert "Oracle metrics are evaluation-only" in text
    assert "materialize_adaptive_region_cohort.py" in text
    assert "NOT HELD-OUT" in text
```

- [ ] **Step 2: Run and verify the documentation test fails**

Run: `pytest -q tests/test_risk_controlled_selection.py::test_readme_documents_source_only_selector_and_oracle_boundary`

Expected: FAIL until README documentation is present.

- [ ] **Step 3: Document the exact workflow and paper claims**

Add a README section named `Risk-controlled source-only mask selection` that includes:

```text
Discovery exports candidate masks and a reliability fallback. Selector fit and
calibration use disjoint development IDs. At inference, classifier probability,
Grad-CAM++, and source segmentation choose one mask before generation; A0 and A11
reuse that decision. Oracle metrics are evaluation-only. Existing fixed-region
outputs may be joined with materialize_adaptive_region_cohort.py only after the
selection manifest is complete and hashed. Any overlap with fit/calibration IDs
must be reported as NOT HELD-OUT.
```

Include the exact fit, selection-only, materialization, and evaluation commands, plus definitions for fallback rate, non-fallback support, Wilson failure UCB, and mask-selection counts.

- [ ] **Step 4: Run documentation and full verification**

Run: `pytest -q && git diff --check`

Expected: tests PASS and `git diff --check` prints no output.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/superpowers/specs/2026-08-01-risk-controlled-source-mask-selection-design.md tests/test_risk_controlled_selection.py
git commit -m "docs: explain risk-controlled mask workflow"
```

- [ ] **Step 6: Review final history and artifact boundary**

Run: `git status --short && git log --oneline -12`

Expected: clean tracked worktree; experiment outputs remain ignored; history contains small discovery, feature, fitting, inference, replay, test, and documentation commits.
