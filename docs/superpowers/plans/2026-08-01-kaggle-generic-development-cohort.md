# Kaggle Generic Development Cohort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one resumable, Kaggle-portable `--data_size` workflow for target-generic A11 region discovery, selector fitting, and calibration on development IDs that are strictly disjoint from the existing evaluation 300.

**Architecture:** Add pure cohort-allocation, beam-search, device-resolution, and content-addressed-cache modules under `src/cci_diff`. A thin orchestration script composes those modules with the existing Grad-CAM++, concept-graph, A11 generation, graph-discovery, feature-extraction, and selector-fitting code. Run manifests reference immutable cache entries so a 30-image run and a later 300-image run share exact interventions without sharing fitted artifacts.

**Tech Stack:** Python 3.10, NumPy, PyTorch, PIL, existing CCI scripts, pytest, JSON/CSV artifacts, CUDA/MPS/CPU.

## Global Constraints

- The existing evaluation-300 IDs are exclusions only and never enter development outcomes.
- `--data_size 30` allocates `4/10/16`; `--data_size 300` allocates `40/100/160`.
- The allocation ratio is always `2:5:8`, with deterministic largest-remainder rounding and stable nested role membership.
- Discovery, fitting, calibration, and primary evaluation use A11 only.
- Candidate masks are target-generic semantic combinations of at most three components.
- Atomic shortlist size is six, beam width is four, and at most six candidates are evaluated per cardinality.
- Production Wilson calibration requirements remain 60 accepted non-fallback observations and failure UCB at most 0.05.
- No smoke-specific algorithm branch or relaxed threshold is allowed.
- All generation reuse is content-addressed and fails closed on provenance mismatch.
- Kaggle must work from `/kaggle/input` with outputs under `/kaggle/working` and no required network download.

---

### Task 1: Deterministic Development Cohorts

**Files:**
- Create: `src/cci_diff/development_cohort.py`
- Create: `tests/test_development_cohort.py`

**Interfaces:**
- Produces: `DevelopmentCounts`, `DevelopmentCohort`, `allocate_development_counts(data_size: int)`, and `assign_development_cohort(eligible_ids, evaluation_ids, data_size, seed)`.
- Consumes: integer source IDs and the immutable evaluation exclusion set.

- [ ] **Step 1: Write failing allocation and nesting tests**

```python
def test_standard_sizes_use_two_five_eight_ratio():
    assert allocate_development_counts(30) == DevelopmentCounts(4, 10, 16)
    assert allocate_development_counts(300) == DevelopmentCounts(40, 100, 160)

def test_larger_cohort_preserves_ids_in_each_role():
    eligible = range(10_000)
    small = assign_development_cohort(eligible, {1, 2, 3}, 30, 42)
    large = assign_development_cohort(eligible, {1, 2, 3}, 300, 42)
    for role in ("discovery", "fit", "calibration"):
        assert set(getattr(small, role)).issubset(getattr(large, role))
    assert small.all_ids.isdisjoint({1, 2, 3})
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_development_cohort.py`

Expected: FAIL because `cci_diff.development_cohort` does not exist.

- [ ] **Step 3: Implement immutable allocation and stable role buckets**

```python
@dataclass(frozen=True)
class DevelopmentCounts:
    discovery: int
    fit: int
    calibration: int

def allocate_development_counts(data_size: int) -> DevelopmentCounts:
    if isinstance(data_size, bool) or data_size < 15:
        raise ValueError("data_size must be at least 15")
    weights = (2, 5, 8)
    exact = tuple(data_size * value / 15 for value in weights)
    counts = [math.floor(value) for value in exact]
    order = sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))
    for index in order[: data_size - sum(counts)]:
        counts[index] += 1
    return DevelopmentCounts(*counts)
```

Assign each eligible ID to one of 15 stable hash buckets using
`sha256(f"{seed}:role:{sample_id}")`, map buckets `0:2`, `2:7`, and `7:15` to
the three roles, order within each role by a second SHA-256 key, exclude all
evaluation IDs before truncation, and fail with required/available counts if a
role is undersupplied.

- [ ] **Step 4: Add validation tests**

Cover `data_size < 15`, duplicate eligible IDs, insufficient role buckets,
pairwise disjointness, deterministic serialization, and exact exclusion.

- [ ] **Step 5: Run the task tests**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_development_cohort.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cci_diff/development_cohort.py tests/test_development_cohort.py
git commit -m "feat: allocate nested development cohorts"
```

### Task 2: Target-Generic Beam Search

**Files:**
- Create: `src/cci_diff/generic_region_discovery.py`
- Create: `tests/test_generic_region_discovery.py`
- Modify: `src/cci_diff/region_screening.py`

**Interfaces:**
- Consumes: source-only component screening rows and `RegionSetEvidence` after each A11 level.
- Produces: `BeamSearchConfig`, `AtomicComponentScore`, `shortlist_atomic_components(rows, config)`, `propose_region_sets(shortlist, beam, cardinality, config)`, and `advance_beam(evidence, config)`.

- [ ] **Step 1: Write failing genericity and budget tests**

```python
def test_beam_search_has_no_target_specific_regions():
    rows = screening_rows_for(["hair", "nose", "mouth", "neck", "eye"])
    shortlist = shortlist_atomic_components(rows, BeamSearchConfig())
    assert shortlist[0] == "hair"
    assert set(shortlist).issubset({"hair", "nose", "mouth", "neck", "eye"})

def test_each_level_is_deterministic_and_budgeted():
    config = BeamSearchConfig(atomic_shortlist_size=6, beam_width=4,
                              level_evaluation_budget=6, max_components=3)
    pairs = propose_region_sets(("a", "b", "c", "d", "e", "f"),
                                (("a",), ("b",), ("c",), ("d",)), 2, config)
    assert len(pairs) <= 6
    assert all(len(item) == 2 for item in pairs)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_generic_region_discovery.py`

Expected: FAIL because the module is missing.

- [ ] **Step 3: Implement source-only shortlist and canonical expansion**

Use aggregate median density, median captured mass, coverage frequency, mean
mask fraction, and canonical component name for stable screening. Expand only
by adding shortlist atoms, canonicalize tuples, remove already-evaluated sets,
pre-rank expansions by union source coverage/density, and cap every level at
six sets.

- [ ] **Step 4: Implement evidence-ranked beam advancement**

Reuse `eligible_candidate_region_sets` and Pareto annotations from
`counterfactual_graph.py`. Rank supported candidates by Pareto status,
positive confidence, flip rate, mean effect, smaller area, and tuple; retain
four. If no supported candidate exists, retain the strongest deterministic
fallback so discovery can report fallback-only status.

- [ ] **Step 5: Add exhaustive edge tests**

Cover ties, duplicate unions, absent components, singleton-to-pair and
pair-to-triple expansion, maximum tuple size three, six-set level budgets,
four-set beam budgets, and fallback-only evidence.

- [ ] **Step 6: Run focused tests**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_generic_region_discovery.py tests/test_counterfactual_graph.py tests/test_region_screening.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cci_diff/generic_region_discovery.py src/cci_diff/region_screening.py tests/test_generic_region_discovery.py
git commit -m "feat: discover generic region combinations"
```

### Task 3: Content-Addressed A11 Intervention Cache

**Files:**
- Create: `src/cci_diff/intervention_cache.py`
- Create: `tests/test_intervention_cache.py`
- Modify: `scripts/run_counterfactual_region_interventions.py`
- Test: `tests/test_counterfactual_region_interventions.py`

**Interfaces:**
- Produces: `InterventionCacheKey`, `CachedIntervention`, `cache_key_for(...)`, `load_cached_intervention(root, key)`, and `store_cached_intervention(root, key, observation, artifacts)`.
- Consumes: exact source, mask, checkpoint, classifier, graph, seed, and A11-policy digests.

- [ ] **Step 1: Write failing cache identity tests**

```python
def test_cache_key_changes_for_every_scientific_input():
    base = cache_key_for(**valid_inputs())
    for field in ("source_sha256", "mask_sha256", "checkpoint_sha256",
                  "classifier_sha256", "policy_sha256", "seed"):
        changed = valid_inputs() | {field: different_value(field)}
        assert cache_key_for(**changed) != base

def test_atomic_cache_round_trip_rejects_partial_entry(tmp_path):
    entry = store_cached_intervention(tmp_path, key, observation, artifacts)
    assert load_cached_intervention(tmp_path, key) == entry
    (entry.path / "complete.json").unlink()
    assert load_cached_intervention(tmp_path, key) is None
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_intervention_cache.py`

Expected: FAIL because the cache module is missing.

- [ ] **Step 3: Implement canonical cache keys and atomic completion**

Serialize a versioned canonical JSON payload, hash it with SHA-256, write
metadata and observation to a temporary sibling directory, fsync files, rename
the directory atomically, and write `complete.json` last. Existing complete
entries are immutable; conflicting metadata raises `ValueError`.

- [ ] **Step 4: Make intervention execution A11-policy driven**

Extend `build_intervention_command` to consume a frozen generation-policy JSON
and emit the exact A11 hook, trust-region controller, projection, inference,
dtype, mask, and smooth-boundary post-attack flags. Remove the hardcoded
feedback/attack-disabled assumptions for this new call path while preserving
legacy defaults for existing callers.

- [ ] **Step 5: Integrate cache lookup and store around subprocess execution**

Before generation, validate and reuse a complete cache entry. After successful
generation and `load_completed_observation`, store the observation and artifact
bindings atomically. Reconstruct CSV rows from validated entries in canonical
sample/region order so restarts never duplicate rows.

- [ ] **Step 6: Test resume without subprocess invocation**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_intervention_cache.py tests/test_counterfactual_region_interventions.py`

Expected: PASS, including a test where the second invocation records zero
subprocess calls.

- [ ] **Step 7: Commit**

```bash
git add src/cci_diff/intervention_cache.py scripts/run_counterfactual_region_interventions.py tests/test_intervention_cache.py tests/test_counterfactual_region_interventions.py
git commit -m "feat: cache resumable A11 interventions"
```

### Task 4: Development Data With External Evaluation IDs

**Files:**
- Modify: `scripts/prepare_adaptive_replay_data.py`
- Modify: `scripts/fit_region_selector.py`
- Modify: `tests/test_prepare_adaptive_replay_data.py`
- Modify: `tests/test_fit_region_selector.py`

**Interfaces:**
- `prepare_adaptive_replay_data(..., development_cohort: DevelopmentCohort, evaluation_ids: Iterable[int])` writes discovery, fitting, calibration, and external evaluation role files.
- `provenance_from_manifests(...)` requires source-feature IDs to equal development IDs only while preserving evaluation IDs solely for held-out validation.

- [ ] **Step 1: Write failing external-evaluation tests**

```python
def test_preparation_accepts_evaluation_ids_without_candidate_outputs(tmp_path):
    cohort = DevelopmentCohort(discovery=(1,), fit=(2,), calibration=(3,))
    result = prepare_adaptive_replay_data(
        candidate_results_for((1, 2, 3)), tmp_path / "out",
        development_cohort=cohort, evaluation_ids=(100, 101))
    assert result["cohorts"]["evaluation"] == [100, 101]
    assert all(100 not in row for row in read_development_rows(tmp_path / "out"))
```

- [ ] **Step 2: Run tests and verify the current same-pool assumption fails**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_prepare_adaptive_replay_data.py tests/test_fit_region_selector.py`

Expected: FAIL because counts currently must partition one candidate-results pool.

- [ ] **Step 3: Implement separate development and evaluation domains**

Read candidate outcomes only for discovery, fitting, and calibration IDs.
Require all four role sets to be non-empty and pairwise disjoint, write the
external evaluation IDs into the split manifest, and keep
`evaluation_outputs_exported_to_selector_data=false`.

- [ ] **Step 4: Tighten feature-manifest validation**

Require source-feature IDs to equal `discovery ∪ fit ∪ calibration`; require
evaluation IDs to be absent. Continue embedding all four role lists into the
frozen selector so inference accepts only the external evaluation cohort.

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_prepare_adaptive_replay_data.py tests/test_fit_region_selector.py tests/test_run_individual_region_cci.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_adaptive_replay_data.py scripts/fit_region_selector.py tests/test_prepare_adaptive_replay_data.py tests/test_fit_region_selector.py
git commit -m "feat: separate development from evaluation IDs"
```

### Task 5: Portable Device and Path Resolution

**Files:**
- Create: `src/cci_diff/runtime_environment.py`
- Create: `tests/test_runtime_environment.py`
- Modify: `scripts/screen_counterfactual_regions.py`
- Modify: `scripts/run_counterfactual_region_interventions.py`
- Modify: `scripts/run_individual_region_cci.py`

**Interfaces:**
- Produces: `resolve_device(requested: str, torch_module) -> str` and `validate_local_artifacts(paths: Mapping[str, Path])`.
- Consumes: `auto|cuda|mps|cpu` and injected torch capability checks.

- [ ] **Step 1: Write failing precedence tests**

```python
@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"),
                          (False, False, "cpu")])
def test_auto_device_precedence(cuda, mps, expected):
    assert resolve_device("auto", fake_torch(cuda, mps)) == expected
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_runtime_environment.py`

Expected: FAIL because the runtime module is missing.

- [ ] **Step 3: Implement strict device resolution and local artifact checks**

Reject unavailable explicitly requested accelerators, prefer CUDA then MPS
then CPU for `auto`, and never download models unless the existing explicit
download flag is passed.

- [ ] **Step 4: Wire `--device auto` through source screening, A11 generation, and held-out selection**

Resolve once at process start, write the resolved value to every manifest,
and derive `float16` only for CUDA when dtype is `auto`; MPS and CPU use
`float32`.

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_runtime_environment.py tests/test_counterfactual_region_interventions.py tests/test_run_individual_region_cci.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cci_diff/runtime_environment.py scripts/screen_counterfactual_regions.py scripts/run_counterfactual_region_interventions.py scripts/run_individual_region_cci.py tests/test_runtime_environment.py
git commit -m "feat: resolve portable inference devices"
```

### Task 6: One `data_size` Development Orchestrator

**Files:**
- Create: `scripts/run_generic_region_development.py`
- Create: `tests/test_run_generic_region_development.py`
- Modify: `scripts/discover_counterfactual_graph.py`

**Interfaces:**
- Produces: `run_development(args) -> dict[str, Any]` and CLI `scripts/run_generic_region_development.py --data_size N ...`.
- Consumes: evaluation exclusion manifest, template graph, dataset/model paths, generation policy, cache root, run root, seed, and device.
- Defines an injectable test boundary:

```python
class DevelopmentBackend(Protocol):
    def screen(self, *, sample_ids: tuple[int, ...], regions: tuple[str, ...]) -> list[dict[str, Any]]: ...
    def intervene(self, *, sample_ids: tuple[int, ...], region_sets: tuple[tuple[str, ...], ...]) -> tuple[InterventionObservation, ...]: ...
    def extract_features(self, *, sample_ids: tuple[int, ...], graph_path: Path) -> Path: ...
    def fit(self, *, graph_path: Path, source_features: Path,
            development_outcomes: Path, split_manifest: Path) -> Path: ...
```

- [ ] **Step 1: Write a failing synthetic orchestration test**

```python
def test_data_size_30_runs_one_parameterized_workflow(tmp_path, fake_backend):
    result = run_development(args_for(tmp_path, data_size=30), backend=fake_backend)
    assert result["counts"] == {"discovery": 4, "fit": 10,
                                "calibration": 16}
    assert result["max_components"] == 3
    assert result["variant"] == "A11"
    assert result["evaluation_overlap"] == []
    assert result["special_mode"] is None
```

- [ ] **Step 2: Run the test and verify RED**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_run_generic_region_development.py`

Expected: FAIL because the orchestrator is missing.

- [ ] **Step 3: Implement phase-manifest orchestration**

Create a canonical `development_run.json` before generation. Implement phases
`cohort`, `screen`, `discover_1`, `discover_2`, `discover_3`, `freeze_graph`,
`development_interventions`, `source_features`, `fit`, and `complete`. Each
phase validates prior artifact digests and can resume independently.

Before freezing the cohort, scan candidate source IDs deterministically while
excluding evaluation IDs. Resolve the target and desired direction from the
template graph, require the source to lie on the opposite classifier side,
require a detected face, source image, and at least one non-empty semantic
component, and continue until every stable role bucket contains its requested
count. Record every accepted/rejected eligibility decision without reading a
generated image.

- [ ] **Step 4: Compose generic discovery levels**

Use every key in `CELEBAMASK_COMPONENT_SUFFIXES` as the initial semantic
universe, screen to six atoms, run/cache at most six singleton A11 sets,
advance the four-wide beam, repeat for pairs and triples, then pass all audited
evidence to `build_influence_graph` with `minimum_samples` equal to the
discovery cohort size. Freeze at most four candidates plus fallback.

- [ ] **Step 5: Compose fitting and calibration**

Generate/cache every frozen candidate on fit and calibration IDs, prepare
development rows with external evaluation IDs, compute source-only features,
and invoke `fit_region_selector`. Preserve the production risk requirements;
record fallback-only calibration as a valid completed result.

- [ ] **Step 6: Add resume and mismatch tests**

Run once with `data_size=30`, rerun with identical arguments and assert zero
new backend calls, then run with `data_size=300` and assert all exact small-run
cache keys are reused. Assert a changed policy or mask digest causes a closed
failure rather than reuse.

- [ ] **Step 7: Add CLI validation tests**

Cover required evaluation exclusions, `data_size >= 15`, local-only models,
arbitrary absolute Kaggle paths, `--device auto`, and canonical output files.

- [ ] **Step 8: Run focused orchestration tests**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_run_generic_region_development.py tests/test_development_cohort.py tests/test_generic_region_discovery.py tests/test_intervention_cache.py`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add scripts/run_generic_region_development.py scripts/discover_counterfactual_graph.py tests/test_run_generic_region_development.py
git commit -m "feat: orchestrate generic development workflow"
```

### Task 7: Held-Out A11 Evaluation Contract

**Files:**
- Modify: `scripts/run_individual_region_cci.py`
- Create: `tests/test_generic_heldout_evaluation.py`
- Modify: `README.md`

**Interfaces:**
- Produces CLI mode `--generate_selected_a11` that validates all evaluation IDs, freezes the complete selection manifest, and then performs one A11 generation per source-selected region set.
- Consumes: frozen selector, influence graph, evaluation IDs, semantic-mask manifest, and generation policy.
- Adds `run_heldout_a11(args, generation_backend=None) -> dict[str, Any]`;
  `generation_backend` defaults to the real content-addressed A11 executor and
  is injectable only for deterministic tests.

- [ ] **Step 1: Write a failing exactly-once evaluation test**

```python
def test_all_decisions_are_frozen_before_one_a11_call_per_source(tmp_path):
    result = run_heldout(args_for_three_sources(tmp_path), backend=recording_backend)
    assert selection_manifest(tmp_path)["sample_ids"] == [1, 2, 3]
    assert recording_backend.calls == [(1, "A11"), (2, "A11"), (3, "A11")]
    assert all(call.manifest_existed_before_call for call in recording_backend.audit)
```

- [ ] **Step 2: Run the test and verify RED**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_generic_heldout_evaluation.py`

Expected: FAIL because generic A11 generation is not connected to frozen
source decisions.

- [ ] **Step 3: Implement selected-mask A11 generation**

Reuse the phase-one selection logic and content-addressed A11 executor. Reject
any evaluation ID absent from the selector artifact or overlapping a
development role. Write and hash all decisions before cache lookup or
generation. Do not emit or invoke A0.

- [ ] **Step 4: Add negative leakage and exactly-once tests**

Cover generated-output fields in decisions, oracle fields in ranking,
development/evaluation overlap, incomplete semantic-mask provenance, retries,
and one invocation per unique evaluation ID.

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_generic_heldout_evaluation.py tests/test_run_individual_region_cci.py tests/test_risk_controlled_selection.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_individual_region_cci.py tests/test_generic_heldout_evaluation.py README.md
git commit -m "feat: generate held-out A11 from frozen masks"
```

### Task 8: Kaggle Entry Point, Documentation, and Full Verification

**Files:**
- Create: `examples/kaggle_generic_development.py`
- Modify: `README.md`
- Create: `tests/test_kaggle_generic_development_example.py`

**Interfaces:**
- Produces: a thin Kaggle example that builds and invokes the same CLI command
  with `/kaggle/input` and `/kaggle/working` paths.

- [ ] **Step 1: Write a failing example-contract test**

```python
def test_kaggle_example_uses_shared_cli_and_only_data_size_changes():
    command = build_kaggle_command(data_size=30, paths=fake_kaggle_paths())
    assert command[1].endswith("run_generic_region_development.py")
    assert command[command.index("--data_size") + 1] == "30"
    assert "/kaggle/working" in " ".join(command)
    assert "smoke" not in " ".join(command).lower()
```

- [ ] **Step 2: Run the test and verify RED**

Run: `PYTHONPATH=. .venv-ml/bin/pytest -q tests/test_kaggle_generic_development_example.py`

Expected: FAIL because the example module is missing.

- [ ] **Step 3: Implement the thin Kaggle command builder**

Read paths from function arguments or environment variables, default device to
`auto`, default cache/output below `/kaggle/working`, and invoke the shared CLI
with `subprocess.run(check=True)`. Do not contain discovery, training, or
calibration logic in the example.

- [ ] **Step 4: Replace the obsolete same-300 development documentation**

Document the evaluation-300 exclusion, `data_size=30` and `data_size=300`
commands, 2:5:8 allocation, generic beam budget, A11-only behavior, Kaggle
mounts, resume/cache locations, expected fallback-only behavior for small
runs, and held-out exactly-once A11 evaluation.

- [ ] **Step 5: Run all targeted tests and CLI smoke checks**

Run:

```bash
PYTHONPATH=. .venv-ml/bin/pytest -q \
  tests/test_development_cohort.py \
  tests/test_generic_region_discovery.py \
  tests/test_intervention_cache.py \
  tests/test_prepare_adaptive_replay_data.py \
  tests/test_fit_region_selector.py \
  tests/test_runtime_environment.py \
  tests/test_run_generic_region_development.py \
  tests/test_generic_heldout_evaluation.py \
  tests/test_kaggle_generic_development_example.py
.venv-ml/bin/python scripts/run_generic_region_development.py --help
.venv-ml/bin/python scripts/run_individual_region_cci.py --help
```

Expected: all tests pass and both help commands exit zero.

- [ ] **Step 6: Run the full repository verification**

Run:

```bash
PYTHONPATH=. .venv-ml/bin/pytest -q
git diff --check
.venv-ml/bin/python -m compileall -q src scripts examples
```

Expected: zero failures and zero whitespace/compilation errors.

- [ ] **Step 7: Run a synthetic Kaggle-oriented dry run**

Run the orchestrator with temporary absolute input/output roots, `--data_size
30`, `--device cpu`, and the synthetic backend fixture. Verify the manifest
reports `4/10/16`, A11 only, maximum three components, zero evaluation overlap,
and no smoke-specific mode.

- [ ] **Step 8: Commit**

```bash
git add examples/kaggle_generic_development.py README.md tests/test_kaggle_generic_development_example.py
git commit -m "docs: add Kaggle generic development workflow"
```

## Final Review Gate

- [ ] Re-read the design specification and map every requirement to a passing
  test or verified artifact.
- [ ] Request a focused code review of the complete implementation range.
- [ ] Fix every Critical and Important issue and rerun the full suite.
- [ ] Confirm `git status --short` is clean on `main` and report the exact
  commit range, test count, and the command for `data_size=30`.
