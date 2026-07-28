# Pareto Region Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed flip-threshold graph selection with deterministic Pareto filtering and target-effect-per-area selection.

**Architecture:** Keep intervention aggregation unchanged. Add Pareto annotations and selection helpers to `counterfactual_graph.py`, then expose those annotations through graph JSON, CSV, provenance, and the discovery report while retaining `required_flip_rate` as legacy metadata.

**Tech Stack:** Python 3, dataclasses, pytest, CSV/JSON artifacts.

## Global Constraints

- A generation candidate must have finite target effect, flip rate, and positive mask area.
- Higher target effect and flip rate are beneficial; smaller mask area is beneficial.
- Dominated candidates remain in evidence but cannot be selected.
- Positive-effect Pareto candidates rank by target effect divided by mask fraction.
- `required_flip_rate` is compatibility metadata and must not affect selection.
- Existing intervention execution and aggregation remain unchanged.

---

### Task 1: Pareto Evidence And Selection

**Files:**
- Modify: `tests/test_counterfactual_graph.py`
- Modify: `src/cci_diff/counterfactual_graph.py`

**Interfaces:**
- Consumes: `Mapping[RegionTuple, RegionSetEvidence]`.
- Produces: `pareto_region_sets(...)`, `target_efficiency(...)`, and threshold-independent `select_region_set(...)`.

- [ ] **Step 1: Write failing tests**

Add tests proving that:

```python
assert broad.regions not in pareto_region_sets(evidence)
assert select_region_set(evidence, required_flip_rate=0.01) == selected
assert select_region_set(evidence, required_flip_rate=0.99) == selected
```

Also cover an effect-area trade-off, deterministic ties, nonpositive fallback,
and rejection of missing or nonpositive mask area.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_counterfactual_graph.py
```

Expected: failures because Pareto helpers and the new behavior do not exist.

- [ ] **Step 3: Implement minimal selector**

Implement strict Pareto dominance over `(mean_effect, flip_rate,
mean_mask_fraction)`, target efficiency
`mean_effect / max(mean_mask_fraction, 1e-12)`, deterministic tie-breaking,
and the nonpositive-effect fallback.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
pytest -q tests/test_counterfactual_graph.py
```

Expected: all selector tests pass.

### Task 2: Graph Status And Artifact Annotations

**Files:**
- Modify: `tests/test_counterfactual_graph.py`
- Modify: `tests/test_counterfactual_graph_cli.py`
- Modify: `src/cci_diff/counterfactual_graph.py`
- Modify: `scripts/discover_counterfactual_graph.py`

**Interfaces:**
- Consumes: Pareto selection helpers from Task 1.
- Produces: evidence rows containing `pareto_optimal`,
  `target_efficiency`, and `dominated_by`; graph statuses
  `pareto_efficient` or `fallback_nonpositive_effect`.

- [ ] **Step 1: Write failing serialization tests**

Assert JSON and CSV artifacts contain Pareto annotations, graph provenance
identifies `pareto_target_efficiency_v1`, and the report no longer describes
the flip threshold as a selection requirement.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_counterfactual_graph.py tests/test_counterfactual_graph_cli.py
```

Expected: failures for missing fields and legacy status text.

- [ ] **Step 3: Implement artifact annotations**

Add a serializable per-region selection annotation, derive graph status from
the selected effect, preserve `required_flip_rate` under compatibility
metadata, and update CSV/report rendering.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
pytest -q tests/test_counterfactual_graph.py tests/test_counterfactual_graph_cli.py
```

Expected: all graph and CLI tests pass.

### Task 3: Regression Verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes: completed implementation.
- Produces: fresh test evidence.

- [ ] **Step 1: Run focused discovery suites**

```bash
pytest -q tests/test_counterfactual_graph.py tests/test_counterfactual_graph_cli.py tests/test_counterfactual_region_interventions.py tests/test_individual_region_selection.py
```

- [ ] **Step 2: Run the complete test suite**

```bash
pytest -q
```

- [ ] **Step 3: Inspect repository changes**

```bash
git diff --check
git status --short
```

Confirm only the approved selector, tests, report artifacts, spec, and plan
are changed.
