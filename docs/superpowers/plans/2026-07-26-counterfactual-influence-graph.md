# Counterfactual Influence Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically discover which semantic face regions are needed to
flip one classifier label, using paired masked interventions rather than
hand-authored target-to-region edges.

**Architecture:** Add a pure analysis module for effect estimation, confidence,
interaction, and target-first selection; a region-screening module for
Grad-CAM/mask overlap and union masks; and two CLIs for intervention
orchestration and graph analysis. Temporary execution graphs feed the existing
clean-CCI runner without changing residual-driven controller weights.

**Tech Stack:** Python 3.10, NumPy, Pillow, PyTorch, existing CelebA ResNet50,
Grad-CAM++, SD2 clean-CCI runner, pytest.

## Global Constraints

- Work in `/Users/hung.domodec.com/my-docs/cci-diff`.
- Do not commit or stage any files.
- Preserve existing dirty-worktree changes and output directories.
- Treat edges as classifier-specific counterfactual influence, not biological causality.
- Use generation-classifier threshold `0.5` for flip rate.
- Use `0.95` as the default cohort flip-rate requirement.
- Keep graph evidence separate from online dual-controller weights.
- Disable post-generation attack during influence discovery.
- Use identical seeds and generation settings across paired region sets.

---

### Task 1: Effect Estimation And Selection

**Files:**
- Create: `src/cci_diff/counterfactual_graph.py`
- Test: `tests/test_counterfactual_graph.py`

**Interfaces:**
- Consumes: intervention rows represented by `InterventionObservation`.
- Produces: `RegionSetEvidence`, `InfluenceGraphResult`,
  `aggregate_region_sets(...)`, `compute_interactions(...)`,
  `select_region_set(...)`, and `build_influence_graph(...)`.

- [ ] **Step 1: Write failing tests for binary direction and canonical regions**

```python
def test_observation_converts_smile_removal_to_desired_probability():
    row = InterventionObservation(
        target="Smiling", desired_value=0, sample_id=1, seed=42,
        regions=("lower_lip", "mouth"), source_probability=0.9,
        output_probability=0.3,
    )
    assert row.regions == ("lower_lip", "mouth")
    assert row.source_desired_probability == pytest.approx(0.1)
    assert row.output_desired_probability == pytest.approx(0.7)
    assert row.target_effect == pytest.approx(0.6)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-ml/bin/python -m pytest \
  -p no:cacheprovider tests/test_counterfactual_graph.py -q
```

Expected: import failure for `cci_diff.counterfactual_graph`.

- [ ] **Step 3: Implement immutable observation and evidence types**

```python
@dataclass(frozen=True)
class InterventionObservation:
    target: str
    desired_value: int
    sample_id: int
    seed: int
    regions: tuple[str, ...]
    source_probability: float
    output_probability: float
    mask_fraction: float | None = None
    identity_cosine: float | None = None
    non_target_drift: float | None = None
    outside_l1: float | None = None
    changed_fraction: float | None = None
    output_path: str | None = None
    audit_path: str | None = None
```

Canonicalize and validate target, desired value, finite probabilities, sample
ID, seed, and non-empty unique regions in `__post_init__`.

- [ ] **Step 4: Add cluster-bootstrap and aggregation tests**

Cover one seed per image, repeated seeds per image, deterministic confidence
intervals, unsupported singleton edges, and missing optional metrics.

- [ ] **Step 5: Implement aggregation**

Use image IDs as bootstrap clusters. For every region tuple compute count,
sample count, target flip rate, mean/median effect, 95% bootstrap interval,
mask fraction, identity, drift, outside L1, and changed fraction.

- [ ] **Step 6: Add interaction and selection tests**

Assert pair synergy equals joint effect minus singleton effects. Assert a
smaller 95%-passing region set beats a larger 100%-passing set, while a 90%
set never beats a 95% set. Assert fallback maximizes flip rate when no set
passes.

- [ ] **Step 7: Implement interaction, selection, and graph serialization**

Return a JSON-serializable graph with verified singleton edges, all region-set
evidence, interactions, selected regions, thresholds, and provenance.

- [ ] **Step 8: Run focused tests**

Expected: all `tests/test_counterfactual_graph.py` tests pass.

---

### Task 2: Semantic Region Screening And Union Masks

**Files:**
- Create: `src/cci_diff/region_screening.py`
- Modify: `src/cci_diff/concept_registry.py`
- Test: `tests/test_region_screening.py`
- Modify: `tests/test_json_graph_compiler.py`

**Interfaces:**
- Consumes: NumPy saliency arrays, component masks, region names.
- Produces: `RegionScreenScore`, `score_region_masks(...)`,
  `canonical_region_sets(...)`, `build_union_mask(...)`, and
  `celebamask_component_path(...)`.

- [ ] **Step 1: Write failing overlap and union tests**

Create synthetic saliency and disjoint masks. Assert captured mass, region
density, proposal score, deterministic ranking, and exact union pixels.

- [ ] **Step 2: Implement screening helpers**

```python
@dataclass(frozen=True)
class RegionScreenScore:
    region: str
    captured_mass: float
    region_density: float
    mask_fraction: float
    proposal_score: float
```

Reject mismatched shapes, non-finite saliency, empty masks, duplicate regions,
and more than eight candidate regions.

- [ ] **Step 3: Implement canonical combination generation**

Generate sorted unique tuples with `itertools.combinations` for sizes
`1..max_set_size`. Reject grids larger than 255 sets.

- [ ] **Step 4: Expand reviewed mask roles**

Register the 19 canonical CelebAMask-HQ components plus `target_region`, while
retaining existing aliases through path resolution. Keep evaluator
registration unchanged.

- [ ] **Step 5: Verify compiler support**

Compile a graph with `audit_role="target_region"` and components
`["mouth", "upper_lip", "lower_lip"]` against temporary masks.

- [ ] **Step 6: Run focused tests**

Expected: region-screening and compiler tests pass.

---

### Task 3: Paired Region Intervention Runner

**Files:**
- Create: `scripts/run_counterfactual_region_interventions.py`
- Test: `tests/test_counterfactual_region_interventions.py`

**Interfaces:**
- Consumes: template graph, source/mask roots, sample IDs, candidate regions,
  seeds, model paths, and generation settings.
- Produces: temporary graphs, bindings, resumable run directories,
  `intervention_manifest.json`, and `intervention_results.csv`.

- [ ] **Step 1: Write failing graph/binding generation tests**

Use a temporary template graph and three 4x4 masks. Assert that generated graph
components equal the active set, audit role is `target_region`, the union mask
is correct, and bindings contain every required role.

- [ ] **Step 2: Implement execution-policy materialization**

```python
def materialize_region_policy(
    template_graph_path: Path,
    source_path: Path,
    component_paths: Mapping[str, Path],
    regions: tuple[str, ...],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    ...
```

Return graph path, binding path, and union-mask path. Preserve target,
constraints, edges, and controller settings from the template.

- [ ] **Step 3: Write command-construction tests**

Assert same seed and all generation flags are present; `clean_constraint`,
`feedback`, graph, binding, classifier, and identity arguments are required;
post-attack arguments are absent.

- [ ] **Step 4: Implement resumable orchestration**

Directory layout:

```text
runs/<sample-id>/seed_<seed>/<region-set>/
```

Before executing, validate an existing `audit.json` and output image. After
execution, parse the audit into one `InterventionObservation`.

- [ ] **Step 5: Implement audit extraction**

Read source/output classifier probabilities at the target index, final
generation-classifier pass, identity cosine, non-target drift, semantic
outside L1, semantic mask fraction, and paths. Compute changed fraction with
the existing spatial measurement helper.

- [ ] **Step 6: Implement manifest and CSV writing**

Record target, label index, desired value, source cohort, seeds, region sets,
template graph digest, checkpoint paths, inference settings, and unresolved
failures. Flush CSV after every completed intervention.

- [ ] **Step 7: Add argument validation and direct-script test**

Require unique sample IDs and seeds, registered regions, `max_set_size` within
candidate count, existing local model files, and output directory. Mock
subprocess execution in tests.

- [ ] **Step 8: Run focused tests**

Expected: all runner tests pass without loading diffusion.

---

### Task 4: Screening And Analysis CLIs

**Files:**
- Create: `scripts/screen_counterfactual_regions.py`
- Create: `scripts/discover_counterfactual_graph.py`
- Test: `tests/test_counterfactual_graph_cli.py`

**Interfaces:**
- Screening produces proposal CSV/summary/manifest.
- Discovery consumes `intervention_results.csv` and produces the learned graph,
  metrics CSVs, and Markdown report.

- [ ] **Step 1: Write CLI parser and synthetic analysis tests**

Assert direct `--help` works. Feed synthetic observations where the triple
`mouth+upper_lip+lower_lip` is the smallest set reaching 95% FR and assert the
generated JSON selects that tuple.

- [ ] **Step 2: Implement screening CLI**

Load the classifier once, compute source Grad-CAM++, resolve aligned masks,
write one row per sample-region, and aggregate proposal ranking. Screening
never writes verified edges.

- [ ] **Step 3: Implement discovery CLI**

Parse CSV rows into observations, aggregate evidence with deterministic
cluster bootstrap, compute interactions, select the region set, and write:

```text
counterfactual_influence_graph.json
region_set_metrics.csv
interaction_metrics.csv
counterfactual_influence_report.md
```

- [ ] **Step 4: Add malformed-input tests**

Reject mixed targets, mixed desired values, missing singleton evidence,
duplicate sample/seed/region-set rows, and fewer than `minimum_samples` for
edge verification.

- [ ] **Step 5: Run focused CLI tests**

Expected: all direct-script and synthetic integration tests pass.

---

### Task 5: Documentation And Verification

**Files:**
- Modify: `docs/cci_latent_guidance_math.md`
- Modify: `paper/cci_conference_v1.tex`

**Interfaces:**
- Documents the distinction between execution-policy edges and measured
  counterfactual influence.

- [ ] **Step 1: Document the learned graph**

Add equations for signed target effect, cluster confidence, interaction, and
lexicographic region-set selection. State explicitly that edge strengths do
not become loss coefficients.

- [ ] **Step 2: Add a paper future-method paragraph**

Describe the implemented discovery mechanism as an experimental extension,
without replacing the reported fixed experimental setup or claiming results
before a held-out run exists.

- [ ] **Step 3: Run focused test files**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-ml/bin/python -m pytest \
  -p no:cacheprovider \
  tests/test_counterfactual_graph.py \
  tests/test_region_screening.py \
  tests/test_counterfactual_region_interventions.py \
  tests/test_counterfactual_graph_cli.py -q
```

- [ ] **Step 4: Run full verification**

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-ml/bin/python -m pytest \
  -p no:cacheprovider
git diff --check
```

Expected: zero test failures and no whitespace errors.

- [ ] **Step 5: Run a synthetic end-to-end smoke**

Generate temporary intervention CSV data, run
`scripts/discover_counterfactual_graph.py`, and verify all four result
artifacts parse successfully and identify the expected minimal region set.

- [ ] **Step 6: Preserve the working tree**

Do not stage, commit, merge, reset, or remove unrelated files. Report all new
paths and verification evidence.
