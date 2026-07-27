# Individual Grad-CAM Region Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select the smallest saliency-covering subset of globally verified
semantic regions for each held-out source image and run exactly one CCI-BLD
generation.

**Architecture:** Add a pure selector that loads frozen influence evidence,
scores exact semantic-mask unions against one source Grad-CAM++ map, and
returns a deterministic region decision. Add a separate resumable evaluation
runner that computes the source saliency, materializes one image-specific CCI
policy, invokes diffusion once, and records selection and output metrics.

**Tech Stack:** Python 3.10, NumPy, Pillow, PyTorch, Grad-CAM++, existing CelebA
ResNet50, existing clean-CCI SD2 runner, pytest.

## Global Constraints

- Work in `/Users/hung.domodec.com/my-docs/cci-diff`.
- Do not commit or stage files.
- Use only source-image information for individual region selection.
- Use only globally verified singleton regions as per-image candidates.
- Default Grad-CAM coverage threshold is `0.80`.
- Run exactly one diffusion generation per held-out image.
- Do not retry, escalate, rerank outputs, or apply post-generation attack.
- Keep the global graph and threshold unchanged throughout held-out evaluation.

---

### Task 1: Frozen Policy And Pure Individual Selector

**Files:**
- Create: `src/cci_diff/individual_region_selection.py`
- Test: `tests/test_individual_region_selection.py`

**Interfaces:**
- Consumes: `influence_graph.json`, a two-dimensional saliency array, and
  image-specific semantic component masks.
- Produces: `FrozenInfluencePolicy`,
  `IndividualRegionSelection`, `load_frozen_influence_policy(...)`, and
  `select_individual_region_set(...)`.

- [ ] **Step 1: Write failing policy-loading tests**

Create a temporary influence graph with verified edges for `mouth`,
`upper_lip`, and `lower_lip`, a global fallback set, and region-set evidence.
Assert:

```python
policy = load_frozen_influence_policy(path)
assert policy.target == "Smiling"
assert policy.desired_value == 0
assert policy.verified_regions == ("lower_lip", "mouth", "upper_lip")
assert policy.fallback_regions == ("lower_lip", "mouth", "upper_lip")
assert policy.global_effect(("lower_lip", "mouth")) == pytest.approx(0.42)
```

Reject unsupported graph types, empty verified edges, fallback regions outside
the verified region set, malformed evidence, and desired values outside
`{0,1}`.

- [ ] **Step 2: Run policy tests and verify red**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-ml/bin/python -m pytest \
  -p no:cacheprovider tests/test_individual_region_selection.py -q
```

Expected: import failure for `cci_diff.individual_region_selection`.

- [ ] **Step 3: Implement immutable policy types and graph loading**

Use:

```python
@dataclass(frozen=True)
class FrozenInfluencePolicy:
    target: str
    desired_value: int
    verified_regions: tuple[str, ...]
    fallback_regions: tuple[str, ...]
    region_set_effects: Mapping[tuple[str, ...], float]
    graph_path: str
    graph_sha256: str

@dataclass(frozen=True)
class IndividualRegionSelection:
    selected_regions: tuple[str, ...]
    available_regions: tuple[str, ...]
    missing_regions: tuple[str, ...]
    coverage: float
    mask_fraction: float
    coverage_threshold: float
    fallback_used: bool
    fallback_reason: str | None
    region_importance: Mapping[str, float]
    candidate_count: int
```

Canonicalize every region tuple. Preserve mappings with `MappingProxyType`.

- [ ] **Step 4: Write failing exact-coverage selection tests**

Use synthetic saliency and partially overlapping masks. Assert:

- union coverage does not double-count overlapping pixels;
- the smallest-area feasible set wins even when it is not the first prefix of
  per-region rankings;
- fewer regions break equal-area ties;
- higher global measured effect breaks remaining ties;
- the complete available union always reaches coverage `1.0`;
- missing globally verified component masks are reported and excluded.

- [ ] **Step 5: Implement deterministic subset selection**

For every non-empty subset from `canonical_region_sets(...)`, compute:

```python
union = logical_or.reduce(component_masks[region] > 0 for region in regions)
coverage = saliency[union].sum() / saliency[global_union].sum()
mask_fraction = union.mean()
```

Filter by `coverage >= coverage_threshold`, then minimize:

```text
mask_fraction
region_count
negative global effect
canonical region tuple
```

Validate two-dimensional finite non-negative saliency, identical mask shapes,
threshold in `(0,1]`, no more than eight verified regions, and at least one
available non-empty mask.

- [ ] **Step 6: Write and implement zero-support fallback tests**

When `sum(saliency * global_union) <= eps`, select the available members of
`fallback_regions`, write coverage `0.0`, and mark:

```text
fallback_used = true
fallback_reason = "zero_saliency_in_verified_union"
```

Raise when none of the fallback regions has an available non-empty mask.

- [ ] **Step 7: Run focused selector tests**

Expected: all `tests/test_individual_region_selection.py` tests pass.

---

### Task 2: One-Pass Individual Evaluation Runner

**Files:**
- Create: `scripts/run_individual_region_cci.py`
- Test: `tests/test_run_individual_region_cci.py`

**Interfaces:**
- Consumes: frozen influence graph, template CCI graph, held-out sample IDs,
  semantic mask root, classifier, identity model, SD2 checkpoint, seed, and
  generation settings.
- Produces: frozen selector policy, per-image selections, temporary execution
  policies, one output per image, manifest, and result CSV.
- Reuses:
  `materialize_region_policy(...)`,
  `build_intervention_command(...)`, and
  `load_completed_observation(...)` from
  `scripts/run_counterfactual_region_interventions.py`.

- [ ] **Step 1: Write failing source-eligibility tests**

Define:

```python
def source_requires_flip(
    probability: float,
    desired_value: int,
    threshold: float = 0.5,
) -> bool:
    ...
```

Assert positive sources are eligible for desired `0`, negative sources are
eligible for desired `1`, and already-satisfied sources are rejected.

- [ ] **Step 2: Write failing policy-materialization tests**

Given a source probability, synthetic Grad-CAM map, component masks, and
frozen policy, call:

```python
selection, graph_path, binding_path, union_path = prepare_individual_policy(
    source_path=source,
    sample_id=7,
    source_probability=0.91,
    saliency=saliency,
    component_paths=component_paths,
    frozen_policy=policy,
    template_graph_path=template_graph,
    coverage_threshold=0.80,
    output_dir=tmp_path / "policy",
)
```

Assert the generated graph contains exactly `selection.selected_regions`, the
binding contains `target_region`, and `selection.json` records the influence
graph digest and source-only decision.

- [ ] **Step 3: Implement source loading, Grad-CAM, and policy preparation**

Load one source tensor at `classifier_input_size`, compute its probability,
and call existing:

```python
gradcam_pp_saliency(
    classifier,
    normalized_source,
    label_index=label_index,
    original_present=source_probability >= 0.5,
)
```

Resolve semantic files with `celebamask_component_path(...)`; missing files are
allowed when at least one verified fallback mask remains available.

- [ ] **Step 4: Write failing one-subprocess-per-image orchestration test**

Mock classifier/saliency preparation and `subprocess.run`. For two sample IDs,
assert:

```text
subprocess call count = 2
```

Assert commands contain clean feedback CCI and do not contain post-attack,
candidate, reranking, retry, or escalation flags. Simulate one failed
generation and assert it is recorded once rather than rerun.

- [ ] **Step 5: Implement resumable one-pass orchestration**

Create:

```text
policies/<sample>/
runs/<sample>/
```

Before reusing a completed run, require:

- valid `selection.json`;
- matching influence graph digest;
- matching threshold, source ID, seed, and selected regions;
- valid output audit and mask artifacts.

Otherwise, run one subprocess. Never loop over alternative region sets.

- [ ] **Step 6: Implement incremental artifacts**

Write after every selection and completed output:

```text
individual_policy.json
individual_manifest.json
individual_selections.csv
individual_results.csv
failures.jsonl
```

`individual_policy.json` freezes graph digests, target, desired value,
verified regions, fallback regions, threshold, classifier path, seed, and
generation settings.

- [ ] **Step 7: Implement argument validation**

Require unique non-negative sample IDs, one integer seed, existing local
graphs/checkpoints, coverage threshold in `(0,1]`, and positive inference
steps. Reject overlap between explicitly supplied discovery IDs and test IDs
when `--discovery_manifest` is provided.

- [ ] **Step 8: Run focused runner tests**

Expected: all `tests/test_run_individual_region_cci.py` tests pass without
loading diffusion.

---

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-26-individual-gradcam-region-selection-design.md`

**Interfaces:**
- Consumes: completed selector and runner CLI.
- Produces: exact held-out evaluation command and verified repository state.

- [ ] **Step 1: Document the one-pass command**

Add a README command with:

```bash
.venv-ml/bin/python scripts/run_individual_region_cci.py \
  --influence_graph outputs/smile_counterfactual_graph/influence_graph.json \
  --template_graph examples/graphs/remove_smile_clean_cci.json \
  --sample_ids <held-out IDs> \
  --coverage_threshold 0.80 \
  --seed 42 \
  --model_path checkpoints/sd2-1-base \
  --classifier_path models/resnet50_multilabel_model.pth \
  --identity_model_path models/facenet_vggface2.ts \
  --output_dir outputs/smile_individual_region_cci \
  --device mps
```

State that each sample generates once and a failed flip is retained.

- [ ] **Step 2: Verify CLI and compilation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-ml/bin/python \
  scripts/run_individual_region_cci.py --help
PYTHONPYCACHEPREFIX=/tmp/cci-pycache .venv-ml/bin/python \
  -m compileall -q src/cci_diff/individual_region_selection.py \
  scripts/run_individual_region_cci.py
```

Expected: both commands exit `0`.

- [ ] **Step 3: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-ml/bin/python -m pytest \
  -p no:cacheprovider \
  tests/test_individual_region_selection.py \
  tests/test_run_individual_region_cci.py -q
```

Expected: all focused tests pass.

- [ ] **Step 4: Run the full suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-ml/bin/python -m pytest \
  -p no:cacheprovider -q
```

Expected: zero failures.

- [ ] **Step 5: Run a source-only dry run**

Use one real CelebAMask-HQ image and `--dry_run`. Verify exactly one
`selection.json`, one temporary graph, one binding, and one hard union mask are
created, while no diffusion output is claimed.

- [ ] **Step 6: Review the feature diff**

Run whitespace checks and inspect only files in this plan. Confirm no
post-attack option, candidate-output ranking, retry loop, or graph mutation
from held-out results exists.
