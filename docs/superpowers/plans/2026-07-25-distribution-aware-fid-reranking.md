# Distribution-Aware FID Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate four deterministic A3 smile-removal candidates for each of ten sources and select a target-valid output set with lower FID than single-seed and random-selection controls.

**Architecture:** Add a pure NumPy/SciPy selection module for reference-fitted PCA, proxy FID, eligibility, and deterministic selectors. Add one orchestration script that reuses `run_clean_cci_pilot.py` for generation, reuses the cached Inception extraction pattern from `evaluate_fid_sfid.py`, materializes four selected sets, and writes a reproducible comparison report.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyTorch, torchvision, pytorch-fid, Pillow, Stable Diffusion 2, MPS, pytest.

## Global Constraints

- Work only under `/Users/hung.domodec.com/my-docs/cci-diff`.
- Create no git commit and stage no file.
- Task: smile removal with A3 clean CCI and smooth-boundary post-attack.
- Generate seeds 42, 43, 44, and 45 for ten deterministic eligible sources.
- Use 35 denoising steps, mask geometry `x4_y4_f3`, MPS, and float32.
- Use post-attack schedule `0.05,0.08,0.10,0.30,0.50` and margin `0.03`.
- Exclude all evaluation source IDs from the 1,000-image reference cohort.
- Require all ten S3 smoke-test outputs to pass the saved-image generation classifier.
- Use PCA-64 proxy FID for selection and standard 2048-dimensional FID for reporting.
- Preserve every generated raw/corrected image and audit.
- Do not run the 100-image experiment unless the smoke-test acceptance conditions pass.

---

### Task 1: Pure Distribution-Aware Selection Core

**Files:**
- Create: `src/cci_diff/fid_reranking.py`
- Create: `tests/test_fid_reranking.py`

**Interfaces:**
- Consumes: candidate rows containing `sample_id`, `seed`, `desired_probability`, `identity_cosine`, `outside_semantic_l1`, `post_attack_linf`, and `raw_target_pass`, plus aligned candidate/reference activation arrays.
- Produces: `select_reference_ids(...)`, `fit_reference_projection(...)`, `project_features(...)`, `frechet_distance(...)`, `candidate_is_eligible(...)`, `select_single_seed(...)`, `select_random_candidates(...)`, `select_independent_candidates(...)`, and `select_global_fid_candidates(...)`.

- [ ] **Step 1: Write failing reference and projection tests**

Add tests that establish exact deterministic behavior:

```python
def test_reference_ids_are_sorted_complete_and_exclude_evaluation_ids(tmp_path):
    for image_id in (0, 1, 2, 3, 4, 5):
        (tmp_path / f"{image_id}.jpg").write_bytes(b"x")

    selected = select_reference_ids(
        tmp_path,
        count=3,
        excluded_ids={1, 3},
    )

    assert [image_id for image_id, _ in selected] == [0, 2, 4]


def test_reference_projection_is_deterministic_and_limited_by_rank():
    values = np.arange(60, dtype=float).reshape(10, 6)
    first = fit_reference_projection(values, dimensions=4)
    second = fit_reference_projection(values, dimensions=4)

    np.testing.assert_allclose(first.mean, second.mean)
    np.testing.assert_allclose(first.components, second.components)
    assert first.components.shape == (4, 6)
```

- [ ] **Step 2: Verify the new tests fail**

Run:

```bash
.venv-ml/bin/python -m pytest -p no:cacheprovider tests/test_fid_reranking.py -q
```

Expected: collection fails because `cci_diff.fid_reranking` does not exist.

- [ ] **Step 3: Implement reference selection and PCA projection**

Create immutable projection state and deterministic helpers:

```python
@dataclass(frozen=True)
class ReferenceProjection:
    mean: np.ndarray
    components: np.ndarray
    projected_mean: np.ndarray
    projected_variance: np.ndarray


def select_reference_ids(
    image_root: str | Path,
    *,
    count: int,
    excluded_ids: Collection[int],
) -> tuple[tuple[int, Path], ...]:
    paths = []
    for path in Path(image_root).glob("*.jpg"):
        if path.stem.isdigit() and int(path.stem) not in excluded_ids:
            paths.append((int(path.stem), path))
    paths.sort(key=lambda item: item[0])
    if len(paths) < count:
        raise ValueError(f"reference cohort requires {count} images; found {len(paths)}")
    return tuple(paths[:count])


def fit_reference_projection(
    activations: np.ndarray,
    *,
    dimensions: int = 64,
) -> ReferenceProjection:
    values = _finite_matrix(activations, "reference activations")
    rank = min(dimensions, len(values) - 1, values.shape[1])
    if rank < 1:
        raise ValueError("reference projection requires at least two activations")
    mean = values.mean(axis=0)
    _, _, right = np.linalg.svd(values - mean, full_matrices=False)
    components = _canonicalize_component_signs(right[:rank])
    projected = (values - mean) @ components.T
    return ReferenceProjection(
        mean=mean,
        components=components,
        projected_mean=projected.mean(axis=0),
        projected_variance=np.maximum(projected.var(axis=0), 1e-12),
    )
```

- [ ] **Step 4: Verify reference/projection tests pass**

Run the focused test file. Expected: reference and projection tests pass while later not-yet-added tests are absent.

- [ ] **Step 5: Write failing FID and eligibility tests**

Test unequal cohort sizes, covariance regularization, and hard constraints:

```python
def test_frechet_distance_accepts_unequal_cohort_sizes_and_identical_moments():
    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    selected = np.array([[-1.0], [1.0]])
    assert frechet_distance(reference, selected, epsilon=1e-6) == pytest.approx(
        0.0, abs=1e-5
    )


def test_candidate_eligibility_requires_target_identity_and_locality():
    valid = {
        "desired_probability": 0.51,
        "identity_cosine": 0.81,
        "outside_semantic_l1": 0.02,
    }
    assert candidate_is_eligible(valid)
    assert not candidate_is_eligible(dict(valid, desired_probability=0.49))
    assert not candidate_is_eligible(dict(valid, identity_cosine=0.79))
    assert not candidate_is_eligible(dict(valid, outside_semantic_l1=0.031))
```

- [ ] **Step 6: Verify FID/eligibility tests fail**

Run the focused test file. Expected: failures because the functions are undefined.

- [ ] **Step 7: Implement regularized FID and eligibility**

Use SciPy’s stable matrix square root:

```python
def frechet_distance(
    reference: np.ndarray,
    selected: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    from scipy.linalg import sqrtm

    reference = _finite_matrix(reference, "reference features")
    selected = _finite_matrix(selected, "selected features")
    if reference.shape[1] != selected.shape[1]:
        raise ValueError("reference and selected features require equal dimensions")
    mean_r, mean_s = reference.mean(axis=0), selected.mean(axis=0)
    cov_r = np.atleast_2d(np.cov(reference, rowvar=False))
    cov_s = np.atleast_2d(np.cov(selected, rowvar=False))
    identity = np.eye(reference.shape[1])
    root = sqrtm((cov_r + epsilon * identity) @ (cov_s + epsilon * identity))
    if np.iscomplexobj(root):
        root = root.real
    value = np.sum((mean_r - mean_s) ** 2) + np.trace(
        cov_r + cov_s - 2.0 * root
    )
    return max(float(value), 0.0)


def candidate_is_eligible(
    row: Mapping[str, Any],
    *,
    identity_minimum: float = 0.80,
    outside_l1_maximum: float = 0.03,
) -> bool:
    return (
        float(row["desired_probability"]) >= 0.5
        and float(row["identity_cosine"]) >= identity_minimum
        and float(row["outside_semantic_l1"]) <= outside_l1_maximum
    )
```

- [ ] **Step 8: Write failing selector tests**

Cover S0-S3, deterministic random selection, tie-breaking, one result per source, and FR protection:

```python
def test_global_selector_reduces_fid_without_dropping_required_passes():
    rows = [
        _candidate(0, 42, desired=0.9),
        _candidate(0, 43, desired=0.9),
        _candidate(1, 42, desired=0.9),
        _candidate(1, 43, desired=0.9),
    ]
    features = np.array([[5.0], [-1.0], [5.0], [1.0]])
    reference = np.array([[-1.0], [1.0], [-1.0], [1.0]])

    selected = select_global_fid_candidates(
        rows,
        features,
        reference,
        minimum_passes=2,
        maximum_passes=8,
    )

    assert [(row["sample_id"], row["seed"]) for row in selected.rows] == [
        (0, 43),
        (1, 43),
    ]
    assert selected.target_passes == 2
    assert selected.final_fid <= selected.initial_fid


def test_global_selector_rejects_lower_fid_swap_that_breaks_fr_constraint():
    rows, features, reference = _fr_constraint_fixture()
    selected = select_global_fid_candidates(
        rows,
        features,
        reference,
        minimum_passes=2,
    )
    assert selected.target_passes == 2
```

- [ ] **Step 9: Implement all four deterministic selectors**

Group candidates by integer `sample_id`. S0 chooses seed 42. S1 uses
`np.random.default_rng(20260725)`. S2 minimizes diagonal Mahalanobis distance
among eligible candidates. S3 initializes from S2 and performs at most eight
coordinate-descent passes:

```python
@dataclass(frozen=True)
class SelectionResult:
    rows: tuple[Mapping[str, Any], ...]
    indices: tuple[int, ...]
    initial_fid: float
    final_fid: float
    target_passes: int
    accepted_swaps: int
    passes: int


def select_global_fid_candidates(
    rows,
    projected_features,
    projected_reference,
    *,
    minimum_passes,
    maximum_passes=8,
    epsilon=1e-6,
) -> SelectionResult:
    current = _independent_indices(rows, projected_features, projected_reference)
    initial = frechet_distance(projected_reference, projected_features[current], epsilon=epsilon)
    accepted = 0
    completed_passes = 0
    for completed_passes in range(1, maximum_passes + 1):
        changed = False
        for sample_id in sorted(_group_indices(rows)):
            for alternative in _ordered_alternatives(rows, sample_id):
                proposal = _replace_sample_index(current, rows, sample_id, alternative)
                if _target_pass_count(rows, proposal) < minimum_passes:
                    continue
                proposal_fid = frechet_distance(
                    projected_reference,
                    projected_features[proposal],
                    epsilon=epsilon,
                )
                current_fid = frechet_distance(
                    projected_reference,
                    projected_features[current],
                    epsilon=epsilon,
                )
                if proposal_fid < current_fid - 1e-8:
                    current = proposal
                    accepted += 1
                    changed = True
                    break
        if not changed:
            break
    final = frechet_distance(projected_reference, projected_features[current], epsilon=epsilon)
    return SelectionResult(
        rows=tuple(rows[index] for index in current),
        indices=tuple(current),
        initial_fid=initial,
        final_fid=final,
        target_passes=_target_pass_count(rows, current),
        accepted_swaps=accepted,
        passes=completed_passes,
    )
```

Tie-break alternatives by eligibility, natural raw target pass, lower
`post_attack_linf`, then lower seed.

- [ ] **Step 10: Run focused selector tests**

Run:

```bash
.venv-ml/bin/python -m pytest -p no:cacheprovider tests/test_fid_reranking.py -q
```

Expected: all tests pass.

---

### Task 2: Four-Seed Smoke-Test Orchestrator

**Files:**
- Create: `scripts/run_fid_rerank_smoke.py`
- Create: `tests/test_fid_rerank_smoke.py`

**Interfaces:**
- Consumes: the four-seed pilot output roots, `pilot_results.csv`, selected corrected images, masks, audits, the image root, and functions from `cci_diff.fid_reranking`.
- Produces: the full artifact tree specified by the design and a CLI that can generate or resume all four seed runs before selecting candidates.

- [ ] **Step 1: Write failing CLI and command-construction tests**

Test exact defaults and the four pilot commands without launching diffusion:

```python
def test_smoke_cli_defaults_match_approved_design():
    args = build_arg_parser().parse_args(
        [
            "--classifier_path", "classifier.pth",
            "--identity_model_path", "identity.ts",
            "--output_dir", "outputs/smoke",
        ]
    )
    assert args.seeds == [42, 43, 44, 45]
    assert args.limit == 10
    assert args.reference_count == 1000
    assert args.proxy_dims == 64
    assert args.minimum_passes == 10


def test_build_pilot_commands_pin_a3_mask_and_attack_configuration(tmp_path):
    commands = build_pilot_commands(_args(tmp_path))
    assert len(commands) == 4
    assert {command[command.index("--seed") + 1] for command in commands} == {
        "42", "43", "44", "45"
    }
    for command in commands:
        assert command[command.index("--variants") + 1] == "A3"
        assert command[command.index("--mask_shapes") + 1] == "4,4,3"
        assert command[command.index("--cci_post_attack") + 1] == "smooth_boundary"
```

- [ ] **Step 2: Verify CLI tests fail**

Run the new test file. Expected: import failure because the script does not
exist.

- [ ] **Step 3: Implement CLI and resumable pilot command construction**

The parser requires classifier, identity, and output paths and exposes the
approved defaults. Build commands equivalent to:

```python
[
    ".venv-ml/bin/python",
    "scripts/run_clean_cci_pilot.py",
    "--features", "smile",
    "--limit", "10",
    "--seed", str(seed),
    "--num_inference_steps", "35",
    "--device", args.device,
    "--model_path", args.model_path,
    "--classifier_path", args.classifier_path,
    "--identity_model_path", args.identity_model_path,
    "--output_dir", str(output_dir / "seeds" / f"seed_{seed}"),
    "--variants", "A3",
    "--mask_shapes", "4,4,3",
    "--cci_post_attack", "smooth_boundary",
    "--cci_post_attack_epsilon_schedule", "0.05,0.08,0.10,0.30,0.50",
    "--cci_post_attack_boundary_margin", "0.03",
    "--continue_on_error",
]
```

Run each command with `subprocess.run(..., check=True)`. Existing pilot
candidate validation and audit reuse make reruns resumable.

- [ ] **Step 4: Write failing candidate-pool tests**

Create synthetic `pilot_results.csv` files and assert:

- all four seed cohorts contain identical sample IDs;
- each selected top-level output is used as the corrected candidate;
- raw output is resolved from the candidate audit;
- 40 rows are produced in sample/seed order;
- duplicate or missing source IDs fail before feature extraction.

- [ ] **Step 5: Implement candidate-pool loading**

Add:

```python
def load_seed_candidate_pool(
    output_dir: Path,
    seeds: Sequence[int],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    cohorts = {}
    for seed in seeds:
        rows = _read_csv(output_dir / "seeds" / f"seed_{seed}" / "pilot_results.csv")
        rows = [row for row in rows if row["feature"] == "smile" and row["variant"] == "A3"]
        normalized = [_normalize_candidate_row(row, seed) for row in rows]
        cohorts[seed] = sorted(normalized, key=lambda row: row["sample_id"])
    _validate_seed_cohorts(cohorts, expected_count)
    return [
        row
        for sample_id in sorted(row["sample_id"] for row in cohorts[seeds[0]])
        for seed in seeds
        for row in cohorts[seed]
        if row["sample_id"] == sample_id
    ]
```

- [ ] **Step 6: Write failing activation and reference-manifest tests**

Inject a fake extractor and assert:

- evaluation IDs are excluded;
- the reference manifest lists 1,000 numeric IDs in order;
- cache files are reused only when path fingerprints match;
- candidate activation order exactly matches candidate row order;
- the reference cache stores the fitted PCA state and checksum metadata.

- [ ] **Step 7: Implement cached Inception extraction**

Reuse `extract_or_load_activations` from `scripts.evaluate_fid_sfid`. Build one
`pytorch_fid.inception.InceptionV3` model and extractor. Write:

```python
reference = select_reference_ids(
    args.image_root,
    count=args.reference_count,
    excluded_ids={row["sample_id"] for row in candidates},
)
reference_activations = extract_or_load_activations(
    [path for _, path in reference],
    output_dir / "reference_features.npz",
    extractor,
)
candidate_activations = extract_or_load_activations(
    [Path(row["output_path"]) for row in candidates],
    output_dir / "candidate_features.npz",
    extractor,
)
```

Persist `reference_manifest.json` with IDs, paths, excluded IDs, feature
dimensions, model name, and cache fingerprint/checksum.

- [ ] **Step 8: Write failing materialization and report tests**

With synthetic rows and features, assert:

- `selected/s0` through `selected/s3` each contain exactly one image per source;
- `selection_s0.csv` through `selection_s3.csv` contain ten rows;
- `selector_metrics.csv` records proxy FID, report FID, FR, identity, locality,
  attack rate, and seed counts;
- `fid_reranking_report.md` names S0-S3 and says 10-image FID is exploratory;
- S3 input/output comparison images exist.

- [ ] **Step 9: Implement selection, materialization, and reporting**

Fit PCA-64 on reference activations, project candidates/reference, run S0-S3,
and compute:

```python
metrics = {
    "selector": name,
    "count": len(selected_rows),
    "generation_fr": np.mean(
        [float(row["desired_probability"]) >= 0.5 for row in selected_rows]
    ),
    "mean_identity_cosine": np.mean(
        [float(row["identity_cosine"]) for row in selected_rows]
    ),
    "mean_outside_semantic_l1": np.mean(
        [float(row["outside_semantic_l1"]) for row in selected_rows]
    ),
    "attack_rate": np.mean(
        [row["post_attack_selected_epsilon"] not in (None, "") for row in selected_rows]
    ),
    "proxy_fid": frechet_distance(projected_reference, projected_selected),
    "report_fid": frechet_distance(source_activations, selected_activations),
}
```

Extract source activations for the ten evaluation sources separately and use
them only for final reporting, never during S2/S3 selection. Copy selected
images with deterministic names and generate S3 pair images using
`create_pair_image` from `cci_diff.comparison_artifacts`.

- [ ] **Step 10: Run orchestrator tests**

Run:

```bash
.venv-ml/bin/python -m pytest -p no:cacheprovider \
  tests/test_fid_reranking.py tests/test_fid_rerank_smoke.py -q
```

Expected: all focused tests pass.

---

### Task 3: Verification and 10-Image Smoke Run

**Files:**
- Verify: `src/cci_diff/fid_reranking.py`
- Verify: `scripts/run_fid_rerank_smoke.py`
- Verify: `tests/test_fid_reranking.py`
- Verify: `tests/test_fid_rerank_smoke.py`
- Produce: `outputs/clean_cci_fid_rerank_smoke10/`

**Interfaces:**
- Consumes: completed Task 1 and Task 2 implementation.
- Produces: verified code and the real 40-candidate smoke-test report.

- [ ] **Step 1: Run focused tests**

```bash
.venv-ml/bin/python -m pytest -p no:cacheprovider \
  tests/test_fid_reranking.py tests/test_fid_rerank_smoke.py \
  tests/test_evaluate_fid_sfid.py tests/test_clean_cci_pilot.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the complete test suite**

```bash
.venv-ml/bin/python -m pytest -p no:cacheprovider -q
```

Expected: zero failures.

- [ ] **Step 3: Run static diff validation**

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Execute or resume the 40-candidate smoke test**

```bash
.venv-ml/bin/python scripts/run_fid_rerank_smoke.py \
  --classifier_path models/resnet50_multilabel_model.pth \
  --identity_model_path models/facenet_vggface2.ts \
  --model_path checkpoints/sd2-1-base \
  --device mps \
  --output_dir outputs/clean_cci_fid_rerank_smoke10
```

Expected: four complete ten-image seed cohorts and no unresolved candidates.

- [ ] **Step 5: Validate generated artifacts**

Check:

```bash
.venv-ml/bin/python scripts/run_fid_rerank_smoke.py \
  --classifier_path models/resnet50_multilabel_model.pth \
  --identity_model_path models/facenet_vggface2.ts \
  --model_path checkpoints/sd2-1-base \
  --device mps \
  --output_dir outputs/clean_cci_fid_rerank_smoke10 \
  --selection_only
```

Expected: activation caches are reused, selector outputs are reproduced, and
all selector CSV/report values remain identical.

- [ ] **Step 6: Apply the acceptance gate**

The smoke test passes only when:

```text
candidate count == 40
S0/S1/S2/S3 selected count == 10 each
S3 generation FR == 1.0
S3 report FID <= S0 report FID
S3 report FID <= S1 report FID
no unresolved candidates
```

If the gate fails, stop before the 100-image run and report which condition
failed, together with selector metrics and candidate diversity.

- [ ] **Step 7: Report results without committing**

Provide direct links to:

- `outputs/clean_cci_fid_rerank_smoke10/fid_reranking_report.md`
- `outputs/clean_cci_fid_rerank_smoke10/selector_metrics.csv`
- `outputs/clean_cci_fid_rerank_smoke10/candidate_metrics.csv`
- `outputs/clean_cci_fid_rerank_smoke10/comparisons/`

State exact S0-S3 FID, FR, identity, locality, attack use, selected seeds, and
whether the 100-image run is supported. Create no commit.
