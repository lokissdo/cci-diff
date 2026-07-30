# Attacked Region 300-Image Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one restartable local scheduler that compares attacked A0 and A11 on the same 300 smile-removal sources for mouth-only and mouth-plus-lips regions, then reports FID, sFID, FVA, FS, MNAC, CD, classifier COUT, and directional FR.

**Architecture:** Extend the existing clean-CCI pilot with immutable per-run graph overrides and manifest-based cohort reuse. Extend the existing independent-metric evaluator so its embedding models remain unchanged while a configured local CelebA classifier supplies all classifier-dependent metrics, including corrected binary COUT. Generalize the FID/sFID reporter to smile-only experiments and combine both region reports with a focused summarizer.

**Tech Stack:** Python 3.10, PyTorch, Diffusers SD2.1, NumPy, Pillow, pytorch-fid, pytest, Bash, macOS MPS and `caffeinate`.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-31-attacked-region-300-evaluation-design.md`.
- Work directly on `main`; do not create another worktree.
- Use exactly 300 unique seed-42 sources shared by A0, A11, and both region jobs.
- Apply the smooth-boundary post-attack to both A0 and A11.
- Use `models/resnet50_multilabel_model.pth` for COUT, FR, MNAC, and CD.
- Use only Smiling index 31 and its complement for binary COUT.
- Evaluate final post-attack images.
- Preserve restartability by reusing valid completed audits.

---

### Task 1: Region and Cohort Overrides

**Files:**
- Modify: `scripts/run_clean_cci_pilot.py`
- Modify: `tests/test_clean_cci_pilot.py`

**Interfaces:**
- Consumes: existing `FEATURES`, JSON concept graphs, and pilot manifests.
- Produces: `resolve_region_components(values)`,
  `write_region_graph(source_path, destination, components)`,
  `--region_components`, and `--sample_ids_manifest`.

- [ ] **Step 1: Write failing parser and mapping tests**

Add tests proving:

```python
args = build_arg_parser().parse_args([
    "--features", "smile",
    "--classifier_path", "classifier.pth",
    "--identity_model_path", "identity.pt",
    "--output_dir", "outputs/test",
    "--region_components", "mouth", "upper_lip", "lower_lip",
])
assert args.region_components == ["mouth", "upper_lip", "lower_lip"]
assert resolve_region_components(["mouth"]) == (
    ("mouth",),
    {"mouth": "mouth"},
)
```

Add a rejection test for region overrides with the hair task and a mutual
exclusion test for `--sample_ids`, `--random_sample_seed`, and
`--sample_ids_manifest`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q \
  tests/test_clean_cci_pilot.py -k 'region_components or sample_ids_manifest'
```

Expected: failures because the new parser options and helpers do not exist.

- [ ] **Step 3: Implement canonical region mapping and graph materialization**

Add canonical mapping:

```python
SMILE_REGION_COMPONENTS = {
    "mouth": "mouth",
    "upper_lip": "u_lip",
    "lower_lip": "l_lip",
}
```

Implement `write_region_graph(source_path, destination, components)` by loading
the source JSON, replacing only `graph["region"]["components"]`, and writing a
stable indented JSON file below the run output. Never modify the tracked
example graph.

Pass the resolved graph path into `build_variant_command` so both `--prompt`
and `--cci_graph` use the run-specific graph. Record canonical and annotation
components in `pilot_manifest.json`.

- [ ] **Step 4: Implement manifest cohort reuse**

Add:

```text
--sample_ids_manifest PATH
```

Load IDs from `manifest["features"][feature]["selected_ids"]`, require unique
integers, and route them through the existing explicit-ID selection path.
Reject simultaneous explicit IDs, random sampling, or manifest sampling.

- [ ] **Step 5: Add command tests for attacked A0 and A11**

Parameterize the existing post-attack forwarding test over `A0` and `A11`.
Assert both commands include:

```text
--cci_post_attack smooth_boundary
--cci_post_attack_epsilon_schedule 0.05,0.08,0.10,0.30,0.50
--cci_post_attack_boundary_margin 0.03
```

- [ ] **Step 6: Verify Task 1 GREEN**

Run:

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q tests/test_clean_cci_pilot.py
```

Expected: all pilot tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add scripts/run_clean_cci_pilot.py tests/test_clean_cci_pilot.py
git commit -m "feat: parameterize attacked pilot regions"
```

### Task 2: Correct Local-Classifier COUT and Attribute Metrics

**Files:**
- Modify: `scripts/evaluate_clean_cci_ace.py`
- Modify: `tests/test_evaluate_clean_cci_ace.py`

**Interfaces:**
- Consumes: final `pilot_results.csv`, local CelebA classifier, ACE VGGFace2 and SimSiam checkpoints.
- Produces: per-image `cout`, local-classifier directional FR/MNAC/CD, classifier provenance, and task summaries.

- [ ] **Step 1: Write failing binary COUT tests**

Add tests for:

```python
curves = np.array([[0.9, 0.5, 0.1]])
scores = binary_cout_from_smile_curves(curves)
assert scores == pytest.approx([0.8])
```

The expected value follows:

```text
AUPC(1 - p_smile) - AUPC(p_smile)
```

Add tests proving that a decreasing smiling curve has positive COUT, an
increasing curve has negative COUT, non-finite inputs fail, and no sigmoid is
applied to values already in `[0, 1]`.

- [ ] **Step 2: Write failing transition construction tests**

Use a tiny deterministic tensor and fake probability callback. Assert:

- transition zero equals the source;
- the final transition equals the output;
- changed pixels are inserted in descending RGB absolute-difference order;
- exactly `steps + 1` probabilities contribute to each curve.

- [ ] **Step 3: Run COUT tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q \
  tests/test_evaluate_clean_cci_ace.py -k cout
```

Expected: failures because COUT helpers are absent.

- [ ] **Step 4: Implement binary COUT**

Implement:

```python
def binary_cout_from_smile_curves(
    smile_probabilities: np.ndarray,
) -> np.ndarray:
    curves = np.asarray(smile_probabilities, dtype=float)
    if curves.ndim != 2 or curves.shape[1] < 2:
        raise ValueError("COUT curves must contain at least two points")
    if not np.isfinite(curves).all() or np.any((curves < 0) | (curves > 1)):
        raise ValueError("COUT curves must contain finite probabilities")
    integrate = getattr(np, "trapezoid", np.trapz)
    source = integrate(curves, axis=1) / (curves.shape[1] - 1)
    desired = integrate(1.0 - curves, axis=1) / (curves.shape[1] - 1)
    return desired - source
```

Use `np.trapezoid` when available and `np.trapz` as the NumPy compatibility
fallback. Desired curves are exactly `1.0 - smile_probabilities`.

Implement batched transition evaluation over aligned source/output tensors.
Use 50 intervals, progressively copy sorted pixels, and call
`classifier_probabilities(model, images, size=512)` without a second sigmoid.

- [ ] **Step 5: Add local classifier selection**

Add required CLI option:

```text
--attribute_classifier_path models/resnet50_multilabel_model.pth
```

Load it with `load_celeba_resnet50`. Use its 40 probabilities for:

- directional FR;
- target-excluded MNAC;
- CD;
- COUT.

Continue using ACE VGGFace2 for FVA and ACE SimSiam for FS. Record checkpoint
path, SHA-256, Smiling index, preprocessing size, and the explicit
non-independent evaluator role in `ace_metrics.json`.

- [ ] **Step 6: Extend rows and summaries**

Write `cout` into every `ace_pair_metrics.csv` row and add these fields to
`ace_task_summary.csv`:

```text
directional_fr
cout
cout_count
```

Retain the existing fields for compatibility. Ensure reports label the column
`COUT (guidance classifier)`.

- [ ] **Step 7: Verify Task 2 GREEN**

Run:

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q \
  tests/test_evaluate_clean_cci_ace.py
```

Expected: all evaluator tests pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add scripts/evaluate_clean_cci_ace.py tests/test_evaluate_clean_cci_ace.py
git commit -m "feat: evaluate binary classifier COUT"
```

### Task 3: Smile-Only FID/sFID Full Table

**Files:**
- Modify: `scripts/evaluate_fid_sfid.py`
- Modify: `tests/test_evaluate_fid_sfid.py`

**Interfaces:**
- Consumes: aligned A0/A11 experiment rows and classifier/embedding metrics.
- Produces: two-row region-level full metric tables with the requested columns.

- [ ] **Step 1: Write failing smile-only loading tests**

Create a synthetic experiment containing only `smile` rows. Assert:

```python
grouped = load_experiment_rows(
    root,
    expected_count=300,
    tasks=("smile",),
)
assert len(grouped["smile"]) == 300
assert "hair" not in grouped
```

Add a parser test for:

```text
--tasks smile
```

- [ ] **Step 2: Write failing full-report COUT test**

Add `cout` to synthetic ACE rows and assert `full_metrics.csv` and
`full_metrics.md` contain:

```text
FID
sFID
FVA
FS
MNAC
CD
COUT
FR (%)
```

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q \
  tests/test_evaluate_fid_sfid.py -k 'smile_only or cout'
```

Expected: failures because tasks are hard-coded and COUT is not propagated.

- [ ] **Step 4: Generalize tasks and propagate COUT**

Add `tasks` to `load_experiment_rows`, validation, activation extraction, and
report loops. Default to `("smile", "hair")` for backward compatibility.
Add `--tasks` to the CLI.

Include mean `cout` in `summarize_pair_rows`, `FULL_METRIC_FIELDS`, CSV, JSON,
and Markdown. Format directional FR as a percentage and use the requested
metric ordering.

- [ ] **Step 5: Verify Task 3 GREEN**

Run:

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q tests/test_evaluate_fid_sfid.py
```

Expected: all FID/sFID tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add scripts/evaluate_fid_sfid.py tests/test_evaluate_fid_sfid.py
git commit -m "feat: report smile-only full metrics"
```

### Task 4: Region Report Combiner and Scheduler

**Files:**
- Create: `scripts/combine_attacked_region_metrics.py`
- Create: `scripts/run_attacked_region_300.sh`
- Create: `tests/test_attacked_region_scheduler.py`

**Interfaces:**
- Consumes: mouth and mouth-plus-lips `full_metrics.csv` files.
- Produces: validated four-row combined CSV/Markdown reports and a sequential restartable scheduler.

- [ ] **Step 1: Write failing combiner tests**

Create two synthetic two-row metric tables. Assert the combiner:

- requires exactly A0 and A11 in each region;
- requires `n == 300`;
- writes four rows;
- adds `region`;
- orders columns as:

```text
region, method, n, fid, sfid, fva_rate, fs, mnac, cd, cout, directional_fr
```

- [ ] **Step 2: Write failing scheduler contract test**

Read the shell script as text and assert it contains:

- `set -euo pipefail`;
- `caffeinate -dimsu`;
- A0/A11 controller modes;
- `--limit 300`;
- seed and random sample seed 42;
- mouth-only region arguments;
- mouth-plus-lips region arguments;
- second-job `--sample_ids_manifest`;
- smooth-boundary attack;
- classifier metric evaluation;
- smile-only FID/sFID evaluation;
- final combiner call.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q \
  tests/test_attacked_region_scheduler.py
```

Expected: failures because both scripts are absent.

- [ ] **Step 4: Implement the combiner**

Build a small CLI accepting:

```text
--region NAME PATH
--region NAME PATH
--output_dir PATH
--expected_count 300
```

Validate finite requested metrics, method identity, sample counts, and schema.
Write `combined_metrics.csv` and `combined_metrics.md`.

- [ ] **Step 5: Implement the scheduler**

Create a Bash script that resolves:

```text
ROOT
PYTHON=$ROOT/.venv-ml/bin/python
ACE_ROOT=$ROOT/../thesis_2025/evaluate/ACE
OUTPUT_ROOT=$ROOT/outputs/attacked_a0_a11_smile300_seed42
```

Run mouth-only generation first with:

```text
--features smile
--limit 300
--seed 42
--random_sample_seed 42
--controller_modes disabled trust_region
--region_components mouth
--mask_dilations 8
--cci_post_attack smooth_boundary
```

Run its local-classifier/embedding evaluator and FID/sFID reporter. Run the
second job with the first manifest and:

```text
--region_components mouth upper_lip lower_lip
```

Then combine both reports. Use explicit paths, quote every expansion, and make
reruns reuse the same output directories.

- [ ] **Step 6: Verify Task 4 GREEN**

Run:

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q \
  tests/test_attacked_region_scheduler.py
bash -n scripts/run_attacked_region_300.sh
```

Expected: tests pass and Bash syntax validation exits zero.

- [ ] **Step 7: Commit Task 4**

```bash
git add scripts/combine_attacked_region_metrics.py \
  scripts/run_attacked_region_300.sh \
  tests/test_attacked_region_scheduler.py
git commit -m "feat: schedule attacked region evaluation"
```

### Task 5: Complete Verification and Handoff

**Files:**
- Verify all modified files.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: a tested command the user can run locally.

- [ ] **Step 1: Run focused suites**

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q \
  tests/test_clean_cci_pilot.py \
  tests/test_evaluate_clean_cci_ace.py \
  tests/test_evaluate_fid_sfid.py \
  tests/test_attacked_region_scheduler.py
```

- [ ] **Step 2: Run complete suite**

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q tests
```

- [ ] **Step 3: Validate script and CLI help**

```bash
bash -n scripts/run_attacked_region_300.sh
PYTHONPATH=src .venv-ml/bin/python scripts/run_clean_cci_pilot.py --help
PYTHONPATH=src .venv-ml/bin/python scripts/evaluate_clean_cci_ace.py --help
PYTHONPATH=src .venv-ml/bin/python scripts/evaluate_fid_sfid.py --help
PYTHONPATH=src .venv-ml/bin/python scripts/combine_attacked_region_metrics.py --help
```

- [ ] **Step 4: Inspect final diff and status**

```bash
git diff --check
git status --short --branch
git log -6 --oneline
```

- [ ] **Step 5: Hand off the run command**

Provide:

```bash
cd /Users/hung.domodec.com/my-docs/cci-diff
bash scripts/run_attacked_region_300.sh
```

State that it runs 1,200 generated outputs sequentially and can be rerun after
interruption to reuse completed audits.
