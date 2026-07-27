# A3 100-Image-Per-Task Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and evaluate 100 spatially selected A3 counterfactuals for smile removal and 100 for blond-hair addition, with three deterministic mask candidates and a paired input artifact for every selected output.

**Architecture:** Extend the existing mask utility and pilot runner instead of adding a second generation stack. Keep candidate scoring, visual comparison rendering, and ACE evaluation in focused modules so generation remains resumable and metric code can be tested without loading diffusion models.

**Tech Stack:** Python 3.10, PyTorch, torchvision, Pillow, NumPy, Stable Diffusion 2, MPS, pytest, ACE CelebA-HQ/VGGFace2/SimSiam checkpoints.

## Global Constraints

- Work only under `/Users/hung.domodec.com/my-docs/cci-diff`; read ACE checkpoints from `/Users/hung.domodec.com/my-docs/thesis_2025/evaluate/ACE`.
- Create no git commit and stage no file.
- Preserve existing A0-A4 defaults unless new CLI flags are supplied.
- Use A3, seed 42, 35 denoising steps, float32, batch size 1, and MPS.
- Generate dilation candidates 0, 4, and 8 at 512-by-512 resolution with 3-pixel feathering.
- Select with the generation classifier; reserve the independent ACE oracle for evaluation.
- Treat target success as primary and spatial minimality as secondary.
- Preserve and resume every valid candidate audit.
- Produce exactly 100 selected smile results, 100 selected hair results, and 600 complete candidate rows.

---

### Task 1: Dilation-Aware Dual Masks

**Files:**
- Modify: `src/cci_diff/masking.py`
- Modify: `scripts/run_sd2_bld_cci.py`
- Modify: `tests/test_clean_cci_cli.py`
- Modify: `tests/test_sd2_clean_cci.py`

**Interfaces:**
- Consumes: aligned binary semantic component paths.
- Produces: `prepare_semantic_masks(..., dilation_radius: int = 0) -> MaskArtifacts` and clean CLI option `--generation_mask_dilation`.

- [ ] **Step 1: Write failing mask tests**

Add tests that create a one-pixel 9-by-9 component mask and assert:

```python
artifacts = prepare_semantic_masks(
    [component],
    feather_radius=0,
    dilation_radius=2,
    hard_output=tmp_path / "semantic.png",
    soft_output=tmp_path / "generation.png",
)
semantic = np.asarray(Image.open(artifacts.semantic_path)) >= 128
generation = np.asarray(Image.open(artifacts.generation_path)) >= 128
assert semantic.sum() == 1
assert generation.sum() == 25
```

Also assert negative and non-integer dilation values fail validation and that omitted dilation preserves current output.

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv-ml/bin/pytest tests/test_sd2_clean_cci.py tests/test_clean_cci_cli.py -q
```

Expected: failures because `dilation_radius` and `--generation_mask_dilation` do not exist.

- [ ] **Step 3: Implement mask dilation**

In `prepare_semantic_masks`, preserve `semantic` before dilation and derive the generation support with Pillow `MaxFilter`:

```python
if isinstance(dilation_radius, bool) or not isinstance(dilation_radius, int):
    raise TypeError("dilation_radius must be an integer")
if dilation_radius < 0:
    raise ValueError("dilation_radius must be non-negative")
dilated = (
    semantic
    if dilation_radius == 0
    else semantic.filter(ImageFilter.MaxFilter(2 * dilation_radius + 1))
)
generation = dilated.filter(ImageFilter.GaussianBlur(feather_radius))
```

Add the CLI integer argument with default 0, pass it through clean and legacy mask preparation, validate non-negativity, and include it in audit metadata.

- [ ] **Step 4: Verify GREEN**

Run the focused tests and the complete existing suite:

```bash
.venv-ml/bin/pytest tests/test_sd2_clean_cci.py tests/test_clean_cci_cli.py -q
.venv-ml/bin/pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Checkpoint without commit**

Run `git diff -- src/cci_diff/masking.py scripts/run_sd2_bld_cci.py tests/test_sd2_clean_cci.py tests/test_clean_cci_cli.py` and leave changes unstaged.

### Task 2: Spatial Metrics and Target-First Candidate Selection

**Files:**
- Create: `src/cci_diff/spatial_selection.py`
- Create: `tests/test_spatial_selection.py`

**Interfaces:**
- Produces: `measure_spatial_change(source_path, output_path, semantic_mask_path, generation_mask_path) -> dict[str, float]`.
- Produces: `select_spatial_candidate(rows: Sequence[Mapping[str, Any]], target_probability: float = 0.8) -> Mapping[str, Any]`.

- [ ] **Step 1: Write failing metric tests**

Use synthetic 4-by-4 RGB images and masks. Assert exact changed fractions at thresholds `1/255`, `5/255`, and `10/255`, exact inside/outside L1 values, and semantic/generation mask fractions.

Use candidate rows to assert:

```python
selected = select_spatial_candidate([
    {"candidate": "d0", "desired_probability": 0.79, "changed_fraction_5": 0.01},
    {"candidate": "d4", "desired_probability": 0.82, "changed_fraction_5": 0.05},
    {"candidate": "d8", "desired_probability": 0.91, "changed_fraction_5": 0.08},
])
assert selected["candidate"] == "d4"
```

Add cases where two candidates pass and the smaller area wins, and where all fail and the highest desired probability wins.

- [ ] **Step 2: Verify RED**

Run `.venv-ml/bin/pytest tests/test_spatial_selection.py -q`.

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement spatial measurements**

Load source/output as aligned RGB float arrays in `[0, 1]`, resize masks with nearest interpolation for semantic and bilinear for generation, and return stable scalar fields:

```python
delta = np.max(np.abs(output - source), axis=2)
for level in (1, 5, 10):
    changed = delta > level / 255.0
    metrics[f"changed_fraction_{level}"] = float(changed.mean())
    metrics[f"outside_semantic_fraction_{level}"] = float(
        np.logical_and(changed, ~semantic).mean()
    )
    metrics[f"outside_generation_fraction_{level}"] = float(
        (changed * (1.0 - generation)).mean()
    )
```

Add mean RGB L1 inside/outside masks, mask fractions, and deterministic candidate selection using tuples:

```python
passing_key = lambda row: (
    row["changed_fraction_5"],
    row["outside_semantic_fraction_5"],
    -row["desired_probability"],
    -value_or_minus_inf(row.get("identity_cosine")),
)
failure_key = lambda row: (
    -row["desired_probability"],
    row["changed_fraction_5"],
)
```

- [ ] **Step 4: Verify GREEN**

Run `.venv-ml/bin/pytest tests/test_spatial_selection.py -q` and `.venv-ml/bin/pytest -q`.

Expected: all tests pass.

### Task 3: Portable Input/Output Comparison Artifacts

**Files:**
- Create: `src/cci_diff/comparison_artifacts.py`
- Create: `tests/test_comparison_artifacts.py`

**Interfaces:**
- Produces: `materialize_selected_artifacts(source, candidate_dir, result_dir, metadata) -> dict[str, str]`.
- Produces: `create_pair_image(source, output, destination, label) -> Path`.
- Produces: `create_paginated_pair_sheets(rows, output_dir, page_size=20) -> list[Path]`.

- [ ] **Step 1: Write failing artifact tests**

Create unequal-aspect source and output fixtures. Assert `input.jpg`, root selected output/audit/masks, `selected.json`, and `input_output.jpg` exist; inspect pair pixels to prove input is left and output is right; assert neither image is stretched.

- [ ] **Step 2: Verify RED**

Run `.venv-ml/bin/pytest tests/test_comparison_artifacts.py -q`.

Expected: import failure.

- [ ] **Step 3: Implement artifact rendering**

Use `ImageOps.contain` inside two fixed 512-by-512 panels, a 36-pixel label band, and atomic temporary-file replacement. Copy source and selected candidate files rather than symlinking. Paginate by feature with at most 20 pairs per page.

- [ ] **Step 4: Verify GREEN**

Run the focused and full test suites. Expected: all tests pass.

### Task 4: Resumable A3 Candidate Orchestration

**Files:**
- Modify: `scripts/run_clean_cci_pilot.py`
- Modify: `tests/test_clean_cci_pilot.py`

**Interfaces:**
- Adds CLI: `--variants`, `--mask_dilations`, `--continue_on_error`.
- Produces: `candidate_results.csv`, selected `pilot_results.csv`, `failures.jsonl`, candidate directories, and selected root artifacts.

- [ ] **Step 1: Write failing orchestration tests**

Add tests proving:

```python
args = parser.parse_args(base_args + [
    "--variants", "A3",
    "--mask_dilations", "0", "4", "8",
    "--continue_on_error",
])
assert args.variants == ["A3"]
assert args.mask_dilations == [0, 4, 8]
```

Mock only subprocess execution while using real audit fixtures. Verify command lists include `--generation_mask_dilation`, valid candidate audits resume, one candidate failure does not stop later samples, candidate selection materializes root artifacts, and default invocation still traverses A0-A4 with dilation 0.

- [ ] **Step 2: Verify RED**

Run `.venv-ml/bin/pytest tests/test_clean_cci_pilot.py -q`.

Expected: parser and orchestration assertions fail.

- [ ] **Step 3: Implement selected variants and candidates**

Validate unique variants and dilations. For each feature/source/variant/dilation, use:

```text
<output>/<feature>/<sample_id>/<variant>/candidates/d<radius>/
```

Pass the dilation to `run_sd2_bld_cci.py`, extract audit metrics, add spatial metrics, and write `candidate_results.csv` after each completion. Once all candidates are valid, call `select_spatial_candidate`, materialize root artifacts, and append the selected row to `pilot_results.csv`.

Write JSON Lines failures containing feature, sample, variant, dilation, exit code, and audit path. Under `--continue_on_error`, continue; otherwise retain current fail-fast behavior. Final completeness checks distinguish unresolved candidates from historical resolved failure lines.

- [ ] **Step 4: Update summaries and contact sheets**

Generate rankings and summaries only from selected rows. Add selected dilation, changed-area fields, source path, input-copy path, output path, and comparison path. Generate paginated pair sheets using Task 3.

- [ ] **Step 5: Verify GREEN**

Run `.venv-ml/bin/pytest tests/test_clean_cci_pilot.py -q` and `.venv-ml/bin/pytest -q`.

Expected: all tests pass.

### Task 5: Independent ACE Evaluator and Statistical Report

**Files:**
- Create: `scripts/evaluate_clean_cci_ace.py`
- Create: `tests/test_evaluate_clean_cci_ace.py`

**Interfaces:**
- CLI consumes `--experiment_root`, `--ace_root`, `--device`, `--batch_size`, and `--bootstrap_seed`.
- Produces: `ace_pair_metrics.csv`, `ace_task_summary.csv`, `ace_metrics.json`, and `ace_paper_comparison.md`.

- [ ] **Step 1: Write failing pure-metric tests**

Test directional success for desired values 0/1, MNAC with target-excluded collateral flips, corrected source-counterfactual cosine pairing, CD from fixed binary matrices, fixed-seed bootstrap intervals, and success-conditioned summaries.

- [ ] **Step 2: Verify RED**

Run `.venv-ml/bin/pytest tests/test_evaluate_clean_cci_ace.py -q`.

Expected: import failure.

- [ ] **Step 3: Implement checkpoint-backed batched scoring**

Load and release one model family at a time:

1. ACE CelebA-HQ oracle for probabilities, directional FR, MNAC, collateral, and CD.
2. VGGFace2 for corrected FVA cosine between each source and its output.
3. SimSiam for FS cosine between each source and its output.

Use exact local ACE preprocessing. Never compute `cosine(cf, cf)`. Process rows in bounded batches and retain candidate spatial fields from `pilot_results.csv`.

- [ ] **Step 4: Implement task summaries**

For each task report unconditional and target-success-conditioned FVA, FS, MNAC, and spatial fields; desired probability mean/median; FR; CD; and 10,000 fixed-seed bootstrap intervals.

- [ ] **Step 5: Implement exploratory FID**

Use `pytorch_fid.fid_score.calculate_fid_given_paths` on temporary paired source/output directories per task. If the package or Inception weights are unavailable, record the exact exception and command needed instead of writing a numeric placeholder.

- [ ] **Step 6: Implement paper comparison report**

Render the supplied prior-paper table, task sample counts, protocol caveats, and selected A3 metrics. Mark COUT unavailable due to the invalid local binary mapping.

- [ ] **Step 7: Verify GREEN**

Run the focused and complete suites. Expected: all tests pass without loading large checkpoints in unit tests.

### Task 6: Four-Result Validation Run

**Files:**
- Generate under: `outputs/clean_cci_a3_100/`

- [ ] **Step 1: Run two samples per task**

Run the pilot with features smile/hair, limit 2, variants A3, dilations 0/4/8, seed 42, 35 steps, MPS, local classifier/identity/SD2 checkpoints, and continue-on-error.

- [ ] **Step 2: Verify generation artifacts**

Assert 12 candidate audits, four `selected.json` files, four copied inputs, four selected outputs, four pair images, 12 candidate CSV rows, and four selected rows.

- [ ] **Step 3: Inspect visual pairs**

Open the four pair images and confirm input-left/output-right layout, nonblank outputs, correct hair/mouth locality, and no incoherent seams.

- [ ] **Step 4: Run ACE validation evaluation**

Run the evaluator, verify four metric rows and two task summaries, and inspect target success plus spatial selection evidence. If generation or evaluation fails, return to the relevant task before scaling.

### Task 7: Full 100-Per-Task Generation

- [ ] **Step 1: Resume with limit 100**

Run the identical pilot command with limit 100. Keep the process attached and poll until it exits. Do not start a second generator on MPS.

- [ ] **Step 2: Retry unresolved failures**

Re-run the same command once. Resume valid candidate audits and regenerate only missing/malformed candidates.

- [ ] **Step 3: Verify exact completeness**

Require 600 candidate rows, 200 selected rows, 600 parseable candidate audits, 200 selected audits, and 200 pair images. Report unresolved sample IDs instead of claiming completion if any count differs.

### Task 8: Full Metrics and Final Comparison

- [ ] **Step 1: Run independent ACE evaluation**

Score all 200 selected pairs and calculate task summaries, CD, FID or a reproducible FID dependency failure, and bootstrap intervals.

- [ ] **Step 2: Verify report consistency**

Recompute row counts and summary means from CSV, assert every path exists, and confirm no independent-oracle value was used for candidate selection.

- [ ] **Step 3: Run final tests**

Run `.venv-ml/bin/pytest -q` and report the exact pass/fail count.

- [ ] **Step 4: Deliver artifacts without commit**

Provide clickable paths to contact sheets, pair metrics, task summaries, JSON protocol record, and paper comparison. State runtime, success counts, unresolved metric limitations, and confirm no commit or staging occurred.
