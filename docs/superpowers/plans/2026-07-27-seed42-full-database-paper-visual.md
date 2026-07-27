# Seed-42 Full-Database Paper Visual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise the standalone CCI manuscript around a fixed seed-42 protocol, genuine image-based pipeline evidence, and a complete-database evaluation protocol without presenting preliminary subset experiments as final results.

**Architecture:** Add an optional, read-only predicted-clean frame observer to the existing clean CCI guidance hook and expose it through the SD2 runner. Use that observer to rerun the seed-42 sample `00131`, vendor selected real trajectory frames and masks under `paper/figures/`, and rewrite the manuscript around the final method and evaluation protocol. The observer must not alter controller state, gradients, denoising outputs, or benchmark artifacts.

**Tech Stack:** Python 3, PyTorch, Pillow, unittest/pytest, Stable Diffusion 2 blended latent diffusion, LaTeX/TikZ, Tectonic, qpdf.

## Global Constraints

- Use generation seed `42`; remove multi-seed FID selection from the method and claims.
- Show only genuine images emitted by the executed CCI pipeline as denoising intermediates.
- Remove quantitative findings from preliminary 10-, 45-, 50-, and 100-image subsets.
- Describe final metrics over the complete eligible evaluation database without assigning unmeasured values to CCI.
- Remove the complete Reproducibility and Provenance section and all local paths and hashes from the manuscript.
- Do not overwrite benchmark outputs.
- Do not create a git commit.

---

### Task 1: Add Read-Only Predicted-Clean Frame Observation

**Files:**
- Modify: `src/cci_diff/adapters/sd2_clean_cci.py`
- Modify: `tests/test_sd2_clean_cci.py`

**Interfaces:**
- Consumes: the existing `CleanCCIGuidanceHook.__call__(step)` path and its already-computed `clean_image` and `post_clean_image` tensors.
- Produces: optional constructor argument `frame_observer: Callable[[dict[str, Any]], None] | None = None`.
- Observer payload: `step`, `timestep`, `progress`, `before_image`, and `after_image`; image tensors are detached snapshots.

- [ ] **Step 1: Write the failing observer test**

Add a focused test that passes `frames.append` to `frame_observer`, executes one eligible denoising step, and asserts:

```python
self.assertEqual(len(frames), 1)
self.assertEqual(frames[0]["step"], 4)
self.assertEqual(frames[0]["timestep"], 500)
self.assertAlmostEqual(frames[0]["progress"], 0.4)
self.assertFalse(frames[0]["before_image"].requires_grad)
self.assertFalse(frames[0]["after_image"].requires_grad)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
.venv-ml/bin/python -m pytest \
  tests/test_sd2_clean_cci.py::SD2CleanCCITests::test_clean_hook_observes_detached_predicted_clean_frames \
  -q
```

Expected: failure because `CleanCCIGuidanceHook` does not accept `frame_observer`.

- [ ] **Step 3: Implement the observer**

Store the optional callback in `CleanCCIGuidanceHook.__init__`. After the post-update predicted-clean image and trace record are complete, invoke:

```python
if self.frame_observer is not None:
    self.frame_observer(
        {
            "step": step.step_index,
            "timestep": record["timestep"],
            "progress": step.progress,
            "before_image": clean_image.detach(),
            "after_image": post_clean_image.detach(),
        }
    )
```

The callback is observational only and must run after all optimization values for the step have been computed.

- [ ] **Step 4: Run the focused test and adapter suite**

Run:

```bash
.venv-ml/bin/python -m pytest tests/test_sd2_clean_cci.py -q
```

Expected: all tests pass.

---

### Task 2: Expose Deterministic Frame Export in the SD2 Runner

**Files:**
- Modify: `scripts/run_sd2_bld_cci.py`
- Modify: `tests/test_clean_cci_cli.py`

**Interfaces:**
- Consumes: `CleanCCIGuidanceHook.frame_observer` from Task 1 and `_save_rgb_grid`.
- Produces: optional CLI argument `--cci_frame_dir`.
- Output files: `step_XX_before.png`, `step_XX_after.png`, and `manifest.json` in the requested directory.

- [ ] **Step 1: Write the failing CLI test**

Add a parser test:

```python
args = build_arg_parser().parse_args(
    ["--output_dir", "out", "--cci_frame_dir", "frames"]
)
self.assertEqual(args.cci_frame_dir, "frames")
```

Add a frame-writer unit test using small RGB tensors and assert the two PNG files and one manifest entry are created with the expected step, timestep, and progress.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv-ml/bin/python -m pytest tests/test_clean_cci_cli.py -q
```

Expected: failure because the parser and writer do not yet support frame export.

- [ ] **Step 3: Implement the frame writer and parser option**

Add:

```python
parser.add_argument(
    "--cci_frame_dir",
    default=None,
    help="Optional directory for predicted-clean before/after frame snapshots.",
)
```

Implement a callable writer that creates the directory, saves both detached images with `_save_rgb_grid`, accumulates only scalar metadata, and writes `manifest.json` atomically after each callback. Instantiate it in `run_clean` only when the option is supplied, and pass it as `frame_observer`.

- [ ] **Step 4: Run the CLI and adapter tests**

Run:

```bash
.venv-ml/bin/python -m pytest \
  tests/test_clean_cci_cli.py tests/test_sd2_clean_cci.py -q
```

Expected: all tests pass.

---

### Task 3: Capture and Curate the Seed-42 Pipeline Example

**Files:**
- Create: `outputs/paper_seed42_smile_00131/`
- Create: `paper/figures/pipeline_seed42_00131/`

**Interfaces:**
- Consumes: `examples/graphs/remove_smile_clean_cci.json`, the existing `smile_00131.json` binding, local classifier/identity checkpoints, and Task 2 frame export.
- Produces: a self-contained paper asset directory with source, masks, selected early/middle/late predicted-clean estimates, raw CCI-BLD output, correction support, final output, and an asset manifest.

- [ ] **Step 1: Run the exact seed-42 exemplar without overwriting prior outputs**

Run:

```bash
.venv-ml/bin/python scripts/run_sd2_bld_cci.py \
  --cci_hook clean_constraint \
  --cci_graph examples/graphs/remove_smile_clean_cci.json \
  --cci_sample_bindings outputs/clean_cci_fid_rerank_100/seeds/seed_42/bindings/smile_00131.json \
  --classifier_path models/resnet50_multilabel_model.pth \
  --identity_model_path models/facenet_vggface2.ts \
  --output_dir outputs/paper_seed42_smile_00131 \
  --cci_frame_dir outputs/paper_seed42_smile_00131/predicted_clean \
  --batch_size 1 \
  --num_inference_steps 35 \
  --seed 42 \
  --device mps \
  --torch_dtype float32 \
  --local_files_only \
  --generation_mask_dilation_x 4 \
  --generation_mask_dilation_y 4 \
  --generation_mask_feather 3 \
  --cci_post_attack smooth_boundary
```

Expected: a complete new run under `outputs/paper_seed42_smile_00131/` with predicted-clean frames and no changes under prior benchmark directories.

- [ ] **Step 2: Verify trajectory provenance**

Check that `manifest.json` step indices match `cci_trace.jsonl`, that all saved images are RGB, and that early/middle/late choices come from increasing progress values. Compare the newly generated raw output with the prior seed-42 sample and record any byte or pixel differences rather than silently substituting artifacts.

- [ ] **Step 3: Visually inspect candidate frames**

Create contact sheets for all `after` frames and inspect them. Select one early, one middle, and one late frame based on legibility at two-column width, not on classifier score.

- [ ] **Step 4: Vendor immutable paper assets**

Copy the selected source, semantic mask, soft generation mask, Grad-CAM++ heatmap, selected predicted-clean frames, raw output, correction support, and final output to `paper/figures/pipeline_seed42_00131/`. Write `asset_manifest.json` containing source paths, selected step indices, progress values, and SHA-256 digests.

---

### Task 4: Rewrite the Manuscript Around the Final Protocol

**Files:**
- Modify: `paper/cci_conference_v1.tex`
- Modify: `paper/README.md`
- Modify if required by citation cleanup: `paper/references.bib`

**Interfaces:**
- Consumes: approved design and vendored assets from Task 3.
- Produces: a conference-style method paper with no preliminary subset claims, no multi-seed FID selector, and no Reproducibility and Provenance section.

- [ ] **Step 1: Replace the abstract and contribution list**

Describe graph-conditioned localized CCI, predicted-clean feedback, target-priority constrained gradients, soft generation/semantic mask roles, latent correction, and smooth saved-image correction. Remove all subset results, multi-seed selection, and final numeric performance claims.

- [ ] **Step 2: Replace the box-only overview figure**

Build a two-column-width image-dominant TikZ figure:

```text
source -> early x0 estimate -> middle x0 estimate -> late x0 estimate
       -> raw CCI-BLD output -> localized correction -> final output
```

Place the semantic mask, soft generation mask, Grad-CAM++ heatmap, and correction support as smaller aligned visual evidence. Keep labels short and images legible.

- [ ] **Step 3: Remove distribution-aware selection**

Delete the `\fidselector` macro, selection method section, selection contribution, related experimental setup, results, limitations, conclusion claims, and any figure/table devoted to seed-pool FID optimization.

- [ ] **Step 4: Replace preliminary results with the full-database protocol**

Retain formal metric definitions for target FR, FVA, FS, MNAC, CD, FID, locality, identity, and evaluator transfer. State that final comparisons apply the frozen seed-42 pipeline to every eligible image in the declared database split. Do not include current-method values until that complete evaluation exists.

- [ ] **Step 5: Remove obsolete appendices and provenance**

Remove subset-result appendices, directional subset sFID tables, prior-context tables containing a provisional CCI row, and the complete Reproducibility and Provenance section. Keep only method-relevant concept-graph material.

- [ ] **Step 6: Rewrite the paper README**

Document the fixed seed-42 protocol, paper asset manifest, build commands, and validation procedure. Remove benchmark-subset inventories and numeric preliminary conclusions.

---

### Task 5: Build, Inspect, and Audit the Paper

**Files:**
- Regenerate: `paper/cci_conference_v1.pdf`
- Regenerate: standard Tectonic intermediates under `paper/`

**Interfaces:**
- Consumes: the revised LaTeX and vendored figures.
- Produces: a checked conference PDF and fresh verification evidence.

- [ ] **Step 1: Run manuscript wording scans**

Run searches for preliminary sample-count claims, FID-selection terminology, local absolute paths, checkpoint hashes, and the removed section title. Any match must be either a bibliography year/reference or rewritten.

- [ ] **Step 2: Build the PDF**

Run:

```bash
tectonic --keep-logs --keep-intermediates cci_conference_v1.tex
```

from `paper/`. Expected: exit code 0, no unresolved citations/references, and no overfull boxes.

- [ ] **Step 3: Validate PDF structure**

Run:

```bash
qpdf --check cci_conference_v1.pdf
```

Expected: exit code 0 with no syntax or stream errors.

- [ ] **Step 4: Render and inspect pages**

Render the PDF pages to images, inspect the overview figure at full-page and cropped resolution, and revise if labels overlap, intermediate images are unreadable, or masks are ambiguous.

- [ ] **Step 5: Run the full test suite**

Run:

```bash
.venv-ml/bin/python -m pytest -q
```

Expected: all tests pass. Record any pre-existing unrelated failure separately; do not describe the manuscript as fully verified if the covering tests or PDF checks fail.

- [ ] **Step 6: Confirm repository scope**

Inspect `git diff -- paper scripts/run_sd2_bld_cci.py src/cci_diff/adapters/sd2_clean_cci.py tests/test_clean_cci_cli.py tests/test_sd2_clean_cci.py docs/superpowers/`. Confirm no benchmark output was overwritten and no commit was created.
