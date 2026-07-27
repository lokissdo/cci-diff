# Anisotropic Perioral Mask Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add backward-compatible anisotropic generation-mask dilation and run a controlled four-candidate mouth-realism experiment on CelebA-HQ image `00000`.

**Architecture:** The masking layer resolves scalar or x/y dilation into a rectangular binary dilation before Gaussian feathering. The clean-CCI CLI forwards geometry into mask construction and audit metadata. The pilot accepts explicit `x,y,feather` candidate specifications while retaining the existing scalar `--mask_dilations` path.

**Tech Stack:** Python 3.11, NumPy, Pillow, PyTorch, pytest, SD2/diffusers, MPS.

## Global Constraints

- Do not change the hard semantic mask, CCI loss, controller weights, prompt, model, seed, or 35-step schedule.
- Existing scalar dilation behavior and CLI calls remain compatible.
- Do not commit or stage any files.
- Work only under `/Users/hung.domodec.com/my-docs/cci-diff`.

---

### Task 1: Rectangular Generation-Mask Dilation

**Files:**
- Modify: `src/cci_diff/masking.py`
- Test: `tests/test_masking.py`

**Interfaces:**
- Extend `build_mask_artifacts(..., dilation_radius=0, dilation_x=None, dilation_y=None)`.
- Scalar dilation resolves to `x=y=dilation_radius`.
- Explicit axes override the corresponding scalar radius.

- [ ] **Step 1: Add failing rectangular-dilation tests**

Add tests that create a one-pixel semantic mask, request `dilation_x=2, dilation_y=1`, and assert a 5-by-3 hard generation support before feathering. Assert the saved semantic mask remains one pixel. Add validation cases for negative, boolean, and non-integer axis radii.

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
.venv-ml/bin/python -m pytest tests/test_masking.py -q
```

Expected: failure because `dilation_x` and `dilation_y` are not accepted.

- [ ] **Step 3: Implement separable rectangular maximum filtering**

Add a private helper that pads the binary NumPy mask with zeros, takes a horizontal maximum across `2*x+1`, then a vertical maximum across `2*y+1`, and converts the result back to `L` mode. Reuse the existing Gaussian blur for feathering.

- [ ] **Step 4: Verify masking tests and scalar compatibility**

Run:

```bash
.venv-ml/bin/python -m pytest tests/test_masking.py -q
```

Expected: all masking tests pass, including the pre-existing scalar test.

### Task 2: CLI And Audit Geometry

**Files:**
- Modify: `scripts/run_sd2_bld_cci.py`
- Test: `tests/test_clean_cci_cli.py`

**Interfaces:**
- Add optional CLI flags `--generation_mask_dilation_x` and `--generation_mask_dilation_y`.
- Existing `--generation_mask_feather` overrides the graph feather for generation-mask construction while leaving the semantic mask unchanged.
- Audit metadata records scalar, x radius, y radius, and effective feather.

- [ ] **Step 1: Add failing parser, validation, and audit tests**

Test acceptance of `x=12`, `y=6`, feather `7`; rejection of negative axis values; and audit serialization of all effective values.

- [ ] **Step 2: Verify the focused tests fail**

Run:

```bash
.venv-ml/bin/python -m pytest tests/test_clean_cci_cli.py -q
```

Expected: parser failure for unknown axis options.

- [ ] **Step 3: Forward effective geometry through clean mask compilation**

Resolve omitted axes from scalar dilation, pass the effective radii and feather to `build_mask_artifacts`, and retain existing defaults of scalar zero and graph feather when no override is supplied.

- [ ] **Step 4: Run focused CLI tests**

Run:

```bash
.venv-ml/bin/python -m pytest tests/test_clean_cci_cli.py tests/test_masking.py -q
```

Expected: all focused tests pass.

### Task 3: Explicit Pilot Mask Candidates

**Files:**
- Modify: `scripts/run_clean_cci_pilot.py`
- Test: `tests/test_clean_cci_pilot.py`

**Interfaces:**
- Add repeatable candidate input `--mask_shapes x,y,feather`.
- When supplied, shape candidates replace `--mask_dilations` for that run.
- Candidate labels use `x{X}_y{Y}_f{F}` and rows include `dilation_x`, `dilation_y`, and `feather_radius`.
- Existing scalar runs retain labels such as `d0`, `d4`, and `d8`.

- [ ] **Step 1: Add failing candidate parser and command tests**

Parse `4,4,3 8,4,5 12,6,7 16,8,9`; assert four unique candidates and exact CLI forwarding. Retain the existing scalar candidate assertions.

- [ ] **Step 2: Verify pilot tests fail**

Run:

```bash
.venv-ml/bin/python -m pytest tests/test_clean_cci_pilot.py -q
```

Expected: unknown `--mask_shapes` option or missing geometry fields.

- [ ] **Step 3: Implement candidate resolution and metadata**

Centralize scalar and anisotropic candidates into one resolved candidate structure. Use its label for directories and its geometry for command construction, manifests, candidate CSV rows, selected rows, and selection JSON.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
.venv-ml/bin/python -m pytest tests/test_clean_cci_pilot.py tests/test_clean_cci_cli.py tests/test_masking.py -q
.venv-ml/bin/python -m pytest -p no:cacheprovider -q
```

Expected: focused tests and the full suite pass.

### Task 4: Controlled Image `00000` Experiment

**Files:**
- Create runtime outputs under: `outputs/clean_cci_smile_00000_perioral/`

**Interfaces:**
- Input: smile feature, limit 1, seed 42, A3, 35 steps.
- Candidates: `4,4,3`, `8,4,5`, `12,6,7`, `16,8,9`.

- [ ] **Step 1: Run all candidates on MPS**

Run:

```bash
.venv-ml/bin/python scripts/run_clean_cci_pilot.py \
  --features smile --limit 1 --seed 42 --num_inference_steps 35 \
  --device mps --model_path checkpoints/sd2-1-base \
  --classifier_path models/resnet50_multilabel_model.pth \
  --identity_model_path models/facenet_vggface2.ts \
  --output_dir outputs/clean_cci_smile_00000_perioral \
  --variants A3 --mask_shapes 4,4,3 8,4,5 12,6,7 16,8,9 \
  --continue_on_error
```

Expected: four candidate outputs, one selected pair, CSV metrics, and no failure records.

- [ ] **Step 2: Compare numerical evidence**

Report target probability, identity cosine, boundary discontinuity, changed fraction at `5/255`, outside-semantic changed fraction, selected geometry, and runtime for every candidate and the seed-42 control.

- [ ] **Step 3: Perform visual rejection review**

Inspect all four outputs at original resolution. Reject doubled lips, residual teeth, mouth ghosts, broken corners, and skin-transition artifacts even when classifier confidence passes.

- [ ] **Step 4: Record the decision**

State whether anisotropic masking should advance to larger testing. Keep the experiment if and only if at least one candidate reaches `0.8` and visibly improves the mouth without material identity or locality regression.
