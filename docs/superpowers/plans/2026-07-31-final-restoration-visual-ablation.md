# Final-Restoration Visual Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible one-image experiment that shows only the visible and metric effect of A11 final latent restoration.

**Architecture:** A standalone Python orchestration script will reuse the pilot's graph, binding, mask, and A11 command builders. It will run identical A11 commands with final restoration disabled and enabled, validate their audits, calculate image and classifier deltas, and render comparison artifacts without changing the production diffusion path.

**Tech Stack:** Python 3.10, argparse, subprocess, pathlib, JSON, NumPy, Pillow, pytest, existing SD2/CCI runner.

## Global Constraints

- Use CelebAMask-HQ sample `26811`, seed `42`, mouth-only graph, dilation `8`, 35 SD2 inference steps, MPS, and float32 by default.
- Set post-attack mode to `none` in both cases.
- The only behavioral switch between cases is `--cci_disable_final_correction`.
- Do not modify the existing 300-image scheduler or production BLD implementation.
- Report MPS reproducibility mismatch rather than hiding it.

---

### Task 1: Controlled command construction

**Files:**
- Create: `scripts/run_final_restoration_ablation.py`
- Create: `tests/test_final_restoration_ablation.py`

**Interfaces:**
- Consumes: `MaskCandidate`, `annotation_paths`, `build_variant_command`, `resolve_binding_roles`, `write_binding`, and `write_region_graph` from `scripts.run_clean_cci_pilot`.
- Produces: `prepare_ablation(args) -> dict[str, Any]` containing graph, binding, source, masks, and two commands.

- [ ] **Step 1: Write failing tests for command parity**

Create fixtures under a temporary directory, call `prepare_ablation`, and assert:

```python
assert "--cci_post_attack" not in enabled
assert "--cci_disable_final_correction" not in enabled
assert disabled[-1] == "--cci_disable_final_correction"

def normalized(command):
    result = list(command)
    output_index = result.index("--output_dir") + 1
    result[output_index] = "<OUTPUT>"
    return result

assert normalized(disabled[:-1]) == normalized(enabled)
```

Also assert the generated graph contains only `["mouth"]` and the binding contains only the `mouth` role.

- [ ] **Step 2: Run the command tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q \
  tests/test_final_restoration_ablation.py -k prepare
```

Expected: fail because `scripts.run_final_restoration_ablation` does not exist.

- [ ] **Step 3: Implement command preparation**

Add:

```python
def prepare_ablation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    source = Path(args.image_root) / f"{args.sample_id}.jpg"
    masks = annotation_paths(
        Path(args.mask_root),
        args.sample_id,
        FEATURES["smile"]["components"],
    )
    graph = write_region_graph(
        FEATURES["smile"]["graph"],
        output_dir / "config" / "remove_smile_mouth_only.json",
        ("mouth",),
    )
    binding = output_dir / "config" / f"smile_{args.sample_id:05d}.json"
    write_binding(
        binding,
        source,
        masks,
        resolve_binding_roles("smile", ["mouth"]),
    )
    common = build_variant_command(
        pilot_args_from_ablation(args),
        feature="smile",
        variant="A11",
        sample_id=args.sample_id,
        source=source,
        masks=masks,
        binding_path=binding,
        output_path=output_dir / "with_final_restoration",
        mask_candidate=MaskCandidate("d8", args.mask_dilation),
        graph_path=graph,
    )
    enabled = list(common)
    disabled = replace_output_dir(
        common,
        output_dir / "without_final_restoration",
    ) + ["--cci_disable_final_correction"]
    return {
        "source": source,
        "masks": masks,
        "graph": graph,
        "binding": binding,
        "enabled_command": enabled,
        "disabled_command": disabled,
    }
```

Validate the executable, source, masks, graph, model, classifier, and identity checkpoint before returning.

- [ ] **Step 4: Run command tests and verify GREEN**

Run the focused test from Step 2. Expected: pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add scripts/run_final_restoration_ablation.py \
  tests/test_final_restoration_ablation.py
git commit -m "feat: prepare final restoration ablation"
```

### Task 2: Audit, image, and report comparison

**Files:**
- Modify: `scripts/run_final_restoration_ablation.py`
- Modify: `tests/test_final_restoration_ablation.py`

**Interfaces:**
- Produces:
  - `compare_ablation(before_dir, after_dir, output_dir, tolerance) -> dict[str, Any]`
  - `render_comparison(before, after, output_dir) -> dict[str, str]`
  - `run(args) -> dict[str, Any]`

- [ ] **Step 1: Write failing metric and artifact tests**

Use two 4×4 RGB fixtures and synthetic audits. Assert:

```python
assert result["pixel"]["mean_absolute_difference"] == pytest.approx(10 / 255)
assert result["pixel"]["maximum_absolute_difference"] == pytest.approx(10 / 255)
assert result["pixel"]["changed_fraction"] == 1.0
assert result["restoration"]["initial_probability"] == pytest.approx(0.2)
assert result["restoration"]["final_probability"] == pytest.approx(0.8)
assert Path(result["artifacts"]["side_by_side"]).is_file()
assert Path(result["artifacts"]["difference_amplified"]).is_file()
```

Add rejection tests for post-attack data, missing restoration records, unequal image sizes, and a consistency gap greater than the configured tolerance.

- [ ] **Step 2: Run metric tests and verify RED**

Run:

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q \
  tests/test_final_restoration_ablation.py -k 'compare or reject'
```

Expected: fail because comparison functions do not exist.

- [ ] **Step 3: Implement comparison and rendering**

Read `audit.json` and `sd2_bld_grid.png` from both case directories. Require:

```python
before["cci"].get("post_attack") is None
after["cci"].get("post_attack") is None
before["cci"].get("trust_region_final_restoration") is None
restoration = after["cci"]["trust_region_final_restoration"]
```

Calculate normalized absolute pixel differences and extract desired probability,
identity cosine, mean non-target drift, wall time, accepted restoration steps,
and restoration attempts. Require:

```python
abs(
    restoration["initial_probability"]
    - before_desired_probability
) <= tolerance
```

Copy the untouched case images to:

- `before_without_final_restoration.png`
- `after_with_final_restoration.png`

Render:

- `before_after_side_by_side.png` with labels above unmodified images;
- `difference_amplified.png` using `clip(abs(after-before) * 8, 0, 1)`.

Write `comparison.json` and `comparison.md`, including both commands and the
consistency result.

- [ ] **Step 4: Implement orchestration**

Run the disabled command first and enabled command second with
`subprocess.run(command, check=True)`. Reuse a valid existing case only when
`--reuse` is explicitly supplied. Return the comparison payload and print the
side-by-side path.

- [ ] **Step 5: Run all new tests**

Run:

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q \
  tests/test_final_restoration_ablation.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add scripts/run_final_restoration_ablation.py \
  tests/test_final_restoration_ablation.py
git commit -m "feat: report final restoration visual ablation"
```

### Task 3: Verification and one-image execution

**Files:**
- Verify: `scripts/run_final_restoration_ablation.py`
- Produce: `outputs/final_restoration_ablation_26811/`

**Interfaces:**
- Consumes the completed standalone script.
- Produces the user-visible comparison images and metric report.

- [ ] **Step 1: Run focused and regression verification**

```bash
PYTHONPATH=src .venv-ml/bin/python -m pytest -q \
  tests/test_final_restoration_ablation.py \
  tests/test_clean_cci_pilot.py
git diff --check
```

Expected: zero failures and no whitespace errors.

- [ ] **Step 2: Run the ablation**

```bash
caffeinate -dimsu env PYTHONPATH=src .venv-ml/bin/python -u \
  scripts/run_final_restoration_ablation.py \
  --sample-id 26811 \
  --output-dir outputs/final_restoration_ablation_26811
```

Expected: two successful SD2 runs and a printed
`before_after_side_by_side.png` path.

- [ ] **Step 3: Validate outputs**

```bash
PYTHONPATH=src .venv-ml/bin/python -c \
  'import json; from pathlib import Path; p=Path("outputs/final_restoration_ablation_26811"); d=json.loads((p/"comparison.json").read_text()); assert d["consistency"]["passed"]; assert (p/"before_after_side_by_side.png").is_file()'
```

Expected: exit 0.

- [ ] **Step 4: Visually inspect**

Open the before image, after image, side-by-side image, and amplified difference
image. Report visible mouth changes separately from amplified diagnostic
differences.

