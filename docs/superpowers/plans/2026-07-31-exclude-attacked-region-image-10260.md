# Exclude Attacked-Region Image 10260 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude only smile image `10260` from the attacked-region 300-image experiment while retaining a deterministic 300-image cohort.

**Architecture:** Use the pilot's existing task-specific `--exclude_ids_json` interface. Store the experiment exclusion in a small JSON configuration file and pass it from the shared attacked-region generation command, leaving detector and error-handling behavior unchanged.

**Tech Stack:** Bash, JSON, pytest

## Global Constraints

- Exclude only image ID `10260` for the `smile` feature.
- Do not enable general continue-on-error behavior.
- Do not change face detection, exact-count validation, or metrics.

---

### Task 1: Configure the targeted attacked-region exclusion

**Files:**
- Create: `examples/attacked_region_excluded_ids.json`
- Modify: `scripts/run_attacked_region_300.sh:7-48`
- Test: `tests/test_attacked_region_scheduler.py`

**Interfaces:**
- Consumes: `run_clean_cci_pilot.py --exclude_ids_json PATH`, where the JSON maps feature names to integer ID arrays.
- Produces: an attacked-region pilot invocation that loads `{"smile": [10260]}`.

- [x] **Step 1: Write the failing scheduler test**

Add JSON loading and assertions to the existing scheduler test:

```python
def test_scheduler_excludes_only_unreliable_face_image():
    script = Path("scripts/run_attacked_region_300.sh").read_text(
        encoding="utf-8"
    )
    exclusions = json.loads(
        Path("examples/attacked_region_excluded_ids.json").read_text(
            encoding="utf-8"
        )
    )

    assert '--exclude_ids_json "$EXCLUDED_IDS"' in script
    assert exclusions == {"smile": [10260]}
```

- [x] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=src .venv-ml/bin/python -m pytest tests/test_attacked_region_scheduler.py::test_scheduler_excludes_only_unreliable_face_image -v`

Expected: FAIL because `examples/attacked_region_excluded_ids.json` does not exist.

- [x] **Step 3: Add the exclusion configuration and launcher argument**

Create `examples/attacked_region_excluded_ids.json`:

```json
{
  "smile": [10260]
}
```

Define the path next to the launcher's other input paths:

```bash
EXCLUDED_IDS="$ROOT/examples/attacked_region_excluded_ids.json"
```

Pass it to every shared pilot invocation:

```bash
        --exclude_ids_json "$EXCLUDED_IDS" \
```

- [x] **Step 4: Run focused tests to verify they pass**

Run: `PYTHONPATH=src .venv-ml/bin/python -m pytest tests/test_attacked_region_scheduler.py tests/test_clean_cci_pilot.py::TestCleanCCIPilot::test_excluded_discovery_ids_are_loaded_per_feature -v`

Expected: all tests PASS.

- [x] **Step 5: Check shell syntax and the diff**

Run: `bash -n scripts/run_attacked_region_300.sh && git diff --check`

Expected: both commands exit successfully with no output.

- [x] **Step 6: Commit**

```bash
git add examples/attacked_region_excluded_ids.json scripts/run_attacked_region_300.sh tests/test_attacked_region_scheduler.py docs/superpowers/plans/2026-07-31-exclude-attacked-region-image-10260.md
git commit -m "fix: exclude unreliable attacked-region image"
```
