# Cross-Device Sharded Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete 300-image discovery and CCI evaluation by using both Kaggle T4 GPUs while preserving a stable single-worker MPS path.

**Architecture:** The standalone runner selects one cohort, partitions IDs
deterministically, launches one subprocess worker per resolved device, and
merges independent shard artifacts. FaceNet performs differentiable input
casting to its fixed float32 parameter dtype.

**Tech Stack:** Python 3.10+, PyTorch, Diffusers, TorchScript FaceNet, pytest,
Kaggle notebooks and CLI.

## Global Constraints

- Keep seed 42 and deterministic sample ordering.
- Use at most two CUDA workers on Kaggle.
- Use exactly one MPS worker locally.
- Do not rerank outputs.
- Preserve existing output schemas where possible.
- Commit and push only files under `cci-diff`.

---

### Task 1: FaceNet dtype compatibility

**Files:**
- Modify: `src/cci_diff/identity/facenet.py`
- Test: `tests/test_facenet_identity.py`

**Interfaces:**
- Produces: `model_input_dtype(model) -> torch.dtype`
- Produces: `FaceNetIdentityConstraint.bind()` and `.measure()` that cast
  crops before model inference without detaching gradients.

- [ ] Add a failing test using a float32 convolutional embedder and a
  float16 differentiable input.
- [ ] Verify the test fails with an input/weight dtype mismatch.
- [ ] Implement model-dtype lookup and differentiable crop casting.
- [ ] Verify identity tests pass.

### Task 2: Deterministic device and ID sharding

**Files:**
- Create: `src/cci_diff/execution_shards.py`
- Test: `tests/test_execution_shards.py`

**Interfaces:**
- Produces: `resolve_worker_devices(device, max_workers, cuda_device_count)`
- Produces: `partition_ids(sample_ids, worker_count)`
- Produces: `merge_csv_files(inputs, output)` and
  `merge_jsonl_files(inputs, output)`.

- [ ] Add failing tests for two CUDA devices, one MPS worker, deterministic
  round-robin partitions, and duplicate-free merges.
- [ ] Implement the minimal pure functions.
- [ ] Verify the new test module passes.

### Task 3: Explicit evaluation cohorts

**Files:**
- Modify: `scripts/run_clean_cci_pilot.py`
- Test: `tests/test_clean_cci_pilot.py`

**Interfaces:**
- Adds CLI option: `--sample_ids ID [ID ...]`.
- `select_eligible_samples()` iterates only supplied IDs when present.

- [ ] Add a failing parser and selection test.
- [ ] Implement explicit-ID iteration while retaining eligibility checks.
- [ ] Verify existing selection behavior remains unchanged without the flag.

### Task 4: Parallel standalone orchestration

**Files:**
- Modify: `scripts/run_kaggle_smile.py`
- Test: `tests/test_run_kaggle_smile.py`

**Interfaces:**
- Adds CLI option: `--max_workers`.
- Produces: `run_sharded_commands(...)` with per-worker device, output, status,
  and elapsed-time records.
- Discovery and evaluation write `shard_manifest.json`.

- [ ] Add failing command-construction and shard-manifest tests.
- [ ] Implement cohort selection once per mode.
- [ ] Launch discovery intervention and evaluation workers concurrently.
- [ ] Merge shard tables and failures.
- [ ] Run graph freezing only after all discovery shards complete.

### Task 5: Notebook configuration

**Files:**
- Modify: `notebooks/01_global_graph_discovery.ipynb`
- Modify: `notebooks/02_full_cci_fixed_vs_adaptive.ipynb`
- Test: `tests/test_kaggle_notebooks.py`

**Interfaces:**
- Both notebooks pass `--max_workers 2` with `DEVICE = 'cuda'`.

- [ ] Add failing notebook contract assertions.
- [ ] Update standalone commands.
- [ ] Verify notebook tests pass.

### Task 6: Verification and remote restart

**Files:**
- Modify only if verification exposes a defect.

- [ ] Run focused tests for identity, sharding, orchestration, and notebooks.
- [ ] Run the full pytest suite.
- [ ] Commit under `lokissdo <lokissdo@users.noreply.github.com>` and push.
- [ ] Verify the global Git identity remains unchanged.
- [ ] Push new versions of both Kaggle notebooks with the full commit SHA.
- [ ] Confirm logs show `cuda:0` and `cuda:1`, no FaceNet dtype error, and
  completed metric rows from both shards.
