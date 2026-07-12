# CCI-Diff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new CCI-Diff repository that can reuse the ESWA SD2 code while adding causal concept intervention metrics, prompt/spec contracts, and eventually in-loop diffusion guidance.

**Architecture:** Keep framework-neutral research logic in `src/cci_diff`. Add the GPU-heavy SD2 adapter later as a thin wrapper around the old `thesis_2025` inference loop. The old code remains the reference implementation until the CCI hook is stable.

**Tech Stack:** Python 3.10+, standard-library tests for the core; optional GPU dependencies are torch, torchvision, diffusers, open-clip-torch, Pillow, numpy, and OpenCV.

## Global Constraints

- Do not copy large old thesis files into this repo unless the adapter needs a small, reviewed function.
- Keep core metrics importable without torch or diffusers.
- Preserve old ESWA behavior when no CCI config is passed to the future adapter.
- Use TDD for production code.

---

### Task 1: Core CCI Metrics

**Files:**
- Create: `src/cci_diff/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `concept_delta`, `target_concept_success`, `preservation_score`, `concept_leakage`, `counterfactual_purity`, `causal_concept_effect`, `bias_audit_matrix`

- [x] **Step 1: Write failing tests for leakage, purity, CCE, and audit matrix.**
- [x] **Step 2: Run tests and verify import failure.**
- [x] **Step 3: Implement pure-Python metric helpers.**
- [x] **Step 4: Run tests and verify pass.**

### Task 2: Intervention Spec And Prompt Contract

**Files:**
- Create: `src/cci_diff/spec.py`
- Create: `src/cci_diff/prompts.py`
- Test: `tests/test_spec.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Produces: `ConceptIntervention`, `GuidanceWeights`, `build_concept_prompt`

- [x] **Step 1: Write failing tests for intervention validation and audit concept derivation.**
- [x] **Step 2: Implement frozen dataclasses with explicit validation.**
- [x] **Step 3: Write failing tests for positive/negative prompt generation.**
- [x] **Step 4: Implement prompt builder.**
- [x] **Step 5: Run all tests.**

### Task 3: Framework-Neutral Guidance Objective

**Files:**
- Create: `src/cci_diff/guidance.py`
- Test: `tests/test_guidance.py`

**Interfaces:**
- Produces: `GuidanceTerms`, `compose_guidance_loss`

- [x] **Step 1: Write failing test for weighted objective composition.**
- [x] **Step 2: Implement tensor-compatible weighted sum without importing torch.**
- [x] **Step 3: Run all tests.**

### Task 4: SD2 Adapter Prototype

**Files:**
- Create: `src/cci_diff/adapters/sd2_cci.py`
- Test: `tests/test_sd2_adapter_contract.py`

**Interfaces:**
- Consumes: `ConceptIntervention`, `GuidanceWeights`, `GuidanceTerms`, `compose_guidance_loss`
- Produces: `apply_cci_latent_guidance(latents, decode_fn, loss_fn, weights, step_size, latent_mask=None)`

- [x] **Step 1: Write tests with a fake scalar/tensor object that proves the adapter calls decode, computes loss, and applies a gradient-like update.**
- [x] **Step 2: Implement a duck-typed adapter that imports torch only inside the function.**
- [x] **Step 3: Add a clear `ImportError` message when torch is unavailable.**
- [x] **Step 4: Run standard tests locally and GPU tests in Kaggle/Colab later.**

### Task 5: ESWA SD2 Hook Script

**Files:**
- Create: `src/cci_diff/config.py`
- Create: `scripts/run_sd2_cci_from_legacy.py`
- Modify later in copied/derived script only: old `text_editing_SD2.py` loop

**Interfaces:**
- Consumes: JSON CCI config with target concept, desired value, preserved concepts, candidate concepts, weights, prompt, image path, mask path, and classifier path.
- Produces: generated images plus `audit.json` containing concept scores, leakage, preservation, CCE-ready before/after classifier scores.

- [x] **Step 1: Write a config parsing test using a small JSON fixture.**
- [x] **Step 2: Implement config loading into `ConceptIntervention` and `GuidanceWeights`.**
- [x] **Step 3: Add adapter command that can call the old SD2 script path without changing old files.**
- [x] **Step 4: Add optional in-loop hook only after the command wrapper works.**

### Task 5.5: Diffusion State Smoke Runner

**Files:**
- Create: `src/cci_diff/diffusion_state.py`
- Create: `src/cci_diff/fake_backend.py`
- Create: `src/cci_diff/diffusers_backend.py`
- Create: `src/cci_diff/runner.py`
- Create: `scripts/run_diffusion_smoke.py`
- Test: `tests/test_diffusion_state.py`
- Test: `tests/test_runner.py`
- Test: `tests/test_diffusers_backend.py`

**Interfaces:**
- Consumes: JSON CCI config and backend choice.
- Produces: generated sample image plus `audit.json` containing prompt, backend name, and per-step diffusion state records.

- [x] **Step 1: Write failing tests for diffusion-state serialization, fake smoke generation, and missing optional ML dependencies.**
- [x] **Step 2: Implement dependency-free fake backend for local smoke tests.**
- [x] **Step 3: Implement optional `diffusers` backend with helpful install errors.**
- [x] **Step 4: Add CLI script for fake or real backend smoke runs.**
- [x] **Step 5: Set up local `.venv`, install the package editable, and run the fake smoke command.**

### Task 6: Paper Experiment Outputs

**Files:**
- Create: `src/cci_diff/audit.py`
- Test: `tests/test_audit_output.py`

**Interfaces:**
- Consumes: per-sample concept scores and classifier scores.
- Produces: per-sample rows and aggregate bias audit matrix.

- [x] **Step 1: Write tests for JSON-serializable audit rows.**
- [x] **Step 2: Implement audit row and aggregate helpers using `metrics.py`.**
- [ ] **Step 3: Add README example for interpreting the matrix.**
