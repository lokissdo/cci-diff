# SD2 BLD GPU Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GPU-heavy Stable Diffusion 2 blended-latent backend that follows the ESWA `bld_reranking` inference flow and exposes a CCI hook inside the denoising loop.

**Architecture:** Keep the existing lightweight smoke backend. Add a separate `sd2_bld_backend` module for image editing with source-latent encoding, mask blending, DDIM scheduling, CFG, CCI guidance hook, state recording, decode, and image saving. The module imports torch/diffusers/PIL/numpy only inside dependency-loading functions so the core test suite still runs without GPU packages.

**Tech Stack:** Python 3.10+, torch, diffusers, PIL, numpy, optional LoRA weights, standard-library unit tests.

## Global Constraints

- Do not modify the old `../thesis_2025/bld_reranking` source files.
- Keep `cci_diff` importable without torch or diffusers.
- Preserve the ESWA SD2 hook position: after classifier-free guidance and before `scheduler.step`.
- Keep generated outputs under ignored `outputs/`.
- Verify with unit tests locally; real GPU execution is expected on Kaggle/Colab or a Python 3.10+ ML venv.

---

### Task 1: SD2 Hook Contract

**Files:**
- Modify: `src/cci_diff/diffusion_state.py`
- Create: `src/cci_diff/sd2_bld_backend.py`
- Test: `tests/test_sd2_bld_backend.py`

**Interfaces:**
- Consumes: existing `DiffusionState`.
- Produces: `SD2DenoisingStep`, `apply_cci_guidance`, `diffusion_state_from_step`, `blending_start_index`, `require_sd2_dependencies`.

- [x] **Step 1: Write failing tests for optional dependency errors, blend-start calculation, and hook replacement of CFG noise.**
- [x] **Step 2: Run ` .venv/bin/python -m unittest tests.test_sd2_bld_backend -v` and verify the module import fails.**
- [x] **Step 3: Implement the hook dataclass and pure helper functions without importing torch at module import time.**
- [x] **Step 4: Run focused tests and verify pass.**

### Task 2: GPU SD2 Blended-Latent Backend

**Files:**
- Modify: `src/cci_diff/sd2_bld_backend.py`
- Test: `tests/test_sd2_bld_backend.py`

**Interfaces:**
- Consumes: `SD2DenoisingStep`, `apply_cci_guidance`, `diffusion_state_from_step`.
- Produces: `BlendedLatentDiffusionSD2Backend.edit_image(...)`.

- [x] **Step 1: Write tests that inspect constructor arguments and ensure dependencies stay optional.**
- [x] **Step 2: Implement model loading from `DiffusionPipeline.from_pretrained`, optional LoRA loading, DDIM scheduler setup, image-to-latent conversion, mask loading, denoising loop, source-latent blending, decoding, and image grid saving.**
- [x] **Step 3: Ensure every denoising iteration records `cci_guidance`, `scheduler_step`, and `blend` states.**
- [x] **Step 4: Run all tests locally.**

### Task 3: SD2 CLI Runner

**Files:**
- Create: `scripts/run_sd2_bld_cci.py`
- Modify: `README.md`
- Test: `tests/test_sd2_bld_cli.py`

**Interfaces:**
- Consumes: CCI config JSON, init image path, mask path, output directory, model path, LoRA path, seed, batch size, device.
- Produces: output grid image and `audit.json`.

- [x] **Step 1: Write CLI parser tests for required image/mask/config arguments and prompt generation from CCI config.**
- [x] **Step 2: Implement the CLI to call `BlendedLatentDiffusionSD2Backend.edit_image(...)`.**
- [x] **Step 3: Document exact Kaggle/Colab command and local dependency limitation.**
- [x] **Step 4: Run all tests and a local import smoke test.**
