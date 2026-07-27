# CCI Component Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a deterministic 10-sample-per-task one-component ablation of CCI.

**Architecture:** Add explicit controller switches with unchanged defaults, seed source-blend noise through the backend generator, expose named variants through the existing pilot runner, and summarize paired component deltas from existing audit and ACE/FID evaluators.

**Tech Stack:** Python 3.10, PyTorch, Diffusers DDIM, pytest, CSV/JSON.

## Global Constraints

- Do not commit changes.
- Do not modify files outside `cci-diff`.
- Preserve existing CLI behavior unless a new ablation flag is supplied.
- Use the same random stream for every matched variant.

---

### Task 1: Deterministic Source Blending

**Files:** `tests/test_sd2_bld_backend.py`, `src/cci_diff/sd2_bld_backend.py`

- [ ] Add a failing test that requires source-blend noise to use the seeded generator.
- [ ] Pass the run generator to every source-noise draw.
- [ ] Run the backend tests.

### Task 2: Controller Ablation Switches

**Files:** `tests/test_constraint_controller.py`, `src/cci_diff/constraint_controller.py`, `tests/test_sd2_clean_cci.py`, `src/cci_diff/adapters/sd2_clean_cci.py`

- [ ] Add failing tests for disabled target guidance, raw gradients, and disabled target budget.
- [ ] Add boolean controller options with current behavior as defaults.
- [ ] Record the switches and effective behavior in each trace.
- [ ] Run controller and adapter tests.

### Task 3: CLI and Variant Matrix

**Files:** `tests/test_clean_cci_cli.py`, `tests/test_clean_cci_pilot.py`, `scripts/run_sd2_bld_cci.py`, `scripts/run_clean_cci_pilot.py`

- [ ] Add failing parser and command-construction tests for each removal variant.
- [ ] Expose flags for target, normalization, budget, schedule, and final correction.
- [ ] Add named variants while retaining A0-A4 compatibility.
- [ ] Run CLI and pilot tests.

### Task 4: Paired Summary

**Files:** `tests/test_clean_cci_ablation.py`, `scripts/summarize_clean_cci_ablation.py`

- [ ] Add failing tests for paired component deltas and trace aggregation.
- [ ] Implement CSV/JSON/Markdown summaries by task and variant.
- [ ] Include independent and guidance validity, preservation, locality, smoothness, distribution quality, runtime, and final-correction rescue fields.
- [ ] Run summary tests and the full focused suite.

### Task 5: Execute and Evaluate

- [ ] Run a one-sample smoke matrix.
- [ ] Run all variants on 10 smile and 10 hair samples at 35 steps.
- [ ] Run ACE and deterministic FID/sFID evaluation.
- [ ] Generate paired component summaries and inspect failed runs.
- [ ] Update the conference paper with results and a ten-sample pilot limitation.

