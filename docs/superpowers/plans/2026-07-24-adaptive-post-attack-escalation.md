# Adaptive Post-Attack Escalation Implementation Plan

**Goal:** Add quantization-aware per-image epsilon escalation to the optional clean
CCI post-attack while preserving raw outputs and fixed-budget ablations.

**Architecture:** Put schedule parsing and adaptive attack selection in the
post-attack module. The clean runner resolves CLI configuration, invokes the
adaptive function, saves the selected image separately, and records all attempts
plus saved-PNG metrics in the existing audit structure.

**Tech Stack:** Python, PyTorch, torchvision, Pillow, pytest.

---

### Task 1: Specify schedule behavior with failing tests

**Files:**
- Modify: `tests/test_post_attack.py`
- Modify: `tests/test_clean_cci_cli.py`

**Steps:**
1. Add tests for valid and invalid comma-separated epsilon schedules.
2. Add a test proving retries restart from the original image.
3. Add tests for quantization-aware early stopping and exhausted schedules.
4. Add CLI tests for the adaptive default and explicit fixed-epsilon override.
5. Run the focused tests and confirm the new assertions fail for missing behavior.

### Task 2: Implement adaptive selection

**Files:**
- Modify: `src/cci_diff/post_attack.py`

**Steps:**
1. Parse and validate finite, positive, strictly increasing epsilon schedules.
2. Add adaptive attack orchestration around the existing single-budget attack.
3. Quantize each candidate to 8-bit RGB before evaluating its stopping condition.
4. Return the selected candidate and structured attempt metadata.
5. Run post-attack unit tests.

### Task 3: Integrate the clean runner and audit

**Files:**
- Modify: `scripts/run_sd2_bld_cci.py`
- Modify: `tests/test_clean_cci_cli.py`

**Steps:**
1. Add the default `0.05,0.08,0.10` schedule option.
2. Retain explicit single-epsilon behavior for ablations.
3. Replace the single attack call with adaptive orchestration.
4. Record attempts, selected epsilon, escalation state, and final PNG metrics.
5. Run CLI and clean-runner tests.

### Task 4: Verify the complete change

**Files:**
- Modify: `docs/superpowers/specs/2026-07-24-adaptive-post-attack-escalation-design.md`
  only if implementation details require clarification.

**Steps:**
1. Run all repository tests.
2. Inspect the diff for unrelated changes.
3. Report the exact behavior, verification counts, and relevant output paths.
4. Do not create a commit.
