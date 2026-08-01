# MaskDiME Comparison and End-to-End Overview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the BLD-containing overview with the complete proposed method flow and add a protocol-aware CelebA-HQ Smile comparison sourced from MaskDiME.

**Architecture:** Use existing evaluated portrait assets to build a hybrid offline-discovery/online-generation Figure 1, then move BLD-versus-ours portraits into a separate qualitative Figure 2. Add one full-width literature table containing the supplied MaskDiME CelebA-HQ Smile rows and a macro-backed row for our attacked protocol, separated and explicitly marked as cross-protocol context.

**Tech Stack:** LaTeX, TikZ, booktabs, pytest, Tectonic, qpdf, Quick Look

## Global Constraints

- Figure 1 and its caption must not mention BLD.
- Figure 1 must connect classifier-causal mask discovery to localized trust-region generation.
- Published values must exactly match the MaskDiME CelebA-HQ Smile block supplied by the user.
- Include Smile only; do not invent or reserve a row for an unverified Age result.
- MaskDiME's sFID is ten-repeat ACE split-FID; ours is a deterministic two-direction split estimate.
- Do not claim a controlled state-of-the-art ranking across the protocol separator.
- Keep the abstract's verified controlled BLD-versus-ours claims unchanged.
- Work directly on `main` and execute inline, as already authorized.

---

### Task 1: Add Manuscript Structure Regression Checks

**Files:**
- Create: `tests/test_cci_paper_structure.py`
- Read: `paper/cci_trust_region.tex`

**Interfaces:**
- Consumes: the committed LaTeX source
- Produces: source-level invariants for Figure 1, the external table, and protocol labeling

- [ ] **Step 1: Write failing tests for the requested structure**

Create tests that extract the text between `\\label{fig:overview}` and its
surrounding `figure*` environment and assert:

```python
assert "BLD" not in overview
assert "Classifier-Causal Mask Discovery" in overview
assert "Localized Trust-Region Counterfactual Generation" in overview
assert "10429_source.jpg" in overview
assert "10429_mask_overlay.png" in overview
assert "10429_ours.jpg" in overview
```

Also require `\\label{tab:published-celebahq}` and every Smile method name:
`DiVE`, `STEEX`, `DiME`, `ACE $\\ell_1$`, `ACE $\\ell_2$`, `LDCE-txt`,
`TiME`, `RCSB`, and `MaskDiME`. Require the phrases `ten-repeat split-FID`,
`deterministic two-direction split-FID`, and `cross-protocol context`.

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
.venv-ml/bin/python -m pytest -q tests/test_cci_paper_structure.py
```

Expected: failures because `fig:overview`, `tab:published-celebahq`, and the
protocol phrases do not yet exist.

- [ ] **Step 3: Commit the red test**

```bash
git add tests/test_cci_paper_structure.py
git commit -m "test: specify CCI paper overview and literature comparison"
```

### Task 2: Replace Figure 1 with the Complete Method Flow

**Files:**
- Modify: `paper/cci_trust_region.tex:93-278`

**Interfaces:**
- Consumes: `paper/figures/cci_teaser/10429_source.jpg`, `10429_mask_overlay.png`, and `10429_ours.jpg`
- Produces: `\\label{fig:overview}` for all method-overview references

- [ ] **Step 1: Build the hybrid overview**

Replace the current introductory figure with a full-width two-band TikZ figure.
The offline band contains:

```text
Discovery images -> Grad-CAM++ proposals -> Paired region interventions
-> Frozen influence graph
```

The online band contains:

```text
Source image -> Source-specific semantic intervention mask
-> Predicted-clean lexicographic trust-region CCI -> Counterfactual
```

Use the three real sample-10429 images for source, mask overlay, and final
counterfactual. Connect the frozen graph to the mask with a dashed verified
region-policy edge. Label the two bands `Classifier-Causal Mask Discovery` and
`Localized Trust-Region Counterfactual Generation`. State the frozen
classifier--generator causal scope in the caption.

- [ ] **Step 2: Remove the redundant framework figure and update references**

Delete the old `fig:framework` environment. Make the Overall Framework section
refer to `Figure~\\ref{fig:overview}`. Keep the method equations and prose
unchanged.

- [ ] **Step 3: Run the structure test and inspect its remaining failures**

Run:

```bash
.venv-ml/bin/python -m pytest -q tests/test_cci_paper_structure.py
```

Expected: Figure 1 assertions pass; literature-table assertions still fail.

### Task 3: Add Qualitative Baseline Figure and CelebA-HQ Literature Table

**Files:**
- Modify: `paper/cci_trust_region.tex:480-559`

**Interfaces:**
- Consumes: generated `EndToEnd*` metric macros and existing evaluated BLD/Ours image assets
- Produces: `\\label{fig:qualitative}` and `\\label{tab:published-celebahq}`

- [ ] **Step 1: Add a separate qualitative comparison figure**

Under Qualitative Results, add Figure 2 with three paired columns for samples
10429, 18004, and 28408. Each pair shows full-face BLD and Ours images with
LaTeX labels. State that images within each pair share source, mask, prompt,
seed, scheduler, attack, and generation settings. Do not describe this figure
as the method overview.

- [ ] **Step 2: Add the published CelebA-HQ Smile table**

After the controlled BLD-versus-ours narrative, add a full-width table with
columns `Method, FID, sFID, FVA, FS, MNAC, CD, COUT, FR`. Transcribe exactly:

```text
DiVE       107.0  -    35.7  -    7.41  -    -     -
STEEX       21.9  -    97.6  -    5.27  -    -     -
DiME        18.10 27.7 96.7  0.67 2.63  1.82 0.65  97.0
ACE l1       3.21 20.2 100.0 0.89 1.56  2.61 0.55  95.0
ACE l2       6.93 22.0 100.0 0.84 1.87  2.21 0.60  95.0
LDCE-txt    13.6  25.8 99.1  0.76 2.44  1.68 0.34  -
TiME        10.98 23.8 96.6  0.79 2.97  2.32 0.63  97.1
RCSB         3.04 20.0 100.0 0.93 1.22  3.22 0.83  98.9
MaskDiME     2.51 18.1 100.0 0.94 1.41  2.67 0.69  99.4
```

Below a `\\midrule`, append our row using only `\\EndToEndAdaptive*` macros.
The caption cites `\\cite{guo2026maskdime}`, identifies the published block as
256 by 256 CelebA-HQ, and calls our row 512 by 512 attacked CelebAMask-HQ
cross-protocol context. Do not bold a winner across the separator.

- [ ] **Step 3: Explain the estimator and protocol boundary**

Add one paragraph stating that MaskDiME reports ten-repeat ACE split-FID while
ours reports deterministic two-direction split-FID, and that resolution,
classifier, cohort, attack, and generation settings differ. Therefore the table
supports literature positioning but not a controlled ranking; Table 1 remains
the causal comparison for the adaptive controller.

- [ ] **Step 4: Run the structure and regression tests**

Run:

```bash
.venv-ml/bin/python -m pytest -q \
  tests/test_cci_paper_structure.py \
  tests/test_build_cci_conference_metrics.py \
  tests/test_build_cci_teaser_assets.py \
  tests/test_trust_region_solver.py \
  tests/test_trust_region_controller.py \
  tests/test_sd2_clean_cci.py
```

Expected: all tests pass.

### Task 4: Compile, Inspect, and Publish

**Files:**
- Modify: `paper/cci_trust_region.pdf`
- Verify: `paper/cci_trust_region.log`

**Interfaces:**
- Consumes: the revised LaTeX source and existing generated assets
- Produces: a visually inspected final PDF and clean committed manuscript

- [ ] **Step 1: Compile the paper**

From `paper/`, run:

```bash
tectonic --keep-logs --keep-intermediates cci_trust_region.tex
```

Expected: exit code zero.

- [ ] **Step 2: Validate source and PDF structure**

Run:

```bash
if rg -n 'Overfull|Undefined|Citation.*undefined|Reference.*undefined' \
  paper/cci_trust_region.log; then exit 1; fi
qpdf --check paper/cci_trust_region.pdf
git diff --check
```

Expected: no diagnostics and a valid PDF.

- [ ] **Step 3: Render and inspect the overview, table, and qualitative pages**

Split the PDF with qpdf and render the relevant pages at 1800 pixels with
Quick Look. Require readable method stages, undistorted real portraits, a
legible literature table, no clipped captions, and no BLD text in Figure 1.

- [ ] **Step 4: Run final tests and commit**

Run the Task 3 test command again, then:

```bash
git add -f paper/cci_trust_region.tex paper/cci_trust_region.pdf
git commit -m "docs: add MaskDiME context and end-to-end overview"
```
