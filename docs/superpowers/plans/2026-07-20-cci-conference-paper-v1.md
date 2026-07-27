# CCI Conference Paper V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a self-contained, two-column LaTeX conference manuscript that accurately describes the implemented clean CCI controller and all completed BLD-versus-CCI benchmarks.

**Architecture:** Keep the manuscript, verified BibTeX database, vendored
rule-selected figures, and build notes in a focused `paper/` directory. The
manuscript reads executed settings from the approved design contract, presents
complete tables from the final metrics report, and copies deterministic
qualitative artifacts without modifying benchmark outputs.

**Tech Stack:** LaTeX `article` class, BibTeX, TikZ, `booktabs`, `graphicx`, local Markdown/JSON/CSV benchmark artifacts, shell/Python source validation.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-20-cci-conference-paper-v1-design.md`.
- Do not stage or commit any file.
- Do not modify benchmark outputs or thesis evaluation code.
- Use a portable two-column conference layout without an external venue class.
- Use executed graph settings, current implementation, and final reports over historical plans.
- Keep target accuracy, directional FR, same-classifier FR, and strong-target rate distinct.
- Report COUT as unavailable, never as zero.
- Mark 100-image FID and sFID as exploratory and not directly comparable to prior-paper values.
- Use `Anonymous Author(s)` for version 1.

---

### Task 1: Verified Bibliography

**Files:**
- Create: `paper/references.bib`

**Interfaces:**
- Consumes: primary publication records collected in the evidence audit.
- Produces: BibTeX keys used by `paper/cci_conference_v1.tex`.

- [x] **Step 1: Add primary method references**

Create entries with verified author lists, titles, venues, years, pages, and DOI
or canonical URL where available:

```text
rombach2022latent
song2021ddim
avrahami2022blended
bansal2023universal
chung2023dps
wallace2023doodl
augustin2024digin
guo2026maskdime
jeanneret2022dime
jeanneret2023ace
vohoang2026human
rodriguez2021dive
jacob2022steex
jeanneret2024time
luu2025eced
yu2020pcgrad
```

Use the CVF, OpenReview, arXiv, NeurIPS, or publisher record as the source.
Record the prior work as the published ESWA article with DOI
`10.1016/j.eswa.2026.131612`.

- [x] **Step 2: Add dataset and evaluation references**

Add:

```text
liu2015celeba
lee2020maskgan
heusel2017fid
chen2021simsiam
cao2018vggface2
schroff2015facenet
```

- [x] **Step 3: Check BibTeX key uniqueness**

Run:

```bash
rg '^@' paper/references.bib
rg '^@' paper/references.bib | sed -E 's/^@[^{]+\{([^,]+),.*/\1/' | sort | uniq -d
```

Expected: the first command lists 22 entries; the second prints nothing.

### Task 2: Conference Manuscript

**Files:**
- Create: `paper/cci_conference_v1.tex`
- Create: `paper/figures/*.jpg`

**Interfaces:**
- Consumes: `paper/references.bib`, the design contract, executed graphs,
  implementation code, final metrics, and existing comparison images.
- Produces: one standalone conference-style manuscript source.

- [x] **Step 1: Create the portable conference preamble**

Use:

```latex
\documentclass[10pt,twocolumn]{article}
\usepackage[letterpaper,margin=0.72in,columnsep=0.22in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,microtype,multirow}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit}
\usepackage[hidelinks]{hyperref}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{subcaption}
```

Define compact table helpers, use `Anonymous Author(s)`, and set the working
title from the design.

- [x] **Step 2: Write the abstract and introduction**

The abstract must report the 35-step directional gains (+7 smile, +2 hair), the
same-classifier gains (+36 smile, +30 hair), the +31-point strong-target gain
for both tasks, mixed preservation, and the 2.68--2.71x 35-step runtime cost.
State that independent transfer is modest and the study is a first 100-sample
evaluation, not a state-of-the-art result.

The introduction must motivate localized counterfactual validity, explain why
fixed manual loss weights are fragile, and state the five supported
contributions from the design contract.

- [x] **Step 3: Write related work and problem formulation**

Cover:

```text
visual counterfactuals: DiME, ACE, DiG-IN, MaskDiME, prior ESWA pipeline
localized diffusion: latent diffusion, DDIM, BLD
training-free guidance: Universal Guidance, DPS, DOODL
```

Define an input image `x`, target concept `c`, desired binary value `y*`, source
classifier, semantic mask, generation mask, and the objective of changing the
target while constraining identity and off-mask change. Clarify that graph
edges encode intervention policy and are not a learned structural causal model.

- [x] **Step 4: Write the method**

Include the exact predicted-clean equation, signed target margin, target
activation, normalized violation, dual update, coefficient, EMA gradient
normalization, conflict projection, target-priority budget, masked trust-region
update, pre-scheduler noise modification, BLD blend, and final correction.

Report the executed settings:

```text
p*=0.8
dual rate=0.2
penalty=0.5
lambda max=4.0
step scale=0.2
trust radius=0.15
EMA beta=0.9
gradient floor=1e-5
active progress=[0.15,0.90]
every two steps
12 final corrections
mask geometry=x4/y4/f3
```

Draw a TikZ pipeline from concept graph through predicted-clean decode,
evaluators, controller, noise update, DDIM step, and BLD blend.

- [x] **Step 5: Write experimental setup**

State:

```text
CelebAMask-HQ
100 samples per task
Smiling 1->0 and Blond_Hair 0->1
35 and 50 scheduler-step settings (27 and 38 executed denoising updates)
BLD A0 and CCI A3
seed 42
512x512
MPS float32
batch size 1
CFG 5.0
blend start 0.25
local Stable Diffusion 2.1 base checkpoint
800 total outputs
```

Document all evaluator checkpoints and hashes from the design/evidence audit.
Define every metric, including the distinction between target accuracy and
directional FR. Explain the deterministic 50/50 cross-split sFID protocol.

- [x] **Step 6: Write complete results**

Insert all eight rows from each section of
`outputs/fid_sfid_bld_cci_steps35_50/full_metrics.md`:

1. Counterfactual success.
2. Preservation and collateral change.
3. Distribution, locality, and runtime.

Include directional sFID components in an appendix. Discuss the
classifier-transfer gap, uninformative 100% FVA threshold, nearly unchanged FS,
mixed MNAC/CD, nearly unchanged locality, inconclusive 100-sample FID/sFID,
failure of 50 steps to improve validity, and runtime overhead.

- [x] **Step 7: Add rule-selected qualitative examples**

Use 35-step pairs selected by evaluator outcome:

```text
transfer gains: smile 00024 and hair 00000
classifier-alignment failures: smile 00009 and hair 00008
```

Copy the selected BLD and CCI `input_output.jpg` files into
`paper/figures/` and reference those portable copies. Captions must state the
selection rule and must not claim human realism scores.

- [x] **Step 8: Write limitations, ethics, conclusion, and appendix**

Limitations must include two target attributes, 100 samples per task, one seed,
one guidance classifier, modest oracle transfer, no human study, graph masks
from annotations, no automatic text-to-graph compiler, mixed preservation, and
MPS runtime. Ethics must cover biometric privacy, demographic bias, classifier
gaming, and the distinction between model counterfactuals and real-world causal
claims.

Append the exact graph roles, controller settings, evaluator hashes, full sFID
directions, package versions, and reproducibility paths.

### Task 3: Build and Reproduction Notes

**Files:**
- Create: `paper/README.md`

**Interfaces:**
- Consumes: manuscript and bibliography.
- Produces: exact build, validation, and evidence-location instructions.

- [x] **Step 1: Document standard build commands**

Document:

```bash
cd paper
pdflatex cci_conference_v1.tex
bibtex cci_conference_v1
pdflatex cci_conference_v1.tex
pdflatex cci_conference_v1.tex
```

State that TeX binaries are absent on the current machine and no PDF was
compiled locally.

- [x] **Step 2: Document evidence roots**

List the four run roots, final metrics report, JSON metric source, two concept
graphs, and current method modules.

### Task 4: Source Validation

**Files:**
- Validate: `paper/cci_conference_v1.tex`
- Validate: `paper/references.bib`
- Validate: `paper/README.md`

**Interfaces:**
- Consumes: all paper deliverables.
- Produces: evidence that keys, figures, metrics, and structural syntax agree.

- [x] **Step 1: Validate citation keys**

Extract every `\cite{...}` key from the manuscript, split comma-separated keys,
and compare them with the BibTeX keys. Expected: no missing keys.

- [x] **Step 2: Validate included graphics**

Extract every `\includegraphics{...}` path relative to `paper/`. Expected: all
referenced vendored images exist. Compare each vendored file byte-for-byte
with its selected benchmark artifact.

- [x] **Step 3: Validate manuscript structure**

Check for one `\begin{document}`, one `\end{document}`, balanced braces, and
required sections. Confirm there are no `TODO`, `TBD`, fabricated COUT values,
or claims of state-of-the-art performance.

- [x] **Step 4: Validate table values**

Compare the 24 main-table rows/cells against
`outputs/fid_sfid_bld_cci_steps35_50/fid_sfid_metrics.json` and
`full_metrics.md`. Confirm all eight rows appear in each main table and all
directional sFID values appear in the appendix.

- [x] **Step 5: Inspect final repository state**

Run:

```bash
git status --short
git diff --check
```

Expected: the manuscript, bibliography, README, eight vendored figures, design,
and plan are untracked or modified as appropriate; no file is staged, and
`git diff --check` reports no whitespace errors.
