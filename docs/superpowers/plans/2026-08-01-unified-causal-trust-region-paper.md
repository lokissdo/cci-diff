# Unified Causal Mask Discovery and Trust-Region Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the current trust-region manuscript as one coherent paper that connects classifier-causal semantic mask discovery to localized adaptive trust-region counterfactual generation and includes a unified vector framework figure.

**Architecture:** Keep `paper/cci_trust_region.tex` as the sole current manuscript and retain the generated evidence macros in `paper/generated/cci_trust_region_metrics.tex`. Reorganize the source into the MaskDiME-style top-level sequence Introduction, Related Work, Methodology, Experimentation, and Conclusions; place both connected method components under Methodology and use a full-width TikZ figure to expose their data flow. Preserve only verified numerical claims, label unfinished evaluations explicitly as pending, and leave the archived conference source unchanged.

**Tech Stack:** LaTeX, TikZ, BibTeX, Tectonic, Python `unittest`, generated TeX evidence macros

## Global Constraints

- Use the scientific names **BLD**, **Fixed-weight CCI**, and **Adaptive trust-region CCI (Ours)**; do not expose internal experiment labels in the manuscript.
- Do not label the method components “Pipeline 1,” “Pipeline 2,” “Pipeline I,” or “Pipeline II.”
- Define causality only within the frozen classifier–generator intervention system; do not imply biological, anatomical, social, or real-world causation.
- Grad-CAM++ proposes candidate semantic regions but never verifies causal influence.
- The 200-image clean results evaluate fixed-mask localized generation, not automatic mask discovery.
- Keep automatic discovery and both attacked 300-image result tables explicitly pending until complete artifacts exist.
- Distinguish clean-study Target@0.8 from attacked-study FR at the 0.5 classifier boundary.
- Do not modify `paper/cci_conference_v1.tex` or hand-edit `paper/generated/cci_trust_region_metrics.tex`.

---

### Task 1: Rewrite the Paper Framing and Structure

**Files:**
- Modify: `paper/cci_trust_region.tex:25-145`

**Interfaces:**
- Consumes: numerical macros from `paper/generated/cci_trust_region_metrics.tex` and citations already present in `paper/references.bib`
- Produces: the manuscript title, abstract, Introduction, Related Work, and `\section{Methodology}` entry point used by the method rewrite

- [ ] **Step 1: Record the current structural and naming violations**

Run:

```bash
rg -n '^\\(title|section|subsection)|A0|A10|A11|Pipeline [12I]' paper/cci_trust_region.tex
```

Expected: the old trust-region-only title, old top-level structure, and internal labels are found.

- [ ] **Step 2: Replace the title, abstract, Introduction, and Related Work**

Use `apply_patch` to make these exact structural changes:

```tex
\title{\textbf{Classifier-Causal Mask Discovery with Lexicographic\\
Trust-Region Guidance for Localized Diffusion Counterfactuals}}

\section{Introduction}
% Motivate both failures: correlational localization and unstable weighted guidance.
% State one connected framework and three contributions: intervention-verified
% discovery, source-specific minimal mask resolution, and adaptive localized CCI.

\section{Related Work}
% Cover diffusion counterfactuals, saliency/mask localization, predicted-clean
% guidance, and constrained multi-objective optimization. Distinguish MaskDiME's
% dynamic gradient mask from intervention-verified semantic regions.

\section{Methodology}
```

The abstract must report the completed 200-image fixed-mask result using scientific method names and must say that discovery and attacked evaluations remain pending. The introduction must define the two dependent questions, “where may the image change?” and “how should it change there?”, without calling them separate pipelines.

- [ ] **Step 3: Verify framing and top-level organization**

Run:

```bash
rg -n '^\\(title|section)' paper/cci_trust_region.tex
rg -n 'A0|A10|A11|Pipeline [12I]' paper/cci_trust_region.tex
```

Expected: the title and first three sections match the specification; the second command produces no output after the full manuscript rewrite is complete.

- [ ] **Step 4: Commit the framing rewrite**

```bash
git add paper/cci_trust_region.tex
git commit -m "docs: reframe CCI paper as unified causal method"
```

### Task 2: Add the Connected Methodology and Unified Framework Figure

**Files:**
- Modify: `paper/cci_trust_region.tex:146-323`

**Interfaces:**
- Consumes: the Methodology section opened by Task 1 and the causal definition in the approved design specification
- Produces: `sec:framework`, `sec:discovery`, `sec:trust-region`, and `fig:framework`, referenced by the experiments and conclusion

- [ ] **Step 1: Add preliminaries and the classifier-level intervention definition**

Insert these subsections and preserve the existing target-residual, non-target-drift, safety, predicted-clean, trust-region, epsilon-map, BLD, and restoration equations:

```tex
\subsection{Preliminaries}
\label{sec:preliminaries}
\begin{equation}
\Delta_{i,s,R}=p^\star\!\left(x_{i,s}^{\operatorname{do}(R)}\right)-p^\star(x_i).
\label{eq:intervention-effect}
\end{equation}

\subsection{Overall Framework}
\label{sec:framework}

\subsection{Classifier-Causal Mask Discovery}
\label{sec:discovery}

\subsection{Localized Trust-Region CCI}
\label{sec:trust-region}
```

Define `\operatorname{do}(R)` as permission for the generator to change exactly region set `R` while source, prompt, seed, model, scheduler, and settings remain paired. State that an influence edge is verified only if its mean effect is positive, its source-ID-clustered bootstrap lower confidence bound exceeds zero, and its minimum discovery count is met.

- [ ] **Step 2: Describe global discovery and held-out source resolution**

Write four compact paragraphs under `\ref{sec:discovery}`:

```tex
\paragraph{Candidate screening.} Grad-CAM++ ranks semantic components by
captured attribution mass and density; it proposes interventions but supplies
no causal verification.

\paragraph{Controlled interventions.} Singleton and joint region sets are
generated with the same source, prompt, seed, model, scheduler, and settings,
with post-generation attack disabled.

\paragraph{Influence-graph verification.} Paired desired-probability effects,
source-ID-clustered intervals, mask cost, outside change, non-target drift, and
synergy determine a frozen classifier-specific graph.

\paragraph{Source-specific resolution.} A held-out source uses only the frozen
graph, its Grad-CAM++ map, and semantic masks to select the smallest verified
union reaching frozen coverage, before any candidate output is observed.
```

Also state that resolution cannot rerank generations, retry failures, or expand the mask after output inspection.

- [ ] **Step 3: Replace the old method-only diagram with the unified TikZ figure**

Build a full-width figure using the existing `box`, `data`, `op`, `final`, and `group` TikZ styles. Its visible flow must be:

```text
Discovery images → Grad-CAM++ screening → paired region interventions
→ causal verification → frozen influence graph

New source + frozen graph → minimal verified mask → predicted-clean measurements
→ lexicographic trust region → DDIM + BLD + final restoration
→ counterfactual + audit
```

Draw a dashed arrow from the frozen influence graph to minimal-mask resolution. Add a compact figure note: “Causal scope: interventions within the frozen classifier–generator system.” The caption must explain that discovery determines where editing is permitted and adaptive CCI determines how the edit proceeds inside that fixed support.

- [ ] **Step 4: Preserve and rename the localized generation method**

Retain the validated equations and explanatory content, but replace implementation comparisons with scientific prose:

```tex
Fixed-weight CCI uses the same predicted-clean measurements, semantic support,
clean-coordinate trust budget, and restoration budget, but applies a permanent
nominal gradient composition. Adaptive trust-region CCI instead solves the
target-first safety-envelope and drift subproblems at the current iterate.
```

Keep the up-to-12-iteration final restoration and the deterministic line-search fractions `\{1,1/2,1/4,1/8\}`. Explicitly say restoration differentiates through the fixed VAE decoder and evaluators without new U-Net or scheduler transitions.

- [ ] **Step 5: Check causal-language and method-content invariants**

Run:

```bash
rg -n 'Grad-CAM\+\+|source-ID-clustered|do\}\(R\)|frozen influence graph|real-world caus' paper/cci_trust_region.tex
rg -n 'A0|A10|A11|Pipeline [12I]|class name|trace key|CLI' paper/cci_trust_region.tex
```

Expected: the first command locates proposal-only screening, intervention verification, graph freezing, and the causal boundary; the second produces no output.

- [ ] **Step 6: Commit the connected methodology**

```bash
git add paper/cci_trust_region.tex
git commit -m "docs: connect causal mask discovery to adaptive CCI"
```

### Task 3: Reorganize Experiments, Results, and Conclusions

**Files:**
- Modify: `paper/cci_trust_region.tex:324-539`

**Interfaces:**
- Consumes: method labels and section references from Tasks 1–2 plus all generated evidence macros
- Produces: the Experimentation section, four result tables, integrated limitations/ethics, and Conclusions

- [ ] **Step 1: Create the specified experiment hierarchy**

Replace the old protocol/results/limitations hierarchy with:

```tex
\section{Experimentation}
\subsection{Evaluation Protocols and Dataset}
\subsection{Results and Analysis}
\section{Conclusions}
```

Under the protocol subsection, describe the completed clean study, the planned disjoint discovery/held-out study, and the two planned attacked studies. Define Target@0.8, identity cosine, outside-mask MAE, non-target drift, FID, sFID, FVA, FS, MNAC, CD, COUT, and FR, and state that COUT, FR, MNAC, and CD use the same frozen multi-label classifier by design.

- [ ] **Step 2: Rename the completed comparison and preserve its evidence**

Use exactly these table row labels:

```tex
BLD
Fixed-weight CCI
Adaptive trust-region CCI (Ours)
```

Map them respectively to `\RawBLD...`, `\FixedCCI...`, and `\AdaptiveCCI...` macros. Report that adaptive CCI lowers non-target drift by `\AdaptiveDriftReductionVsFixedPct\%` relative to fixed-weight CCI at a one-percentage-point lower Target@0.8 rate, with close identity and outside-mask MAE. Do not call this statistical superiority because no paired confidence interval is available.

- [ ] **Step 3: Keep the restoration ablation and add the discovery table**

Retain the source-`\RestorationSampleID` table as a one-case mechanism illustration. Add this explicit automatic-discovery table:

```tex
\begin{tabular}{lccccc}
\toprule
Target & Verified region(s) & Effect & 95\% CI & Mask area & Held-out success \\
\midrule
Smile removal & \multicolumn{5}{c}{Pending disjoint discovery/held-out run} \\
\bottomrule
\end{tabular}
```

Its caption must state that the fixed-mask 200-image study does not estimate these quantities.

- [ ] **Step 4: Preserve the two attacked tables with scientific names**

For mouth-only and mouth-plus-upper/lower-lip support, keep the eight columns FID, sFID, FVA, FS, MNAC, CD, COUT, and FR. Use only these two rows in each table:

```tex
BLD & \multicolumn{8}{c}{Pending complete 300-image paired run} \\
Adaptive trust-region CCI (Ours) & \multicolumn{8}{c}{Pending complete 300-image paired run} \\
```

Do not insert partial-cohort values.

- [ ] **Step 5: Integrate limitations, causal scope, and ethics**

Move the essential limitations into the end of Results and Conclusions: one attribute/dataset/backbone, descriptive 200-image statistics, same-classifier evaluation, local rather than global optimization guarantees, segmentation dependence, runtime overhead, facial biometric-data caution, and generated-evidence disclosure. Delete the standalone `\section{Limitations}`.

Conclude with the connected claim: intervention-verified localization determines where editing is permitted, and adaptive trust-region guidance determines how the edit proceeds inside that region. Label discovery and attacked evidence as pending.

- [ ] **Step 6: Verify evidence boundaries and section structure**

Run:

```bash
rg -n '^\\section|^\\subsection' paper/cci_trust_region.tex
rg -n 'Pending|200 unique|fixed perioral|Target@0\.8|FR \(%\)' paper/cci_trust_region.tex
rg -n 'A0|A10|A11|section\{Limitations\}|feasibility' paper/cci_trust_region.tex
```

Expected: only the five required top-level sections occur; pending evidence and metric boundaries are explicit; the final command produces no output.

- [ ] **Step 7: Commit the experiment and conclusion rewrite**

```bash
git add paper/cci_trust_region.tex
git commit -m "docs: align CCI experiments with unified method"
```

### Task 4: Compile, Inspect, and Validate the Final Paper

**Files:**
- Modify: `paper/cci_trust_region.pdf`
- Verify: `paper/cci_trust_region.tex`
- Verify: `paper/cci_trust_region.log`
- Test: `tests/test_build_cci_paper_metrics.py`
- Test: `tests/test_cci_trust_region.py`

**Interfaces:**
- Consumes: the complete rewritten LaTeX manuscript and existing bibliography/evidence files
- Produces: a compiled, visually inspected PDF with verified citations, cross-references, figure layout, and evidence macros

- [ ] **Step 1: Regenerate and test the evidence macros**

Run:

```bash
.venv-ml/bin/python -m unittest tests.test_build_cci_paper_metrics tests.test_cci_trust_region -v
```

Expected: all tests pass.

- [ ] **Step 2: Compile the manuscript**

Run from `paper/`:

```bash
tectonic --keep-logs --keep-intermediates cci_trust_region.tex
```

Expected: exit code 0 and `paper/cci_trust_region.pdf` is regenerated.

- [ ] **Step 3: Check LaTeX diagnostics and manuscript invariants**

Run:

```bash
rg -n 'Overfull|Undefined|Citation.*undefined|Reference.*undefined' paper/cci_trust_region.log
rg -n 'A0|A10|A11|Pipeline [12I]|section\{Limitations\}|feasibility' paper/cci_trust_region.tex
git diff --check
```

Expected: all three commands produce no diagnostic or policy violations.

- [ ] **Step 4: Render and inspect the framework page**

Run:

```bash
mkdir -p /tmp/cci-paper-preview
pdftoppm -f 1 -singlefile -png -r 150 paper/cci_trust_region.pdf /tmp/cci-paper-preview/page1
```

Inspect `/tmp/cci-paper-preview/page1.png`. Expected: the full-width framework figure is legible at paper scale, arrows do not cross node labels, the dashed graph-to-mask link is visible, and no text is clipped.

- [ ] **Step 5: Inspect PDF metadata and text extraction**

Run:

```bash
pdfinfo paper/cci_trust_region.pdf
pdftotext paper/cci_trust_region.pdf - | rg 'Classifier-Causal Mask Discovery|Overall Framework|Pending complete 300-image'
```

Expected: a non-empty PDF with the new title, unified framework text, and pending-result labels.

- [ ] **Step 6: Commit the compiled manuscript**

```bash
git add -f paper/cci_trust_region.tex paper/cci_trust_region.pdf
git commit -m "docs: publish unified causal CCI paper"
```

