# Unified Causal Mask Discovery and Trust-Region Paper Design

## Purpose

Revise `paper/cci_trust_region.tex` into one coherent method paper that joins
classifier-causal mask discovery with localized lexicographic trust-region
counterfactual generation. The manuscript should follow the compact structural
style of MaskDiME: Introduction, Related Work, Methodology, Experimentation,
and Conclusions.

The paper must read as one method. It must not call the components “Pipeline
1” and “Pipeline 2,” and it must not expose implementation labels such as A0,
A10, or A11 in the main text, figures, or tables.

## Working Title

**Classifier-Causal Mask Discovery with Lexicographic Trust-Region Guidance
for Localized Diffusion Counterfactuals**

## Central Method Story

The method answers two dependent questions:

1. **Where may the counterfactual change?** A discovery cohort is used to
   screen semantic regions, perform controlled masked interventions, and
   freeze a classifier-specific counterfactual influence graph. For a new
   source, that graph and source Grad-CAM++ evidence resolve the smallest
   likely sufficient semantic region set before generation.
2. **How should the counterfactual change inside that region?** The resolved
   hard semantic mask and soft generation mask localize predicted-clean
   measurements, lexicographic trust-region guidance, BLD source blending, and
   preservation-aware final latent restoration.

The output of mask discovery is therefore the input policy for localized
generation. They are not independent systems.

## Causal Definition and Claim Boundary

The paper uses classifier-level interventional causality. For region set
\(R\), source \(x_i\), fixed seed \(s\), and desired-class probability
\(p^\star\), define the paired intervention effect

\[
\Delta_{i,s,R}
=
p^\star\!\left(x_{i,s}^{\operatorname{do}(R)}\right)
-p^\star(x_i).
\]

The notation \(\operatorname{do}(R)\) means that the generator is permitted to
modify exactly the declared semantic region set while source, prompt, seed,
model, and generation settings remain fixed. Grad-CAM++ proposes candidate
regions but does not verify causal influence.

A target-to-region edge is called verified only when the singleton region has
a positive mean intervention effect, the source-ID-clustered bootstrap confidence
interval excludes zero, and the minimum discovery sample count is met.

This is causality inside the frozen classifier–generator system. The paper must
state explicitly that it does not establish biological, anatomical, social, or
real-world causation.

## Manuscript Structure

### 1. Introduction

Motivate visual counterfactuals through two failure modes:

- saliency alone is correlational and does not establish that changing a
  highlighted region affects the decision;
- once a region is selected, a permanent weighted guidance loss cannot
  reliably prioritize target success, identity/locality safety, and
  non-target preservation.

Present the proposed solution as one framework that verifies intervention
regions first and then solves localized counterfactual generation
lexicographically.

State three contributions:

1. classifier-causal semantic mask discovery through paired same-seed
   interventions and confidence-based graph verification;
2. source-specific minimal region resolution from a frozen influence graph;
3. predicted-clean lexicographic trust-region guidance and preservation-aware
   final restoration inside the resolved support.

Do not include internal variant identifiers.

### 2. Related Work

Use a compact narrative covering:

- visual counterfactual explanations and diffusion-based methods;
- saliency and mask-based localization;
- predicted-clean external guidance;
- constrained and multi-objective gradient optimization.

Differentiate the proposed method from MaskDiME carefully. MaskDiME derives
dynamic gradient masks during diffusion. This method uses Grad-CAM++ only for
screening and source-specific selection among intervention-verified semantic
regions; its causal evidence comes from controlled diffusion interventions.

### 3. Methodology

#### 3.1 Preliminaries

Define:

- source image, explained classifier, target attribute, desired value, and
  desired probability;
- latent diffusion, predicted-clean estimation, and BLD source blending;
- semantic component masks;
- classifier-level intervention effect and the limits of the causal claim.

#### 3.2 Overall Framework

Explain the complete connection before either component is detailed:

1. use a disjoint discovery cohort to build a frozen influence graph;
2. combine the graph with a new source’s Grad-CAM++ map and component masks;
3. select one minimal verified semantic region set before generation;
4. construct a hard semantic mask and soft generation mask;
5. run localized predicted-clean trust-region guidance;
6. apply BLD source blending and final latent restoration without expanding
   the discovered region;
7. save one counterfactual and its audit record.

Make clear which artifacts are global and reusable (influence graph) and which
are source-specific (selected semantic mask and counterfactual).

#### 3.3 Classifier-Causal Mask Discovery

Use four compact parts:

1. **Candidate screening.** Rank semantic components using Grad-CAM++ captured
   mass and density. Screening creates proposals only.
2. **Controlled region interventions.** Generate singleton and joint semantic
   region interventions with identical source, prompt, seed, scheduler, and
   model settings. Disable post-generation attack during discovery.
3. **Influence-graph verification.** Estimate mean desired-probability effect,
   source-ID-clustered bootstrap intervals, mask cost, outside change,
   non-target drift, and region synergy. Retain verified edges and choose the
   smallest region set reaching the frozen target-success requirement.
4. **Source-specific resolution.** On a held-out source, select the smallest
   verified region union reaching frozen saliency coverage. Selection may use
   only the source image, frozen graph, source Grad-CAM++ map, and semantic
   masks; it cannot inspect generated candidates, retry a failure, or expand a
   region after observing the output.

#### 3.4 Localized Trust-Region CCI

Retain the current scientifically necessary content but remove code-oriented
details. Use four compact parts:

1. hard semantic mask versus soft generation mask;
2. predicted-clean VAE measurements with the U-Net detached;
3. target-first safety-envelope and non-target-drift lexicographic
   trust-region subproblems in clean-latent coordinates;
4. epsilon-coordinate mapping, BLD blending, and preservation-aware final
   latent restoration with deterministic backtracking.

Keep equations for the target residual, non-target drift, two lexicographic
subproblems, clean-to-epsilon mapping, and BLD blend. Omit class names, CLI
modes, JSON trace keys, and internal variant nomenclature.

### 4. Experimentation

#### 4.1 Evaluation Protocols and Dataset

Describe CelebAMask-HQ smile removal, source eligibility, classifier and
identity models, paired seeds, SD2.1/DDIM settings, and metric definitions.

Use scientific method names only:

- **BLD**;
- **Fixed-weight CCI**;
- **Adaptive trust-region CCI (Ours)**.

Completed clean evidence consists of 200 unique paired sources using the fixed
perioral semantic union and no post-generation attack. It evaluates the
localized generation component, not automatic mask discovery.

The planned discovery evaluation uses disjoint discovery and held-out cohorts.
It will report verified regions, intervention effects and confidence,
source-specific mask area, target success, identity, locality, and non-target
drift.

The planned attacked evaluation uses 300 paired images with mouth-only support
and then the same 300 IDs with mouth, upper-lip, and lower-lip support. It will
report FID, sFID, FVA, FS, MNAC, CD, COUT, and FR.

#### 4.2 Results and Analysis

Present:

1. the completed 200-image comparison under scientific method names;
2. the illustrative final-restoration ablation;
3. one explicit automatic mask-discovery placeholder table;
4. two explicit 300-image attacked-result placeholder tables.

Do not infer mask-discovery performance from the fixed-mask clean experiment.
Do not fill incomplete tables from partial cohorts. Do not report feasibility
as a headline metric. Distinguish Target@0.8 in the clean experiment from FR at
the 0.5 boundary in the attacked experiment.

### 5. Conclusions

Summarize the combined contribution: intervention-verified localization
defines where editing is permitted, and lexicographic trust-region guidance
determines how to edit within that support. Integrate limitations and facial-
data ethics into the conclusion rather than retaining a separate long section.

The conclusion may cite the completed 200-image trust-region comparison but
must label mask-discovery and attacked results as pending.

## Unified Method Figure

Replace the current generation-only figure with one full-width vector figure.
It must have two visually connected bands without numbering them as pipelines.

### Upper band: Classifier-Causal Mask Discovery

```text
Discovery images
  → Grad-CAM++ candidate screening
  → same-seed controlled region interventions
  → causal verification
  → frozen influence graph
```

### Lower band: Localized Trust-Region Counterfactual Generation

```text
New source + frozen graph
  → minimal verified mask
  → predicted-clean measurements
  → lexicographic trust region
  → DDIM + BLD + final restoration
  → counterfactual + audit
```

A dashed connection must show that the frozen influence graph guides
source-specific mask selection. A compact note must state the causal claim
boundary. The final paper figure should be implemented as TikZ/vector graphics
for exact typography and scaling.

The approved visual draft is `/tmp/cci_unified_pipeline_draft.svg`. It is a
layout reference, not a tracked paper asset.

## Material to Remove or Compress

- Remove A0/A10/A11 identifiers from prose and tables.
- Remove a separate `Limitations` main section; integrate its essential content
  into Results and Conclusions.
- Compress controller implementation settings to one reproducibility sentence
  or a small table.
- Avoid internal mode names, hook names, trace fields, and command-line flags.
- Do not label the components “Pipeline I” or “Pipeline II.”
- Keep only equations needed to explain the scientific method.

## Evidence Boundaries

- The 200-image clean table is completed evidence for fixed-mask localized
  generation.
- The sample-26811 restoration comparison is illustrative, not population
  evidence.
- Automatic causal mask discovery has no completed result artifacts in the
  current workspace and must remain pending.
- The two attacked 300-image experiments remain pending until their complete
  paired metrics exist.
- Classifier-dependent COUT, FR, MNAC, and CD use the same frozen multi-label
  classifier that guides CCI, by design.

## Validation

Before completion:

1. confirm the paper uses only scientific method names;
2. confirm the overall-framework subsection connects global discovery,
   source-specific mask resolution, and localized generation;
3. confirm Grad-CAM++ is never described as causal verification;
4. confirm controlled interventions and confidence intervals define verified
   causal edges;
5. confirm the 200-image results are not attributed to automatic discovery;
6. confirm all incomplete discovery and attacked cells remain explicitly
   pending;
7. compile the PDF with no overfull boxes or undefined references;
8. visually inspect the unified figure at final paper scale;
9. run the existing evidence-builder and trust-region regression tests.

## Non-Goals

- Running the mask-discovery experiment during the paper rewrite.
- Filling partial attacked metrics.
- Copying MaskDiME wording, equations, or figures.
- Claiming real-world causality.
- Rewriting the archived `paper/cci_conference_v1.tex`.
