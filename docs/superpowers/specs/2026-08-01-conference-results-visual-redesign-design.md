# Conference Results and Visual Redesign

## Purpose

Revise paper/cci_trust_region.tex from a method-first draft with incomplete
tables into a result-led conference paper. The revision will headline the
completed end-to-end evaluation, replace pending and low-priority evidence with
one verified results table, and add a real image-to-image teaser that makes the
localized counterfactual effect legible at print size.

The classifier-causal mask discovery and localized trust-region CCI components
remain connected under Methodology. The paper must not invent automatic
mask-discovery results or describe the fixed mouth mask used in the completed
evaluation as causally discovered.

## Publication Framing

### Abstract

The abstract will:

- use the phrase **“on the full paired end-to-end evaluation cohort”**;
- omit the cohort size;
- omit the earlier clean fixed-weight comparison and its Target@0.8 result;
- report the strongest verified end-to-end findings;
- describe both target effectiveness and preservation/image-quality behavior.

The headline verified values are:

- FR: \(81.0\%\);
- COUT: \(0.115\);
- CD: \(2.889\);
- FVA: \(100.0\%\);
- FS: \(0.9958\);
- FID: \(17.43\);
- symmetric FID: \(72.39\).

The strongest verified baseline-relative findings are:

- FR improves by \(16.67\) percentage points over BLD;
- COUT improves by \(0.21337\);
- CD decreases by \(1.85\%\);
- FID, symmetric FID, FVA, and FS remain closely matched.

The abstract does not need to enumerate MNAC because it is not a headline
finding; the complete results table will report it.

### Experimental Protocol

The sample count \(N=300\) appears exactly once in the Experimental Protocol,
as required for reproducibility. It will be stated neutrally and will not be
framed as a limitation.

The protocol will say that:

- both methods use the same eligible source IDs, seed, diffusion settings,
  mouth-region semantic support, and localized smooth-boundary post-attack;
- BLD uses no constraint controller;
- Adaptive trust-region CCI uses the proposed controller;
- all reported aggregates use complete paired outputs;
- classifier-dependent COUT, FR, MNAC, and CD use the same frozen multi-label
  classifier that guides CCI, by design.

No other section, table caption, figure caption, abstract sentence, or
conclusion sentence will repeat the cohort size.

### Results

Results prose will report verified metrics and improvements without discussing
the sample count. The main quantitative table will contain:

| Method | FID ↓ | symmetric FID ↓ | FVA ↑ | FS ↑ | MNAC ↓ | CD ↓ | COUT ↑ | FR (%) ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLD | 17.3720 | 72.5453 | 100.0 | 0.99615 | 2.4967 | 2.9437 | -0.09818 | 64.33 |
| Adaptive trust-region CCI (Ours) | 17.4345 | 72.3917 | 100.0 | 0.99577 | 2.6233 | 2.8894 | 0.11519 | 81.00 |

The narrative will not claim dominance on every metric. It will state that the
method improves FR, COUT, CD, and symmetric FID while retaining closely matched
FID, FVA, and FS. MNAC is fully visible in the table and is not selectively
described as an improvement.

The previous 200-image clean comparison, one-case restoration table, automatic
discovery placeholder, and unfinished attacked-region table will be removed
from the main paper. The incomplete mouth-plus-upper/lower-lip run will not be
reported or mentioned.

## Figure Design

### Figure 1: Qualitative Teaser

Create one full-width image-to-image figure near the Introduction. It will use
real outputs from the completed evaluation and show:

    Source image → semantic intervention mask → BLD → Adaptive trust-region CCI

The primary row uses sample 10429. Supporting rows use samples 18004 and 28408.
Each method column uses the paired source, seed, mask, diffusion settings, and
post-attack protocol.

The figure will include enlarged mouth crops below or inset into the full-face
outputs. These crops are necessary because the intervention is spatially small
and the BLD/adaptive distinction becomes hard to see when a 512 by 512 face is
reduced to two-column print scale.

The mask must be labeled **“semantic intervention mask”**, not “causally
discovered mask” or “discovered mask.” The caption will state that these
examples use the fixed mouth-region protocol from the completed evaluation.
The examples are qualitative illustrations; aggregate conclusions come from
the full paired evaluation table.

The source assets are:

- data/CelebAMask-HQ/CelebA-HQ-img/10429.jpg
- outputs/attacked_a0_a11_smile300_seed42/mouth/smile/10429/A11/semantic_mask.png
- outputs/attacked_a0_a11_smile300_seed42/mouth/smile/10429/A0/candidates/d8/sd2_bld_grid_corrected.png
- outputs/attacked_a0_a11_smile300_seed42/mouth/smile/10429/A11/candidates/d8/sd2_bld_grid_corrected.png
- corresponding source, mask, BLD, and adaptive paths for 18004 and 28408.

Paper-owned copies and derived crop/overlay assets will be stored under
paper/figures/ so the manuscript does not depend on ignored experiment outputs
or absolute local paths.

### Figure 2: Compact Overall Framework

Retain a separate vector method figure under Overall Framework, but reduce its
visual weight. It will show:

    discovery images
      → Grad-CAM++ proposals
      → paired region interventions
      → verified influence graph
      → source-specific semantic support
      → predicted-clean trust-region CCI
      → localized counterfactual

This figure explains the connection between global graph construction and
source-specific generation. It retains the causal-scope note and does not use
numbered component labels.

## Manuscript Restructuring

The paper keeps the conference sequence:

1. Introduction
2. Related Work
3. Methodology
   - Preliminaries
   - Overall Framework
   - Classifier-Causal Mask Discovery
   - Localized Trust-Region CCI
4. Experimentation
   - Evaluation Protocol
   - Quantitative Results
   - Qualitative Results
   - Discussion
5. Conclusions

Changes relative to the current draft:

- make Figure 1 and the abstract result-led;
- move the compact framework to Figure 2;
- remove all pending tables and “planned” experiment prose;
- remove the earlier fixed-weight clean result from the main paper;
- remove the one-case restoration result while preserving restoration as a
  method component;
- remove references to the incomplete second region run;
- correct every expansion of sFID from “spatial FID” to “symmetric FID”;
- retain the causal boundary and same-classifier metric provenance;
- keep limitations compact and technical rather than framing the completed
  evaluation as weak.

## Evidence and Validation

Primary quantitative source:

outputs/attacked_a0_a11_smile300_seed42/mouth/metrics/full_metrics.csv

Supporting sources:

- outputs/attacked_a0_a11_smile300_seed42/mouth/ace_pair_metrics.csv
- outputs/attacked_a0_a11_smile300_seed42/mouth/metrics/fid_sfid_metrics.json
- per-sample selected.json, masks, and corrected outputs for 10429, 18004, and
  28408.

Before publication:

1. verify exactly 300 matched IDs per method and no duplicate method/source
   keys;
2. regenerate paper metric macros from the completed aggregate source rather
   than hand-copying values into multiple locations;
3. verify FR, FVA, FS, MNAC, CD, and COUT against the saved pair-level metrics;
4. verify all paper-owned figure assets are byte-identical copies or
   deterministically derived crops/overlays from the declared source paths;
5. confirm the sample count occurs exactly once in the manuscript;
6. confirm “spatial FID,” all pending-result language, and the earlier
   clean-result headline no longer occur;
7. compile with no overfull boxes, unresolved citations, or unresolved
   references;
8. inspect both figures at final PDF scale.

## Claim Boundary

The paper claims classifier-level interventional causality only within the
frozen classifier–generator system. Grad-CAM++ proposes semantic candidates;
paired controlled interventions verify graph edges. The qualitative masks in
Figure 1 come from the fixed mouth-region end-to-end protocol and therefore
illustrate localized generation, not automatic causal mask discovery.

The paper will not call the evaluation cohort the entire CelebA-HQ dataset.

## Non-Goals

- Completing or reporting the still-running mouth-plus-upper/lower-lip study.
- Inventing automatic mask-discovery measurements.
- Claiming independent-oracle validation for classifier-dependent metrics.
- Claiming improvement on MNAC, FID, FVA, or FS when the saved values do not
  support that statement.
- Modifying the generation or evaluation algorithms.
