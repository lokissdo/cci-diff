# CCI Conference Paper V1 Design

**Date:** 2026-07-20

## Objective

Create a first standalone conference-style paper describing Constraint-Feedback
Causal Concept Intervention (CCI) for localized diffusion counterfactuals. The
manuscript must be a faithful account of the current implementation and the
completed 100-image-per-task benchmarks, including mixed and negative results.

The first version will use a self-contained, two-column conference layout based
on the standard LaTeX `article` class. This avoids depending on a venue-specific
class that is not present in the local environment and keeps later conversion
to IEEE, CVPR, or another venue mechanical. The submission-facing author line
will read `Anonymous Author(s)` until the user supplies final metadata.

## Working Title

**Constraint-Feedback Causal Concept Intervention for Localized Diffusion
Counterfactuals**

## Paper Positioning

The paper is methods-first, with auditability as a secondary contribution.
It presents CCI as an extension of blended latent diffusion (BLD), not as a
replacement diffusion model.

The supported contributions are:

1. A versioned concept graph that separates the intervention target, allowed
   correlated changes, preservation constraints, and audit-only concepts.
2. Predicted-clean guidance that evaluates the target and constraints on an
   estimate of the clean image during denoising.
3. A target-priority primal-dual controller that derives preservation
   coefficients from normalized constraint violations instead of fixed manual
   weights.
4. Per-step traces of target confidence, violations, dual multipliers, gradient
   norms, gradient conflicts, and update norms.
5. A matched-cohort BLD-versus-CCI evaluation producing 800 edited outputs over
   two 100-sample task cohorts, two methods, and two denoising-step settings.
   Because per-step source-blend noise is not shared, this version is not a
   strictly noise-controlled ablation.

The paper must not claim:

- state-of-the-art performance;
- direct comparability with the prior-paper table;
- that CCI satisfies all graph constraints;
- that guidance-classifier gains transfer fully to an independent classifier;
- that 100-sample FID or sFID estimates are conclusive;
- that increasing inference steps improves counterfactual validity;
- that the current JSON graphs are generated automatically from free text.

## Evidence Hierarchy

When sources disagree, use this order:

1. Executed graph JSON, run audits, pair-level CSV files, and final aggregate
   reports.
2. Current implementation.
3. Current design specifications.
4. Historical plans and legacy guidance documentation.

In particular, the executed graphs use active progress `[0.15, 0.90]`.
The earlier `[0.15, 0.65]` proposal must not appear as the executed setting.

## Source of Truth

Method:

- `examples/graphs/remove_smile_clean_cci.json`
- `examples/graphs/blond_hair_clean_cci.json`
- `src/cci_diff/constraint_controller.py`
- `src/cci_diff/adapters/sd2_clean_cci.py`
- `src/cci_diff/sd2_bld_backend.py`
- `scripts/run_sd2_bld_cci.py`
- `docs/superpowers/specs/2026-07-14-constraint-feedback-clean-cci-design.md`

Evaluation:

- `scripts/evaluate_clean_cci_ace.py`
- `scripts/evaluate_fid_sfid.py`
- `outputs/fid_sfid_bld_cci_steps35_50/full_metrics.md`
- `outputs/fid_sfid_bld_cci_steps35_50/fid_sfid_metrics.json`
- the `ace_pair_metrics.csv` and audit files under the four evaluated run roots.

Evaluated run roots:

- `outputs/raw_bld_a0_100_steps35_target`
- `outputs/raw_bld_a0_100_steps50_target`
- `outputs/clean_cci_a3_100_steps35_target`
- `outputs/clean_cci_a3_100_steps50_target`

## Method Contract

### BLD Backbone

Use Stable Diffusion 2.1 base from the local `checkpoints/sd2-1-base` checkpoint,
DDIM sampling, classifier-free guidance scale 5.0, blending start 0.25, seed 42,
512 by 512 resolution, float32, Apple MPS, and batch size 1.

BLD restores the noised source latent outside the soft generation mask after
each scheduler step. A0 uses the same generation path and graph-derived masks,
but its controller returns a zero update.

### Concept Graphs

Smile removal changes `Smiling` from 1 to 0, permits
`Mouth_Slightly_Open` to change, and uses the union of mouth, upper-lip, and
lower-lip masks.

Blond-hair editing changes `Blond_Hair` from 0 to 1 and permits
`Black_Hair`, `Brown_Hair`, and `Gray_Hair` to change. It uses the hair mask.

Both graphs preserve identity, outside-mask locality, and masked residual total
variation with tolerances 0.08, 0.02, and 0.015. Their generation masks use
four-pixel horizontal and vertical dilation and a three-pixel feather radius.

### Predicted-Clean Guidance

For an epsilon-prediction scheduler, estimate the clean latent as

```text
z0_hat = (z_t - sqrt(1 - alpha_t) * epsilon_theta) / sqrt(alpha_t).
```

Detach the U-Net prediction, decode `z0_hat / 0.18215` through the VAE, and
differentiate evaluators through the clean estimate and VAE to the current
latent.

For desired binary value `y*`, classifier logit `f(x0_hat)`, and required
probability `p*=0.8`, define

```text
s_t = (2y* - 1) f(x0_hat)
kappa = log(p* / (1 - p*))
L_target = max(0, kappa - s_t)
a_t = clip((kappa - s_t) / max(|kappa|, 1), 0, 1).
```

### Constraint Feedback

For constraint value `d_k` and tolerance `epsilon_k`,

```text
r_k = d_k / epsilon_k - 1
v_k = max(0, r_k)
lambda_k <- clip(lambda_k + 0.2 r_k, 0, 4)
w_k = lambda_k_before + 0.5 max(r_k, 0).
```

Normalize target and active constraint gradients with an exponential moving
average using beta 0.9 and floor `1e-5`. Remove the component of a constraint
gradient that opposes the target gradient. While the target is infeasible, cap
the aggregate preservation-gradient norm according to target activation.

Apply the soft latent mask, a sinusoidal step scale with maximum 0.2, and a
trust radius of 0.15. Guidance is active over normalized progress 0.15 through
0.90 every two steps. Add the resulting delta to the predicted noise before
the DDIM scheduler step. After the final BLD blend, apply at most 12 masked
target-only latent corrections with backtracking line search.

## Experimental Contract

### Cohorts

Use CelebAMask-HQ images and parsing masks. Each task contains exactly 100
eligible samples selected by the generation classifier:

- smile removal: source `Smiling` probability at least 0.5;
- blond-hair addition: source `Blond_Hair` probability below 0.5.

The same task-specific sample IDs are used for BLD and CCI at 35 and 50
scheduler-step settings (27 and 38 executed denoising updates after the
0.25 start index). This yields 800 edited outputs.

### Evaluators

- Guidance and same-classifier FR: the local 40-label CelebA ResNet50,
  SHA-256
  `9387f298caab711a4e8e354f5bc5492f77e8755f11f585385da9240338f3788c`.
- Independent target evaluation, MNAC, and CD: the ACE oracle checkpoint
  `thesis_2025/evaluate/ACE/models/checkpoint.tar`.
- FVA: VGGFace2 ResNet50 features.
- FS: local SimSiam ResNet50 checkpoint.
- FID and deterministic symmetric FID: `pytorch-fid` 0.3.0,
  2048-dimensional InceptionV3 features on CPU.

### Metric Semantics

Keep the following quantities separate:

- **Target accuracy:** fraction whose output is in the desired class under the
  independent ACE oracle.
- **Directional FR:** fraction whose independent-oracle source label flips to
  the desired output label.
- **Same-classifier FR:** fraction whose guidance-classifier desired
  probability is at least 0.5. This is the closest current value to the local
  legacy FR script.
- **Strong target:** fraction whose guidance-classifier desired probability is
  at least 0.8.

FVA is the fraction above its identity threshold; FVA cosine retains the
continuous signal. FS is SimSiam cosine similarity. MNAC excludes the intended
target attribute and counts collateral attribute flips. CD is the summed
absolute change-correlation score. Locality reports inside/outside L1 and
changed-pixel fractions.

FID uses all 100 source-output pairs per task. Deterministic sFID shuffles
aligned IDs with NumPy seed 42, uses two 50-sample splits, computes both
cross-split directions, and reports their arithmetic mean. These distribution
metrics are exploratory.

## Results Contract

The main counterfactual table must report all eight method/step/task rows:

| Method | Steps | Task | Target accuracy | Directional FR | Same-classifier FR | Strong target |
|---|---:|---|---:|---:|---:|---:|
| BLD | 35 | Hair | 23 | 22 | 32 | 10 |
| BLD | 35 | Smile | 17 | 7 | 20 | 9 |
| BLD | 50 | Hair | 23 | 21 | 29 | 9 |
| BLD | 50 | Smile | 18 | 8 | 18 | 10 |
| CCI | 35 | Hair | 25 | 24 | 62 | 41 |
| CCI | 35 | Smile | 24 | 14 | 56 | 40 |
| CCI | 50 | Hair | 24 | 22 | 63 | 35 |
| CCI | 50 | Smile | 22 | 12 | 58 | 37 |

The primary comparison is 35-step CCI versus 35-step BLD:

- directional FR improves by 7 percentage points for smile removal and
  2 points for blond-hair addition;
- same-classifier FR improves by 36 and 30 points;
- strong-target rate improves by 31 points for both tasks.

The discussion must emphasize the classifier-transfer gap. The improvement is
large for the classifier used in guidance but modest for the independent ACE
oracle.

The preservation and distribution tables must include every value in
`outputs/fid_sfid_bld_cci_steps35_50/full_metrics.md`. The text must state:

- FVA is 100 percent for every run and therefore uninformative at its current
  threshold.
- FS is nearly unchanged.
- MNAC and CD are mixed; CCI is worse for smile collateral change.
- Spatial locality is almost unchanged between BLD and CCI.
- At 35 steps, hair FID/sFID improve slightly under CCI, whereas smile
  FID/sFID worsen slightly.
- Fifty steps do not improve counterfactual validity.
- Median CCI runtime is 2.68--2.71 times the instrumented BLD-A0 control at
  35 steps and
  2.55--2.56 times the instrumented BLD-A0 control at 50 steps.

The prior-paper table may appear only in a clearly labeled context section,
with a warning that dataset sizes, splits, generation/reranking procedures,
and FR definitions differ. It must not be merged into the main experimental
comparison.

## Manuscript Structure

1. Abstract
2. Introduction
3. Related Work
4. Background and Problem Formulation
5. Constraint-Feedback CCI
6. Experimental Setup
7. Results and Analysis
8. Limitations and Future Work
9. Ethical Considerations
10. Conclusion
11. Appendix: complete metrics, concept graphs, and reproducibility details

## Figures and Tables

The first draft will contain:

1. A method diagram drawn directly in TikZ: concept graph, predicted-clean
   decode, evaluators, controller, masked noise update, DDIM, and BLD blend.
2. A qualitative figure using deterministic paired artifacts from the 35-step
   BLD and CCI outputs. Examples will be selected by an explicit
   outcome-based rule and will include both a transfer success and a failure.
3. The complete counterfactual-success table.
4. The complete preservation/collateral table.
5. The complete distribution/locality/runtime table.
6. An appendix table with the directional sFID components.

## Bibliography

Use BibTeX entries verified against primary sources for Stable Diffusion,
DDIM, BLD, Universal Guidance, Diffusion Posterior Sampling, DOODL, DiG-IN,
MaskDiME, CelebA/CelebAMask-HQ, FID, SimSiam, VGGFace2, ACE, and the methods in
the prior-paper context table when exact citations can be established.

Do not invent missing author lists, venues, years, DOIs, or method expansions.
Omit an uncertain citation from version 1 or mark the prose generically until a
primary source verifies it.

## Deliverables

Create:

```text
paper/
  cci_conference_v1.tex
  references.bib
  README.md
  figures/
    transfer_smile_00024_bld.jpg
    transfer_smile_00024_cci.jpg
    transfer_hair_00000_bld.jpg
    transfer_hair_00000_cci.jpg
    gap_smile_00009_bld.jpg
    gap_smile_00009_cci.jpg
    gap_hair_00008_bld.jpg
    gap_hair_00008_cci.jpg
```

The LaTeX source must compile without external venue files in a normal TeX
Live installation. The local machine currently has no `pdflatex`, `latexmk`,
or `bibtex`, so verification here will consist of source validation, citation
key checks, file-existence checks for included figures, table-value checks
against the final metrics JSON, and an explicit note that PDF compilation was
not available.

## Constraints

- Do not stage or commit any file.
- Do not modify benchmark outputs or thesis evaluation code.
- Do not overstate independent-oracle performance.
- Do not call target accuracy “FR.”
- Do not call same-classifier success independent validation.
- Do not report COUT as zero; report it as unavailable.
- Use executed settings and measurements, not intended settings from plans.
