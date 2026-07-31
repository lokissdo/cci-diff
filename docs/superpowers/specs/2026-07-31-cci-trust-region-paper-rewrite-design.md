# Lexicographic Trust-Region CCI Paper Rewrite Design

## Purpose

Replace the obsolete constraint-feedback manuscript with a concise,
method-first paper describing the implemented lexicographic trust-region CCI
method. Preserve `paper/cci_conference_v1.tex` as the archived primal-dual
draft and create a separate manuscript for the current method.

The rewrite must distinguish completed evidence from planned evidence. The
completed paired 200-image clean experiment may support quantitative claims.
The two 300-image attacked experiments remain explicit placeholders until
their cohorts and metrics are complete.

## Scientific Claim

The paper's central claim is narrow:

> Localized diffusion counterfactual guidance can be posed as a sequence of
> small lexicographic trust-region problems that prioritize target progress,
> enforce the best attainable identity/locality envelope, and then minimize
> non-target attribute drift.

The paper does not claim causal identification, global optimization,
classifier independence, or superiority on unfinished end-to-end results.
"Causal concept intervention" refers to a controlled intervention on the
decision evidence of the explained classifier.

## Method Identity

The proposed method is A11 (`trust_region`). Its matched fixed comparator is
A10 (`fixed_trust_matched`), and A0 (`disabled`) is the BLD baseline under the
same source, prompt, mask, seed, and scheduler configuration.

The method has four essential components:

1. predicted-clean evaluation through the fixed VAE decoder during SD2.1
   denoising;
2. a target-progress constraint, identity/locality safety envelope, and
   non-target drift objective ordered lexicographically;
3. clean-latent trust-region updates mapped explicitly to the scheduler's
   epsilon-prediction coordinates;
4. preservation-aware final latent restoration with deterministic
   backtracking and a cumulative displacement radius.

All 39 non-target CelebA attributes are included in the differentiable drift
objective. Residual TV is audit-only. Final restoration is latent optimization
after denoising; it does not invoke another U-Net or scheduler step.

## Paper Structure

### 1. Introduction

Motivate localized visual counterfactuals and the failure of permanently
weighted objectives. State the target/safety/preservation hierarchy and list
three contributions:

- lexicographic predicted-clean guidance in BLD;
- matched fixed/adaptive comparison with clean-coordinate trust budgets;
- preservation-aware final restoration with auditable backtracking.

Do not introduce global graph discovery as a main contribution.

### 2. Related Work

Use one compact section covering diffusion counterfactuals, blended diffusion,
predicted-clean guidance, and gradient coordination. Avoid a long attribution
survey because region discovery is fixed infrastructure in this paper.

### 3. Method

Merge problem formulation into the method section. Use these subsections:

1. **Localized counterfactual objective and masks.** Define desired target
   probability, semantic mask, generation mask, identity, locality, and
   non-target drift.
2. **Predicted-clean measurements.** Define predicted-clean latent/image and
   explain the fixed-decoder Jacobian path.
3. **Lexicographic trust-region update.** Present the two-stage safety-envelope
   and drift-minimization subproblems, target-feasible guard, small active-set
   solve, and clean-to-epsilon mapping.
4. **Preservation-aware final restoration.** Explain the maximum iteration
   budget, deterministic fractions \(1, 1/2, 1/4, 1/8\), acceptance rules,
   cumulative radius, and uint8 re-evaluation.

Keep implementation detail subordinate to the optimization logic. Describe
the fixed VAE decoder as a differentiable measurement map, not as a generative
sampling step.

### 4. Experimental Protocol

Describe CelebAMask-HQ smile removal, mouth-only semantic support, SD2.1/35
steps/seed 42/MPS float32, one output per source, and paired A0/A10/A11 arms.

Separate two protocols:

- **Completed clean study:** two disjoint random cohorts of 100 images each,
  combined as 200 images per method, with no post-generation attack.
- **Planned attacked study:** 300 paired images for mouth only followed by the
  same 300 IDs for mouth + upper lip + lower lip, comparing A0 and A11.

State explicitly that classifier-dependent COUT, FR, MNAC, and CD use the same
frozen multi-label classifier that guides CCI, by design. This is an explained-
model evaluation, not an independent-oracle result.

### 5. Results

Use three result blocks:

1. **Completed clean A0/A10/A11 results.** Aggregate the two 100-image cohorts
   and report target flip rate, feasibility, identity cosine, outside-mask
   error, non-target drift when validly aggregated, and runtime. Distinguish
   generation-classifier measurements from independent metrics.
2. **Final-restoration mechanism ablation.** Present sample 26811 as an
   illustrative mechanism ablation, not population evidence. Report target,
   identity, drift, runtime, pixel change, and accepted-step trace summary.
3. **End-to-end attacked evaluation placeholders.** Provide two visibly
   incomplete tables for mouth-only and expanded-lip regions with columns
   `Method`, `FID`, `sFID`, `FVA`, `FS`, `MNAC`, `CD`, `COUT`, and `FR (%)`.
   Placeholder cells must say `Pending (300-image run)` and cannot influence
   the abstract, conclusion, or claims.

Do not call preliminary summaries statistically significant. Do not claim
that A11 beats A10 unless the displayed completed aggregation directly
supports the stated metric.

### 6. Limitations and Conclusion

Merge discussion, ethics, and conclusion into a compact limitations section
plus conclusion. Cover same-classifier optimization/evaluation, MPS runtime,
one task and dataset, fixed mask policy, incomplete attacked evaluation, and
the absence of a convergence guarantee. Include a short facial-data ethics
paragraph without a standalone full section unless a venue later requires it.

## Removed or Demoted Material

Remove from the main narrative:

- accumulated dual multipliers and residual feedback;
- EMA gradient normalization, target-priority projection, and its budget rule;
- classifier-specific global graph discovery equations;
- the saved-image quantization correction as the core method;
- blond-hair experiments and their fixed policy table;
- independent ACE oracle language that no longer matches the selected metric
  policy;
- claims of complete-database evaluation;
- the large frozen-controller parameter appendix.

Retain only details required to reproduce A0/A10/A11 and interpret current
metrics. Additional solver tolerances and trace fields may be summarized in a
small appendix or referenced through the implementation.

## Figures and Tables

Reuse real repository images only. The main method figure should show:

`source -> BLD denoising -> predicted-clean measurement -> lexicographic
trust-region update -> source blend -> final restoration -> output`.

The figure must visually distinguish the denoising loop from final restoration
and indicate that final restoration uses the VAE decoder and evaluators but no
U-Net/scheduler transition.

Use:

- one method figure;
- one completed 200-image clean comparison table;
- one restoration-ablation figure/table;
- two attacked-result placeholder tables.

Avoid decorative figures that do not support a claim.

## Artifact Strategy

Create `paper/cci_trust_region.tex` and compile it to
`paper/cci_trust_region.pdf`. Preserve the old manuscript unchanged. Reuse
`paper/references.bib`, adding citations only when the rewrite requires them.

The manuscript must compile locally without network access. Generated LaTeX
auxiliary files remain untracked/ignored according to the existing repository
policy.

## Validation

Before completion:

1. derive all completed table values from saved JSON/CSV artifacts rather than
   manual transcription where practical;
2. verify that every method description matches the current A11 code path;
3. search the new manuscript for obsolete terms (`dual multiplier`,
   `feedback`, `conflict projection`, `complete database`, and unsupported
   `independent oracle` claims);
4. compile the PDF and resolve LaTeX errors and missing references;
5. inspect page count, tables, and figure placement;
6. ensure all 300-image cells remain explicit placeholders until complete.

## Non-Goals

- Re-running experiments as part of the manuscript rewrite.
- Filling incomplete attacked metrics from partial cohorts.
- Revising the archived `cci_conference_v1.tex`.
- Claiming comparison with published methods under unmatched protocols.
- Treating the one-image restoration ablation as aggregate evidence.
