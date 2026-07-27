# Distribution-Aware FID Reranking Design

## Goal

Reduce FID for smile-removal counterfactuals while retaining a
generation-classifier flip ratio above 95%. For a 100-image evaluation this
means at least 96 successful outputs. The first experiment is a deterministic
10-image smoke test used to validate mechanics and candidate-selection
behavior, not to estimate publishable FID.

## Motivation

The current 100-image A3 run reaches 100% flip ratio under the generation
classifier, but its exploratory FID is 30.80. Increasing the post-attack
epsilon schedule improves classifier success without improving the image
distribution. FID is a set-level statistic, so selecting candidates
independently by target probability or locality does not directly optimize the
reported metric.

The proposed method generates several valid counterfactuals per source, then
chooses one candidate per source using a constrained set-level objective.
This separates generation from distribution matching and does not add another
gradient to the denoising process.

## Experiment Scope

- Task: smile removal.
- Method: A3 clean CCI with smooth-boundary post-attack.
- Inputs: the first 10 eligible smile-positive samples from the established
  deterministic manifest.
- Candidate count: four seeds per source.
- Seeds: 42, 43, 44, and 45.
- Denoising steps: 35.
- Mask geometry: `x4_y4_f3`.
- Post-attack epsilon schedule: `0.05,0.08,0.10,0.30,0.50`.
- Post-attack boundary margin: 0.03.
- Device and dtype: MPS and float32.
- No model training or checkpoint changes.

The smoke test requires at least 10 of 10 generation-classifier passes because
10 samples cannot represent a strict 96% threshold. The final 100-image run
will require at least 96 passes.

## Candidate Generation

Each source image produces four candidates with identical prompts, masks,
guidance settings, and CCI configuration. Only the random seed changes.

For every candidate, preserve both:

- the raw diffusion output before post-attack;
- the corrected output after the adaptive post-attack.

If a raw candidate already passes the target classifier, it remains eligible
without attack. Correct every failed raw candidate with the adaptive attack.
When two eligible candidates have equal distribution score within `1e-8`,
select the naturally successful candidate before comparing attack magnitude
and seed.

## Reference Statistics

The selector must not compute its reference moments from the final evaluation
outputs or from the paired evaluation source set. It uses cached Inception
features from a disjoint CelebA-HQ reference subset.

For the smoke test, use the first 1,000 available CelebA-HQ image IDs after
excluding all ten evaluated source IDs. Sort candidates numerically before
selection. Record every reference ID and the feature-cache checksum. The final
benchmark reuses this fixed reference split and preserves the same exclusion
rule.

## Candidate Measurements

For each raw and corrected candidate, record:

- generation-classifier target probability and target pass;
- Inception pool3 feature;
- diagonal Mahalanobis distance to the reference feature distribution;
- FaceNet identity cosine;
- inside- and outside-semantic-mask L1;
- changed-pixel fractions at 1, 5, and 10 intensity levels;
- post-attack selected epsilon, actual L-infinity change, and changed fraction;
- runtime and source/candidate paths.

Candidate feature extraction and classifier evaluation must operate on the
saved 8-bit image so selection matches the actual artifact.

## Selection Methods

The experiment compares four selectors over the same candidate pool.

### S0: Single-Seed Baseline

Choose seed 42 for every source. This reproduces the current sampling budget.

### S1: Random Four-Seed Control

Choose one of the four candidates per source using selector seed `20260725`.
This measures whether additional sampling alone explains an improvement.

### S2: Independent Per-Image Selection

Among candidates that satisfy all hard constraints, minimize diagonal
Mahalanobis distance in the reference-fitted feature space. Apply the
tie-breaking rules below. If no candidate passes, select the highest target
probability.

### S3: Constrained Global FID Selection

Fit a 64-dimensional PCA projection using only the fixed reference features.
Choose one candidate per source to minimize proxy FID between the projected
selected candidate features and projected reference features:

\[
\operatorname{FID}(S,R)=
\|\mu_S-\mu_R\|_2^2+
\operatorname{Tr}\left(
\Sigma_S+\Sigma_R-2(\Sigma_S\Sigma_R)^{1/2}
\right).
\]

Initialize S3 from S2. Perform deterministic coordinate descent over source
IDs: test each alternative candidate and accept a swap only if it lowers FID
while preserving the target-pass requirement and hard identity/locality
bounds. Stop after a complete pass with no accepted swap or after a configured
maximum of eight passes.

For the 10-image smoke test, Inception covariance is rank-deficient and FID is
high variance. Add `1e-6 I` to both proxy covariance matrices. Use proxy FID
only for selection; report the standard 2048-dimensional evaluator FID after
selection. These smoke-test FID values validate relative behavior only.

## Constraints and Tie-Breaking

For 100 images, S3 must retain at least 96 generation-classifier passes.
For the 10-image smoke test, all 10 outputs must pass.

Candidate eligibility uses:

- identity cosine at least 0.80;
- outside semantic L1 no greater than 0.03 in normalized RGB units;
- a saved-image target pass at threshold 0.5.

If no candidate satisfies every hard constraint, preserve target validity and
rank violations lexicographically:

1. target pass;
2. identity threshold;
3. outside-mask threshold;
4. lower reference distance;
5. lower attack perturbation;
6. lower seed.

## Outputs

Write the smoke test under:

`outputs/clean_cci_fid_rerank_smoke10`

Required artifacts:

- one directory per source and seed containing raw/corrected images and audit;
- `candidate_metrics.csv`;
- `reference_manifest.json`;
- `reference_features.npz`;
- `selection_s0.csv` through `selection_s3.csv`;
- a selected image directory for each selector;
- `selector_metrics.csv`;
- `fid_reranking_report.md`;
- input/output comparison images for S3.

The report must show target success, FID, identity, locality, attack usage, and
the selected seed distribution for all four selectors.

## Validation

Unit tests cover:

- deterministic seed expansion;
- exclusion of evaluation IDs from the reference set;
- exact target-pass constraint enforcement;
- one-candidate-per-source selection;
- deterministic tie-breaking;
- global swaps that reduce a synthetic FID objective;
- refusal to accept a lower-FID selection that violates the FR constraint;
- manifest and output-path integrity.

The smoke test succeeds when:

- all 40 candidate runs resolve;
- each selector emits exactly 10 images;
- S3 retains 10 target passes;
- S3 FID is no worse than S0 and S1 on the fixed smoke reference;
- all metrics can be recomputed from saved artifacts.

Failure to improve smoke-test FID blocks the 100-image run and triggers
inspection of reference statistics, candidate diversity, and selector
regularization rather than increasing attack strength.

## Interpretation

The method's contribution is constrained distribution-aware candidate
selection, not merely best-of-four sampling. Evidence for that claim requires
S3 to outperform both the single-seed baseline and random four-seed control
using the same generated candidate pool.

The generation-classifier flip ratio remains the paper-compatible FR. Any
independent oracle target rate is reported separately as a transfer diagnostic
and never substituted into the FR column.
