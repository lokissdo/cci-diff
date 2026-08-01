# Risk-Controlled Source-Only Mask Selection Design

## Goal

Choose the smallest intervention-verified semantic mask that is likely to
produce a valid and preservation-safe counterfactual for one source image. The
choice is made before diffusion from source-only evidence, after which the
pipeline generates exactly once.

The algorithm is generic across binary target labels and desired directions.
The initial paper experiment specializes it to smile removal with two nested
candidates:

```text
S1 = mouth
S2 = mouth + upper_lip + lower_lip
```

The method must improve support resolution without using held-out generated
outputs, the independent evaluation oracle, FID, sFID, or any other test
metric to choose a mask.

## Decisions

1. Global discovery supplies intervention-verified candidate region sets and
   cohort-level effect evidence.
2. A target-, direction-, and generation-policy-specific logistic selector is
   fitted on a disjoint development cohort and calibrated on a second disjoint
   cohort.
3. The selector predicts joint target, identity, and locality feasibility for
   every candidate from source-only features.
4. Candidates must pass both saliency coverage and a calibrated risk gate.
5. The smallest feasible mask wins. Predicted feasibility and global effect
   break ties only after mask area.
6. If no candidate passes, use a frozen high-reliability global fallback.
7. The mask choice is frozen before A0 or A11 runs; matched experimental arms
   receive the same selected support for a given source.
8. The independent oracle is evaluation-only. It never trains or calibrates
   the selector and never participates in inference.

This specification supersedes the coverage-only selection rule in
`2026-07-26-individual-gradcam-region-selection-design.md` and changes the
`required_flip_rate` role described in
`2026-07-29-pareto-region-selection-design.md` from unused compatibility
metadata to a reliability requirement for the frozen fallback.

## Scientific Separation

Four ID-disjoint cohorts have explicit roles:

1. **Graph discovery** proposes semantic regions, runs paired same-seed region
   interventions, and exports cohort-level evidence.
2. **Selector fitting** runs every approved candidate under the declared final
   generation policy and fits the source-only feasibility model.
3. **Selector calibration** calibrates probabilities and freezes the selection
   threshold without changing model coefficients.
4. **Held-out evaluation** selects from sources, generates once, and reports
   final metrics.

All split manifests record exact sample IDs and are checked for pairwise
disjointness before fitting or evaluation. Candidate runs from held-out images
may be precomputed for deterministic policy replay, but the selection program
must not read their outputs, audits, paths, probabilities, or failures.
Selection and replay are separate processes. The selector first writes and
hashes the complete source-only decision manifest; only then may the
materializer read candidate paths and resolve each frozen decision to one
existing output. The paper describes the deployable one-generation policy,
and provenance records whether evaluation used direct execution or source-only
replay.

Existing fixed-mask attacked outputs may be used in one of two roles, never
both:

- development/calibration data followed by a new held-out cohort; or
- held-out policy replay after fitting and calibration on separate IDs.

## Global Candidate Policy

Discovery continues to aggregate, annotate, and export every evaluated region
set. An automatic selector candidate is eligible when:

- its sample count reaches the configured discovery minimum;
- mean desired-class effect is positive;
- the clustered confidence-interval lower bound is positive;
- flip rate and semantic mask area are finite;
- every component belongs to the graph's verified region pool.

Eligible Pareto-optimal sets form the default candidate pool. A task manifest
may predeclare a smaller audited family, such as `mouth` and
`mouth+upper_lip+lower_lip`, but it cannot add a set absent from discovery
evidence.

The global fallback is selected before per-image calibration:

1. retain eligible sets whose discovery flip rate is at least
   `required_flip_rate`;
2. among those sets, minimize mask area, then maximize mean effect and flip
   rate;
3. if none reaches the requirement, maximize flip rate, then mean effect, then
   minimize mask area, and mark the fallback `below_required_flip_rate`.

The influence graph adds `candidate_region_sets`, `fallback_regions`, and
`fallback_status`. The legacy `selected_regions` field remains as an alias for
`fallback_regions` during migration.

## Source-Only Features

For source image \(x_i\), target label \(t\), desired value
\(y_t^*\in\{0,1\}\), candidate set \(S\), source target probability
\(p_t(x_i)\), Grad-CAM++ map \(A_i\geq0\), candidate mask \(M_{i,S}\), and
full verified-union mask \(M_{i,U}\), define target difficulty

\[
d_i=-(2y_t^*-1)\operatorname{logit}(p_t(x_i)),
\]

saliency coverage

\[
C_i(S)=
\frac{\sum_p A_i(p)M_{i,S}(p)}
     {\sum_p A_i(p)M_{i,U}(p)+\epsilon},
\]

saliency density

\[
D_i(S)=
\frac{\sum_p A_i(p)M_{i,S}(p)}
     {\sum_p M_{i,S}(p)+\epsilon},
\]

and normalized semantic area

\[
a_i(S)=\frac{\sum_p M_{i,S}(p)}{HW}.
\]

The feature vector is

\[
\phi_i(S)=
[d_i,C_i(S),D_i(S),a_i(S),|S|,
  E(S),F(S),L(S)],
\]

where \(E(S)\), \(F(S)\), and \(L(S)\) are frozen global mean effect, flip
rate, and effect confidence lower bound. Candidate masks use exact binary
unions, so overlapping component masks are never double-counted.

Features are generic across labels. Each target label and desired direction has
its own fitted coefficients because semantic support and intervention
difficulty are task-specific. A model artifact is also tied to a generation
policy signature containing the diffusion checkpoint, prompt, seed policy,
controller, attack, mask preprocessing, and evaluator digests.

## Offline Safe-Success Label

For a development output generated with candidate \(S\), define joint safe
success as

\[
Y_{i,S}=\mathbb{1}[
p_{\mathrm{desired}}\geq0.5+m_T
\;\land\;
1-\operatorname{cos}(e(x_i),e(\hat x_{i,S}))\leq\varepsilon_I
\;\land\;
L_{\mathrm{outside}}(x_i,\hat x_{i,S},M_{i,S})\leq\varepsilon_L].
\]

Defaults reuse the declared pipeline constraints:

- target decision margin `m_T = 0.03`;
- identity distance limit `epsilon_I = 0.08`;
- outside-locality limit `epsilon_L = 0.02`.

The label uses the generation guidance classifier, generation identity model,
and declared locality evaluator. It does not use the independent oracle,
VGGFace2 FVA evaluator, SimSiam FS evaluator, Inception features, MNAC, CD, or
COUT. Non-target drift remains a reported diagnostic and global tie-break
input rather than a new hard threshold absent from the concept graph.

Every development row records both the joint label and each component pass so
calibration failures remain diagnosable.

## Feasibility Model

For each target and desired direction, fit an L2-regularized logistic model

\[
q^{\mathrm{raw}}_i(S)=
\sigma(\beta_0+\beta^\top\widetilde\phi_i(S)),
\]

where continuous features are standardized using fitting-cohort statistics.
Zero-variance features receive scale one. Rows sharing a source ID always
remain in the same fold. Use five deterministic folds assigned from the sorted
source IDs. Select the L2 coefficient by grouped cross-validation over the
fixed grid
`[1e-4, 1e-3, 1e-2, 1e-1, 1.0]`, minimizing mean log loss. Fit deterministically
on CPU with NumPy float64 damped Newton updates and no additional machine
learning dependency. Regularize slopes but not the intercept, stop when the
maximum absolute coefficient update is at most `1e-10`, reject non-finite
updates, and reject a fit that does not converge within 200 iterations.

Fit a two-parameter Platt calibrator on the calibration cohort only with the
same deterministic solver and a fixed `1e-6` slope regularizer:

\[
q_i(S)=\sigma(\gamma_0+\gamma_1\operatorname{logit}
(q^{\mathrm{raw}}_i(S))).
\]

The saved artifact contains feature order, standardization values,
coefficients, regularization, calibration coefficients, graph and generation
digests, cohort hashes, and software versions.

## Risk Threshold

The calibrated score threshold is selected without held-out outputs. For each
candidate threshold over the calibration scores, consider accepted
non-fallback candidate rows with \(q_i(S)\geq\tau_q\). Compute the one-sided
95% Wilson upper confidence bound on their empirical joint failure rate using
`z = 1.6448536269514722`.

Choose the lowest threshold satisfying both:

```text
accepted rows >= 60
failure-rate upper confidence bound <= 0.05
```

If no threshold qualifies, no non-fallback candidate is risk-approved. The
selector then always uses the frozen fallback. This conservative behavior is
recorded rather than weakening the risk requirement after seeing held-out
results.

The 60-row minimum and 5% risk bound are selector protocol constants. Changing
them defines a new predeclared experiment and a new selector digest.

## Per-Image Selection

Let

\[
\mathcal F_i=
\{S:\ C_i(S)\geq\tau_{\mathrm{cov}}
\land q_i(S)\geq\tau_q\},
\]

with default saliency-coverage threshold
`tau_coverage = 0.80`. Select lexicographically

\[
S_i^*=\operatorname{lexmin}_{S\in\mathcal F_i}
(a_i(S),-q_i(S),-E(S),S).
\]

Thus reliability and coverage are constraints, not terms that a small mask can
trade away. Area is minimized only among candidates that pass both gates. If
`F_i` is empty, choose the frozen global fallback.

For the initial nested smile candidates, this reduces to

\[
S_i^*=\begin{cases}
\text{mouth}, & C_i(\text{mouth})\geq0.80
\land q_i(\text{mouth})\geq\tau_q,\\
\text{mouth+upper/lower lips}, & \text{otherwise}.
\end{cases}
\]

The selector does not need to score the larger smile candidate unless it is
used for auditing because it is the predeclared fallback.

## Inference and Matched Arms

For each held-out source:

1. validate source eligibility and semantic mask availability;
2. compute the source target probability and one source Grad-CAM++ map;
3. calculate candidate features and select one mask;
4. write an immutable selection record and execution graph;
5. derive the dilated and feathered generation mask from the selected exact
   semantic union;
6. generate exactly once for each declared experimental arm using the already
   frozen selection;
7. retain failures without retry, output reranking, or region expansion;
8. evaluate completed outputs afterward.

The same source selection file is consumed by A0 and A11. Neither arm may fit
its own selector or alter the mask after generation. Fixed-mouth and
fixed-perioral experiments remain necessary to isolate the support resolver
from the controller contribution. The selector is calibrated for the declared
proposed generation policy. Reusing its frozen support in A0 is a matched-mask
controller comparison, not a claim that its risk probability is calibrated
for A0.

## Evaluation Boundary

The independent multi-attribute oracle evaluates final target FR, MNAC, and CD
only after generation. It is distinct from the guidance classifier and does
not participate in graph discovery, selector fitting, calibration, inference,
attack, or reranking.

Final held-out reporting includes:

- FID and deterministic sFID as cohort-level distribution diagnostics;
- FVA and FS as paired identity/perceptual metrics;
- oracle FR, MNAC, and CD;
- explicitly attributed COUT;
- selected semantic area, mouth-selection rate, fallback rate, and observed
  safe-success rate by selected mask;
- one-generation compute and runtime.

FID and sFID cannot be optimized per image and are never selector features or
calibration objectives. The expected mechanism is that risk-approved small
masks reduce unnecessary change on easier sources while the reliable fallback
preserves target success on harder sources.

Required baselines on identical held-out IDs are:

1. fixed mouth;
2. fixed mouth plus upper/lower lips;
3. source Grad-CAM coverage only;
4. proposed risk-controlled source selector;
5. a post-hoc best-mask ceiling, clearly labeled non-deployable and excluded
   from the main method comparison.

## Artifacts

```text
selector_data_manifest.json
selector_fit_rows.csv
selector_calibration_rows.csv
selector_model.json
selector_calibration_report.json
adaptive_policy.json
adaptive_selections.csv
policies/<sample>/selection.json
policies/<sample>/graph.json
policies/<sample>/binding.json
policies/<sample>/target_region.png
runs/<arm>/<sample>/
adaptive_results.csv
```

Each selection row records source probability, source difficulty, candidate
coverage/density/area, frozen global evidence, raw and calibrated feasibility,
threshold, selected regions, fallback status, and every relevant digest. It
records no generated-output field until the evaluation join occurs in a
separate result artifact. `adaptive_selections.csv` and its companion canonical
JSON manifest are finalized before materialization; the manifest SHA-256 is
computed over canonical JSON excluding the digest field itself.

## Code Boundaries

- `src/cci_diff/counterfactual_graph.py` exports the candidate pool and reliable
  fallback while retaining full evidence.
- `src/cci_diff/individual_region_selection.py` retains all globally verified
  candidate sets instead of collapsing availability to legacy
  `generation_regions`.
- `src/cci_diff/risk_controlled_selection.py` owns feature construction,
  deterministic logistic fitting, Platt calibration, Wilson risk calibration,
  artifact validation, and lexicographic selection.
- `scripts/fit_region_selector.py` builds disjoint fitting/calibration rows and
  writes the frozen model.
- `scripts/run_individual_region_cci.py` loads the frozen selector, performs
  source-only selection, and writes policies before generation.
- `scripts/materialize_adaptive_region_cohort.py` accepts only a completed,
  hashed selection manifest plus candidate roots, then joins each frozen
  decision to one already-completed deterministic output. It supports
  economical policy replay but is not required by deployable inference.

No module combines selector fitting with final metric calculation.

## Validation and Failure Handling

Reject fitting or inference when:

- cohort ID sets overlap;
- a candidate lacks global intervention evidence;
- fitting rows are not paired across the declared candidate family;
- graph, classifier, generation-policy, or semantic-mask digests disagree;
- source features contain non-finite values;
- a required semantic mask is unavailable;
- a model artifact has an unknown feature schema or protocol version.

Zero saliency inside the verified union, calibration with no qualifying risk
threshold, or no feasible candidate causes deterministic fallback. It does not
cause an output-dependent retry.

## Testing

Unit tests cover:

- generic signed target difficulty for desired values zero and one;
- exact-union saliency coverage without overlap double counting;
- safe-success labels at every target, identity, and locality boundary;
- grouped source splitting and split-overlap rejection;
- deterministic logistic fitting and artifact round trips;
- Platt calibration and Wilson upper bounds;
- threshold selection, including no-qualifying-threshold fallback;
- lexicographic selection where a small unreliable mask loses;
- deterministic tie-breaking and missing-mask rejection;
- graph migration from legacy `selected_regions`;
- selector/generation digest mismatch rejection;
- absence of oracle and generated-output fields from selector inputs.

Integration tests use synthetic graph evidence and paired candidate rows to
show:

- an easy smile source selects `mouth`;
- a difficult or lip-distributed smile selects the full perioral mask;
- A0 and A11 consume the identical frozen selection;
- exactly one subprocess is launched per source and arm;
- held-out output probabilities cannot change the chosen mask;
- policy replay and direct one-generation execution select identical image
  paths for deterministic runs.

A replay integration test also verifies that the existing fixed-mouth and
fixed-perioral layouts can produce mixed A0 and A11 manifests without invoking
the diffusion runner. The materializer must reject incomplete candidate pairs,
variant mismatches, source-ID mismatches, and any selection manifest whose hash
changes after selection.

## Acceptance Criteria

The implementation is ready for a paper run when:

1. graph, fitting, calibration, and held-out manifests are pairwise disjoint;
2. the graph exports at least one eligible candidate and one deterministic
   fallback for the target;
3. selector fitting and calibration are bitwise reproducible on the same
   platform and inputs;
4. every held-out selection is source-only and written before generation;
5. matched arms use identical selections and exactly one generation each;
6. the independent oracle is absent from selector provenance;
7. reports separate guidance-classifier diagnostics from oracle metrics;
8. fixed-mask, coverage-only, and risk-controlled baselines run on identical
   IDs, seeds, and generation settings;
9. no held-out FID, sFID, FVA, FS, MNAC, CD, COUT, or FR value affects model,
   threshold, candidate, or fallback selection.

## Claim Boundary

The method resolves the smallest **risk-approved, intervention-verified**
semantic support under a frozen classifier-generator system. It does not claim
to recover a unique human-causal facial region, guarantee per-image success,
or optimize distribution metrics during inference. Generality is architectural;
each target label and desired direction requires its own discovery evidence and
calibration artifact. The initial quantitative validation is smile removal,
with a second label required before claiming cross-attribute empirical
generality.
