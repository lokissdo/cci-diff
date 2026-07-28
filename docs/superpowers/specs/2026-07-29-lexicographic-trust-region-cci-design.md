# Lexicographic Trust-Region CCI Design

## Purpose

Replace the current accumulated primal-dual preservation weighting with a
target-progress-constrained trust-region optimizer. The new adaptive CCI
method must improve non-target attribute preservation relative to fixed-weight
CCI at matched target success. Identity and outside-mask locality remain
safety constraints.

This change addresses five measured failure modes in the existing controller:

1. preservation multipliers integrate unreliable early predicted-clean
   violations and saturate;
2. target-first budgeting frequently removes the entire preservation update;
3. global trust clipping erases differences between fixed and adaptive
   coefficients;
4. the update radius is measured in noise-space units whose predicted-clean
   effect changes with timestep;
5. the shared target-only final correction dominates both fixed and adaptive
   outputs without checking preservation.

The existing `disabled`, `fixed_equal`, and `feedback` modes remain available
for compatibility. The new proposed method is exposed as an additional
controller mode rather than silently changing archived experiment semantics.

## Optimization Objective

For source image \(x_s\), decoded predicted-clean image \(x(z)\), target
attribute \(T\), desired target sign \(y_T\), target probability threshold
\(p^\star\), and target logit \(f_T\), define:

\[
\kappa = \log\frac{p^\star}{1-p^\star},
\qquad
c_T(z) = \kappa - (2y_T-1)f_T(x(z)).
\]

The requested target is feasible when \(c_T(z)\le0\).

The primary differentiable preservation objective is the mean smooth drift
over every non-target CelebA attribute:

\[
D_{\mathrm{NT}}(z)=
\frac{1}{K-1}\sum_{j\ne T}
\operatorname{Huber}_\delta\left(
p_j(x(z))-p_j(x_s)
\right).
\]

No non-target attribute may be exempted. The generation classifier supplies
this optimization surrogate. Confirmatory evaluation uses an independent ACE
oracle and therefore does not reuse the optimization model for the primary
claim.

The hard preservation surface contains:

\[
c_I(z)=d_{\mathrm{identity}}(z)-\epsilon_I,
\qquad
c_L(z)=d_{\mathrm{outside}}(z)-\epsilon_L.
\]

Identity and locality are feasible when their residuals are non-positive.
Residual TV remains an audit metric in the first implementation because the
current tolerance is infeasible for every measured adaptive output. It can
return as a hard constraint only after a separate calibration demonstrates a
non-empty feasible region.

## Lexicographic Local Subproblem

At a guided step, compute gradients with respect to a predicted-clean latent
optimization variable:

\[
g_T=\nabla c_T,\quad
g_D=\nabla D_{\mathrm{NT}},\quad
g_I=\nabla c_I,\quad
g_L=\nabla c_L.
\]

Apply the effective generation support to each gradient before normalization,
Gram-matrix construction, or optimization. This makes the solver reason about
the update that can actually survive the generation mask rather than masking
an already-composed direction.

When the target is infeasible, request achievable local progress:

\[
q_t=\min\left(
\max(c_T,0),
\chi_t\Delta_t\lVert g_T\rVert
\right),
\qquad 0<\chi_t<1.
\]

The first subproblem finds the smallest unavoidable safety violation while
preserving the target-progress request:

\[
\begin{aligned}
\tau^\star=\min_{d,\tau}\quad &\tau\\
\mathrm{s.t.}\quad
&g_T^\top d\le-q_t,\\
&c_I+g_I^\top d\le\tau,\\
&c_L+g_L^\top d\le\tau,\\
&\lVert d\rVert_2\le\Delta_t.
\end{aligned}
\]

The second subproblem optimizes non-target preservation inside that best
feasibility envelope:

\[
\begin{aligned}
d^\star=\arg\min_d\quad
&g_D^\top d+\frac{1}{2\eta_t}\lVert d\rVert_2^2\\
\mathrm{s.t.}\quad
&g_T^\top d\le-q_t,\\
&c_I+g_I^\top d\le\tau^\star+\zeta,\\
&c_L+g_L^\top d\le\tau^\star+\zeta,\\
&\lVert d\rVert_2\le\Delta_t.
\end{aligned}
\]

When the target is already feasible, replace the progress constraint with the
linearized guard \(c_T+g_T^\top d\le0\). This permits preservation restoration
without crossing back over the target boundary.

The QP's KKT multipliers are recorded as the adaptive coefficients. They are
recomputed from current residuals and gradient geometry at every step. No
multiplier is carried across changing DDIM timesteps.

## Solver

The problem has at most four gradient vectors and three inequality families,
so the solve occurs in their gradient span rather than full latent dimension.
The controller builds a small float64 CPU Gram matrix from detached float32
masked gradients and solves by deterministic active-set enumeration.

The solver must:

- have no SciPy or network-installed dependency;
- use a deterministic constraint ordering;
- regularize nearly singular Gram systems;
- return the minimum-norm solution when several active sets are equivalent;
- validate the returned linearized constraints within a configured numerical
  tolerance;
- expose infeasibility, degeneracy, and fallback reasons in the trace.

If the requested target progress is infeasible within the trust region, reduce
\(q_t\) by deterministic bisection. If no reliable target descent is possible,
skip the update and record `unreliable_target_gradient`. Do not silently fall
back to a fixed weighted sum.

## Coordinates, Schedule, and BLD Interaction

The trust radius is defined in predicted-clean latent coordinates. For epsilon
prediction,

\[
\hat z_0 =
\frac{z_t-\sqrt{1-\alpha_t}\,\epsilon_t}{\sqrt{\alpha_t}},
\]

so a desired predicted-clean displacement \(d\) maps to:

\[
\Delta\epsilon_t =
-\sqrt{\frac{\alpha_t}{1-\alpha_t}}\,d.
\]

The adapter performs this conversion explicitly. The controller never clips a
noise-space vector under a clean-latent radius.

The new mode uses a reliability schedule based on scheduler alpha/SNR rather
than only normalized loop progress. The threshold is one frozen
calibration-time configuration, shared across fixed and adaptive comparisons.
Early predicted-clean observations remain traceable but do not update the
optimizer.

The adapter records the requested clean displacement, mapped noise
displacement, post-scheduler displacement, and post-BLD displacement. These
retention measurements verify where guidance is attenuated. The
implementation must account for the known soft-blend operator so generation
support is not intentionally applied twice. Any inverse compensation is
bounded; mask values below a support floor receive no update rather than an
unbounded inverse-mask gain.

## Preservation-Aware Final Restoration

The existing `FinalTargetLatentCorrectionHook` remains unchanged for archived
modes. The new mode uses a separate final trust-region hook that evaluates the
actual post-BLD latent.

It uses the same target, non-target drift, identity, and locality definitions
and the same lexicographic subproblems. Candidate steps are accepted by
backtracking only when:

- an infeasible target makes strict progress, or a feasible target remains
  feasible;
- the best achievable identity/locality envelope does not worsen beyond the
  solver tolerance;
- once target and safety are feasible, non-target drift decreases;
- the cumulative displacement remains inside a configured final radius.

Backtracking fractions are deterministic. Every attempted candidate records
target probability, non-target drift, identity distance, locality, step norm,
cumulative norm, and acceptance reason. The saved uint8 output is re-evaluated
and reported separately.

The old target-only final correction is disabled for the primary
fixed-versus-adaptive optimizer ablation. A full-system comparison may compare
a fixed-weight final optimizer with the adaptive final QP, provided both have
identical step, radius, stopping, and evaluation budgets.

## Components

### Non-target drift evaluator

Add a differentiable evaluator that:

- caches source logits/probabilities during binding;
- excludes exactly the target attribute index;
- returns one scalar smooth drift objective;
- reports mean absolute probability drift and per-attribute values for audit;
- performs one shared 40-output classifier forward.

### Trust-region controller

Add a new controller rather than overloading
`ConstraintFeedbackController`. It owns:

- gradient preparation after effective masking;
- reliability and finite-value checks;
- the two lexicographic subproblems;
- trust-radius state;
- active-set and KKT diagnostics;
- clean-coordinate update records.

It does not own VAE decoding, classifier evaluation, or DDIM conversion.

### Clean CCI adapter

The adapter owns:

- construction of predicted-clean latents/images;
- target, drift, identity, and locality observations;
- clean-latent to scheduler-output conversion;
- post-update measurement;
- step-retention tracing.

### Final restoration hook

Add a preservation-aware hook for the new mode. Do not change archived
`FinalTargetLatentCorrectionHook` behavior.

### Configuration

Extend the graph controller specification with optional trust-region settings
that have validated defaults:

- initial, minimum, and maximum clean trust radii;
- target-progress fraction \(\chi\);
- feasibility envelope tolerance \(\zeta\);
- active/reliability alpha or SNR threshold;
- Huber transition for non-target drift;
- support floor and maximum blend compensation;
- final cumulative radius and maximum final iterations.

Serialization must preserve version-one graph compatibility. Audit output
records resolved defaults so experiments are reproducible.

## Modes and Fair Comparators

Keep existing meanings:

- `disabled`: raw BLD;
- `fixed_equal`: archived fixed-equal controller;
- `feedback`: archived accumulated primal-dual controller.

Add:

- `trust_region`: proposed adaptive lexicographic QP.

For the optimizer-specific benchmark, add a fixed-weight comparator using the
same clean-coordinate mapping, schedule, support handling, trust-radius
budget, and final-optimizer budget as `trust_region`. Its only difference is a
frozen scalar composition in place of the QP. Name it
`fixed_trust_matched` to avoid confusing it with archived `fixed_equal`.

## Trace and Diagnostics

Every active step records:

- all raw and masked objective-gradient norms;
- the complete cosine and Gram matrices;
- target and safety residuals before and after the update;
- requested and achieved linearized target progress;
- \(\tau^\star\), active constraints, KKT multipliers, and solver residuals;
- clean, noise, scheduler, and blend displacement norms;
- trust radius before and after adaptation;
- skip/fallback reason;
- non-target drift before and after.

Aggregate diagnostics report:

- solver infeasibility and degeneracy rates;
- target-progress satisfaction rate;
- identity/locality active-set rates;
- update-retention ratios through scheduler and blend;
- final restoration acceptance and stopping reasons;
- runtime broken down into decode, evaluators, gradients, QP, and candidate
  evaluation.

## Evaluation

The primary scientific comparison is adaptive `trust_region` versus
`fixed_trust_matched`. Raw BLD and archived modes are contextual baselines.

Use disjoint discovery, calibration, and held-out test cohorts split by
identity cluster. Each source uses paired seeds, prompts, latents, scheduler,
mask, and execution settings across arms. Give fixed and adaptive methods the
same calibration search budget.

Construct success-preservation frontiers on calibration by varying only
pre-registered target effort settings. Freeze the settings and interpolation
weights before test. On the test set:

- target success is measured by the independent ACE oracle;
- primary preservation is independent continuous non-target probability drift
  over all 39 non-target attributes;
- identity uses the independent face representation already used by the ACE
  evaluation;
- locality uses normalized outside-semantic L1;
- MNAC, FID/sFID, FS, residual TV, boundary metrics, and runtime are secondary.

Report preservation at the highest common calibrated target-success operating
point and normalized area under the safety-feasible success-drift frontier.
Use paired identity-cluster bootstrap confidence intervals. Do not condition
the primary preservation metric on successful samples.

The adaptive method is supported only if:

- its target-success difference from the fixed comparator is within
  pre-registered ±5 percentage-point equivalence bounds;
- it reduces independent non-target drift by at least 10%, with a paired
  confidence interval excluding zero;
- its normalized frontier area is lower with a paired interval excluding
  zero;
- joint identity/locality pass rate is at least 95% and is non-inferior to
  fixed within 2 percentage points;
- all planned paired results are present.

## Error Handling

- Non-finite evaluator outputs or gradients reject the step and leave trust
  state unchanged.
- Two consecutive non-finite steps raise as in the archived controller.
- Solver infeasibility reduces requested progress before skipping.
- Singular systems use bounded regularization and are explicitly traced.
- Empty masks, unreliable target gradients, or missing bound evaluators fail
  before generation.
- Final restoration never returns a candidate that was not explicitly
  accepted.
- Archived modes and their audit semantics remain unchanged.

## Testing

### Pure optimizer tests

- feasible and infeasible target-progress requests;
- target-feasible margin guard;
- identity/locality active-set combinations;
- minimum-norm tie breaking;
- singular and nearly collinear gradients;
- deterministic bisection of target progress;
- no target regression inside the linear model;
- KKT and primal residual validation;
- trust-radius compliance;
- float32 latent and float64 Gram interoperability.

### Evaluator tests

- target exclusion from all 40 attributes;
- cached source probabilities;
- zero drift for identical images;
- Huber behavior and differentiability;
- one classifier forward for the full attribute vector.

### Adapter tests

- exact epsilon-to-clean displacement conversion;
- timestep-invariant clean trust radius;
- mask applied once in the intended effective operator;
- bounded soft-blend compensation;
- reliability scheduling;
- scheduler/blend retention records;
- unchanged archived controller behavior.

### Final restoration tests

- rejects target-improving candidates that worsen the safety envelope;
- restores preservation while maintaining a feasible target;
- enforces cumulative radius;
- deterministic backtracking and early stopping;
- uint8 saved-output re-evaluation.

### End-to-end smoke tests

- `disabled`, archived modes, `fixed_trust_matched`, and `trust_region`;
- complete audit and trace serialization;
- paired fixed/adaptive command construction;
- no network access or model download in tests;
- fake backends for all unit and smoke coverage.

## Rollout

1. Implement the pure evaluator, QP solver, and controller behind the new
   modes.
2. Add adapter coordinate conversion and diagnostics without enabling final
   restoration.
3. Run deterministic fake-backend and small real-model trace diagnostics.
4. Verify update retention and calibrate a reliability threshold on the
   calibration cohort.
5. Implement the preservation-aware final hook.
6. Run a paired 10-image fixed/adaptive smoke test with the old final
   correction disabled.
7. Proceed to the frozen calibration and held-out matched-success benchmark
   only if solver and retention diagnostics pass.

## Non-Goals

- Retuning or rewriting archived A0-A9 results.
- Optimizing against the independent ACE oracle.
- Adding a new diffusion model, scheduler, or classifier checkpoint.
- Claiming theoretical convergence for a changing non-convex DDIM trajectory.
- Reintroducing residual TV as a hard constraint before feasibility
  calibration.
- Using post-generation attacks to rescue the primary optimizer comparison.
