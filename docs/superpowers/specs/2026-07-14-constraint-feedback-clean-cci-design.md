# Constraint-Feedback Clean CCI Design

## Status

This document specifies the next CCI-Diff experiment and implementation. The
direction was approved on 2026-07-14. The implementation remains gated on the
user's review of this written specification.

No commits are part of this work. All source, documentation, tests, and runtime
artifacts must remain under `my-docs/cci-diff`.

## Goal

Build a classifier-valid counterfactual editing method in which:

1. The requested classifier label flips with a configurable confidence margin.
2. Identity, locality, preserved concepts, and artifact controls are explicit
   constraints rather than manually balanced loss terms.
3. Guidance is evaluated on the predicted clean image, not a directly decoded
   noisy diffusion latent.
4. Constraint weights are generated online from measured violations.
5. A future text or label compiler can construct the intervention graph without
   changing the SD2 optimizer.
6. A complete per-step trace explains why every CCI weight changed.

The first supported interventions are:

- `Smiling: 1 -> 0`, using mouth, upper-lip, and lower-lip masks.
- `Blond_Hair: 0 -> 1`, using the hair mask.

The method must continue to run in float32 on Apple MPS with batch size 1 and
target an end-to-end cost of no more than approximately three times the BLD
baseline for one seed.

## Current Failure And Evidence

The current latent hook runs after the DDIM scheduler step and decodes the
still-noisy latent directly:

```text
z_t -> DDIM step -> z_(t-1) -> VAE decode -> classifier loss -> latent update
```

The external CelebA classifier was trained on clean images. It therefore sees
an out-of-distribution image during most guided steps. The saved robust trace
for image 0 demonstrates the mismatch:

```text
final Smiling probability: approximately 0.9966
in-loop remove-smile BCE:  approximately 0.005
```

For a clean image with Smiling probability near 0.9966 and target 0, the loss
should be large. The near-zero in-loop loss means the noisy intermediate is
mistaken for an already successful counterfactual. Normalizing that tiny
gradient to unit norm then amplifies an unreliable direction.

The 15-image experiment confirms the weak aggregate effect:

```text
Smile no-hook: 1/15 successful flips
Smile CCI:     2/15 successful flips

Hair no-hook: 11/15 successful flips
Hair CCI:     11/15 successful flips
```

The hair outputs also show that a higher target-model score can coexist with a
visually gray or otherwise imperfect result. The explained classifier is the
authority for counterfactual validity, but it cannot also be the only semantic
or realism evaluator.

## Research Basis

This design follows four established observations:

- Universal Guidance evaluates clean-image guidance models on a denoiser's
  predicted clean sample to close the noisy/clean domain gap.
- Diffusion Posterior Sampling evaluates a differentiable condition through an
  estimate of the clean sample rather than applying a clean measurement model
  directly to a noisy state.
- Limited-interval guidance shows that guidance is most useful in the middle
  of denoising and can harm quality at very high and very low noise levels.
- DOODL and DiG-IN show that optimizing losses on final clean pixels improves
  objective alignment, but full-trajectory backpropagation exceeds the chosen
  MPS time and memory budget.

Related work:

- Universal Guidance: https://arxiv.org/abs/2302.07121
- Diffusion Posterior Sampling: https://arxiv.org/abs/2209.14687
- Limited-interval guidance: https://openreview.net/forum?id=nAIhvNy15T
- DOODL: https://arxiv.org/abs/2303.13703
- DiG-IN: https://openaccess.thecvf.com/content/CVPR2024/html/Augustin_DiG-IN_Diffusion_Guidance_for_Investigating_Networks_-_Uncovering_Classifier_Differences_CVPR_2024_paper.html
- MaskDiME: https://openaccess.thecvf.com/content/CVPR2026/html/Guo_MaskDiME_Adaptive_Masked_Diffusion_for_Precise_and_Efficient_Visual_Counterfactual_CVPR_2026_paper.html

The individual ingredients are not claimed as novel in isolation. The intended
contribution is their auditable integration for causal concept intervention:
a clean-estimate, target-priority controller whose preservation weights emerge
from graph-defined constraint violations.

## Terminology

- `intervention request`: target concept, desired value, and optional text.
- `concept graph`: validated target, allowed-change, and invariant nodes.
- `compiled CCI plan`: graph nodes resolved to evaluators, masks, thresholds,
  and a controller policy.
- `audit mask`: hard semantic mask used only for strict final measurements.
- `generation mask`: fixed feathered semantic mask used by guidance and BLD.
- `predicted clean latent`: the scheduler estimate `z0_hat(z_t, epsilon_t)`.
- `constraint multiplier`: automatically updated non-negative value `lambda_k`.
- `weight trajectory`: the sequence of multipliers and target activation over
  denoising steps.

## Non-Goals

- Do not backpropagate through the U-Net or the full denoising trajectory.
- Do not train or fine-tune SD2.
- Do not infer causal truth from text alone.
- Do not implement MaskDiME-style top-k gradient mask discovery in this pass.
- Do not make CLIP text loss a default target objective.
- Do not tune a separate manual scalar weight for every loss.
- Do not replace the existing no-hook, latent-color, or latent-classifier modes.
- Do not claim identity preservation from a generic classifier feature vector.

## Architecture

```text
InterventionRequest ----+
                        |
SampleBindings ---------+--> ConceptGraphCompiler ----> ConceptRegistry
                                  |                           |
                                  v                           v
                         ValidatedConceptGraph ------> CompiledCCIPlan
                                                               |
                                                               v
                                                       CleanCCIController
                                                               |
                                               +---------------+---------------+
                                               |               |               |
                                               v               v               v
                                         target evaluator  constraints    mask policy
                                               |               |               |
                                               +---------------+---------------+
                                                               |
                                                               v
                                                  SD2 predicted-clean hook
                                                               |
                                                               v
                                                  DDIM step -> BLD soft blend
                                                               |
                                                               v
                                                  trace.jsonl + audit.json
```

The compiler and runtime controller are deliberately separate. A future text
compiler decides which graph nodes apply. It does not guess runtime weights.
The controller obtains weights from the actual generated sample and measured
constraint violations.

## Concept Graph

### Node Roles

Every graph node has exactly one role:

- `target`: the classifier decision that must reach a signed margin.
- `allowed_change`: a concept that may legitimately move after intervention.
- `constraint`: a concept or measurement that must remain within tolerance.
- `audit_only`: an independent evaluator recorded but not differentiated.

Supported edge relations are:

- `may_affect`: a target is allowed to change the destination concept.
- `must_preserve`: the destination must remain within its tolerance.
- `measured_by`: a concept is resolved through an evaluator adapter.

The graph must be acyclic. A target cannot simultaneously have
`must_preserve` and `may_affect` edges to the same node.

### Version 1 JSON

Version 1 is loaded from a validated JSON file. It contains no per-loss weights:

```json
{
  "version": 1,
  "intervention": {
    "concept": "Smiling",
    "desired_value": 0,
    "target_probability": 0.8
  },
  "region": {
    "audit_role": "mouth",
    "components": ["mouth", "upper_lip", "lower_lip"],
    "feather_radius": 3.0
  },
  "nodes": [
    {
      "id": "smiling",
      "role": "target",
      "evaluator": "celeba_attribute",
      "attribute": "Smiling"
    },
    {
      "id": "mouth_open",
      "role": "allowed_change",
      "evaluator": "celeba_attribute",
      "attribute": "Mouth_Slightly_Open"
    },
    {
      "id": "identity",
      "role": "constraint",
      "evaluator": "facenet_identity",
      "tolerance": 0.08
    },
    {
      "id": "outside_locality",
      "role": "constraint",
      "evaluator": "outside_l1",
      "tolerance": 0.02
    },
    {
      "id": "residual_tv",
      "role": "constraint",
      "evaluator": "masked_residual_tv",
      "tolerance": 0.015
    }
  ],
  "edges": [
    {"source": "smiling", "target": "mouth_open", "relation": "may_affect"},
    {"source": "smiling", "target": "identity", "relation": "must_preserve"},
    {"source": "smiling", "target": "outside_locality", "relation": "must_preserve"},
    {"source": "smiling", "target": "residual_tv", "relation": "must_preserve"}
  ],
  "controller": {
    "dual_rate": 0.2,
    "penalty": 0.5,
    "lambda_max": 4.0,
    "step_scale": 0.2,
    "trust_radius": 0.15,
    "norm_ema_beta": 0.9,
    "gradient_floor": 0.00001,
    "active_progress": [0.15, 0.65],
    "every_n_steps": 2
  }
}
```

The values under `controller` are global optimizer settings, not semantic loss
weights. Constraint tolerances are measurable acceptance boundaries. They must
be calibrated once on a held-out validation subset and then frozen across
interventions reported in the same experiment.

Image-specific files are supplied separately as `SampleBindings`:

```json
{
  "source_image": "data/0.jpg",
  "masks": {
    "mouth": "data/00000_mouth.png",
    "upper_lip": "data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_u_lip.png",
    "lower_lip": "data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_l_lip.png"
  }
}
```

This keeps one concept graph reusable across all images. Compilation fails when
a graph role has no sample binding or when a binding supplies an unused role.

### Compiler Interface

```python
class ConceptGraphCompiler(Protocol):
    def compile(
        self,
        request: InterventionRequest,
        bindings: SampleBindings,
        registry: ConceptRegistry,
    ) -> CompiledCCIPlan:
        ...
```

Version 1 implements `JsonConceptGraphCompiler`. A later
`TextConceptGraphCompiler` may parse `"remove smiling"`, retrieve a validated
graph template, bind its target value, and resolve evaluator and mask adapters.

The text layer may propose unknown nodes, but graph validation must reject them
unless they exist in the registry. Text generation never silently creates a
causal edge. New causal edges require a reviewed graph template or supporting
data analysis.

## Clean Prediction

At a selected reverse step, let:

```text
z_t       = current noisy latent
epsilon_t = detached CFG noise prediction
alpha_t   = scheduler cumulative alpha
```

For an epsilon-prediction DDIM scheduler:

```text
z0_hat = (z_t - sqrt(1 - alpha_t) * epsilon_t) / sqrt(alpha_t)
x0_hat = VAE.decode(z0_hat / 0.18215)
```

`epsilon_t` is detached. Gradients pass through the affine clean estimate, VAE
decoder, and evaluator models to `z_t`, but not through the U-Net. This keeps
the memory cost practical on MPS.

The implementation must use the scheduler's `prediction_type` and support
`epsilon` and `v_prediction`. It rejects direct `sample` prediction because a
detached model output would make `z0_hat` independent of `z_t`; supporting it
would require U-Net backpropagation, which is outside this design. It also
fails clearly for an unknown prediction type.

## Counterfactual Validity

Let `f(x0_hat)` be the target classifier logit and `y_star` be 0 or 1. Define
the signed target logit:

```text
s_t = (2 * y_star - 1) * f(x0_hat)
kappa = log(p_goal / (1 - p_goal))
```

The target is successful when `s_t >= kappa`. Use the non-saturating margin
loss:

```text
L_target = max(0, kappa - s_t)
```

This avoids the near-zero gradients that can occur with an already saturated
binary cross-entropy and prevents continued pressure after the requested
confidence is reached.

Target activation is recorded as:

```text
a_t = clamp((kappa - s_t) / max(abs(kappa), 1), 0, 1)
```

When `a_t == 0`, target guidance is disabled for that step. If the margin stays
satisfied for two selected steps, CCI target guidance remains off unless the
margin later falls below `kappa`.

## Constraints And Automatic Weights

Each constraint returns a non-negative measurement `d_k(x0_hat)` and a fixed
tolerance `epsilon_k`. Its normalized residual is:

```text
r_k = d_k / epsilon_k - 1
v_k = max(0, r_k)
```

The controller maintains one non-negative multiplier per constraint:

```text
lambda_k(t+1) = clip(
    lambda_k(t) + dual_rate * r_k,
    0,
    lambda_max
)
```

Weights therefore decrease when a constraint is comfortably satisfied and
increase when it is violated. All multipliers start at zero. An augmented term
causes a newly violated constraint to act immediately:

```text
L_constraint_k = lambda_k * v_k + 0.5 * penalty * v_k^2
```

There are no manual `identity=0.7`, `locality=1.3`, or similar task-specific
weights.

### Version 1 Evaluators

- `celeba_attribute`: differentiable target or preserved-attribute logit.
- `outside_l1`: mean absolute clean-image residual outside the generation mask,
  measured against the cached VAE reconstruction of the source.
- `masked_residual_tv`: total variation of the edit residual inside the
  generation mask.
- `facenet_identity`: cosine distance between frozen VGGFace2 FaceNet embeddings
  of source and predicted clean faces.
- `clip_image_audit`: optional source-image similarity for audit only; it must
  not be reported as face identity.

The identity evaluator uses a fixed source face crop for both images. The crop
is detected once from the source and then reused, avoiding a non-differentiable
detector inside every denoising step. If the FaceNet dependency or checkpoint
is unavailable and the graph requires identity, execution fails instead of
silently substituting another metric.

FaceNet support is a lazy optional dependency. Add an `identity` project extra
containing `facenet-pytorch`, and add an explicit download command that stores
the VGGFace2 checkpoint under `models/`. Generation never downloads weights
implicitly. The audit records the checkpoint digest and package version.

## Gradient Composition

Compute target and active constraint gradients separately with respect to
`z_t`. Let `g_target = grad(L_target)` and `g_k = grad(v_k)`. For each gradient,
maintain a per-run exponential moving average of its norm:

```text
n_k(t) = beta * n_k(t-1) + (1 - beta) * ||g_k||
gbar_k = g_k / max(n_k(t), gradient_floor)
```

The floor prevents a tiny unreliable gradient from being inflated to unit norm.
Only violated constraints contribute. The augmented-Lagrangian coefficient is:

```text
w_k = lambda_k + penalty * v_k
g_constraint = sum_k w_k * gbar_k
g_update = a_t * gbar_target + g_constraint
```

Target validity has priority. If an active constraint direction would make the
first-order target loss worse, remove only its target-opposing component:

```text
if dot(gbar_target, g_constraint) < 0:
    g_constraint = g_constraint
                 - dot(g_constraint, gbar_target)
                   / ||gbar_target||^2 * gbar_target
```

Skip target-priority projection when the target is inactive or its raw gradient
norm is below `gradient_floor`. In that case, active constraints may still
repair preservation violations within the same trust radius.

Form the masked update and clip that complete update to the latent trust radius:

```text
delta_t = eta_t * M_latent * g_update
delta_t = delta_t * min(1, trust_radius / max(||delta_t||, gradient_floor))
```

The trace records every pre-projection cosine, projection decision, raw norm,
normalized norm, pre-clip update norm, and final update norm.

The target-priority projection does not declare an infeasible sample valid. A
final candidate succeeds only if both the target margin and all hard audit
constraints pass.

## Noise-Prediction Update

CCI acts through the existing pre-scheduler noise hook rather than injecting a
latent after the scheduler step:

```text
epsilon_cci = epsilon_cfg
            + sqrt(1 - alpha_t) * delta_t

z_(t-1) = DDIM.step(epsilon_cci, t, z_t)
```

`M_latent` is the feathered generation mask at latent resolution. The existing
BLD blend runs afterward and restores the noised source outside the generation
region.

This makes the denoiser participate in realizing the intervention and gives the
remaining reverse steps an opportunity to repair texture.

## Guidance Schedule

The schedule is defined by normalized denoising progress rather than absolute
step indices so that 35-step and 70-step runs are comparable:

```text
progress = selected_reverse_index / max(selected_reverse_steps - 1, 1)
active when 0.15 <= progress <= 0.65
```

Within the active interval, use a smooth bell gate:

```text
u = (progress - start) / (end - start)
eta_t = step_scale * sin(pi * u)^2
```

The confidence activation `a_t` further reduces target pressure near success.
No CCI gradient is applied during the final texture-refinement interval.

## Mask Policy

Keep three distinct spatial objects:

- `audit_mask`: original hard mask used for strict metrics only.
- `semantic_mask`: hard union of allowed anatomical components.
- `generation_mask`: feathered semantic union used for guidance and soft BLD.

The masks are fixed for a compiled plan. This differs from MaskDiME's adaptive
top-k gradient masks and keeps the causal question explicit: intervene in the
known semantic region while adapting constraint strength.

Target gradients use the generation mask. Boundary and outside constraints use
their own measurement masks before differentiation; their resulting latent
gradients are restricted to the generation region plus its feathered boundary.
The audit denominator never changes with the generation mask.

## Runtime Trace And Weight Graphs

Write one JSON object per selected guidance step to `cci_trace.jsonl`:

```json
{
  "step": 8,
  "timestep": 684,
  "progress": 0.31,
  "target": {
    "logit": -0.42,
    "target_probability": 0.60,
    "required_probability": 0.80,
    "margin_residual": 0.98,
    "activation": 0.71,
    "gradient_norm": 0.014
  },
  "constraints": {
    "identity": {
      "value": 0.04,
      "tolerance": 0.08,
      "residual": -0.50,
      "lambda": 0.0
    },
    "outside_locality": {
      "value": 0.026,
      "tolerance": 0.02,
      "residual": 0.30,
      "lambda": 0.18
    }
  },
  "update": {
    "eta": 0.12,
    "projected": false,
    "norm": 0.10
  }
}
```

Add `scripts/plot_cci_trace.py` to produce:

- target probability and required margin by step;
- every `lambda_k(t)` by step;
- normalized constraint residuals by step;
- gradient norms and target/constraint cosine similarities;
- actual update norm and schedule gate.

The JSONL and CSV exports are authoritative. PNG plots are derived artifacts.
The plotting script must not be required to run generation.

## Source Boundaries

Keep the implementation out of the already-large runner where possible:

- `src/cci_diff/concept_graph.py`
  - graph and sample-binding dataclasses, JSON parsing, DAG and role validation.
- `src/cci_diff/concept_registry.py`
  - concept, evaluator, and mask-component registrations.
- `src/cci_diff/compilers/json_graph.py`
  - `JsonConceptGraphCompiler` and `CompiledCCIPlan` construction.
- `src/cci_diff/constraints.py`
  - evaluator protocol and locality, TV, attribute, and identity adapters.
- `src/cci_diff/adapters/sd2_clean_cci.py`
  - scheduler-aware clean prediction, controller state, dual updates, gradient
    projection, trust clipping, and trace records.
- `src/cci_diff/sd2_bld_backend.py`
  - expose predicted-clean guidance context before `scheduler.step`; preserve
    all existing hook behavior.
- `scripts/run_sd2_bld_cci.py`
  - CLI wiring, model loading, plan compilation, audit assembly.
- `scripts/plot_cci_trace.py`
  - trace-to-CSV/PNG reporting.
- `scripts/download_identity_model.py`
  - explicit FaceNet VGGFace2 checkpoint acquisition and digest reporting.
- `examples/graphs/remove_smile_clean_cci.json`
- `examples/graphs/blond_hair_clean_cci.json`
- `examples/bindings/sample_0_mouth.json`

The future text compiler implements the same compiler protocol in a separate
module. It must not be imported by the MPS generation path unless selected.

## CLI

Add an opt-in mode without changing existing commands:

```text
--cci_hook clean_constraint
--cci_graph examples/graphs/remove_smile_clean_cci.json
--cci_sample_bindings examples/bindings/sample_0_mouth.json
--classifier_path models/resnet50_multilabel_model.pth
--identity_model_path models/facenet_vggface2.pt
--cci_trace outputs/.../cci_trace.jsonl
```

Graph values are the single source of truth for target probability, semantic
mask roles, constraints, tolerances, and controller settings. Sample bindings
are the single source of truth for the source image and role-to-file mapping.
Duplicating those values in CLI flags is an error rather than an override.

The audit records:

- graph path and SHA-256 digest;
- compiled graph with resolved evaluator versions;
- model checkpoint paths and digests;
- source and final evaluator measurements;
- final feasibility decision and failed constraints;
- trace and plot paths;
- runtime and peak MPS allocation when available.

## Error Handling

Reject the run before model generation when:

- the graph version is unsupported;
- the graph is cyclic;
- the target is missing or has a non-binary desired value;
- the target probability is not strictly between 0.5 and 1.0;
- a node is both allowed to change and required to remain invariant;
- an evaluator or concept is absent from the registry;
- a required checkpoint or mask is missing;
- a tolerance is non-positive;
- mask components have mismatched dimensions or an empty union;
- controller bounds are invalid.

During generation:

- skip a guidance step and trace the reason when any loss or gradient is
  non-finite;
- abort after two consecutive non-finite selected steps;
- mark `unreliable_target_gradient` when the target margin is unmet but its raw
  gradient remains below the floor for two selected steps;
- never relabel an infeasible final image as successful because its weighted
  aggregate score is high.

## Testing

### Unit Tests

- Parse and round-trip the version 1 graph.
- Reject cycles, role conflicts, unknown concepts, and invalid thresholds.
- Compile identical plans from equivalent graph JSON.
- Verify clean prediction for epsilon and v-prediction schedulers.
- Verify direct-sample prediction is rejected without U-Net backpropagation.
- Verify target margin direction for desired values 0 and 1.
- Verify a satisfied target produces zero target activation.
- Verify dual weights increase on violation and decrease on satisfaction.
- Verify augmented constraints act on the first violated step.
- Verify target-priority projection preserves a target-descent direction.
- Verify trust clipping bounds each update.
- Verify zero and tiny gradients do not become NaN or full-strength updates.
- Verify target and constraint masks are distinct and correctly applied.
- Verify traces contain all required fields and serialize deterministically.
- Verify old hooks remain behaviorally unchanged.

Use fake VAE, classifier, identity, scheduler, and constraint adapters for unit
tests. No network access or large checkpoint is allowed in the unit suite.

### Local Integration Tests

- Load the existing CelebA ResNet50 checkpoint in float32.
- Load the identity checkpoint in float32 and keep parameters frozen.
- Run a tiny CPU clean-guidance step and verify gradients reach the latent but
  not evaluator parameters.
- Run one 35-step MPS sample for remove-smile and one for blond-hair.
- Verify all JSONL records parse and all plot series have matching step counts.

### Experimental Ablation

Use identical images, masks, prompts, seeds, and BLD settings:

```text
A0: BLD, no hook
A1: current noisy-latent fixed-weight CCI
A2: predicted-clean fixed-weight CCI
A3: predicted-clean constraint-feedback CCI
A4: A3 without target-priority projection
```

Run the 15-image smile/hair pilot first. Only after mechanism and pilot checks
pass should the final evaluation expand to at least 50 images per task and four
seeds per image, generated sequentially on MPS.

## Metrics

Report paired metrics, not only selected examples:

- target-model flip rate and signed margin;
- independent semantic agreement where an oracle is available;
- FaceNet identity cosine similarity;
- strict audit-mask inside MAE;
- semantic-mask inside MAE;
- outside-mask MAE and perceptual distance;
- non-target CelebA attribute drift excluding `may_affect` nodes;
- residual TV and boundary discontinuity;
- candidate feasibility rate;
- runtime and peak memory;
- blinded artifact review for the pilot grid.

The target classifier determines counterfactual validity. Independent metrics
detect classifier shortcuts; they do not replace the target decision.

## Success Criteria

Mechanism checks:

- In-loop target measurements are made on `x0_hat` and are numerically
  consistent with progressively cleaner final predictions.
- An unmet target no longer produces a near-zero remove-target loss solely
  because the latent is noisy.
- Every generated weight has a recorded constraint residual explaining it.

Fifteen-image pilot:

- Remove-smile A3 reaches at least 8/15 valid flips, compared with the current
  2/15 CCI result.
- Blond-hair A3 reaches at least the current 11/15 valid flips.
- Mean outside-mask MAE is no more than 10 percent above the paired no-hook
  result.
- Mean FaceNet identity cosine similarity is at least 0.90 and no more than
  0.02 below the paired no-hook result.
- At least 12/15 outputs per task pass blinded artifact review.
- A3 improves flip rate over A2 or achieves equal flip rate with lower identity
  or locality violation. Otherwise adaptive weighting is not supported by the
  experiment and must be reported as a negative result.

Efficiency:

- Batch-size-1 peak MPS memory stays below the device limit without allocator
  fallback.
- Median one-seed runtime is no more than three times the paired no-hook run.

## Future Text-To-Graph Layer

The future layer accepts an intervention such as:

```text
"make blond hair false while preserving identity and age"
```

It performs:

1. concept and desired-value parsing;
2. retrieval of a reviewed concept-graph template;
3. allowed-change and invariant-node binding;
4. evaluator and semantic-mask resolution through the registry;
5. validation and compilation to the same `CompiledCCIPlan`.

It does not emit arbitrary runtime weights. The online controller continues to
produce those from observed violations. This keeps generated plans inspectable
and makes text parsing errors separable from optimization errors.

## Delivery Sequence

1. Implement graph schema, validation, and JSON compiler.
2. Add explicit FaceNet checkpoint acquisition and the frozen identity adapter.
3. Implement clean prediction and target-margin unit tests.
4. Implement constraint adapters and automatic dual controller.
5. Integrate the pre-scheduler hook without changing legacy modes.
6. Add trace export and plotting.
7. Add remove-smile and blond-hair graphs plus per-image bindings.
8. Run focused CPU tests, the full unit suite, and shell validation.
9. Run one-image MPS mechanism checks.
10. Run the paired 15-image A0-A4 pilot sequentially.
11. Decide from the stated criteria whether A3 is supported before scaling the
    experiment.

## Implementation Compatibility Note

`facenet-pytorch==2.6.0` pins torch `<2.3`, torchvision `<0.18`, NumPy `<2`,
and Pillow `<10.3`. Those constraints conflict with the main `.venv-ml`
environment, which currently uses torch 2.13, torchvision 0.28, NumPy 2.2,
and Pillow 12.3. This does not change the identity-constraint algorithm.
Acquisition and export run in `.venv-facenet-export`; production runtime loads
the resulting VGGFace2 TorchScript model and uses the already-installed OpenCV
detector.
