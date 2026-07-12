# Smile Classifier CCI Design

## Goal

Add a differentiable CelebA ResNet50 classifier hook to the SD2 blended-latent
denoising loop, then use it to remove the smile from `data/0.jpg` inside
`data/00000_mouth.png`.

The experiment compares no-hook BLD and classifier-guided BLD with the same
source, mask, prompt, seed, scheduler settings, and number of steps. The source
already has a ResNet50 `Smiling` probability of `0.999687`, so the intervention
uses `desired_value: 0`.

## Non-Goals

- Do not use the existing RGB color prior for expression editing.
- Do not combine `Smiling` with `Mouth_Slightly_Open` in the first experiment.
- Do not retrain or fine-tune the supplied classifier checkpoint.
- Do not implement CLIP guidance in this pass.
- Do not change the soft-mask design or make it a dependency of this hook.

## Approach

Use the supplied `models/resnet50_multilabel_model.pth` checkpoint as a frozen,
differentiable image classifier. The model has the exact 40-label head produced
by the thesis ResNet50 training code. In canonical CelebA attribute order,
`Smiling` is zero-based output index `31`.

For each selected denoising step:

1. Decode the current latent without detaching it.
2. Resize and ImageNet-normalize the decoded image using differentiable torch
   operations.
3. Compute the ResNet50 `Smiling` logit.
4. Compute binary cross-entropy against the intervention's desired value.
5. Add outside-mask source preservation loss.
6. Differentiate the weighted loss with respect to the current latent.
7. Restrict the gradient to the latent mouth mask, optionally normalize it, and
   update the latent before BLD restores the source outside the mask.

## Classifier Component

Add a self-contained classifier module under `src/cci_diff/classifiers`. It
recreates the checkpoint architecture without importing the thesis training
module, pandas, dataset code, or training utilities.

The module will:

- construct torchvision ResNet50 with `weights=None`, because the local
  checkpoint already contains all backbone weights;
- recreate the `2048 -> 1024 -> 512 -> 40` classifier head and matching batch
  normalization/dropout layers;
- load the state dictionary with `weights_only=True` where supported;
- expose logits before sigmoid for numerically stable guidance;
- set evaluation mode and freeze every model parameter while preserving input
  gradients;
- provide the canonical CelebA attribute names and resolve `smile`/`smiling` to
  index `31`.

The implementation must not use `torch.no_grad()` around classifier inference,
because gradients must flow from the classifier output through the decoded
image into the latent.

## Loss

Let:

- `z_t` be the current latent;
- `D(z_t)` be the VAE-decoded image in `[0, 1]`;
- `N` be differentiable resize and ImageNet normalization;
- `C_31` be the frozen ResNet50 `Smiling` logit;
- `y = 0` be the desired no-smile value;
- `m` be the latent mouth mask.

The classifier loss is:

```text
L_classifier = BCEWithLogits(C_31(N(D(z_t))), y)
```

The existing outside-region reconstruction loss remains:

```text
L_outside = MSE((1 - m_image) * D(z_t),
                (1 - m_image) * D(z_source))
```

The first implementation maps only real, independent terms into
`GuidanceTerms`:

```text
target       = 0
preservation = 0
leakage      = 0
classifier   = L_classifier
outside_mask = L_outside
```

Therefore:

```text
L_CCI = w_classifier * L_classifier
      + w_outside_mask * L_outside
```

The latent update is:

```text
g = m * gradient_z(L_CCI)
z'_t = z_t - eta * normalize(g)
```

Using distinct terms avoids the current color prototype's duplicate mapping of
one loss into several conceptual weight slots.

## CLI And Configuration

Extend the existing hook option with `latent_classifier` and add:

- `--classifier_path`
- `--classifier_label_index`, defaulting to automatic concept resolution
- `--classifier_input_size`, default `512`

The experiment gets a separate configuration such as
`examples/remove_smile_intervention.json` with `target_concept: "smile"` and
`desired_value: 0`. The existing smile and hair configurations remain unchanged.

The MPS wrapper will accept classifier settings through normal CLI arguments.
The first comparison should use normalized gradients and conservative guidance,
then adjust one variable at a time only if the classifier score does not move.

## Audit

Record the following in `audit.json`:

- hook type;
- classifier checkpoint path;
- classifier attribute and index;
- desired value;
- classifier preprocessing size;
- guidance schedule and step size;
- source and output `Smiling` probabilities;
- all configured CCI weights.

Classifier scores must be computed in evaluation mode with the same
preprocessing used by guidance.

## Error Handling

- Reject a missing classifier checkpoint before model generation starts.
- Reject classifier indices outside `[0, 39]`.
- Reject unsupported classifier target concepts when no explicit index is given.
- Report state-dictionary incompatibility with the checkpoint path and missing
  or unexpected keys.
- Keep the classifier and decoded tensors in float32 on MPS.
- If MPS runs out of memory, report the failing stage instead of silently moving
  only part of the graph to CPU.

## Testing

Add focused tests that do not load the 100 MB checkpoint unless explicitly
running an integration test:

- model structure accepts a checkpoint-shaped state dictionary;
- `smile` resolves to CelebA index `31`;
- preprocessing preserves input gradients and produces normalized tensors;
- desired value `0` creates a gradient that reduces a synthetic smile logit;
- classifier parameters remain frozen while latent gradients are produced;
- CLI parses classifier-hook arguments;
- audit records classifier metadata and before/after scores;
- color and no-hook behavior remains backward compatible.

Then run two full MPS generations with identical settings:

```text
no hook:             --cci_hook none
classifier guidance: --cci_hook latent_classifier --cci_normalize_grad
```

## Success Criteria

- The classifier-hook output has a lower `Smiling` probability than both the
  source and same-seed no-hook output.
- The visible mouth expression is less smiling without duplicated teeth or an
  incoherent mouth interior.
- Outside-mask change remains much lower than inside-mask change.
- The audit proves that the classifier hook executed during denoising.
- Any failure to produce a neutral expression is reported as an experimental
  result, not hidden by selecting a favorable seed.

## Limitations

The supplied mask covers the mouth but not smile-related cheeks or eye creases.
Classifier guidance may therefore lower the score by changing teeth or lips
without producing a fully neutral facial expression. A later ablation can
compare the strict mouth mask with a safely expanded soft generation mask while
retaining the original mask for auditing.

## Deferred CLIP Experiment

A later, separate experiment can add a CLIP image-text objective comparing
prompts such as `neutral expression` and `smiling expression`:

```text
L_CLIP = 1 - cosine(E_image(D(z_t)), E_text("neutral expression"))
```

That experiment should compare ResNet50-only, CLIP-only, and combined guidance
under identical seeds. It is deferred because CLIP guidance is broader and less
attribute-specific; mixing it into the first run would make it unclear whether
the observed change came from the trained CelebA smile classifier or from text
semantics.
