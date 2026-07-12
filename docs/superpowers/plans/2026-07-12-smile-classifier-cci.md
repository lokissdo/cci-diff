# Smile Classifier CCI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a frozen CelebA ResNet50 `Smiling` classifier loss to the SD2 latent CCI hook and compare same-seed no-hook and classifier-hook runs that remove the smile from `data/0.jpg`.

**Architecture:** Add a self-contained, lazily imported classifier adapter that recreates the thesis checkpoint architecture without training dependencies. The SD2 runner will decode each selected latent, evaluate the `Smiling` logit, compose classifier and outside-mask losses, and use the existing masked latent-gradient adapter before BLD blending. A dedicated shell runner will produce repeatable baseline and guided outputs with classifier scores in each audit.

**Tech Stack:** Python 3.10, PyTorch, torchvision ResNet50, diffusers SD2, Pillow, unittest, Bash, Apple MPS float32

## Global Constraints

- Use `models/resnet50_multilabel_model.pth`; do not retrain or download classifier weights.
- Use only CelebA `Smiling`, zero-based index `31`, with `desired_value: 0`.
- Keep classifier parameters frozen while preserving gradients to decoded images and latents.
- Construct torchvision ResNet50 with `weights=None`; loading must not access the network or `~/.cache/torch`.
- Keep classifier and decoded tensors in float32 on MPS.
- Map only `classifier` and `outside_mask` to non-zero CCI loss terms for this hook.
- Keep `latent_color` and `none` behavior backward compatible.
- Keep CLIP guidance deferred to the experiment documented in the design spec.
- Preserve user changes already present in the dirty worktree; stage only files owned by each task.

---

### Task 1: CelebA ResNet50 Classifier Adapter

**Files:**
- Create: `src/cci_diff/classifiers/__init__.py`
- Create: `src/cci_diff/classifiers/celeba_resnet50.py`
- Create: `tests/test_celeba_resnet50.py`

**Interfaces:**
- Consumes: checkpoint state dictionaries created by `ResNet50MultiLabel` in the thesis training repository.
- Produces: `CELEBA_ATTRIBUTES`, `resolve_celeba_attribute_index(concept: str) -> int`, `load_celeba_resnet50(path, *, device, dtype) -> Any`, `preprocess_classifier_images(images, *, size: int) -> Any`, `classifier_logits(model, images, *, size: int) -> Any`, and `classifier_probabilities(model, images, *, size: int) -> Any`.

- [ ] **Step 1: Write failing attribute-resolution and preprocessing tests**

```python
class TestCelebAResNet50(unittest.TestCase):
    def test_smile_resolves_to_canonical_celeba_index(self):
        from cci_diff.classifiers.celeba_resnet50 import (
            resolve_celeba_attribute_index,
        )

        self.assertEqual(resolve_celeba_attribute_index("smile"), 31)
        self.assertEqual(resolve_celeba_attribute_index("Smiling"), 31)
        with self.assertRaises(ValueError):
            resolve_celeba_attribute_index("unknown concept")

    def test_preprocessing_is_differentiable_and_normalized(self):
        import torch
        from cci_diff.classifiers.celeba_resnet50 import (
            preprocess_classifier_images,
        )

        image = torch.full((1, 3, 8, 8), 0.5, requires_grad=True)
        result = preprocess_classifier_images(image, size=16)
        result.sum().backward()

        self.assertEqual(tuple(result.shape), (1, 3, 16, 16))
        self.assertIsNotNone(image.grad)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv-ml/bin/python -m unittest tests.test_celeba_resnet50 -v`

Expected: FAIL because `cci_diff.classifiers.celeba_resnet50` does not exist.

- [ ] **Step 3: Implement the lazy classifier adapter**

Use the canonical 40 CelebA attributes and these core implementations:

```python
def resolve_celeba_attribute_index(concept: str) -> int:
    normalized = concept.casefold().replace("_", " ").strip()
    aliases = {"smile": "smiling"}
    normalized = aliases.get(normalized, normalized)
    for index, attribute in enumerate(CELEBA_ATTRIBUTES):
        if attribute.casefold().replace("_", " ") == normalized:
            return index
    raise ValueError(f"No CelebA classifier output for {concept!r}")


def preprocess_classifier_images(images, *, size: int):
    import torch
    import torch.nn.functional as functional

    if size <= 0:
        raise ValueError("classifier input size must be positive")
    images = functional.interpolate(
        images.float(), size=(size, size), mode="bilinear", align_corners=False
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std
```

`load_celeba_resnet50` must define the exact `base_model`, `fc1`, `bn1`,
`relu1`, `drop1`, `fc2`, `bn2`, `relu2`, and `fc3` names expected by the
checkpoint, expose logits before sigmoid, load with `weights_only=True`, call
`eval()`, call `requires_grad_(False)`, and move to the requested device/dtype.

- [ ] **Step 4: Add a checkpoint-shape integration test**

```python
def test_local_checkpoint_loads_without_pretrained_download(self):
    from pathlib import Path
    import torch
    from cci_diff.classifiers.celeba_resnet50 import load_celeba_resnet50

    path = Path("models/resnet50_multilabel_model.pth")
    if not path.exists():
        self.skipTest("local classifier checkpoint is not available")
    model = load_celeba_resnet50(path, device="cpu", dtype=torch.float32)
    self.assertFalse(model.training)
    self.assertEqual(model.fc3.out_features, 40)
    self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
```

- [ ] **Step 5: Run adapter tests and verify GREEN**

Run: `.venv-ml/bin/python -m unittest tests.test_celeba_resnet50 -v`

Expected: all tests PASS without network or pandas imports.

- [ ] **Step 6: Commit the adapter**

```bash
git add src/cci_diff/classifiers/__init__.py \
  src/cci_diff/classifiers/celeba_resnet50.py \
  tests/test_celeba_resnet50.py
git commit -m "feat: add CelebA ResNet50 classifier adapter"
```

### Task 2: Single-Attribute Latent Classifier Guidance

**Files:**
- Modify: `scripts/run_sd2_bld_cci.py:17-194`
- Modify: `tests/test_sd2_bld_cli.py`
- Create: `tests/test_smile_classifier_hook.py`

**Interfaces:**
- Consumes: classifier adapter functions from Task 1, `CCIConfig`, `SD2DenoisingStep`, and `apply_cci_latent_guidance`.
- Produces: CLI hook `latent_classifier` and `build_cci_latent_guidance_hook(...)` behavior that returns a masked semantic latent update.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_parser_accepts_latent_classifier_options(self):
    args = build_arg_parser().parse_args([
        "--cci_config", "examples/remove_smile_intervention.json",
        "--init_image", "data/0.jpg",
        "--mask", "data/00000_mouth.png",
        "--output_dir", "outputs/remove_smile",
        "--cci_hook", "latent_classifier",
        "--classifier_path", "models/resnet50_multilabel_model.pth",
        "--classifier_label_index", "31",
        "--classifier_input_size", "512",
    ])
    self.assertEqual(args.cci_hook, "latent_classifier")
    self.assertEqual(args.classifier_label_index, 31)
    self.assertEqual(args.classifier_input_size, 512)
```

- [ ] **Step 2: Run the CLI test and verify RED**

Run: `.venv-ml/bin/python -m unittest tests.test_sd2_bld_cli -v`

Expected: FAIL because `latent_classifier` and its arguments are not recognized.

- [ ] **Step 3: Add CLI arguments and hook-specific validation**

Add `latent_classifier` to `--cci_hook`, then add:

```python
parser.add_argument("--classifier_path", default=None)
parser.add_argument("--classifier_label_index", type=int, default=None)
parser.add_argument("--classifier_input_size", type=int, default=512)
```

Validate that classifier hook runs have an existing checkpoint, input size is
positive, and explicit indices are in `[0, 39]`. Resolve the concept through
`resolve_celeba_attribute_index` when no index is supplied. Keep classifier
loading entirely outside the per-step closure.

- [ ] **Step 4: Write a failing semantic-gradient test with a tiny classifier**

```python
def test_remove_smile_loss_updates_only_masked_latent(self):
    import torch
    from torch import nn
    from cci_diff.adapters.sd2_cci import apply_cci_latent_guidance
    from cci_diff.guidance import GuidanceTerms
    from cci_diff.spec import GuidanceWeights

    classifier = nn.Sequential(nn.Flatten(), nn.Linear(2, 1, bias=False))
    classifier[1].weight.data.fill_(1.0)
    classifier.requires_grad_(False)
    latents = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0]])

    def loss_fn(decoded):
        logit = classifier(decoded).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, torch.zeros_like(logit)
        )
        zero = loss * 0.0
        return GuidanceTerms(zero, zero, zero, loss, zero)

    guided = apply_cci_latent_guidance(
        latents,
        decode_fn=lambda value: value,
        loss_fn=loss_fn,
        weights=GuidanceWeights(
            target=0, preservation=0, leakage=0, classifier=1, outside_mask=0
        ),
        step_size=0.1,
        latent_mask=mask,
    )
    self.assertLess(guided[0, 0], latents[0, 0])
    self.assertEqual(guided[0, 1], latents[0, 1])
```

- [ ] **Step 5: Implement classifier loss composition in the hook**

For `latent_classifier`, calculate:

```python
logits = classifier_logits(classifier, decoded, size=args.classifier_input_size)
target = torch.full_like(logits[:, label_index], config.intervention.desired_value)
classifier_loss = functional.binary_cross_entropy_with_logits(
    logits[:, label_index], target
)
outside_delta = _masked_mse(decoded, source_image, outside_mask)
zero = classifier_loss * 0.0
return GuidanceTerms(
    target=zero,
    preservation=zero,
    leakage=zero,
    classifier=classifier_loss,
    outside_mask=outside_delta,
)
```

Preserve the existing color-hook loss unchanged in its own branch. Pass the
latent mouth mask and normalization settings through the existing
`apply_cci_latent_guidance` call.

- [ ] **Step 6: Run hook and existing CLI/adapter tests**

Run: `.venv-ml/bin/python -m unittest tests.test_smile_classifier_hook tests.test_sd2_bld_cli tests.test_sd2_adapter_contract -v`

Expected: all tests PASS.

- [ ] **Step 7: Commit classifier hook integration**

```bash
git add scripts/run_sd2_bld_cci.py \
  tests/test_sd2_bld_cli.py \
  tests/test_smile_classifier_hook.py
git commit -m "feat: guide SD2 latents with smile classifier"
```

### Task 3: Remove-Smile Configuration And Audit Scores

**Files:**
- Create: `examples/remove_smile_intervention.json`
- Modify: `scripts/run_sd2_bld_cci.py:209-262`
- Modify: `tests/test_smile_classifier_hook.py`

**Interfaces:**
- Consumes: the classifier instance and resolved label metadata created while building the hook.
- Produces: a remove-smile CCI configuration and `audit.json` classifier metadata with source/output probabilities.

- [ ] **Step 1: Add the remove-smile configuration**

```json
{
  "target_concept": "smile",
  "desired_value": 0,
  "preserved_concepts": [
    "identity",
    "hair",
    "age-like appearance",
    "gender presentation"
  ],
  "candidate_concepts": [
    "smile",
    "identity",
    "hair",
    "age-like appearance",
    "gender presentation",
    "makeup",
    "eyeglasses"
  ],
  "weights": {
    "target": 0.0,
    "preservation": 0.0,
    "leakage": 0.0,
    "classifier": 1.0,
    "outside_mask": 1.0
  }
}
```

- [ ] **Step 2: Write a failing audit-metadata test**

Extract a small pure helper, `classifier_audit_metadata(args, config, runtime)`,
and test that classifier runs return:

```python
self.assertEqual(metadata["attribute"], "Smiling")
self.assertEqual(metadata["label_index"], 31)
self.assertEqual(metadata["desired_value"], 0)
self.assertEqual(metadata["input_size"], 512)
self.assertIsNone(metadata["target_rgb"])
```

- [ ] **Step 3: Run the audit test and verify RED**

Run: `.venv-ml/bin/python -m unittest tests.test_smile_classifier_hook -v`

Expected: FAIL because classifier audit metadata is not implemented.

- [ ] **Step 4: Implement source/output scoring and audit metadata**

Return both the hook and a small runtime object containing classifier, label
index, and attribute. After generation, load the source and output through
differentiable preprocessing under `torch.no_grad()`, compute sigmoid
probabilities, and record:

```json
{
  "classifier": {
    "path": "models/resnet50_multilabel_model.pth",
    "attribute": "Smiling",
    "label_index": 31,
    "desired_value": 0,
    "input_size": 512,
    "source_probability": 0.999687,
    "output_probabilities": [0.0]
  }
}
```

For batch grids, split the output image into `args.batch_size` crops of
`args.width` pixels before scoring. For `none` and `latent_color`, set the
classifier audit field to `null` and preserve the current target RGB audit.

- [ ] **Step 5: Run focused and full unit tests**

Run: `.venv-ml/bin/python -m unittest tests.test_smile_classifier_hook tests.test_sd2_bld_cli -v`

Run: `.venv-ml/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS; optional dependency tests may skip only when their
documented dependency is absent.

- [ ] **Step 6: Commit config and audit support**

```bash
git add examples/remove_smile_intervention.json \
  scripts/run_sd2_bld_cci.py \
  tests/test_smile_classifier_hook.py
git commit -m "feat: audit smile classifier guidance"
```

### Task 4: Reproducible MPS A/B Experiment

**Files:**
- Create: `scripts/run_remove_smile_classifier_mps.sh`
- Modify: `README.md`
- Runtime output: `outputs/sample_0_sd2_bld_remove_smile_no_hook_mps/`
- Runtime output: `outputs/sample_0_sd2_bld_remove_smile_classifier_mps/`

**Interfaces:**
- Consumes: the Task 2 hook, Task 3 configuration, `data/0.jpg`, `data/00000_mouth.png`, local SD2 checkpoint, and classifier checkpoint.
- Produces: same-seed no-hook and classifier-guided images plus auditable score comparison.

- [ ] **Step 1: Create the executable A/B runner**

The script must use `set -euo pipefail`, resolve the project root, and execute:

```bash
COMMON_ENV=(
  CCI_CONFIG=examples/remove_smile_intervention.json
  INIT_IMAGE=data/0.jpg
  MASK=data/00000_mouth.png
  AUTO_MASK=0
  NUM_INFERENCE_STEPS=35
  SEED=42
  BATCH_SIZE=1
)

env "${COMMON_ENV[@]}" \
  OUTPUT_DIR=outputs/sample_0_sd2_bld_remove_smile_no_hook_mps \
  ./scripts/run_sd2_bld_mps.sh --cci_hook none

env "${COMMON_ENV[@]}" \
  OUTPUT_DIR=outputs/sample_0_sd2_bld_remove_smile_classifier_mps \
  ./scripts/run_sd2_bld_mps.sh \
    --cci_hook latent_classifier \
    --classifier_path models/resnet50_multilabel_model.pth \
    --classifier_label_index 31 \
    --classifier_input_size 512 \
    --cci_step_size 0.5 \
    --cci_every_n_steps 2 \
    --cci_normalize_grad
```

- [ ] **Step 2: Document the experiment command and outputs**

Add a short README section pointing to
`scripts/run_remove_smile_classifier_mps.sh`, stating that it removes a smile
with CelebA index 31 and that CLIP guidance remains a later experiment.

- [ ] **Step 3: Verify shell syntax and run unit tests**

Run: `bash -n scripts/run_remove_smile_classifier_mps.sh`

Run: `.venv-ml/bin/python -m unittest discover -s tests -v`

Expected: shell syntax exit 0 and all unit tests PASS.

- [ ] **Step 4: Run the MPS A/B experiment**

Run: `./scripts/run_remove_smile_classifier_mps.sh`

Expected: both output folders contain `sd2_bld_grid.png` and `audit.json`; the
guided audit contains `cci_latent_guidance` states and classifier metadata.

- [ ] **Step 5: Inspect and quantify both outputs**

Check both images visually. Report source, baseline, and guided smile
probabilities, inside/outside mask MAE, and any mouth artifacts. The guided
result succeeds only if it lowers the smile probability relative to the same-seed
baseline without unacceptable mouth corruption.

If the score does not move, rerun only the guided case with
`--cci_step_size 1.0`; do not change seed, prompt, schedule, and step size at the
same time. If the score moves but visual quality degrades, retain the first run
as evidence and report that classifier optimization found an adversarial visual
shortcut.

- [ ] **Step 6: Commit the reproducible runner and documentation**

```bash
git add scripts/run_remove_smile_classifier_mps.sh README.md
git commit -m "docs: add remove-smile classifier experiment"
```

## Final Verification

- [ ] Run `.venv-ml/bin/python -m unittest discover -s tests -v` and record the exact pass/skip/failure counts.
- [ ] Run `bash -n scripts/run_remove_smile_classifier_mps.sh`.
- [ ] Confirm the classifier checkpoint loads without network access.
- [ ] Confirm no-hook, latent-color, and latent-classifier CLI modes parse.
- [ ] Confirm the source image scores near the previously measured `0.999687` for `Smiling`.
- [ ] Confirm both output audits and images exist and are readable.
- [ ] Compare the generated images visually and report artifacts honestly.
- [ ] Confirm the CLIP experiment remains documentation-only.
