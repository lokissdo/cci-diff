# Robust Smile Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a mouth-plus-both-lips semantic mask, robust multi-scale smile loss, realism gradients, and middle-step scheduling can remove the smile from `data/0.jpg` without the tooth artifacts produced by hard-mouth classifier guidance.

**Architecture:** Build and save a hard semantic union plus a 3-pixel feathered operational mask. Extend SD2 BLD to use the feathered mask only for robust guidance and soft blending while preserving old hard-mask behavior. Add a separate robust latent adapter that normalizes smile, boundary, and TV gradients independently, then run one same-seed MPS comparison and audit both mouth-only and semantic-union regions.

**Tech Stack:** Python 3.10, PyTorch float32, torchvision, Pillow, NumPy, diffusers SD2, unittest, Bash, Apple MPS

## Global Constraints

- Use exactly `00000_mouth.png`, `00000_u_lip.png`, and `00000_l_lip.png` for the semantic union.
- Do not morphologically dilate the semantic union.
- Feather the union by 3 pixels only for generation; never use the feathered mask as an audit denominator.
- Keep the original mouth mask as the strict audit mask.
- Keep the existing `none`, `latent_color`, and hard `latent_classifier` paths backward compatible.
- Apply robust guidance only at steps `4, 6, 8, 10, 12, 14, 16`.
- Use deterministic classifier scales `256,384,512`, Gaussian kernel 5, sigma 1.0.
- Use weights `smile=1.0`, `boundary=0.3`, `tv=0.05`, base step size `0.20` with linear decay.
- Keep all robust guidance tensors in float32 on MPS.
- Do not add CLIP, MaskDiME, model downloads, commits, or changes outside `my-docs/cci-diff`.

---

### Task 1: Semantic Mask Union And Soft BLD

**Files:**
- Create: `src/cci_diff/masking.py`
- Modify: `src/cci_diff/sd2_bld_backend.py`
- Create: `tests/test_masking.py`
- Modify: `tests/test_sd2_bld_backend.py`

**Interfaces:**
- Consumes: aligned component-mask paths and a feather radius.
- Produces: `prepare_semantic_masks(component_paths, *, feather_radius, hard_output, soft_output) -> MaskArtifacts`, `read_mask_tensor(path, *, size, binary)`, and `blend_soft_latents(edited, generation_mask, source)`.

- [ ] **Step 1: Write failing semantic-mask tests**

```python
def test_semantic_union_ors_all_components_and_feathers_edges(self):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        mouth = root / "mouth.png"
        upper = root / "upper.png"
        lower = root / "lower.png"
        Image.new("L", (16, 16), 0).save(mouth)
        # Draw three non-overlapping white regions in the fixture masks.
        artifacts = prepare_semantic_masks(
            [mouth, upper, lower],
            feather_radius=3,
            hard_output=root / "semantic.png",
            soft_output=root / "generation.png",
        )
        hard = np.array(Image.open(artifacts.semantic_path))
        soft = np.array(Image.open(artifacts.generation_path))
        self.assertGreater((hard > 0).sum(), 0)
        self.assertTrue(((soft > 0) & (soft < 255)).any())
```

Also test missing paths, mismatched dimensions, and negative feather radius.

- [ ] **Step 2: Run mask tests and verify RED**

Run: `.venv-ml/bin/python -m unittest tests.test_masking -v`

Expected: FAIL because `cci_diff.masking` does not exist.

- [ ] **Step 3: Implement component union and feathering**

```python
@dataclass(frozen=True)
class MaskArtifacts:
    semantic_path: str
    generation_path: str
    semantic_fraction: float


def prepare_semantic_masks(component_paths, *, feather_radius, hard_output, soft_output):
    masks = [Image.open(path).convert("L") for path in component_paths]
    semantic = masks[0].point(lambda value: 255 if value >= 128 else 0)
    for mask in masks[1:]:
        semantic = ImageChops.lighter(
            semantic, mask.point(lambda value: 255 if value >= 128 else 0)
        )
    generation = semantic.filter(ImageFilter.GaussianBlur(feather_radius))
    semantic.save(hard_output)
    generation.save(soft_output)
    fraction = float((np.array(semantic) >= 128).mean())
    return MaskArtifacts(
        semantic_path=str(hard_output),
        generation_path=str(soft_output),
        semantic_fraction=fraction,
    )
```

- [ ] **Step 4: Write failing soft-blend tests**

```python
def test_soft_blend_interpolates_edited_and_source_latents(self):
    edited = torch.tensor([10.0, 10.0])
    source = torch.tensor([2.0, 2.0])
    mask = torch.tensor([0.0, 0.25])
    result = blend_soft_latents(edited, mask, source)
    self.assertTrue(torch.allclose(result, torch.tensor([2.0, 4.0])))
```

- [ ] **Step 5: Implement optional generation mask in the backend**

Extend `edit_image` with optional `generation_mask`. Read the original `mask`
as binary audit state. Read `generation_mask` without thresholding, repeat both
to the batch size, expose the generation tensor as `step.latent_mask`, and use:

```python
latents = generation_mask * latents + (1.0 - generation_mask) * noise_source_latents
```

only when an explicit generation mask is supplied. Keep the old `where` blend
for every existing command.

- [ ] **Step 6: Run focused and backend tests**

Run: `.venv-ml/bin/python -m unittest tests.test_masking tests.test_sd2_bld_backend -v`

Expected: all tests PASS.

### Task 2: Robust Multi-Term Latent Guidance

**Files:**
- Create: `src/cci_diff/adapters/sd2_robust.py`
- Create: `tests/test_sd2_robust_adapter.py`

**Interfaces:**
- Consumes: decoded images, source images, classifier, label index, semantic masks, boundary masks, and schedule settings.
- Produces: `multi_scale_classifier_loss`, `boundary_loss`, `residual_tv_loss`, `robust_step_size`, and `apply_robust_latent_guidance` returning `(updated_latents, stats)`.

- [ ] **Step 1: Write failing objective tests**

```python
def test_tv_penalizes_checkerboard_more_than_smooth_residual(self):
    smooth = torch.zeros((1, 3, 4, 4))
    checker = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]] * 2).float()
    checker = checker.view(1, 1, 4, 4).repeat(1, 3, 1, 1)
    mask = torch.ones((1, 1, 4, 4))
    self.assertGreater(residual_tv_loss(checker, smooth, mask), 0)

def test_schedule_decays_and_rejects_steps_outside_active_window(self):
    self.assertIsNone(robust_step_size(3, start=4, end=16, every=2, base=0.2))
    self.assertAlmostEqual(
        robust_step_size(4, start=4, end=16, every=2, base=0.2), 0.2
    )
    self.assertIsNotNone(
        robust_step_size(16, start=4, end=16, every=2, base=0.2)
    )
```

Add tests that boundary loss is zero for identical images and multi-scale loss
backpropagates through all requested deterministic views.

- [ ] **Step 2: Run objective tests and verify RED**

Run: `.venv-ml/bin/python -m unittest tests.test_sd2_robust_adapter -v`

Expected: FAIL because `cci_diff.adapters.sd2_robust` does not exist.

- [ ] **Step 3: Implement robust losses and schedule**

```python
def robust_step_size(step, *, start, end, every, base):
    if step < start or step > end or (step - start) % every:
        return None
    return base * (end - step + 1) / (end - start + 1)

def residual_tv_loss(decoded, source, mask):
    residual = mask * (decoded - source)
    vertical = (residual[:, :, 1:] - residual[:, :, :-1]).abs().mean()
    horizontal = (residual[:, :, :, 1:] - residual[:, :, :, :-1]).abs().mean()
    return vertical + horizontal
```

Use `torchvision.transforms.functional.gaussian_blur` for kernel 5/sigma 1.0,
resize each view to its requested scale, and average BCE-with-logits.

- [ ] **Step 4: Write failing separate-gradient test**

Test that a semantic loss updates a masked latent, a zero boundary term remains
finite, and returned stats contain each loss and raw masked gradient norm.

- [ ] **Step 5: Implement separate gradient normalization**

```python
for name, loss in terms.items():
    gradient = torch.autograd.grad(loss, guided_latents, retain_graph=True)[0]
    gradient = gradient * generation_mask
    norm = per_sample_norm(gradient)
    normalized = torch.where(norm > eps, gradient / norm.clamp_min(eps), 0.0)
    update = update + weights[name] * normalized
return (latents - step_size * update).detach(), stats
```

- [ ] **Step 6: Run robust adapter tests**

Run: `.venv-ml/bin/python -m unittest tests.test_sd2_robust_adapter -v`

Expected: all tests PASS without NaN values.

### Task 3: CLI, Hook, Audit, And Test Runner

**Files:**
- Modify: `scripts/run_sd2_bld_cci.py`
- Modify: `scripts/run_remove_smile_classifier_mps.sh`
- Modify: `tests/test_sd2_bld_cli.py`
- Modify: `tests/test_smile_classifier_hook.py`

**Interfaces:**
- Consumes: Task 1 mask artifacts and Task 2 robust adapter.
- Produces: opt-in robust CLI flags, robust hook runtime traces, saved masks, strict/semantic metrics, and a reproducible output folder.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_parser_accepts_robust_classifier_guidance_options(self):
    args = build_arg_parser().parse_args([
        "--cci_config", "examples/remove_smile_intervention.json",
        "--init_image", "data/0.jpg",
        "--mask", "data/00000_mouth.png",
        "--output_dir", "outputs/robust",
        "--cci_hook", "latent_classifier",
        "--classifier_path", "models/resnet50_multilabel_model.pth",
        "--robust_classifier_guidance",
        "--generation_mask_component", "data/00000_mouth.png",
        "--generation_mask_component", "data/00000_u_lip.png",
        "--generation_mask_component", "data/00000_l_lip.png",
        "--generation_mask_feather", "3",
        "--classifier_scales", "256,384,512",
        "--classifier_blur_sigma", "1.0",
        "--boundary_weight", "0.3",
        "--tv_weight", "0.05",
    ])
    self.assertTrue(args.robust_classifier_guidance)
    self.assertEqual(len(args.generation_mask_component), 3)
    self.assertEqual(args.generation_mask_feather, 3.0)
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `.venv-ml/bin/python -m unittest tests.test_sd2_bld_cli -v`

Expected: FAIL because robust flags are unknown.

- [ ] **Step 3: Add robust flags and prepare masks before backend execution**

Add the exact defaults from Global Constraints. When robust mode is enabled,
require exactly the three supplied semantic component paths, write
`semantic_mask.png` and `generation_mask.png` into the output directory, and
pass the generated path to the backend.

- [ ] **Step 4: Integrate robust hook and audit trace**

Use the cached decoded source, upsample generation/semantic masks to image
resolution, build the robust terms, apply the decayed robust step, and append
losses plus gradient norms to the runtime. Audit source/output smile scores,
seven applied indices, mask paths/coverage, strict mouth MAE, semantic-union
MAE, and outside-region MAE.

- [ ] **Step 5: Update the MPS test command**

Add a third guided run that writes
`outputs/sample_0_sd2_bld_remove_smile_robust_mps` with:

```bash
--robust_classifier_guidance \
--generation_mask_component data/00000_mouth.png \
--generation_mask_component data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_u_lip.png \
--generation_mask_component data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_l_lip.png \
--generation_mask_feather 3 \
--cci_start_step 4 --cci_end_step 16 --cci_every_n_steps 2 \
--cci_step_size 0.20
```

- [ ] **Step 6: Run complete tests and shell validation**

Run: `.venv-ml/bin/python -m unittest discover -s tests -v`

Run: `bash -n scripts/run_remove_smile_classifier_mps.sh`

Expected: all tests PASS and shell syntax exits 0.

### Task 4: MPS Experiment And Honest Evaluation

**Files:**
- Runtime output: `outputs/sample_0_sd2_bld_remove_smile_robust_mps/`

**Interfaces:**
- Consumes: the robust command from Task 3 and saved same-seed baseline outputs.
- Produces: robust image, masks, audit trace, score/realism comparison, and a clear success or failure classification.

- [ ] **Step 1: Run only the new robust MPS case**

Run the robust command from the updated shell script using seed 42 and 35
requested inference steps.

- [ ] **Step 2: Verify artifacts and metrics**

Confirm all output files are readable. Compare source, no-hook, hard-classifier,
and robust smile probability; mouth-only MAE; semantic-mask MAE; outside-mask
MAE; and applied indices.

- [ ] **Step 3: Inspect the robust image**

Classify the result as:

- success: score below `0.5`, visibly non-smiling, no mouth corruption;
- adversarial shortcut: lower score with visible artifacts;
- insufficient intervention: realistic but score remains at or above `0.5`.

- [ ] **Step 4: Final verification without committing**

Run the full test suite and `git diff --check`, report exact results, preserve all
changes uncommitted, and do not modify files outside `my-docs/cci-diff`.
