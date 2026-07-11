# ESWA SD2 Reuse Notes

Source repo:

`/Users/hung.domodec.com/Documents/my-docs/thesis_2025`

## Reusable Pieces

- `bld_reranking/generate_bld_img.py`: dataset loop, mask loading, old prompt construction, and command wiring.
- `bld_reranking/bld/scripts/text_editing_SD2.py`: SD2 VAE/tokenizer/text encoder/UNet loading, DDIM loop, source-latent blending, CLIP scoring, classifier reranking.
- `adversarial_attack/pgd_utils.py`: masked gradient pattern that can inspire classifier-gradient guidance.
- `face_parts_retrieval/CelebAMask-HQ/face_parsing/prompt_face_part_extractor.py`: face-part mask selection idea.

## Current ESWA Inference Flow

1. Build one binary edit mask.
2. Build one hard-coded smile/neutral prompt.
3. Encode source image to latent.
4. Sample random latents.
5. Run DDIM denoising with classifier-free guidance.
6. Blend source latent outside the mask at every step.
7. Decode candidates.
8. Rank candidates using CLIP and the smile classifier.

## CCI-Diff Hook Point

In `text_editing_SD2.py`, the hook belongs between:

```python
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
latents = self.scheduler.step(noise_pred, t, latents).prev_sample
```

The CCI adapter should temporarily enable gradients on `latents`, decode an
approximate image, compute target/preservation/leakage/classifier losses, and
apply a small latent update before `scheduler.step`.

## First GPU Adapter Behavior

- Keep the old prompt and mask path working.
- Add `--cci_config path/to/config.json`.
- If no CCI config is passed, preserve old ESWA behavior.
- If CCI config is passed, replace hard-coded smile logic with `ConceptIntervention`.
- Produce JSON audit output per generated sample.
