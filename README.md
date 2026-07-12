# CCI-Diff

CCI-Diff is a clean research scaffold for **Causal Concept Intervention Diffusion**.
It is meant to grow from the ESWA thesis code in `../thesis_2025` without copying
the old pipeline wholesale.

## Current Slice

- Pure Python concept intervention specs.
- Framework-neutral guidance loss composition.
- CCI metrics for concept leakage, preservation, causal concept effect, purity, and bias audit matrices.
- Prompt helper for converting an intervention spec into target/preservation prompt text.
- Diffusion smoke runner with a dependency-free fake backend and an optional `diffusers` backend.
- Standard-library tests, runnable without GPU dependencies.

## Local Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install setuptools
.venv/bin/python -m pip install -e . --no-build-isolation
```

The `--no-build-isolation` flag keeps pip from downloading build dependencies
again after `setuptools` is already present in the venv.

## Local Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Run A Small Diffusion Smoke Test

This path does not need torch, diffusers, a GPU, or model downloads. It writes a
tiny deterministic `.ppm` image and an `audit.json` with the prompt and recorded
diffusion states.

```bash
.venv/bin/python scripts/run_diffusion_smoke.py \
  --cci_config examples/smile_intervention.json \
  --output_dir outputs/fake_smoke \
  --backend fake \
  --num_inference_steps 3 \
  --seed 7
```

## Optional Real Diffusers Smoke Test

Use Python 3.10.19 or another `torch`-compatible Python version, then install
the requirements file:

```bash
python3.10 -m venv .venv-ml
.venv-ml/bin/python -m pip install -r requirements.txt
.venv-ml/bin/python scripts/run_diffusion_smoke.py \
  --cci_config examples/smile_intervention.json \
  --output_dir outputs/tiny_diffusers_smoke \
  --backend diffusers \
  --model_id hf-internal-testing/tiny-stable-diffusion-pipe \
  --device cpu \
  --num_inference_steps 2 \
  --seed 7
```

## Run The GPU SD2 Blended-Latent Backend

This follows the ESWA SD2 path in `../thesis_2025/bld_reranking`: encode the
source image, read the mask, run DDIM denoising with classifier-free guidance,
apply the CCI hook point before `scheduler.step`, blend source latents back
through the mask, decode, and write an audit trace.

Run this in the Python 3.10.19 ML environment with `torch` and `diffusers`
installed:

If your runtime cannot fetch from Hugging Face during inference, download the
model snapshot first:

```bash
.venv-ml/bin/python scripts/download_hf_model.py \
  --model_id stabilityai/stable-diffusion-2-base \
  --local_dir checkpoints/sd2-base
```

If Hugging Face asks for access, run `huggingface-cli login` first or pass
`--token hf_...` to the download script.

```bash
.venv-ml/bin/python scripts/run_sd2_bld_cci.py \
  --cci_config examples/smile_intervention.json \
  --init_image data/1.jpg \
  --mask outputs/sample_1/mask.png \
  --output_dir outputs/sample_1_sd2_bld \
  --model_path checkpoints/sd2-base \
  --batch_size 4 \
  --device cuda \
  --torch_dtype float16 \
  --num_inference_steps 50 \
  --guidance_scale 5.0 \
  --blending_start_percentage 0.25 \
  --initial_latent_mode random \
  --local_files_only
```

`--initial_latent_mode random` preserves the old SD2 script behavior.
`--initial_latent_mode source_noise` uses the source-noised initialization from
the SDXL variant.

## Compare Remove-Smile Classifier Guidance On MPS

Run a same-seed A/B experiment on `data/0.jpg` using the supplied mouth mask and
the frozen CelebA ResNet50 `Smiling` output at index 31:

```bash
./scripts/run_remove_smile_classifier_mps.sh
```

The script writes separate no-hook and classifier-guided output folders under
`outputs/`. The guided audit records source/output smile probabilities and the
classifier settings used during denoising. CLIP expression guidance is reserved
for a later experiment so this result remains attributable to one classifier.

## Dry-Run The Legacy SD2 Bridge

```bash
PYTHONPATH=src python3 scripts/run_sd2_cci_from_legacy.py \
  --cci_config examples/smile_intervention.json \
  --legacy_script ../thesis_2025/bld_reranking/bld/scripts/text_editing_SD2.py \
  --init_image data/1.jpg \
  --mask data/1_mask.png \
  --classifier_path models/classifier.pth \
  --output_dir outputs/sample_1 \
  --batch_size 2 \
  --device cuda \
  --dry-run
```

## GPU Implementation Target

The first GPU adapter should wrap the old SD2 inference loop in:

`../thesis_2025/bld_reranking/bld/scripts/text_editing_SD2.py`

The CCI hook belongs between classifier-free guidance and `scheduler.step`, where
the current loop computes `noise_pred` and updates `latents`.
