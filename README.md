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

## Discover A Counterfactual Influence Graph

This workflow replaces fixed target-to-face-part assignments with measured,
classifier-specific evidence. Grad-CAM++ only proposes regions; same-seed
masked diffusion interventions verify their effects. The resulting edges
describe this classifier and intervention pipeline, not biological causality.

Screen candidate regions:

```bash
.venv-ml/bin/python scripts/screen_counterfactual_regions.py \
  --template_graph examples/graphs/remove_smile_clean_cci.json \
  --classifier_path models/resnet50_multilabel_model.pth \
  --sample_ids 0 1 2 3 4 5 6 7 8 9 \
  --candidate_regions skin mouth upper_lip lower_lip \
  --max_selected_regions 4 \
  --saliency_coverage_threshold 0.80 \
  --cohort_frequency_threshold 0.90 \
  --minimum_captured_saliency 0.02 \
  --output_dir outputs/smile_region_screening \
  --device mps
```

Screening preserves the supplied semantic masks. It ranks eligible regions by
median in-mask Grad-CAM++ intensity, median captured saliency, availability,
and then smaller area. A broad region such as `skin` therefore does not outrank
a concentrated mouth region merely because it contains more total heatmap
mass.

Run paired interventions. Every region set uses the same source IDs, seeds,
diffusion settings, classifier, and identity model. Post-generation attack is
disabled so the measured effect belongs to diffusion:

```bash
.venv-ml/bin/python scripts/run_counterfactual_region_interventions.py \
  --template_graph examples/graphs/remove_smile_clean_cci.json \
  --sample_ids 0 1 2 3 4 5 6 7 8 9 \
  --candidate_regions mouth upper_lip lower_lip \
  --max_set_size 4 \
  --stop_flip_rate 0.96 \
  --seeds 42 \
  --model_path checkpoints/sd2-1-base \
  --classifier_path models/resnet50_multilabel_model.pth \
  --identity_model_path models/facenet_vggface2.ts \
  --output_dir outputs/smile_region_interventions \
  --device mps
```

Before generation, the runner compares every requested hard union mask over
the complete discovery cohort. Exactly equivalent sets are reduced to one
canonical treatment. For example, when every mouth pixel is already contained
by `skin`, `skin + mouth` is recorded as an alias of `skin` and is not generated
again. Strict supersets that add any pixel are retained. The requested sets,
canonical sets, signatures, and aliases are stored in
`intervention_manifest.json`.

Analyze the completed CSV and emit both the evidence graph and selected
execution policy:

```bash
.venv-ml/bin/python scripts/discover_counterfactual_graph.py \
  --results outputs/smile_region_interventions/intervention_results.csv \
  --template_graph examples/graphs/remove_smile_clean_cci.json \
  --required_flip_rate 0.95 \
  --minimum_samples 10 \
  --output_dir outputs/smile_counterfactual_graph
```

The selected region set first has to satisfy the requested generation
classifier flip rate. Among passing sets, selection minimizes semantic mask
area, outside-mask change, total changed area, non-target drift, identity loss,
and region count in that order. Online constraint weights remain
residual-driven; graph discovery does not predict a permanent loss-weight
vector.

The full 100-discovery/100-held-out smile workflow is resumable:

```bash
bash scripts/run_smile_graph_individual_100.sh
```

Its defaults include `MINIMUM_REGION_COVERAGE=0.90` and
`MINIMUM_CAPTURED_SALIENCY=0.02`. Override either as an environment variable
when running a documented ablation.

## Run The Two-Stage Kaggle Experiment

The two notebooks separate graph discovery from held-out generation:

1. `notebooks/01_global_graph_discovery.ipynb` builds and freezes one global
   graph per target using a discovery cohort. Grad-CAM++ proposes no more than
   four regions, and same-seed interventions test region sets progressively.
2. `notebooks/02_full_cci_fixed_vs_adaptive.ipynb` excludes the discovery IDs
   and runs a matched three-arm comparison: raw BLD (`A0`), fixed-equal CCI
   (`A2`), and adaptive-feedback CCI (`A3`).

Both notebooks default to 300 images for smile removal and 300 images for
blond-hair addition, seed 42, CUDA, and float16 diffusion. Configure the path
cell for the imported SD2 checkpoint, CelebA classifier, identity model, and
CelebAMask-HQ dataset before running all cells. Outputs are written beneath
`/kaggle/working`; completed rows are reused when a run is resumed.

The held-out policies are intentionally independent of graph discovery:
smile removal uses `mouth + upper_lip + lower_lip`, while blond-hair addition
uses `hair`. This isolates controller behavior from region-policy selection.

## Run One-Pass Individual Region CCI

After discovery, freeze the influence graph and use source Grad-CAM++ to select
the smallest saliency-covering subset of globally verified regions for every
held-out image:

```bash
.venv-ml/bin/python scripts/run_individual_region_cci.py \
  --influence_graph outputs/smile_counterfactual_graph/influence_graph.json \
  --template_graph examples/graphs/remove_smile_clean_cci.json \
  --sample_ids 100 101 102 103 104 \
  --coverage_threshold 0.80 \
  --seed 42 \
  --model_path checkpoints/sd2-1-base \
  --classifier_path models/resnet50_multilabel_model.pth \
  --identity_model_path models/facenet_vggface2.ts \
  --output_dir outputs/smile_individual_region_cci \
  --device mps
```

The selector sees only the source classifier decision, source Grad-CAM++ map,
and source semantic masks. It generates once per image. Failed flips are
retained as failures; the runner does not expand the mask, generate
alternatives, rerank outputs, or apply post-generation attack. Pass
`--discovery_manifest` to reject accidental overlap between discovery and
held-out IDs.

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
