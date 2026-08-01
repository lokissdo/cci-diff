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

Discovery exports every supported positive Pareto candidate plus a reliable
fallback. `required_flip_rate` is the fallback reliability threshold; it is
not a license for a low-success tiny mask to become the inference default.
Per-image adaptive selection is fitted and calibrated separately as described
below. Online constraint weights remain residual-driven; graph discovery does
not predict a permanent loss-weight vector.

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
2. `notebooks/02_full_cci_fixed_vs_adaptive.ipynb` independently assumes the
   reviewed region policy and runs a matched three-arm comparison: raw BLD
   (`A0`), fixed-equal CCI (`A2`), and adaptive-feedback CCI (`A3`).

Both notebooks currently run the smile-removal task only and default to 300
images, seed 42, CUDA, and float16 diffusion. They clone a pinned revision of
this public repository, attach `ipythonx/celebamaskhq`, load
`sd2-community/stable-diffusion-2-1` from Hugging Face, and read the classifier
and identity checkpoints from the private `cci-assets` Kaggle dataset.
Outputs are written beneath `/kaggle/working`; completed rows are reused when
a run is resumed.

Notebook 2 assumes the reviewed smile policy
`mouth + upper_lip + lower_lip`, selects its own eligible cohort, and does not
require Notebook 1 output. Experiment scripts run inline through `runpy`, with
unbuffered timestamped progress printed in the active notebook cell. Git clone
and dependency installation remain external setup commands.

### Compare Matched Fixed and Adaptive Trust-Region CCI

Run the clean-coordinate fixed comparator and lexicographic adaptive optimizer
with identical samples, masks, seeds, diffusion settings, trust-radius budget,
and preservation-aware final-restoration budget:

```bash
python scripts/run_clean_cci_pilot.py \
  --controller_modes fixed_trust_matched trust_region \
  --features smile \
  --limit 10 \
  --random_sample_seed 42 \
  --num_inference_steps 35 \
  --seed 42 \
  --cci_post_attack none
```

`A10` is the matched clean-coordinate fixed comparator and `A11` is the
adaptive lexicographic trust-region optimizer. The archived `A2` fixed-equal
and `A3` primal-dual feedback definitions are unchanged. A10/A11 do not use
the archived target-only final correction; they share preservation-aware final
restoration instead.

Use `scripts/evaluate_clean_cci_ace.py` to report independent ACE
`independent_non_target_drift`, the primary continuous preservation metric,
alongside MNAC. For a calibrated effort sweep, use
`scripts/evaluate_matched_success.py` to freeze the common target-success grid
on calibration data and compute paired identity-cluster bootstrap intervals on
held-out rows.

### Launch Kaggle Remotely

Authenticate the official Kaggle CLI once, then run:

```bash
.venv-kaggle/bin/python scripts/run_kaggle_two_stage.py
```

The launcher uploads or versions the local evaluator files in `models/` as the
private dataset `a210462khihng/cci-assets`, injects the current Git commit into
both notebooks, and can run them sequentially. They remain independent and may
also be started separately. Completed outputs are downloaded to
`outputs/kaggle_remote/`.

Useful controls:

```bash
# Validate packages and generated metadata without contacting Kaggle.
.venv-kaggle/bin/python scripts/run_kaggle_two_stage.py --prepare_only

# Start graph discovery and return immediately.
.venv-kaggle/bin/python scripts/run_kaggle_two_stage.py --no_wait

# Run Notebook 2 independently.
.venv-kaggle/bin/python scripts/run_kaggle_two_stage.py \
  --start_at evaluation \
  --skip_datasets
```

The Kaggle and Hugging Face tokens are never included in uploaded files or
kernel metadata. The selected Hugging Face model is public, so no Hugging Face
secret is needed inside Kaggle.

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

## Risk-controlled source-only mask selection

### Generic 30-to-300 development workflow

The current paper workflow is target- and direction-generic. It screens every
CelebAMask-HQ semantic component from source images, retains six atomic
components, and evaluates a four-wide A11 beam with at most six singleton,
pair, and triple sets per level. It does not hardcode mouth or lip masks.

One `--data_size` argument controls the development cohort. `30` allocates
4 discovery, 10 fitting, and 16 calibration images; `300` allocates
40/100/160. Other values of at least 15 use the same deterministic 2:5:8
allocation. There is no separate smoke algorithm or relaxed calibration rule:

```bash
.venv-ml/bin/python scripts/run_generic_region_development.py \
  --data_size 30 \
  --eligible_ids_manifest data/candidate-source-ids.json \
  --evaluation_ids_manifest data/paper-evaluation-ids.json \
  --template_graph examples/graphs/remove_smile_clean_cci.json \
  --generation_policy examples/replay_smile_a11_policy.json \
  --image_root data/CelebAMask-HQ/CelebA-HQ-img \
  --mask_root data/CelebAMask-HQ/CelebAMask-HQ-mask-anno \
  --model_path checkpoints/sd2-1-base \
  --classifier_path models/resnet50_multilabel_model.pth \
  --identity_model_path models/facenet_vggface2.ts \
  --cache_dir outputs/generic-development/a11-cache \
  --output_dir outputs/generic-development/n30 \
  --device auto
```

The candidate manifest is scanned using only source availability, target
direction, face detection, and semantic-mask availability. The evaluation
manifest must contain the external paper evaluation IDs; those IDs are
excluded before discovery and need no candidate generations. The existing
`outputs/attacked_a0_a11_smile300_seed42` cohort remains evaluation data and
is not consumed by this development command.

For the full development run, change `--data_size 30` to `300` and use a new
run directory such as `n300`, while keeping the same `--cache_dir`. Stable
role assignment preserves the 30-image members inside the 300-image cohort,
and exact source-mask-policy A11 interventions are reused. Discovery, fitting,
and calibration are rebuilt from the larger cohorts. The N=30 selector and
graph are pipeline-validation artifacts, not final paper artifacts.

On Kaggle, edit the path bundle in
`examples/kaggle_generic_development.py`. It invokes this same CLI under
`/kaggle/working`; set `data_size=30` first, then `300`. The frozen generation
policy must name the attached checkpoint directory and contain its exact
`checkpoint_files` inventory. `--device auto` selects CUDA, then MPS, then
CPU, and model download remains disabled.

After the N=300 selector is frozen, generate the external evaluation cohort
exactly once per source-selected mask:

```bash
.venv-ml/bin/python scripts/run_individual_region_cci.py \
  --generate_selected_a11 \
  --influence_graph outputs/generic-development/n300/influence_graph.json \
  --template_graph examples/graphs/remove_smile_clean_cci.json \
  --sample_ids $(.venv-ml/bin/python -c 'import json; print(*json.load(open("data/paper-evaluation-ids.json"))["sample_ids"])') \
  --model_path checkpoints/sd2-1-base \
  --classifier_path models/resnet50_multilabel_model.pth \
  --identity_model_path models/facenet_vggface2.ts \
  --generation_policy_manifest examples/replay_smile_a11_policy.json \
  --semantic_mask_manifest outputs/generic-development/n300/semantic_mask_manifest.json \
  --selector_model outputs/generic-development/n300/backend/selector/selector_model.json \
  --intervention_cache_dir outputs/generic-development/a11-cache \
  --output_dir outputs/generic-development/paper-evaluation-a11 \
  --device auto
```

This mode validates and hashes every source-only decision before the first
cache lookup or generation. It runs A11 only—no A0 arm, mask escalation,
generated-output reranking, oracle ranking, or metric-based selection.

The risk-controlled selector chooses the smallest candidate only after it
passes both source saliency coverage and calibrated joint target, identity,
and locality risk gates. The initial smile family is `mouth` versus
`mouth + upper_lip + lower_lip`, but the implementation is generic across
binary labels and desired directions. A selector artifact is specific to its
target, direction, classifier, influence graph, semantic preprocessing, and
generation policy.

Oracle metrics are evaluation-only. The independent oracle, FID, sFID, FVA,
FS, MNAC, CD, COUT, generated probabilities, post-attack outcomes, and output
paths cannot enter discovery-time source features, selector fitting features,
calibration decisions, or inference ranking. A0 and A11 reuse the same frozen
source decision. The selector writes and hashes the complete decision manifest
before either direct generation or fixed-output replay can read a result.

Four sample-ID roles are pairwise disjoint: graph discovery, selector fitting,
selector calibration, and held-out evaluation. The 300-image replay requires
both fixed candidate roots to contain all 600 A0/A11 rows; do not run the
commands below while the companion perioral generation is incomplete. Once
complete, prepare a predeclared 40/80/100/80 split and export only discovery
interventions and fit/calibration outcomes:

```bash
.venv-ml/bin/python scripts/prepare_adaptive_replay_data.py \
  --candidate_results mouth=outputs/attacked_a0_a11_smile300_seed42/mouth/pilot_results.csv \
  --candidate_results lower_lip+mouth+upper_lip=outputs/attacked_a0_a11_smile300_seed42/mouth_upper_lower_lip/pilot_results.csv \
  --sample_ids_manifest outputs/attacked_a0_a11_smile300_seed42/mouth/pilot_manifest.json \
  --discovery_count 40 \
  --fit_count 80 \
  --calibration_count 100 \
  --evaluation_count 80 \
  --random_seed 42 \
  --variant A11 \
  --output_dir outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared
```

Build the candidate graph from the discovery IDs only:

```bash
.venv-ml/bin/python scripts/discover_counterfactual_graph.py \
  --results outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared/discovery_interventions.csv \
  --template_graph examples/graphs/remove_smile_clean_cci.json \
  --required_flip_rate 0.95 \
  --minimum_samples 40 \
  --output_dir outputs/attacked_a0_a11_smile300_seed42/adaptive_graph
```

Freeze the exact semantic-mask bytes for all four predeclared cohorts before
extracting any selector feature:

```bash
.venv-ml/bin/python scripts/build_semantic_mask_manifest.py \
  --influence_graph outputs/attacked_a0_a11_smile300_seed42/adaptive_graph/influence_graph.json \
  --sample_ids $(.venv-ml/bin/python -c 'import json; d=json.load(open("outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared/split_manifest.json")); print(*sum(d["cohorts"].values(), []))') \
  --mask_root data/CelebAMask-HQ/CelebAMask-HQ-mask-anno \
  --output outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared/semantic_mask_manifest.json
```

Compute the eight source-only features for all IDs. This mode loads the source
classifier, Grad-CAM++, and segmentation masks, but performs no selection and
no diffusion:

```bash
.venv-ml/bin/python scripts/run_individual_region_cci.py \
  --influence_graph outputs/attacked_a0_a11_smile300_seed42/adaptive_graph/influence_graph.json \
  --template_graph examples/graphs/remove_smile_clean_cci.json \
  --sample_ids $(.venv-ml/bin/python -c 'import json; d=json.load(open("outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared/split_manifest.json")); print(*sum(d["cohorts"].values(), []))') \
  --model_path checkpoints/sd2-1-base \
  --classifier_path models/resnet50_multilabel_model.pth \
  --identity_model_path models/facenet_vggface2.ts \
  --generation_policy_manifest examples/replay_smile_a11_policy.json \
  --semantic_mask_manifest outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared/semantic_mask_manifest.json \
  --source_features_only \
  --output_dir outputs/attacked_a0_a11_smile300_seed42/adaptive_source_features \
  --device mps
```

Fit coefficients on the fit IDs, calibrate on the calibration IDs, choose the
lowest risk threshold with at least 60 accepted non-fallback rows and a
one-sided 95% Wilson failure upper bound no larger than 0.05, then freeze the
artifact:

```bash
.venv-ml/bin/python scripts/fit_region_selector.py \
  --influence_graph outputs/attacked_a0_a11_smile300_seed42/adaptive_graph/influence_graph.json \
  --source_features outputs/attacked_a0_a11_smile300_seed42/adaptive_source_features/selector_source_features.csv \
  --development_outcomes outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared/development_outcomes.csv \
  --source_feature_manifest outputs/attacked_a0_a11_smile300_seed42/adaptive_source_features/source_feature_manifest.json \
  --split_manifest outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared/split_manifest.json \
  --discovery_ids outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared/discovery_ids.json \
  --evaluation_ids outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared/evaluation_ids.json \
  --output_dir outputs/attacked_a0_a11_smile300_seed42/adaptive_selector
```

Select masks for the 80 held-out sources. `--selection_only` finalizes the
manifest without launching diffusion:

```bash
.venv-ml/bin/python scripts/run_individual_region_cci.py \
  --influence_graph outputs/attacked_a0_a11_smile300_seed42/adaptive_graph/influence_graph.json \
  --template_graph examples/graphs/remove_smile_clean_cci.json \
  --sample_ids $(.venv-ml/bin/python -c 'import json; print(*json.load(open("outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared/evaluation_ids.json"))["sample_ids"])') \
  --model_path checkpoints/sd2-1-base \
  --classifier_path models/resnet50_multilabel_model.pth \
  --identity_model_path models/facenet_vggface2.ts \
  --generation_policy_manifest examples/replay_smile_a11_policy.json \
  --semantic_mask_manifest outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared/semantic_mask_manifest.json \
  --selector_model outputs/attacked_a0_a11_smile300_seed42/adaptive_selector/selector_model.json \
  --selection_only \
  --output_dir outputs/attacked_a0_a11_smile300_seed42/adaptive_heldout_selections \
  --device mps
```

Join each frozen decision to the already-generated post-attack A0 and A11
rows. `materialize_adaptive_region_cohort.py` never imports the diffusion
runner:

```bash
.venv-ml/bin/python scripts/materialize_adaptive_region_cohort.py \
  --selection_manifest outputs/attacked_a0_a11_smile300_seed42/adaptive_heldout_selections/adaptive_selection_manifest.json \
  --candidate_results mouth=outputs/attacked_a0_a11_smile300_seed42/mouth/pilot_results.csv \
  --candidate_results lower_lip+mouth+upper_lip=outputs/attacked_a0_a11_smile300_seed42/mouth_upper_lower_lip/pilot_results.csv \
  --candidate_manifest mouth=outputs/attacked_a0_a11_smile300_seed42/mouth/pilot_manifest.json \
  --candidate_manifest lower_lip+mouth+upper_lip=outputs/attacked_a0_a11_smile300_seed42/mouth_upper_lower_lip/pilot_manifest.json \
  --generation_policy_manifest examples/replay_smile_a11_policy.json \
  --selector_model outputs/attacked_a0_a11_smile300_seed42/adaptive_selector/selector_model.json \
  --evaluation_ids outputs/attacked_a0_a11_smile300_seed42/adaptive_prepared/evaluation_ids.json \
  --expected_count 80 \
  --expected_variants A0 A11 \
  --output_dir outputs/attacked_a0_a11_smile300_seed42/adaptive_heldout
```

`adaptive_results.csv` and its compatibility alias `pilot_results.csv` can be
passed to the existing metric scripts. Report selected area, mouth-selection
rate, fallback rate, safe-success by mask, and runtime alongside FID, sFID,
FVA, FS, MNAC, CD, COUT, and FR. These final metrics evaluate the frozen
policy; they never tune its per-image ranking.

An all-300 replay is useful only as a diagnostic because 220 of those IDs were
used for discovery, fitting, or calibration. Run selection and materialization
with `--exploratory`; every resulting report must be titled `NOT HELD-OUT` and
must not be used as the paper's primary quantitative claim.

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
