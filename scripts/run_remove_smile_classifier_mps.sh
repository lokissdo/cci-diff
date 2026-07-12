#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

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
  ./scripts/run_sd2_bld_mps.sh \
  --cci_hook none

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

env "${COMMON_ENV[@]}" \
  OUTPUT_DIR=outputs/sample_0_sd2_bld_remove_smile_robust_mps \
  ./scripts/run_sd2_bld_mps.sh \
  --cci_hook latent_classifier \
  --classifier_path models/resnet50_multilabel_model.pth \
  --classifier_label_index 31 \
  --classifier_input_size 512 \
  --robust_classifier_guidance \
  --generation_mask_component data/00000_mouth.png \
  --generation_mask_component data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_u_lip.png \
  --generation_mask_component data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_l_lip.png \
  --generation_mask_feather 3 \
  --classifier_scales 256,384,512 \
  --classifier_blur_sigma 1.0 \
  --boundary_weight 0.3 \
  --tv_weight 0.05 \
  --cci_start_step 4 \
  --cci_end_step 16 \
  --cci_every_n_steps 2 \
  --cci_step_size 0.20
