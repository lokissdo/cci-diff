#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv-ml/bin/python}"
CCI_CONFIG="${CCI_CONFIG:-examples/hair_intervention.json}"
INIT_IMAGE="${INIT_IMAGE:-data/1.jpg}"
LABEL_MAP="${LABEL_MAP:-data/1.png}"
ATTRIBUTE2PARTS="${ATTRIBUTE2PARTS:-../thesis_2025/face_parts_retrieval/CelebAMask-HQ/face_parsing/attribute2parts.json}"
MASK_PARTS="${MASK_PARTS:-hair}"
GENERATED_MASK="${GENERATED_MASK:-outputs/generated_masks/1_hair.png}"
MASK="${MASK:-${GENERATED_MASK}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sample_1_sd2_bld_mps}"
MODEL_PATH="${MODEL_PATH:-checkpoints/sd2-1-base}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
BLENDING_START_PERCENTAGE="${BLENDING_START_PERCENTAGE:-0.25}"
SEED="${SEED:-42}"
INITIAL_LATENT_MODE="${INITIAL_LATENT_MODE:-random}"
AUTO_MASK="${AUTO_MASK:-1}"
MASK_PART_ARGS=()
read -r -a MASK_PART_ARGS <<< "${MASK_PARTS}"

if [[ "${AUTO_MASK}" != "0" && "${MASK}" == "${GENERATED_MASK}" ]]; then
  "${PYTHON_BIN}" scripts/make_celebamask_mask.py \
    --cci_config "${CCI_CONFIG}" \
    --label_map "${LABEL_MAP}" \
    --attribute2parts "${ATTRIBUTE2PARTS}" \
    --parts "${MASK_PART_ARGS[@]}" \
    --output "${MASK}"
fi

"${PYTHON_BIN}" scripts/run_sd2_bld_cci.py \
  --cci_config "${CCI_CONFIG}" \
  --init_image "${INIT_IMAGE}" \
  --mask "${MASK}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_path "${MODEL_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --device mps \
  --torch_dtype float32 \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --blending_start_percentage "${BLENDING_START_PERCENTAGE}" \
  --seed "${SEED}" \
  --initial_latent_mode "${INITIAL_LATENT_MODE}" \
  --local_files_only \
  "$@"
