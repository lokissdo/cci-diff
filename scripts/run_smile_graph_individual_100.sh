#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_smile_graph_individual_100.sh

Runs the complete smile counterfactual-region experiment:
  1. Select 100 discovery and 100 disjoint held-out source images.
  2. Screen the semantic-region ontology with source Grad-CAM++.
  3. Run discovery interventions over automatically selected region combinations.
  4. Build the frozen influence graph from classifier counterfactual effects.
  5. Run held-out individual-region CCI once for each of 100 images.

The script is resumable. Run the same command again after an interruption.

Important environment overrides:
  DEVICE=mps
  NUM_INFERENCE_STEPS=35
  COVERAGE_THRESHOLD=0.80
  DISCOVERY_COUNT=100
  TEST_COUNT=100
  MAX_SELECTED_REGIONS=4
  SALIENCY_COVERAGE_THRESHOLD=0.80
  SALIENCY_COHORT_FREQUENCY=0.90
  STOP_FLIP_RATE=0.96
  MINIMUM_REGION_COVERAGE=0.90
  MINIMUM_CAPTURED_SALIENCY=0.02
  MAX_PASSES=3
  PYTHON_BIN=.venv-ml/bin/python
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  usage >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-.venv-ml/bin/python}"
DEVICE="${DEVICE:-mps}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-35}"
COVERAGE_THRESHOLD="${COVERAGE_THRESHOLD:-0.80}"
DISCOVERY_COUNT="${DISCOVERY_COUNT:-100}"
TEST_COUNT="${TEST_COUNT:-100}"
MAX_SELECTED_REGIONS="${MAX_SELECTED_REGIONS:-4}"
SALIENCY_COVERAGE_THRESHOLD="${SALIENCY_COVERAGE_THRESHOLD:-0.80}"
SALIENCY_COHORT_FREQUENCY="${SALIENCY_COHORT_FREQUENCY:-0.90}"
STOP_FLIP_RATE="${STOP_FLIP_RATE:-0.96}"
MINIMUM_REGION_COVERAGE="${MINIMUM_REGION_COVERAGE:-0.90}"
MINIMUM_CAPTURED_SALIENCY="${MINIMUM_CAPTURED_SALIENCY:-0.02}"
MAX_REGION_SET_SIZE="${MAX_REGION_SET_SIZE:-4}"
MAX_PASSES="${MAX_PASSES:-3}"
MAX_IMAGE_ID="${MAX_IMAGE_ID:-30000}"
SEED="${SEED:-42}"

IMAGE_ROOT="${IMAGE_ROOT:-data/CelebAMask-HQ/CelebA-HQ-img}"
MASK_ROOT="${MASK_ROOT:-data/CelebAMask-HQ/CelebAMask-HQ-mask-anno}"
MODEL_PATH="${MODEL_PATH:-checkpoints/sd2-1-base}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-models/resnet50_multilabel_model.pth}"
IDENTITY_MODEL_PATH="${IDENTITY_MODEL_PATH:-models/facenet_vggface2.ts}"
TEMPLATE_GRAPH="${TEMPLATE_GRAPH:-examples/graphs/remove_smile_clean_cci.json}"
DISCOVERY_SOURCE_MANIFEST="${DISCOVERY_SOURCE_MANIFEST:-outputs/clean_cci_a3_100_steps35_target/pilot_manifest.json}"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-outputs/smile_individual_experiment}"
COHORTS_JSON="${COHORTS_JSON:-${EXPERIMENT_DIR}/cohorts.json}"
SCREENING_DIR="${SCREENING_DIR:-outputs/smile_global_region_screening_100}"
DISCOVERY_DIR="${DISCOVERY_DIR:-outputs/smile_global_graph_discovery_100}"
GRAPH_DIR="${GRAPH_DIR:-outputs/smile_global_graph_100}"
INDIVIDUAL_DIR="${INDIVIDUAL_DIR:-outputs/smile_individual_region_100}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/cci-matplotlib}"

for required_path in \
  "${PYTHON_BIN}" \
  "${IMAGE_ROOT}" \
  "${MASK_ROOT}" \
  "${MODEL_PATH}" \
  "${CLASSIFIER_PATH}" \
  "${IDENTITY_MODEL_PATH}" \
  "${TEMPLATE_GRAPH}" \
  "${DISCOVERY_SOURCE_MANIFEST}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${EXPERIMENT_DIR}"

echo
echo "=== Stage 1/5: deterministic disjoint cohorts ==="
COHORTS_JSON="${COHORTS_JSON}" \
DISCOVERY_SOURCE_MANIFEST="${DISCOVERY_SOURCE_MANIFEST}" \
DISCOVERY_COUNT="${DISCOVERY_COUNT}" \
TEST_COUNT="${TEST_COUNT}" \
MAX_IMAGE_ID="${MAX_IMAGE_ID}" \
IMAGE_ROOT="${IMAGE_ROOT}" \
MASK_ROOT="${MASK_ROOT}" \
CLASSIFIER_PATH="${CLASSIFIER_PATH}" \
DEVICE="${DEVICE}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from argparse import Namespace
from pathlib import Path

import torch

from cci_diff.classifiers.celeba_resnet50 import load_celeba_resnet50
from cci_diff.identity.facenet import build_face_detector
from scripts.run_clean_cci_pilot import select_eligible_samples

cohorts_path = Path(os.environ["COHORTS_JSON"])
discovery_count = int(os.environ["DISCOVERY_COUNT"])
test_count = int(os.environ["TEST_COUNT"])

if cohorts_path.is_file():
    payload = json.loads(cohorts_path.read_text(encoding="utf-8"))
    discovery_ids = [int(value) for value in payload["discovery_ids"]]
    test_ids = [int(value) for value in payload["test_ids"]]
    if len(discovery_ids) != discovery_count:
        raise ValueError(
            f"Existing cohort has {len(discovery_ids)} discovery IDs; "
            f"expected {discovery_count}"
        )
    if len(test_ids) != test_count:
        raise ValueError(
            f"Existing cohort has {len(test_ids)} test IDs; expected {test_count}"
        )
    overlap = sorted(set(discovery_ids).intersection(test_ids))
    if overlap:
        raise ValueError(f"Existing cohorts overlap: {overlap}")
    print(f"Reusing validated cohorts: {cohorts_path}")
    raise SystemExit(0)

source_manifest_path = Path(os.environ["DISCOVERY_SOURCE_MANIFEST"])
source_manifest = json.loads(
    source_manifest_path.read_text(encoding="utf-8")
)
discovery_ids = [
    int(value)
    for value in source_manifest["features"]["smile"]["selected_ids"]
][0:discovery_count]
if len(discovery_ids) != discovery_count:
    raise ValueError(
        f"Found only {len(discovery_ids)} discovery IDs; "
        f"required {discovery_count}"
    )

device = os.environ["DEVICE"]
classifier = load_celeba_resnet50(
    os.environ["CLASSIFIER_PATH"],
    device=device,
    dtype=torch.float32,
)
detector = build_face_detector()
args = Namespace(
    max_image_id=int(os.environ["MAX_IMAGE_ID"]),
    image_root=os.environ["IMAGE_ROOT"],
    mask_root=os.environ["MASK_ROOT"],
    classifier_input_size=512,
    device=device,
    limit=discovery_count + test_count,
)
eligible, decisions = select_eligible_samples(
    args,
    feature="smile",
    classifier=classifier,
    detector=detector,
)
discovery_set = set(discovery_ids)
test_ids = [
    int(sample_id)
    for sample_id, _, _ in eligible
    if sample_id not in discovery_set
][0:test_count]
if len(test_ids) != test_count:
    raise ValueError(
        f"Found only {len(test_ids)} disjoint held-out IDs; "
        f"required {test_count}. Increase MAX_IMAGE_ID."
    )
overlap = sorted(discovery_set.intersection(test_ids))
if overlap:
    raise AssertionError(f"Discovery/test overlap: {overlap}")

payload = {
    "version": 1,
    "task": "Smiling -> not Smiling",
    "discovery_ids": discovery_ids,
    "test_ids": test_ids,
    "discovery_count": discovery_count,
    "test_count": test_count,
    "disjoint": True,
    "source_manifest": str(source_manifest_path),
    "classifier_path": os.environ["CLASSIFIER_PATH"],
    "eligibility_rule": {
        "source_smiling_probability_at_least": 0.5,
        "face_detection_required": True,
        "complete_mouth_and_lip_masks_required": True,
    },
    "scanned_count": len(decisions),
}
cohorts_path.parent.mkdir(parents=True, exist_ok=True)
cohorts_path.write_text(
    json.dumps(payload, indent=2, allow_nan=False),
    encoding="utf-8",
)
print(f"Wrote disjoint cohorts: {cohorts_path}")
PY

read_ids() {
  local key="$1"
  "${PYTHON_BIN}" -c \
    'import json,sys; print(" ".join(str(v) for v in json.load(open(sys.argv[1]))[sys.argv[2]]))' \
    "${COHORTS_JSON}" "${key}"
}

DISCOVERY_IDS_TEXT="$(read_ids discovery_ids)"
TEST_IDS_TEXT="$(read_ids test_ids)"
IFS=' ' read -r -a DISCOVERY_IDS <<< "${DISCOVERY_IDS_TEXT}"
IFS=' ' read -r -a TEST_IDS <<< "${TEST_IDS_TEXT}"

if [[ -n "${REGION_UNIVERSE:-}" ]]; then
  IFS=' ' read -r -a SCREEN_REGIONS <<< "${REGION_UNIVERSE}"
else
  SCREEN_REGIONS=(
    background skin nose eye_glasses left_eye right_eye
    left_brow right_brow left_ear right_ear mouth upper_lip lower_lip
    hair hat ear_ring necklace neck cloth
  )
fi

echo
echo "=== Stage 2/5: automatic Grad-CAM++ region screening ==="
"${PYTHON_BIN}" scripts/screen_counterfactual_regions.py \
  --template_graph "${TEMPLATE_GRAPH}" \
  --classifier_path "${CLASSIFIER_PATH}" \
  --sample_ids "${DISCOVERY_IDS[@]}" \
  --candidate_regions "${SCREEN_REGIONS[@]}" \
  --max_selected_regions "${MAX_SELECTED_REGIONS}" \
  --saliency_coverage_threshold "${SALIENCY_COVERAGE_THRESHOLD}" \
  --cohort_frequency_threshold "${SALIENCY_COHORT_FREQUENCY}" \
  --minimum_coverage_frequency "${MINIMUM_REGION_COVERAGE}" \
  --minimum_captured_saliency "${MINIMUM_CAPTURED_SALIENCY}" \
  --image_root "${IMAGE_ROOT}" \
  --mask_root "${MASK_ROOT}" \
  --output_dir "${SCREENING_DIR}" \
  --device "${DEVICE}"

CANDIDATE_REGIONS_TEXT="$(
  "${PYTHON_BIN}" -c \
    'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["selected_candidate_regions"]))' \
    "${SCREENING_DIR}/screening_manifest.json"
)"
IFS=' ' read -r -a CANDIDATE_REGIONS <<< "${CANDIDATE_REGIONS_TEXT}"
if [[ "${#CANDIDATE_REGIONS[@]}" -eq 0 ]]; then
  echo "Grad-CAM++ screening selected no intervention candidates." >&2
  exit 1
fi
if [[ "${MAX_REGION_SET_SIZE}" -gt "${#CANDIDATE_REGIONS[@]}" ]]; then
  EFFECTIVE_MAX_REGION_SET_SIZE="${#CANDIDATE_REGIONS[@]}"
else
  EFFECTIVE_MAX_REGION_SET_SIZE="${MAX_REGION_SET_SIZE}"
fi
echo "Selected regions: ${CANDIDATE_REGIONS[*]}"

run_discovery_interventions() {
  "${PYTHON_BIN}" scripts/run_counterfactual_region_interventions.py \
    --template_graph "${TEMPLATE_GRAPH}" \
    --sample_ids "${DISCOVERY_IDS[@]}" \
    --candidate_regions "${CANDIDATE_REGIONS[@]}" \
    --max_set_size "${EFFECTIVE_MAX_REGION_SET_SIZE}" \
    --stop_flip_rate "${STOP_FLIP_RATE}" \
    --seeds "${SEED}" \
    --image_root "${IMAGE_ROOT}" \
    --mask_root "${MASK_ROOT}" \
    --model_path "${MODEL_PATH}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --identity_model_path "${IDENTITY_MODEL_PATH}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --output_dir "${DISCOVERY_DIR}" \
    --device "${DEVICE}" \
    --python_executable "${PYTHON_BIN}" \
    "$@"
}

echo
echo "=== Stage 2b/5: exact cohort-wide union-mask deduplication ==="
run_discovery_interventions --dry_run
REGION_SET_COUNT="$(
  "${PYTHON_BIN}" -c \
    'import json,sys; print(len(json.load(open(sys.argv[1]))["region_sets"]))' \
    "${DISCOVERY_DIR}/intervention_manifest.json"
)"
REQUESTED_REGION_SET_COUNT="$(
  "${PYTHON_BIN}" -c \
    'import json,sys; print(len(json.load(open(sys.argv[1]))["requested_region_sets"]))' \
    "${DISCOVERY_DIR}/intervention_manifest.json"
)"
EXPECTED_DISCOVERY_ROWS="$(
  "${PYTHON_BIN}" -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["expected_rows"]))' \
    "${DISCOVERY_DIR}/intervention_manifest.json"
)"
echo "Intervention grid: ${REGION_SET_COUNT}/${REQUESTED_REGION_SET_COUNT} canonical sets x ${DISCOVERY_COUNT} images"

json_int_or_zero() {
  local path="$1"
  local key="$2"
  if [[ ! -f "${path}" ]]; then
    echo 0
    return
  fi
  "${PYTHON_BIN}" -c \
    'import json,sys; print(int(json.load(open(sys.argv[1])).get(sys.argv[2], 0)))' \
    "${path}" "${key}"
}

echo
echo "=== Stage 3/5: ${EXPECTED_DISCOVERY_ROWS} discovery interventions ==="
completed_rows=0
for ((pass = 1; pass <= MAX_PASSES; pass++)); do
  execution_complete="$(
    "${PYTHON_BIN}" -c \
      'import json,sys; print("1" if json.load(open(sys.argv[1])).get("execution_complete") else "0")' \
      "${DISCOVERY_DIR}/intervention_manifest.json"
  )"
  if [[ "${execution_complete}" -eq 1 ]]; then
    break
  fi
  completed_rows="$(
    json_int_or_zero "${DISCOVERY_DIR}/intervention_manifest.json" completed_rows
  )"
  echo "Discovery pass ${pass}/${MAX_PASSES}; currently ${completed_rows}/${EXPECTED_DISCOVERY_ROWS}"
  if ! run_discovery_interventions --continue_on_error; then
    echo "Discovery pass ${pass} exited early; completed artifacts will be reused." >&2
  fi
done

execution_complete="$(
  "${PYTHON_BIN}" -c \
    'import json,sys; print("1" if json.load(open(sys.argv[1])).get("execution_complete") else "0")' \
    "${DISCOVERY_DIR}/intervention_manifest.json"
)"
if [[ "${execution_complete}" -ne 1 ]]; then
  completed_rows="$(
    json_int_or_zero "${DISCOVERY_DIR}/intervention_manifest.json" completed_rows
  )"
  echo "Discovery incomplete after ${completed_rows} rows." >&2
  echo "Run this same script again to resume." >&2
  exit 1
fi

echo
echo "=== Stage 4/5: frozen influence graph ==="
"${PYTHON_BIN}" scripts/discover_counterfactual_graph.py \
  --results "${DISCOVERY_DIR}/intervention_results.csv" \
  --template_graph "${TEMPLATE_GRAPH}" \
  --required_flip_rate 0.95 \
  --minimum_samples "${DISCOVERY_COUNT}" \
  --bootstrap_samples 2000 \
  --confidence 0.95 \
  --random_seed "${SEED}" \
  --output_dir "${GRAPH_DIR}"

echo
echo "=== Stage 5/5: ${TEST_COUNT} held-out individual-region CCI generations ==="
completed_generations=0
for ((pass = 1; pass <= MAX_PASSES; pass++)); do
  completed_generations="$(
    json_int_or_zero "${INDIVIDUAL_DIR}/individual_manifest.json" completed_generations
  )"
  if [[ "${completed_generations}" -eq "${TEST_COUNT}" ]]; then
    break
  fi
  echo "Held-out pass ${pass}/${MAX_PASSES}; currently ${completed_generations}/${TEST_COUNT}"
  if ! "${PYTHON_BIN}" scripts/run_individual_region_cci.py \
    --influence_graph "${GRAPH_DIR}/influence_graph.json" \
    --template_graph "${TEMPLATE_GRAPH}" \
    --sample_ids "${TEST_IDS[@]}" \
    --coverage_threshold "${COVERAGE_THRESHOLD}" \
    --seed "${SEED}" \
    --image_root "${IMAGE_ROOT}" \
    --mask_root "${MASK_ROOT}" \
    --model_path "${MODEL_PATH}" \
    --classifier_path "${CLASSIFIER_PATH}" \
    --identity_model_path "${IDENTITY_MODEL_PATH}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --discovery_manifest "${DISCOVERY_DIR}/intervention_manifest.json" \
    --output_dir "${INDIVIDUAL_DIR}" \
    --device "${DEVICE}" \
    --python_executable "${PYTHON_BIN}" \
    --continue_on_error; then
    echo "Held-out pass ${pass} exited early; completed artifacts will be reused." >&2
  fi
done

completed_generations="$(
  json_int_or_zero "${INDIVIDUAL_DIR}/individual_manifest.json" completed_generations
)"
if [[ "${completed_generations}" -ne "${TEST_COUNT}" ]]; then
  echo "Held-out run incomplete: ${completed_generations}/${TEST_COUNT} images." >&2
  echo "Run this same script again to resume." >&2
  exit 1
fi

echo
echo "Experiment complete."
echo "Cohorts:          ${COHORTS_JSON}"
echo "Region screening: ${SCREENING_DIR}/screening_manifest.json"
echo "Discovery data:   ${DISCOVERY_DIR}/intervention_results.csv"
echo "Frozen graph:     ${GRAPH_DIR}/influence_graph.json"
echo "Discovery report: ${GRAPH_DIR}/discovery_report.md"
echo "Held-out results: ${INDIVIDUAL_DIR}/individual_results.csv"
echo "Held-out manifest:${INDIVIDUAL_DIR}/individual_manifest.json"
