#!/bin/bash
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON="$ROOT/.venv-ml/bin/python"
ACE_ROOT="$ROOT/../thesis_2025/evaluate/ACE"
OUTPUT_ROOT="$ROOT/outputs/attacked_a0_a11_smile300_seed42"
MOUTH="$OUTPUT_ROOT/mouth"
LIPS="$OUTPUT_ROOT/mouth_upper_lower_lip"
CLASSIFIER="$ROOT/models/resnet50_multilabel_model.pth"
IDENTITY="$ROOT/models/facenet_vggface2.ts"
MODEL="$ROOT/checkpoints/sd2-1-base"
IMAGE_ROOT="$ROOT/data/CelebAMask-HQ/CelebA-HQ-img"
MASK_ROOT="$ROOT/data/CelebAMask-HQ/CelebAMask-HQ-mask-anno"
EXCLUDED_IDS="$ROOT/examples/attacked_region_excluded_ids.json"

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/scheduler.log") 2>&1

test -x "$PYTHON"
test -d "$ACE_ROOT"
test -f "$CLASSIFIER"
test -f "$IDENTITY"
test -d "$MODEL"
test -d "$IMAGE_ROOT"
test -d "$MASK_ROOT"

run_generation() {
    output_dir=$1
    shift
    cd "$ROOT"
    caffeinate -dimsu env PYTHONPATH=src "$PYTHON" -u \
        scripts/run_clean_cci_pilot.py \
        --features smile \
        --limit 300 \
        --seed 42 \
        --num_inference_steps 35 \
        --device mps \
        --python_executable "$PYTHON" \
        --torch_dtype float32 \
        --model_path "$MODEL" \
        --classifier_path "$CLASSIFIER" \
        --identity_model_path "$IDENTITY" \
        --output_dir "$output_dir" \
        --image_root "$IMAGE_ROOT" \
        --mask_root "$MASK_ROOT" \
        --exclude_ids_json "$EXCLUDED_IDS" \
        --controller_modes disabled trust_region \
        --mask_dilations 8 \
        --cci_post_attack smooth_boundary \
        --cci_post_attack_epsilon_schedule 0.05,0.08,0.10,0.30,0.50 \
        --cci_post_attack_boundary_margin 0.03 \
        "$@"
}

validate_generation() {
    output_dir=$1
    "$PYTHON" - "$output_dir" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
table = root / "pilot_results.csv"
if not table.is_file():
    raise SystemExit(f"missing result table: {table}")
with table.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
selected = [
    row for row in rows
    if row.get("feature") == "smile" and row.get("variant") in {"A0", "A11"}
]
counts = Counter(row["variant"] for row in selected)
if counts != Counter({"A0": 300, "A11": 300}):
    raise SystemExit(f"incomplete selected results: {dict(counts)}")
ids = {
    variant: {int(float(row["sample_id"])) for row in selected if row["variant"] == variant}
    for variant in ("A0", "A11")
}
if ids["A0"] != ids["A11"] or len(ids["A0"]) != 300:
    raise SystemExit("A0 and A11 cohorts are not the same 300 IDs")
if any(not Path(row["output_path"]).is_file() for row in selected):
    raise SystemExit("one or more final post-attack outputs are missing")
PY
}

run_metrics() {
    output_dir=$1
    metrics_dir="$output_dir/metrics"
    cd "$ROOT"
    caffeinate -dimsu env PYTHONPATH=src "$PYTHON" -u \
        scripts/evaluate_clean_cci_ace.py \
        --experiment_root "$output_dir" \
        --ace_root "$ACE_ROOT" \
        --attribute_classifier_path "$CLASSIFIER" \
        --device mps \
        --batch_size 8 \
        --bootstrap_seed 42 \
        --classifier_input_size 512 \
        --cout_steps 50
    caffeinate -dimsu env PYTHONPATH=src "$PYTHON" -u \
        scripts/evaluate_fid_sfid.py \
        --experiment A0 35 "$output_dir" \
        --experiment A11 35 "$output_dir" \
        --output-dir "$metrics_dir" \
        --seed 42 \
        --batch-size 16 \
        --num-workers 0 \
        --dims 2048 \
        --device cpu \
        --expected-count 300 \
        --tasks smile
}

run_generation "$MOUTH" \
    --random_sample_seed 42 \
    --region_components mouth
validate_generation "$MOUTH"
run_metrics "$MOUTH"

run_generation "$LIPS" \
    --sample_ids_manifest "$MOUTH/pilot_manifest.json" \
    --region_components mouth upper_lip lower_lip
validate_generation "$LIPS"
run_metrics "$LIPS"

"$PYTHON" "$ROOT/scripts/combine_attacked_region_metrics.py" \
    --region mouth "$MOUTH/metrics/full_metrics.csv" \
    --region mouth_upper_lower_lip "$LIPS/metrics/full_metrics.csv" \
    --output_dir "$OUTPUT_ROOT" \
    --expected_count 300

echo "Completed attacked A0/A11 region evaluation: $OUTPUT_ROOT"
