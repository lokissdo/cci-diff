# Deterministic FID and sFID Evaluation Design

**Date:** 2026-07-16

## Objective

Install `pytorch-fid` in the existing `.venv-ml` environment and compute standard
FID plus deterministic symmetric FID (sFID) for BLD and CCI at 35 and 50
inference steps, separately for the 100-image smile and hair cohorts.

## Evaluated Runs

1. `outputs/raw_bld_a0_100_steps35_target`
2. `outputs/raw_bld_a0_100_steps50_target`
3. `outputs/clean_cci_a3_100_steps35_target`
4. `outputs/clean_cci_a3_100_steps50_target`

Every run must contain exactly 100 smile and 100 hair rows in
`ace_pair_metrics.csv`. The evaluator rejects missing files, duplicate
feature/sample pairs, mismatched cohorts, and incomplete source/output paths.

## Dependency Installation

The `.venv-ml/bin/pip` launcher contains a stale path from the removed
`Documents/my-docs` checkout. Install through the working interpreter instead:

```bash
.venv-ml/bin/python -m pip install pytorch-fid
```

Record the installed `pytorch-fid`, PyTorch, torchvision, NumPy, and SciPy
versions in the output metadata. Do not modify the global Python environment.

## Metric Protocol

Use the `pytorch-fid` InceptionV3 feature extractor with the standard
2048-dimensional feature block. Run feature extraction on CPU for compatibility
and reproducibility.

For each task, sort the 100 aligned pairs by numeric sample ID, then shuffle
their indices with NumPy `default_rng(42)`. The first 50 indices form split 1
and the remaining 50 form split 2. Apply the same indices to source and output
features.

Compute:

```text
FID = FID(source_all, output_all)

sFID_1 = FID(source_split_1, output_split_2)
sFID_2 = FID(source_split_2, output_split_1)
sFID = (sFID_1 + sFID_2) / 2
```

This preserves both directional values printed by the legacy
`compute_sFID.sh` while adding a single explicit mean for comparison. The fixed
split makes repeated runs deterministic. The report must call these 100-image
estimates exploratory because FID is biased and unstable at small sample sizes.

## Architecture

Add one focused evaluator script under `scripts/`. It will:

1. Load and validate all four experiment tables.
2. Confirm identical task cohorts across runs.
3. Load InceptionV3 once.
4. Extract source features once per task because all runs share source images.
5. Extract output features once per run and task.
6. Cache feature arrays under the comparison output directory for resumability.
7. Compute FID and sFID from cached arrays using `calculate_frechet_distance`.
8. Write machine-readable and human-readable reports.

The evaluator will not copy images into temporary directory trees and will not
use the random, CUDA-only legacy shell script.

## Outputs

Write to:

```text
outputs/fid_sfid_bld_cci_steps35_50/
  fid_sfid_metrics.csv
  fid_sfid_metrics.json
  fid_sfid_comparison.md
  features/
```

Each metric row records method, inference steps, task, sample count, FID,
`sFID_1`, `sFID_2`, mean sFID, seed, split sizes, feature dimension, device,
source root, and output root.

## Testing

Focused tests will use synthetic feature arrays and temporary CSV/image
fixtures to prove:

- deterministic split membership for seed 42;
- aligned source/output indices;
- sFID is the arithmetic mean of its two directional values;
- duplicate, incomplete, and mismatched cohorts are rejected;
- valid feature caches resume without recomputation;
- CSV, JSON, and Markdown outputs contain all eight rows.

After focused tests, run the complete project test suite. Then run the real
eight-row evaluation twice and verify the second run reproduces every metric
from cached features.

## Constraints

- Do not stage or commit files.
- Do not modify the thesis evaluator in place.
- Do not use CUDA-specific paths.
- Do not present these 100-image estimates as directly comparable to paper
  results obtained with a different sample count or split protocol.
