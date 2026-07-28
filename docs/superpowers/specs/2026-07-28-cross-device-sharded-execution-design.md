# Cross-Device Sharded Execution Design

## Objective

Run the smile graph-discovery and raw BLD/fixed/adaptive CCI notebooks over
300 eligible images without silently truncating work at Kaggle's 12-hour cell
limit. Use both T4 GPUs remotely and retain a compatible single-worker MPS
path for local execution.

## Failure Evidence

The previous discovery job selected 300 eligible images after scanning 656
sources and completed Grad-CAM++ screening for all 300. It then planned 900
diffusion interventions: 300 images multiplied by three region sets. Only four
sample directories were started before timeout, and no metric row completed.
The evaluation job likewise selected 300 images but completed no candidate
row. Both jobs ended as `CANCEL_ACKNOWLEDGED` after 43,200 seconds.

The first intervention failures came from passing decoded `float16` tensors to
a FaceNet TorchScript model whose parameters are `float32`. Both orchestration
scripts also execute every generated candidate serially and start a fresh
Python process for each candidate.

## Execution Model

The top-level runner resolves worker devices once:

- `cuda` expands to `cuda:0`, `cuda:1`, up to `--max_workers`.
- An explicit device such as `cuda:1` creates one worker.
- `mps` always creates one worker.
- `cpu` always creates one worker.

Eligible sample IDs are selected once, sorted, and partitioned round-robin.
Each worker receives an explicit sample-ID shard and a separate output
directory. Workers run concurrently. A worker failure terminates the parent
stage after all workers are collected, while successful shard outputs remain
available for diagnosis and resume.

## Discovery

Grad-CAM++ screening remains a single short stage because it completed 300
images in about 74 seconds. The selected region list is frozen before worker
launch. The intervention IDs are split across devices, and each shard executes
the same canonical region-set grid for its own IDs.

After all shards finish, the parent merges:

- `intervention_results.csv`;
- `failures.jsonl`;
- execution counts and shard provenance.

The global graph compiler reads the merged result table. No sample appears in
more than one shard.

## Evaluation

The parent selects the common 300-image cohort once and passes explicit IDs to
each `run_clean_cci_pilot.py` worker. Each worker runs A0, A2, and A3 for only
its shard. The parent merges:

- `candidate_results.csv`;
- `pilot_results.csv`;
- `pilot_ranked.csv`;
- `failures.jsonl`;
- a summary containing worker devices and per-shard counts.

Output images remain in shard directories, and merged CSV paths continue to
point to those files.

## Numeric Behavior

FaceNet remains `float32` on every platform. Source and generated face crops
are converted to the model's parameter dtype immediately before inference.
The cast is differentiable, so identity gradients still propagate to decoded
images and diffusion latents.

SD2 uses:

- `float16` on CUDA;
- `float32` on MPS and CPU.

MPS never starts concurrent model workers because multiple full SD2 processes
compete for unified memory and reduce stability.

## Resume And Logging

Each shard uses existing artifact validation to skip completed candidates.
The parent writes `shard_manifest.json` before launch and records command,
device, assigned IDs, return code, and elapsed time. Logs use worker prefixes
and remain unbuffered.

The notebooks invoke the same standalone runner:

- Kaggle: `--device cuda --max_workers 2`;
- local Apple Silicon: `--device mps --max_workers 1`.

## Validation

Tests cover:

1. FaceNet accepts half-precision decoded images and returns a differentiable
   float32 identity loss.
2. Round-robin shards are complete, disjoint, and deterministic.
3. CUDA auto-expansion respects available GPU count and worker limits.
4. MPS always resolves to one worker.
5. Explicit sample IDs prevent each evaluation worker from rescanning and
   selecting the same cohort.
6. CSV and JSONL shard merges preserve all rows exactly once.
7. Both notebooks declare the intended worker count.

Before a 300-image rerun, each notebook runs a two-image remote smoke stage.
The smoke succeeds only when both CUDA devices start, FaceNet reports no dtype
error, and each shard produces at least one complete metric row.
