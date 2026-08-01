# Kaggle-Portable Generic Development Cohort Design

## Goal

Build one parameterized workflow that discovers, fits, and calibrates a
target-generic semantic-mask selector from a new development cohort while
keeping the existing 300 CCI images permanently evaluation-only. The same
workflow must run with `--data_size 30`, `--data_size 300`, or another valid
size without a smoke-specific code path.

## Scientific Contract

- The existing evaluation-300 IDs are immutable exclusions from discovery,
  fitting, and calibration.
- Development and evaluation outcomes never cross cohort boundaries.
- Development generation uses A11 only.
- Held-out evaluation freezes every source-only mask decision before running
  exactly one new A11 generation per evaluation image.
- Oracle scores and final FID, sFID, FVA, FS, MNAC, CD, COUT, and FR metrics
  are evaluation-only and cannot tune discovery, fitting, calibration, or
  per-image selection.
- A small `data_size` run exercises the identical algorithm. It receives no
  relaxed statistical thresholds and makes no paper-level claim.

## Parameterized Cohort Allocation

`--data_size` is the only size control required for the standard workflow.
The development roles use the fixed ratio `2:5:8`:

- discovery: `2/15`
- fitting: `5/15`
- calibration: `8/15`

Thus `data_size=30` produces `4/10/16`, while `data_size=300` produces
`40/100/160`. For other sizes, counts are computed with the largest-remainder
method. Fractional ties are resolved in the stable order discovery, fitting,
calibration. `data_size` must be at least 15, and every role must be non-empty.

Eligible sources are assigned to stable role buckets before truncation. A
SHA-256 ordering over `(seed, sample_id)` assigns two fifteenths of the
eligible universe to discovery, five fifteenths to fitting, and eight
fifteenths to calibration. The workflow takes the first required count from
each bucket. Consequently, increasing `data_size` from 30 to 300 preserves
the original IDs in their original roles.

The cohort manifest records the evaluation exclusion manifest and digest,
eligible-source rules, seed, ordered IDs by role, source-image digests, and
classifier digest. Cohort IDs are frozen before any A11 generation.

## Target-Generic Region Discovery

Discovery must not contain label-specific mask families. It operates over all
available semantic components for each source:

1. Compute source classifier probability and Grad-CAM++ saliency.
2. Aggregate component prevalence, saliency coverage, and saliency density
   across discovery sources.
3. Deterministically retain the top six atomic components. Ties use canonical
   component name.
4. Evaluate A11 interventions for up to six candidates at each cardinality:
   singleton, pair, and triple. Before A11 evaluation, unseen expansions are
   pre-ranked using source-only aggregate coverage and density.
5. After each cardinality, retain a beam of four candidates using positive
   effect confidence, Pareto target-effect-versus-area status, flip rate,
   mean effect, area, and canonical region tuple in that order.
6. Export no more than four supported Pareto candidates plus one deterministic
   reliability fallback. The fallback may duplicate a candidate.

The maximum region-set size is three. The maximum discovery intervention
family is 18 sets: six at each cardinality. Every evaluated set and pruning
decision is written to an audit artifact. If fewer than two supported
candidates remain, the workflow completes with a fallback-only selector and
an explicit reason rather than inventing a candidate.

## Sequential Data Flow

The workflow is one orchestration command with the following phases:

1. Validate Kaggle/local paths, evaluation exclusions, models, masks, and
   policy artifacts.
2. Freeze the parameterized development cohort.
3. Run generic A11 beam discovery on discovery IDs.
4. Freeze the influence graph and candidate family.
5. Generate A11 outcomes for the frozen family on fitting and calibration
   IDs.
6. Extract source-only selector features for all development IDs.
7. Fit coefficients on fitting IDs.
8. Calibrate probabilities and choose the Wilson risk threshold on calibration
   IDs. The existing minimum of 60 accepted non-fallback observations and
   0.05 failure-UCB bound remain unchanged.
9. Freeze the selector, cohort, candidate, feature, and calibration artifacts.

With `data_size=30`, calibration can legitimately become fallback-only. This
is an output of the normal risk rule, not a special small-run behavior.

The full `data_size=300` run rebuilds discovery from all 40 discovery IDs. If
its candidate family differs from the 30-image run, it refits and recalibrates
from scratch. Compatible cached A11 interventions remain reusable.

## Resume and Reuse

Generated interventions are stored in a content-addressed cache keyed by:

- source ID and source SHA-256
- canonical region tuple
- A11 generation-policy signature
- checkpoint inventory digest
- classifier digest
- semantic-mask component digests
- seed and graph/controller settings

Run manifests for different `data_size` values reference cache entries rather
than owning them. Increasing 30 to 300 therefore reuses every exact
image-region-policy result while preventing stale or incompatible reuse.

Each completed intervention is recorded atomically before the next begins.
Resume reconstructs consolidated CSVs from validated cache records. It rejects
changed cohort IDs, source bytes, masks, model inventory, classifier, seed,
beam parameters, target direction, or A11 policy. Interrupted partial files
are never treated as complete.

## Kaggle Portability

The orchestration logic remains in tested Python modules and scripts. A thin
Kaggle example invokes the same CLI; it does not duplicate pipeline logic.

All filesystem locations are arguments, including:

- repository root
- image and semantic-mask roots under `/kaggle/input/...`
- checkpoint and classifier paths under `/kaggle/input/...`
- evaluation exclusion manifest
- cache and run outputs under `/kaggle/working/...`

Device selection supports `auto`, `cuda`, `mps`, and `cpu`. `auto` prefers
CUDA, then MPS, then CPU. Kaggle uses local checkpoint files and never requires
network model download. The command emits phase progress, completed/remaining
counts, and resumable manifests frequently enough for notebook session
inspection.

The local evaluation-300 process is not a data dependency. Kaggle CUDA may run
development concurrently because it is a separate device and workspace.
Local MPS development must not compete with an active local MPS evaluation.

## Held-Out Evaluation

After the full selector is frozen, a separate evaluation command:

1. validates that all 300 evaluation IDs are absent from every development
   role;
2. extracts source-only features and selects one mask per image;
3. writes and hashes the complete 300-decision manifest;
4. runs exactly one A11 generation per image using the frozen mask;
5. computes oracle and final metrics only after generation completes.

The existing mouth and perioral evaluation outputs may be retained as
historical references. They neither constrain generic discovery nor enter
selector training.

## Failure Handling

- Missing source, mask, checkpoint, classifier, or evaluation exclusion files
  fail before generation.
- Any development/evaluation ID overlap fails before generation.
- Insufficient eligible IDs in any stable role bucket fails with the required
  and available counts.
- Candidate or cache provenance mismatch fails closed.
- Individual generation failures are recorded with sample ID, region tuple,
  phase, and error; resume retries only failed or absent keys.
- Discovery with inadequate positive evidence freezes a fallback-only graph
  and reports the evidence deficiency.
- Small calibration sets retain the production risk rule and may freeze a
  fallback-only selector.

## Testing and Acceptance

Automated tests must cover:

- `30 -> 4/10/16` and `300 -> 40/100/160` allocation;
- deterministic allocation for arbitrary valid sizes;
- nested role membership when increasing data size;
- strict exclusion of the evaluation-300 IDs;
- generic singleton/pair/triple expansion with no target-name special cases;
- deterministic beam budgets, tie-breaking, Pareto filtering, and fallback;
- content-addressed cache hits and provenance mismatch rejection;
- interrupted-run reconstruction without duplicate generation;
- fallback-only behavior when calibration cannot accept 60 observations;
- CLI path portability and automatic CUDA/MPS/CPU selection;
- a synthetic end-to-end 30-image orchestration test with no real diffusion;
- preservation of all existing tests.

Acceptance requires a clean full test suite, CLI help smoke tests, deterministic
artifact reproduction, a successful Kaggle-oriented dry run, and an explicit
report showing zero overlap with the evaluation-300 manifest.

## Non-Goals

- No A0 generation for discovery, fitting, calibration, or primary evaluation.
- No hardcoded smiling/mouth candidate family.
- No oracle-driven mask ranking.
- No lowered risk threshold for small datasets.
- No notebook-only fork of the implementation.
- No use of the currently running evaluation outcomes as development data.
