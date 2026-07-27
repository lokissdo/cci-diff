# Kaggle Two-Stage CCI Design

## Purpose

Provide two reproducible, resumable pipelines:

1. discover and freeze a target-specific semantic influence graph on a
   discovery cohort;
2. compare raw BLD, fixed-weight CCI, and adaptive CCI on a disjoint
   evaluation cohort.

The implementation supports Kaggle T4 execution with user-supplied SD2,
classifier, identity, and CelebAMask-HQ assets.

## Global Graph Discovery

Grad-CAM++ screening uses source images only. It proposes at most four
semantic regions for each target label. Intervention discovery evaluates
region sets progressively by cardinality from one through four, without a
post-generation attack.

For region set \(R\), target effect is

\[
E_R = \frac{1}{N}\sum_i
\left[p_{\mathrm{desired}}(x_i^R)-p_{\mathrm{desired}}(x_i)\right].
\]

Raw flip rate uses the generation classifier at probability threshold 0.5.
Before a set reaches the configured raw-flip requirement, sets are ordered by
mean target effect, its confidence bound, and then minimal-change metrics.
Once target sufficiency is reached, selection minimizes mask area before
outside change, non-target drift, identity loss, and component count.

The discovery manifest records every executed cardinality and whether the
threshold was reached. A fallback graph is allowed when no set reaches the
threshold, but it is marked unsupported and must not be described as a
sufficient graph.

## Full CCI Evaluation

Notebook 2 is independent from graph discovery and selects its own evaluation
cohort. For the current smile-only benchmark, the region policy is fixed to
`mouth`, `upper_lip`, and `lower_lip`.

Every source is run with identical seed, prompt, masks, scheduler, and model
under three controller modes:

- `disabled`: raw BLD without CCI target, constraint, or final-correction
  updates;
- `fixed_equal`: CCI target and constraint gradients with fixed equal
  constraint coefficients;
- `feedback`: CCI target and constraint gradients with adaptive dual
  coefficients.

Output rows identify the controller mode directly rather than relying only on
A-number aliases.

## Graph Roles

Version two removes `allowed_change`. Graph nodes have only:

- `target`;
- `constraint`;
- `audit_only`.

All non-target classifier attributes are included in non-target drift. No
attribute receives a manually selected metric exemption.

## Kaggle Notebooks

`notebooks/01_global_graph_discovery.ipynb`:

- installs the local package without downloading checkpoints;
- validates imported paths;
- selects deterministic task-specific discovery cohorts;
- screens at most four regions;
- runs resumable progressive interventions;
- freezes one graph per target;
- displays target-effect, flip-rate, area, and preservation tables.

`notebooks/02_full_cci_fixed_vs_adaptive.ipynb`:

- validates the same imported assets;
- selects its own smile-removal evaluation cohort;
- runs smile removal with the fixed mouth-and-lips region policy;
- compares raw BLD (`disabled`), `fixed_equal`, and `feedback`;
- writes source/output pairs and aggregate metrics.

Both notebooks default to 300 smile-removal images, CUDA, float16 diffusion,
seed 42, and configurable paths under `/kaggle/input` and `/kaggle/working`.
Experiment entry points execute inline through `runpy` and print unbuffered,
timestamped progress in the active notebook cell.

## Validation

Tests cover dynamic candidate selection, maximum cardinality four,
target-effect-first fallback selection, progressive stopping provenance,
graphs without `allowed_change`, explicit controller-mode orchestration,
notebook structure, path configuration, and disjoint cohorts.
