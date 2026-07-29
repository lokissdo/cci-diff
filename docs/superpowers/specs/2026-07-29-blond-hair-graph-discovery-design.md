# Blond-Hair Graph Discovery Design

## Objective

Extend the existing Kaggle discovery workflow to test the counterfactual
transition `Blond_Hair: 0 -> 1` without rerunning or modifying the completed
smile experiment.

The hair experiment must use the same discovery methodology as smile:
classifier-based cohort selection, Grad-CAM++ semantic-region screening,
masked diffusion interventions, Pareto filtering, and target-efficiency
selection.

## Scope

This change adds a reusable task selector to the existing Kaggle runner and
notebook. The initial new task is `blond_hair`. Age discovery is explicitly
outside this change.

Existing smile behavior and output directories remain supported, but the
hair-only Kaggle submission invokes only `blond_hair`.

## Task Configuration

Each discovery task resolves to an immutable configuration containing:

- task key and output directory name;
- pilot feature used for eligible-source selection;
- intervention graph template;
- classifier attribute and desired value;
- Grad-CAM++ semantic-region candidates.

The blond-hair configuration is:

- task key: `blond_hair`;
- classifier transition: `Blond_Hair: 0 -> 1`;
- graph template: `examples/graphs/blond_hair_clean_cci.json`;
- output subtree: `blond_hair/`;
- eligible sources: images classified as not blond;
- candidate regions: `hair`, `skin`, `left_brow`, `right_brow`,
  `left_eye`, `right_eye`, `left_ear`, `right_ear`, and `hat`;
- maximum screened regions: four.

## Execution Flow

1. Select the requested number of unique not-blond source images.
2. Save the cohort under a task-specific key in `discovery_ids.json`.
3. Run Grad-CAM++ screening for the desired blond class.
4. Retain up to four automatically ranked semantic regions.
5. Generate every canonical non-empty subset of those regions for each source
   image, using seed 42 and 35 denoising steps.
6. Merge both GPU shards and reject incomplete runs.
7. Freeze a classifier-specific influence graph using the existing Pareto
   target-effect, flip-rate, and mask-area frontier.
8. Select the Pareto candidate with the greatest target effect per unit mask
   area.

The smile subtree and its existing artifacts are never read as resumable hair
state, overwritten, or regenerated.

## Validation and Full Run

Before the full submission, run two blond-hair samples end to end and require:

- two unique eligible sources;
- all expected region-subset rows;
- zero intervention failures;
- a valid frozen graph and selected execution graph;
- all output paths under `blond_hair/`.

After validation, submit a separate 100-image Kaggle run using both T4 GPUs.
The final review reports coverage, failures, flip rate, mean and median target
effect, mask fraction, identity cosine, non-target drift, Pareto membership,
and the selected generation regions.

## Interpretation

Edges represent classifier-specific effects under masked diffusion
interventions. They are not biological causal relationships. The discovery
cohort is used to select a global graph; a later held-out evaluation is
required to establish downstream CCI performance.

