# Seed-42 Full-Database Paper Visual Design

## Goal

Revise the standalone CCI manuscript so it describes the final method without
presenting subset experiments as final evidence. The final benchmark will use
the complete evaluation database and one deterministic generation seed,
42. The method figure will show real pipeline artifacts rather than only text
boxes.

## Scope

- Use seed 42 as the generation protocol.
- Remove multi-seed candidate generation, set-level FID selection, and all
  claims attributed to that mechanism.
- Remove quantitative CCI findings obtained from 10, 45, 50, or 100-image
  subsets.
- Retain metric definitions and a full-database evaluation protocol so final
  values can be inserted after the complete run.
- Remove the Reproducibility and Provenance appendix, including local paths,
  hashes, and run-specific implementation caveats.
- Preserve equations and explanations for graph policy, predicted-clean
  feedback, target priority, constraint handling, BLD blending, latent
  correction, and localized saved-image boundary correction.

## Pipeline Figure

The figure will use one visually strong smile-to-neutral seed-42 example and
show:

1. source image;
2. semantic face-part mask and Grad-CAM++ saliency;
3. early, middle, and late predicted-clean estimates captured from the actual
   CCI-BLD denoising trajectory;
4. raw decoded CCI-BLD output;
5. localized correction mask;
6. final corrected output.

Arrows and short stage labels may explain operations, but images remain the
dominant visual signal. Every displayed intermediate must come from the
executed pipeline; no synthetic approximation will be labeled as a denoising
state.

## Example Selection

Candidate examples are filtered using generation-classifier success, identity
similarity, and locality, then inspected for visible target change and obvious
artifacts. Sample `00131` is used because it shows a clear, realistic
smile-to-neutral transition and genuinely exercises the localized saved-image
correction. Its rerun raw and corrected outputs are byte-identical to the
previous seed-42 artifacts.

## Evaluation Presentation

The manuscript will describe FID, FVA, FS, MNAC, CD, locality, target flip
rate, and transfer diagnostics as full-database metrics. Existing subset
tables and numerical conclusions will be removed rather than relabeled.
Published comparison numbers may remain explicitly identified as prior work;
the current method will not receive a row until its full-database evaluation
is complete.

## Validation

- The LaTeX document must compile without unresolved references or overfull
  boxes.
- The pipeline figure must render legibly at two-column page width.
- A text scan must find no claims based on 10, 45, 50, or 100-image subsets and
  no set-level FID-selection method.
- The PDF must pass `qpdf --check`.
- No benchmark outputs are overwritten and no git commit is created.
