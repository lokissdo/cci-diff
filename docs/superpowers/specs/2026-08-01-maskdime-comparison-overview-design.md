# End-to-End Overview and MaskDiME Comparison Design

## Goal

Revise the conference paper so its first figure explains the complete proposed
method, without presenting BLD as a stage of that method, and add literature
context using the CelebA-HQ Smile values reported by MaskDiME.

## Selected Design

The user's instruction selects a hybrid overview plus a separate comparison:

1. Figure 1 becomes the end-to-end method overview. Its offline row shows
   discovery images, Grad-CAM++ proposals, paired region interventions, and the
   frozen influence graph. Its online row uses real evaluated imagery to show
   source image, source-specific semantic intervention mask, predicted-clean
   lexicographic trust-region CCI, and final counterfactual. The graph connects
   to the source-specific mask through a dashed verified-policy edge. BLD does
   not appear in this figure or its caption.
2. A later qualitative figure contains the BLD-versus-ours image pairs. It is
   explicitly a baseline comparison, not part of the method flow.
3. A full-width literature table reproduces the MaskDiME paper's CelebA-HQ Smile
   rows supplied by the user and appends our metrics below a protocol separator.
   Published values are cited to MaskDiME and are not independently recomputed.

## Comparison Semantics

The literature table is contextual, not a controlled leaderboard. MaskDiME
uses CelebA-HQ at 256 by 256 and reports ACE-style split-FID averaged over ten
random splits. Our experiment uses CelebAMask-HQ images at 512 by 512, a fixed
mouth support, a localized post-generation attack, and a deterministic
two-direction split estimator. The caption and results prose must state these
differences and must not bold a winner across the protocol separator.

Only the Smile block is included because the current paper evaluates smile
removal and has no verified Age result. Missing values remain dashes exactly as
reported. Our row continues to use generated macros rather than copied numeric
literals.

## Alternatives Considered

- Keeping BLD in Figure 1 would preserve the existing comparison but conflicts
  with the requested interpretation of Figure 1 as the proposed method.
- A diagram-only overview would be structurally correct but would lose the
  realistic input-to-output evidence requested for the paper.
- Mixing published CelebA-HQ results and our attacked CelebAMask-HQ values without
  a separator would look simpler but would imply comparability that the source
  protocols do not support.

## Manuscript Changes

- Replace the current introductory image comparison with the hybrid method
  overview and remove the now-redundant standalone framework figure.
- Update Introduction and Overall Framework references to Figure 1.
- Add the qualitative BLD-versus-ours figure under Qualitative Results.
- Add the MaskDiME-sourced Smile comparison after the controlled BLD comparison
  and explain the protocol boundary in one concise paragraph.
- Keep the abstract's verified controlled claims unchanged.

## Verification

- Figure 1 and its caption contain no BLD reference.
- The literature table contains every supplied CelebA-HQ Smile row and eight
  metrics in the supplied order.
- The MaskDiME citation resolves, table provenance is explicit, and no direct
  state-of-the-art claim is made from cross-protocol values.
- The PDF has no overflow, undefined reference, clipped figure, or unreadable
  table diagnostics.
- Existing metric builders and trust-region tests continue to pass.
