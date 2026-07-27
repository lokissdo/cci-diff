# Adaptive Post-Attack Escalation Design

## Goal

Make the optional smooth boundary post-attack satisfy difficult target flips without
applying the strongest perturbation to every image. The default policy escalates the
maximum perturbation only when the saved-image classifier still fails.

## Policy

For each CCI-BLD candidate:

1. Keep the original generated candidate unchanged on disk.
2. Try epsilon budgets `0.05`, `0.08`, `0.10`, `0.30`, and `0.50` in
   order. The larger budgets are reached only when all smaller budgets fail
   after 8-bit quantization.
3. Start every attempt from the same original candidate, never from the preceding
   attacked result.
4. Quantize each attempted result to 8-bit RGB before checking the target margin.
   This models the PNG that evaluation will read and avoids selecting a candidate
   whose in-memory score is lost during serialization.
5. Stop at the first attempt that satisfies the configured target margin.
6. If every budget fails, retain the result from the largest budget and record the
   failure explicitly.

An explicitly supplied single epsilon remains available for controlled ablations.

## Minimal-Change Rationale

The schedule is a per-image line search over an upper perturbation bound. Easy
examples retain the `0.05` result, while only unresolved examples can reach
`0.08`, `0.10`, `0.30`, or `0.50`. Because retries restart from the original
image, `epsilon` remains a true global bound rather than accumulating across
attempts. A `0.03` classifier margin is used by default so an in-memory
boundary crossing remains valid after 8-bit PNG serialization.

The existing soft anatomical mask, boundary smoothing, masked update, and
quantization-aware boundary refinement remain unchanged. Escalation changes only
the permitted search radius.

## Audit Contract

The post-attack audit records:

- the configured epsilon schedule;
- one attempt record per executed epsilon;
- internal and quantized target probabilities;
- target and margin pass states;
- iterations and boundary-refinement iterations;
- the selected epsilon and whether escalation occurred;
- final metrics recomputed from the saved corrected PNG.

The raw `sd2_bld_grid.png` remains the uncorrected CCI-BLD output.
`sd2_bld_grid_corrected.png` contains the selected attacked result.

## Failure Handling

The schedule must contain finite, positive, strictly increasing values. Invalid
schedules fail before generation. If no attempt reaches the target margin, the
audit reports `success: false`; the system does not silently claim a flip.

## Verification

Tests cover schedule parsing, fixed-epsilon compatibility, retry restart semantics,
early stopping, quantized-score selection, exhausted schedules, and audit fields.
The complete test suite is run after focused tests.
