# CCI Component Ablation Design

## Goal

Measure whether each CCI component contributes independently on 10 matched smile-removal and 10 matched blond-hair interventions.

## Experimental Matrix

Use 35 configured DDIM steps, seed 42, batch size one, the x4/y4/f3 generation mask, and identical task cohorts. Compare raw BLD, full CCI, and one-removal variants for predicted-clean guidance, target guidance, EMA gradient normalization, adaptive dual feedback, conflict projection, target-first constraint budgeting, the middle-only guidance schedule, and final correction.

All variants must consume identical initial and per-step source-blend random tensors. Runs are sequential on Apple MPS.

## Measurements

Primary validity is independent ACE directional flip rate. Also report ACE target accuracy and desired probability, same-classifier flip and strong-target rates, FVA cosine, FS, MNAC, CD, online identity/locality/TV pass rates, FID, sFID, runtime, and paired per-sample deltas.

Internal trace summaries report target probability change, constraint residuals and multipliers, projection frequency, constraint scale, update norm, and final-correction rescue count.

## Interpretation

A component is supported only when the full system outperforms its one-removal variant on the component's intended metric without a larger adverse change elsewhere. Ten samples per task are a diagnostic pilot, not population-level proof. Report uncertainty and avoid significance claims when the sample is underpowered.

