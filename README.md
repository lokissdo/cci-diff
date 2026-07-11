# CCI-Diff

CCI-Diff is a clean research scaffold for **Causal Concept Intervention Diffusion**.
It is meant to grow from the ESWA thesis code in `../thesis_2025` without copying
the old pipeline wholesale.

## Current Slice

- Pure Python concept intervention specs.
- Framework-neutral guidance loss composition.
- CCI metrics for concept leakage, preservation, causal concept effect, purity, and bias audit matrices.
- Prompt helper for converting an intervention spec into target/preservation prompt text.
- Standard-library tests, runnable without GPU dependencies.

## Local Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## GPU Implementation Target

The first GPU adapter should wrap the old SD2 inference loop in:

`../thesis_2025/bld_reranking/bld/scripts/text_editing_SD2.py`

The CCI hook belongs between classifier-free guidance and `scheduler.step`, where
the current loop computes `noise_pred` and updates `latents`.
