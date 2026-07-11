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

## Dry-Run The Legacy SD2 Bridge

```bash
PYTHONPATH=src python3 scripts/run_sd2_cci_from_legacy.py \
  --cci_config examples/smile_intervention.json \
  --legacy_script ../thesis_2025/bld_reranking/bld/scripts/text_editing_SD2.py \
  --init_image data/1.jpg \
  --mask data/1_mask.png \
  --classifier_path models/classifier.pth \
  --output_dir outputs/sample_1 \
  --batch_size 2 \
  --device cuda \
  --dry-run
```

## GPU Implementation Target

The first GPU adapter should wrap the old SD2 inference loop in:

`../thesis_2025/bld_reranking/bld/scripts/text_editing_SD2.py`

The CCI hook belongs between classifier-free guidance and `scheduler.step`, where
the current loop computes `noise_pred` and updates `latents`.
