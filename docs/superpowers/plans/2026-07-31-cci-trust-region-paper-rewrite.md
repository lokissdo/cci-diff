# Lexicographic Trust-Region CCI Paper Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a concise, method-first LaTeX paper for A11 lexicographic trust-region CCI, supported by the completed paired 200-image clean experiment and explicit placeholders for the unfinished 300-image attacked experiments.

**Architecture:** Preserve the archived primal-dual manuscript and create a separate `paper/cci_trust_region.tex`. Generate completed table values reproducibly from the two saved clean-cohort CSVs and the restoration-ablation JSON, then compile and audit the manuscript against the implemented A11 code path and the approved editorial specification.

**Tech Stack:** Python 3.10+, standard-library CSV/JSON/statistics, pytest, LaTeX `article`, TikZ, BibTeX through Tectonic, repository experiment artifacts.

## Global Constraints

- Follow `docs/superpowers/specs/2026-07-31-cci-trust-region-paper-rewrite-design.md`.
- Preserve `paper/cci_conference_v1.tex` unchanged.
- Create `paper/cci_trust_region.tex` and `paper/cci_trust_region.pdf`.
- Use A11 (`trust_region`) as the proposed method, A10 (`fixed_trust_matched`) as the matched fixed comparator, and A0 (`disabled`) as the BLD baseline.
- Treat the clean evidence as two disjoint 100-image cohorts, 200 unique paired source IDs per method, with no post-generation attack.
- Do not derive claims from partial attacked cohorts.
- Every attacked-result cell must say `Pending (300-image run)` until the corresponding run is complete.
- State that COUT, FR, MNAC, and CD use the same frozen multi-label classifier that guides CCI.
- Describe final restoration as fixed-VAE latent optimization with no U-Net or scheduler transition.
- Do not claim causal identification, convergence, classifier independence, or superiority on incomplete results.
- `paper/` is intentionally ignored; use `git add -f` only for the new manuscript, its generated evidence files, and the compiled PDF.

---

### Task 1: Reproducible Paper Evidence

**Files:**
- Create: `scripts/build_cci_paper_metrics.py`
- Create: `tests/test_build_cci_paper_metrics.py`
- Create: `paper/generated/cci_trust_region_metrics.json`
- Create: `paper/generated/cci_trust_region_metrics.tex`

**Interfaces:**
- Consumes: one or more clean `pilot_results.csv` files, a required per-cohort sample count, and `outputs/final_restoration_ablation_26811/comparison.json`.
- Produces: `build_metrics(cohort_paths: list[Path], expected_per_cohort: int, ablation_path: Path) -> dict[str, object]`, a provenance JSON file, and LaTeX commands consumed through `\input{generated/cci_trust_region_metrics.tex}`.

- [ ] **Step 1: Write focused aggregation tests**

Create `tests/test_build_cci_paper_metrics.py` with fixtures that write two
disjoint two-source cohorts containing paired A0/A10/A11 rows. Verify count,
flip rate, feasibility, means, median runtime, restoration fields, and duplicate
source rejection:

```python
import csv
import json
from pathlib import Path

import pytest

from scripts.build_cci_paper_metrics import build_metrics, write_tex


FIELDS = (
    "feature", "sample_id", "variant", "target_pass", "feasible",
    "identity_cosine", "strict_outside_mae", "non_target_drift",
    "runtime_seconds",
)


def write_cohort(path: Path, ids: tuple[int, ...], offset: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for sample_id in ids:
            for variant, flip, feasible, identity, outside, drift, runtime in (
                ("A0", False, False, 0.80, 4.0, 0.08, 20.0),
                ("A10", True, True, 0.90, 4.1, 0.07, 80.0),
                ("A11", True, True, 0.91, 4.2, 0.06, 90.0),
            ):
                writer.writerow({
                    "feature": "smile",
                    "sample_id": sample_id,
                    "variant": variant,
                    "target_pass": flip,
                    "feasible": feasible,
                    "identity_cosine": identity + offset,
                    "strict_outside_mae": outside,
                    "non_target_drift": drift,
                    "runtime_seconds": runtime,
                })


def write_ablation(path: Path) -> None:
    path.write_text(json.dumps({
        "sample_id": 26811,
        "before": {
            "desired_probability": 0.10,
            "identity_cosine": 0.83,
            "mean_non_target_drift": 0.03,
            "wall_seconds": 53.0,
        },
        "after": {
            "desired_probability": 0.79,
            "identity_cosine": 0.87,
            "mean_non_target_drift": 0.06,
            "wall_seconds": 85.0,
        },
        "pixel": {
            "mean_absolute_difference": 0.0008,
            "maximum_absolute_difference": 0.29,
            "changed_fraction": 0.04,
        },
        "restoration": {"accepted_steps": 8},
    }), encoding="utf-8")


def test_build_metrics_aggregates_disjoint_paired_cohorts(tmp_path: Path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    ablation = tmp_path / "ablation.json"
    write_cohort(first, (1, 2), 0.00)
    write_cohort(second, (3, 4), 0.02)
    write_ablation(ablation)

    result = build_metrics([first, second], 2, ablation)

    assert result["cohort"]["unique_sources"] == 4
    assert result["methods"]["A11"]["count"] == 4
    assert result["methods"]["A11"]["target_pass_rate"] == pytest.approx(1.0)
    assert result["methods"]["A11"]["mean_identity_cosine"] == pytest.approx(0.92)
    assert result["restoration"]["accepted_steps"] == 8

    tex_path = tmp_path / "metrics.tex"
    write_tex(result, tex_path)
    assert "\\newcommand{\\AdaptiveCCISampleCount}{4}" in tex_path.read_text()
    assert "\\newcommand{\\AdaptiveCCITargetPassRatePct}{100.0}" in tex_path.read_text()


def test_build_metrics_rejects_cross_cohort_duplicate_sources(tmp_path: Path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    ablation = tmp_path / "ablation.json"
    write_cohort(first, (1, 2), 0.00)
    write_cohort(second, (2, 3), 0.00)
    write_ablation(ablation)

    with pytest.raises(ValueError, match="cohorts are not disjoint"):
        build_metrics([first, second], 2, ablation)
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
.venv-ml/bin/python -m pytest -q tests/test_build_cci_paper_metrics.py
```

Expected: FAIL because `scripts.build_cci_paper_metrics` does not exist.

- [ ] **Step 3: Implement the evidence builder**

Create `scripts/build_cci_paper_metrics.py` with:

```python
#!/usr/bin/env python3
"""Build reproducible clean-study and restoration metrics for the CCI paper."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Iterable


VARIANTS = ("A0", "A10", "A11")
PREFIXES = {"A0": "RawBLD", "A10": "FixedCCI", "A11": "AdaptiveCCI"}


def _truth(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty cohort: {path}")
    return rows


def _mean(rows: Iterable[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def build_metrics(
    cohort_paths: list[Path],
    expected_per_cohort: int,
    ablation_path: Path,
) -> dict[str, object]:
    all_rows: list[dict[str, str]] = []
    seen_cohort_ids: set[int] = set()
    cohort_records: list[dict[str, object]] = []
    for path in cohort_paths:
        rows = _read_rows(path)
        ids_by_variant = {
            variant: {int(row["sample_id"]) for row in rows if row["variant"] == variant}
            for variant in VARIANTS
        }
        if any(len(ids) != expected_per_cohort for ids in ids_by_variant.values()):
            raise ValueError(f"unexpected paired count in {path}: {ids_by_variant}")
        if not (ids_by_variant["A0"] == ids_by_variant["A10"] == ids_by_variant["A11"]):
            raise ValueError(f"variants are not paired in {path}")
        cohort_ids = ids_by_variant["A0"]
        overlap = seen_cohort_ids & cohort_ids
        if overlap:
            raise ValueError(f"cohorts are not disjoint: {sorted(overlap)}")
        seen_cohort_ids.update(cohort_ids)
        cohort_records.append({"path": str(path), "source_count": len(cohort_ids)})
        all_rows.extend(rows)

    methods: dict[str, object] = {}
    for variant in VARIANTS:
        rows = [row for row in all_rows if row["variant"] == variant]
        methods[variant] = {
            "count": len(rows),
            "target_pass_rate": statistics.fmean(_truth(row["target_pass"]) for row in rows),
            "mean_identity_cosine": _mean(rows, "identity_cosine"),
            "mean_strict_outside_mae": _mean(rows, "strict_outside_mae"),
            "mean_non_target_drift": _mean(rows, "non_target_drift"),
            "median_runtime_seconds": statistics.median(
                float(row["runtime_seconds"]) for row in rows
            ),
        }

    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    restoration = {
        "sample_id": int(ablation["sample_id"]),
        "before": ablation["before"],
        "after": ablation["after"],
        "pixel": ablation["pixel"],
        "accepted_steps": int(ablation["restoration"]["accepted_steps"]),
    }
    return {
        "cohort": {
            "cohorts": cohort_records,
            "unique_sources": len(seen_cohort_ids),
            "post_attack": False,
        },
        "methods": methods,
        "restoration": restoration,
    }


def write_tex(payload: dict[str, object], path: Path) -> None:
    lines = ["% Generated by scripts/build_cci_paper_metrics.py; do not edit."]
    methods = payload["methods"]
    for variant in VARIANTS:
        values = methods[variant]
        prefix = PREFIXES[variant]
        commands = {
            "SampleCount": f'{values["count"]}',
            "TargetPassRatePct": f'{100 * values["target_pass_rate"]:.1f}',
            "IdentityCosine": f'{values["mean_identity_cosine"]:.4f}',
            "OutsideMAE": f'{values["mean_strict_outside_mae"]:.3f}',
            "NonTargetDrift": f'{values["mean_non_target_drift"]:.4f}',
            "RuntimeMedian": f'{values["median_runtime_seconds"]:.1f}',
        }
        lines.extend(
            f"\\newcommand{{\\{prefix}{name}}}{{{value}}}"
            for name, value in commands.items()
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", action="append", type=Path, required=True)
    parser.add_argument("--expected_per_cohort", type=int, default=100)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--json_out", type=Path, required=True)
    parser.add_argument("--tex_out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_metrics(args.cohort, args.expected_per_cohort, args.ablation)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_tex(payload, args.tex_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests and generate the real evidence files**

Run:

```bash
.venv-ml/bin/python -m pytest -q tests/test_build_cci_paper_metrics.py
.venv-ml/bin/python scripts/build_cci_paper_metrics.py \
  --cohort outputs/trust_region_random100_seed42/pilot_results.csv \
  --cohort outputs/trust_region_random100_seed42_part2/pilot_results.csv \
  --expected_per_cohort 100 \
  --ablation outputs/final_restoration_ablation_26811/comparison.json \
  --json_out paper/generated/cci_trust_region_metrics.json \
  --tex_out paper/generated/cci_trust_region_metrics.tex
```

Expected: tests PASS; generated JSON reports 200 unique sources and 200 rows
for each of A0/A10/A11.

- [ ] **Step 5: Commit the evidence builder and generated evidence**

```bash
git add scripts/build_cci_paper_metrics.py tests/test_build_cci_paper_metrics.py
git add -f paper/generated/cci_trust_region_metrics.json paper/generated/cci_trust_region_metrics.tex
git commit -m "docs: generate trust-region paper evidence"
```

---

### Task 2: Method-First Manuscript

**Files:**
- Create: `paper/cci_trust_region.tex`
- Read: `paper/references.bib`
- Read: `src/cci_diff/trust_region_controller.py`
- Read: `src/cci_diff/adapters/sd2_clean_cci.py`
- Read: `src/cci_diff/trust_region_solver.py`

**Interfaces:**
- Consumes: `paper/generated/cci_trust_region_metrics.tex`, the approved rewrite specification, the current A11 implementation, and existing BibTeX keys.
- Produces: a standalone two-column manuscript with completed clean results, an illustrative restoration ablation, and two attacked-result placeholder tables.

- [ ] **Step 1: Create the manuscript shell**

Create `paper/cci_trust_region.tex` with the portable preamble and evidence
input:

```latex
\documentclass[10pt,twocolumn]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[letterpaper,margin=0.72in,columnsep=0.22in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,microtype,tikz}
\usetikzlibrary{arrows.meta,positioning,fit}
\usepackage[hidelinks]{hyperref}
\usepackage[font=small,labelfont=bf]{caption}
\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.08}
\newcommand{\method}{\textsc{TR-CCI}}
\newcommand{\bld}{\textsc{BLD}}
\input{generated/cci_trust_region_metrics.tex}
\title{Lexicographic Trust-Region Guidance for Localized Diffusion Counterfactuals}
\author{Anonymous Author(s)}
\date{}
```

Use exactly these main sections:

```latex
\section{Introduction}
\section{Related Work}
\section{Lexicographic Trust-Region CCI}
\section{Experimental Protocol}
\section{Results}
\section{Limitations}
\section{Conclusion}
```

- [ ] **Step 2: Write the abstract, introduction, and related work**

The abstract must state the method hierarchy and completed study scope without
claiming attacked results. The introduction must explain why a permanent
weighted sum is fragile, define the target/safety/drift ordering, and state
only these contributions:

```text
1. predicted-clean lexicographic trust-region guidance inside BLD;
2. a matched fixed/adaptive comparison in clean-latent coordinates;
3. preservation-aware final restoration with deterministic acceptance traces.
```

Compress related work to diffusion counterfactuals, BLD, predicted-clean
guidance, and multi-objective gradient coordination using existing keys in
`paper/references.bib`.

- [ ] **Step 3: Write the method and pipeline figure**

Include definitions for desired target probability, the semantic/generation
masks, all-39-attribute non-target drift, identity, and outside-mask locality.
Present the predicted-clean equations:

```latex
\hat z_0=\frac{z_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(z_t,t)}
                 {\sqrt{\bar\alpha_t}},\qquad
\hat x_0=D(\hat z_0/s_{\mathrm{VAE}}).
```

Explain the decoder gradient as
`$g=J_D(\hat z_0)^\top\nabla_x L$`, with the U-Net prediction detached.
Present the two lexicographic subproblems from the specification: first minimize
the best attainable identity/locality envelope under requested target
progress, then minimize non-target drift inside that envelope. State that the
small problem is solved deterministically in the span of the masked gradients.

Include the clean-to-epsilon mapping:

```latex
\Delta\epsilon_t=-\sqrt{\frac{\bar\alpha_t}{1-\bar\alpha_t}}\,d_t.
```

Use a TikZ figure with a dashed box around the denoising loop and a separate
solid box for final restoration. Label restoration `VAE decode + evaluators +
backtracking; no U-Net/scheduler`.

Describe final restoration with at most 12 iterations, fractions
`$\{1,1/2,1/4,1/8\}$`, a cumulative clean-latent radius, target/safety/drift
acceptance rules, and saved uint8 re-evaluation.

- [ ] **Step 4: Write the experimental protocol and completed clean table**

State the exact completed protocol: CelebAMask-HQ smile removal, fixed
mouth/upper-lip/lower-lip semantic union,
two disjoint 100-image random cohorts, 200 unique paired sources, seed 42,
SD2.1 base, DDIM, 35 steps, CFG 5.0, blend start 0.25, 512 square pixels,
float32 MPS, batch one, and no post-generation attack.

Describe the arms as A0 raw BLD, A10 matched fixed CCI, and A11 adaptive
lexicographic CCI. Do not describe A0 runtime as CCI runtime.

Populate the table exclusively with generated commands:

```latex
\begin{tabular}{lrrrrr}
\toprule
Method & Target@0.8 & Identity & Outside & NT drift & Time \\
\midrule
A0 BLD & \RawBLDTargetPassRatePct &
\RawBLDIdentityCosine & \RawBLDOutsideMAE &
\RawBLDNonTargetDrift & \RawBLDRuntimeMedian \\
A10 fixed & \FixedCCITargetPassRatePct &
\FixedCCIIdentityCosine & \FixedCCIOutsideMAE &
\FixedCCINonTargetDrift & \FixedCCIRuntimeMedian \\
A11 adaptive & \AdaptiveCCITargetPassRatePct &
\AdaptiveCCIIdentityCosine & \AdaptiveCCIOutsideMAE &
\AdaptiveCCINonTargetDrift & \AdaptiveCCIRuntimeMedian \\
\bottomrule
\end{tabular}
```

Define Target@0.8 as reaching the declared desired probability 0.8 under the
explained classifier. Reserve FR for the pending attacked evaluation at the
0.5 decision boundary. Identity is FaceNet cosine, outside is strict
outside-mask MAE, and NT drift is mean absolute
probability drift over all non-target attributes, and time as per-image median
seconds.

- [ ] **Step 5: Write cautious results and restoration ablation**

Discuss only differences visible in the generated 200-image aggregation.
Avoid significance language because no confidence interval is computed here.
State explicitly that the clean study evaluates the same classifier used for
guidance.

Add the sample-26811 restoration table using values from
`paper/generated/cci_trust_region_metrics.json`; label it an illustrative
mechanism ablation. Report before/after desired probability, identity cosine,
non-target drift, runtime, pixel MAE, maximum pixel change, changed fraction,
and eight accepted steps. Explain that this example demonstrates mechanism,
not population-level improvement.

- [ ] **Step 6: Add the two attacked-result placeholder tables**

Create separate tables titled:

```text
End-to-end attacked evaluation, mouth-only support (300 paired images)
End-to-end attacked evaluation, mouth + upper/lower lip support (same 300 IDs)
```

Each table has rows `A0 BLD` and `A11 adaptive CCI`, columns `Method`, `FID`,
`sFID`, `FVA`, `FS`, `MNAC`, `CD`, `COUT`, and `FR (\%)`, and every metric
cell contains `\multicolumn{8}{c}{Pending (300-image run)}` or an equivalent
explicitly pending row. State that COUT, FR, MNAC, and CD will use the same
frozen multi-label classifier that guides CCI.

- [ ] **Step 7: Write limitations and conclusion**

Cover same-classifier evaluation, one task/dataset, fixed masks, runtime from
repeated decoder/evaluator backward passes and restoration line searches,
absence of convergence guarantees, unfinished attacked results, and facial
data ethics. The conclusion may summarize the formulation and completed clean
study scope but must not claim superiority from the pending tables.

- [ ] **Step 8: Commit the manuscript source**

```bash
git add -f paper/cci_trust_region.tex
git commit -m "docs: write lexicographic trust-region CCI paper"
```

---

### Task 3: Manuscript Verification and PDF

**Files:**
- Modify: `paper/cci_trust_region.tex` only if verification reveals errors
- Create: `paper/cci_trust_region.pdf`

**Interfaces:**
- Consumes: the completed manuscript, generated metric commands, and `paper/references.bib`.
- Produces: a successfully compiled PDF and an evidence-backed claim audit.

- [ ] **Step 1: Verify the evidence builder against real artifacts**

Run:

```bash
.venv-ml/bin/python -m pytest -q tests/test_build_cci_paper_metrics.py
.venv-ml/bin/python scripts/build_cci_paper_metrics.py \
  --cohort outputs/trust_region_random100_seed42/pilot_results.csv \
  --cohort outputs/trust_region_random100_seed42_part2/pilot_results.csv \
  --expected_per_cohort 100 \
  --ablation outputs/final_restoration_ablation_26811/comparison.json \
  --json_out paper/generated/cci_trust_region_metrics.json \
  --tex_out paper/generated/cci_trust_region_metrics.tex
jq '.cohort, .methods, .restoration.accepted_steps' \
  paper/generated/cci_trust_region_metrics.json
```

Expected: tests PASS; 200 unique sources; A0/A10/A11 each contain 200 rows;
restoration reports eight accepted steps.

- [ ] **Step 2: Run the obsolete-claim and placeholder audit**

Run:

```bash
rg -n -i 'dual multiplier|conflict projection|complete.database|independent oracle|blond.hair' \
  paper/cci_trust_region.tex
rg -n -F 'Pending (300-image run)' paper/cci_trust_region.tex
```

Expected: the obsolete-term search prints nothing; the placeholder search
finds both attacked-result tables.

- [ ] **Step 3: Compile with Tectonic**

Run from `paper/`:

```bash
tectonic --keep-logs --keep-intermediates cci_trust_region.tex
```

Expected: exit code 0 and `paper/cci_trust_region.pdf` exists. Resolve any
undefined references, missing BibTeX keys, overfull tables, or missing files,
then rerun until the build is clean apart from harmless font warnings.

- [ ] **Step 4: Inspect text and page geometry**

Run:

```bash
pdfinfo paper/cci_trust_region.pdf | rg 'Pages|Page size'
pdftotext paper/cci_trust_region.pdf - | rg -n \
  'Lexicographic Trust-Region|200 unique|Pending \(300-image run\)|no U-Net'
qlmanage -t -s 1400 -o /tmp paper/cci_trust_region.pdf
```

Expected: letter-sized pages; extracted text includes the method title,
completed cohort size, pending-result labels, and final-restoration distinction.
Inspect the generated preview for clipped equations, overlapping tables, and
unreadable figure labels.

- [ ] **Step 5: Run repository regression tests proportionate to the change**

Run:

```bash
.venv-ml/bin/python -m pytest -q \
  tests/test_build_cci_paper_metrics.py \
  tests/test_trust_region_controller.py \
  tests/test_sd2_clean_cci.py \
  tests/test_final_restoration_ablation.py
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit the verified evidence and PDF**

```bash
git add scripts/build_cci_paper_metrics.py tests/test_build_cci_paper_metrics.py
git add -f paper/cci_trust_region.tex \
  paper/cci_trust_region.pdf \
  paper/generated/cci_trust_region_metrics.json \
  paper/generated/cci_trust_region_metrics.tex
git commit -m "docs: verify trust-region CCI manuscript"
```
