# Result-Led Conference Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Turn the current manuscript into a result-led conference paper with verified end-to-end metrics and a real Source → Mask → BLD → Ours qualitative teaser.

**Architecture:** Add one focused evidence builder that validates the complete paired cohort and emits stable LaTeX macros, and one focused image builder that creates paper-owned overlays and crops with provenance hashes. Rewrite the current LaTeX source to consume those generated artifacts, replace incomplete and earlier clean results with one completed end-to-end table, and keep a compact vector framework figure separate from the qualitative teaser.

**Tech Stack:** Python 3.10, csv/json/hashlib, Pillow, pytest, LaTeX, TikZ, Tectonic, qpdf

## Global Constraints

- The abstract must say “on the full paired end-to-end evaluation cohort” and must not state the cohort size.
- State \(N=300\) exactly once, neutrally, in Experimental Protocol.
- Do not repeat the cohort size in results, captions, conclusions, or tables.
- Use only verified values from outputs/attacked_a0_a11_smile300_seed42/mouth/metrics/full_metrics.csv and the aligned pair table.
- Expand sFID as “symmetric FID,” never “spatial FID.”
- Report BLD and Adaptive trust-region CCI (Ours); do not expose A0/A11 in the paper.
- Remove the old clean comparison, restoration result, pending tables, and incomplete second-region study from the paper.
- Label Figure 1 masks “semantic intervention mask”; do not imply automatic discovery.
- Keep classifier-causal claims inside the frozen classifier–generator system.
- Do not modify the generation or evaluation algorithms.
- Work directly on main, as previously authorized; do not create a worktree.

---

### Task 1: Build Verified End-to-End Paper Metrics

**Files:**
- Create: scripts/build_cci_conference_metrics.py
- Create: tests/test_build_cci_conference_metrics.py
- Create: paper/generated/cci_end_to_end_metrics.json
- Create: paper/generated/cci_end_to_end_metrics.tex

**Interfaces:**
- Consumes: full_metrics.csv with one smile row for A0 and A11; ace_pair_metrics.csv with paired feature/sample/variant rows
- Produces: build_metrics(full_metrics_path: Path, pair_metrics_path: Path, expected_count: int) -> dict[str, object] and write_outputs(payload: dict[str, object], json_path: Path, tex_path: Path) -> None

- [ ] **Step 1: Write failing tests for cohort validation, aggregation reconciliation, and TeX formatting**

Create tests/test_build_cci_conference_metrics.py with synthetic full and pair
tables. The primary test must assert:

    result = build_metrics(full_path, pair_path, expected_count=2)
    assert result["cohort"]["paired_count"] == 2
    assert result["methods"]["A0"]["directional_fr"] == pytest.approx(0.5)
    assert result["methods"]["A11"]["directional_fr"] == pytest.approx(1.0)
    assert result["deltas"]["fr_percentage_points"] == pytest.approx(50.0)
    assert result["deltas"]["cout_gain"] == pytest.approx(0.4)
    assert result["deltas"]["cd_reduction_percent"] == pytest.approx(10.0)

Use a second formatting test whose synthetic aggregate rows contain the exact
completed values. After write_outputs, assert these exact commands occur:

    \newcommand{\EndToEndBLDFID}{17.3720}
    \newcommand{\EndToEndAdaptiveFRPct}{81.0}
    \newcommand{\EndToEndFRGainPctPoints}{16.7}
    \newcommand{\EndToEndCOUTGain}{0.2134}
    \newcommand{\EndToEndCDReductionPct}{1.8}

Add rejection tests for duplicate feature/sample/variant keys, mismatched A0/A11
ID sets, unexpected counts, absent methods, non-finite metrics, and disagreement
between pair-level FR/FS/MNAC/COUT/FVA and full_metrics.csv.

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

    .venv-ml/bin/python -m pytest -q tests/test_build_cci_conference_metrics.py

Expected: collection fails because scripts.build_cci_conference_metrics does
not exist.

- [ ] **Step 3: Implement the minimal validated evidence builder**

Create scripts/build_cci_conference_metrics.py with this control flow and
these exact helper interfaces:

    METHODS = ("A0", "A11")
    REQUIRED_AGGREGATES = (
        "fid", "sfid", "fva_rate", "fs", "mnac",
        "cd", "cout", "directional_fr",
    )

    def build_metrics(
        full_metrics_path: Path,
        pair_metrics_path: Path,
        expected_count: int,
    ) -> dict[str, object]:
        aggregate_rows = _read_csv(full_metrics_path)
        pair_rows = _read_csv(pair_metrics_path)
        methods = _validate_aggregates(aggregate_rows, expected_count)
        paired_count = _validate_and_reconcile_pairs(
            pair_rows, methods, expected_count
        )
        return {
            "cohort": {"paired_count": paired_count},
            "methods": methods,
            "deltas": _compute_deltas(methods),
        }

    def write_outputs(
        payload: dict[str, object],
        json_path: Path,
        tex_path: Path,
    ) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        tex_path.write_text(
            "\n".join(_tex_command_lines(payload)) + "\n",
            encoding="utf-8",
        )

    def _read_csv(path: Path) -> list[dict[str, str]]
    def _validate_aggregates(
        rows: list[dict[str, str]], expected_count: int
    ) -> dict[str, dict[str, float]]
    def _validate_and_reconcile_pairs(
        rows: list[dict[str, str]],
        methods: dict[str, dict[str, float]],
        expected_count: int,
    ) -> int
    def _compute_deltas(
        methods: dict[str, dict[str, float]]
    ) -> dict[str, float]
    def _tex_command_lines(payload: dict[str, object]) -> list[str]

Implementation requirements:

1. Require exactly one smile aggregate row for each method and no extra selected
   rows.
2. Require each aggregate n to equal expected_count.
3. Require finite values for every REQUIRED_AGGREGATES field.
4. Parse pair rows for feature smile and variants A0/A11; reject duplicate
   keys and require equal ID sets of expected_count.
5. Independently recompute directional FR, mean FS cosine, mean MNAC, mean
   COUT, and FVA rate using fva_cosine > 0.5. Compare each to the aggregate
   with absolute tolerance 1e-9.
6. Compute:

       fr_percentage_points = 100 * (adaptive_fr - bld_fr)
       cout_gain = adaptive_cout - bld_cout
       cd_reduction_percent = 100 * (bld_cd - adaptive_cd) / bld_cd

7. Emit JSON with allow_nan=False and stable TeX commands with four decimals
   for FID/symmetric FID/FS/MNAC/CD/COUT, one decimal for FVA/FR percentages,
   one decimal for FR gain and CD reduction, and four decimals for COUT gain.
8. The CLI accepts --full_metrics, --pair_metrics, --expected_count,
   --json_out, and --tex_out.

Use the prefixes EndToEndBLD for A0 and EndToEndAdaptive for A11. Emit FID,
SymmetricFID, FVAPct, FS, MNAC, CD, COUT, and FRPct for each method, followed
by EndToEndFRGainPctPoints, EndToEndCOUTGain, and
EndToEndCDReductionPct. Do not emit a sample-count command.

- [ ] **Step 4: Run the new tests and verify they pass**

Run:

    .venv-ml/bin/python -m pytest -q tests/test_build_cci_conference_metrics.py

Expected: all tests pass.

- [ ] **Step 5: Generate paper metrics from the completed experiment**

Run:

    .venv-ml/bin/python scripts/build_cci_conference_metrics.py \
      --full_metrics outputs/attacked_a0_a11_smile300_seed42/mouth/metrics/full_metrics.csv \
      --pair_metrics outputs/attacked_a0_a11_smile300_seed42/mouth/ace_pair_metrics.csv \
      --expected_count 300 \
      --json_out paper/generated/cci_end_to_end_metrics.json \
      --tex_out paper/generated/cci_end_to_end_metrics.tex

Expected headline output values:

    BLD: FID 17.3720, symmetric FID 72.5453, FVA 100.0,
         FS 0.9961, MNAC 2.4967, CD 2.9437, COUT -0.0982, FR 64.3
    Ours: FID 17.4345, symmetric FID 72.3917, FVA 100.0,
          FS 0.9958, MNAC 2.6233, CD 2.8894, COUT 0.1152, FR 81.0

- [ ] **Step 6: Commit the evidence builder**

    git add scripts/build_cci_conference_metrics.py \
      tests/test_build_cci_conference_metrics.py
    git add -f paper/generated/cci_end_to_end_metrics.json \
      paper/generated/cci_end_to_end_metrics.tex
    git commit -m "docs: generate verified end-to-end paper metrics"

### Task 2: Build Deterministic Qualitative Figure Assets

**Files:**
- Create: scripts/build_cci_teaser_assets.py
- Create: tests/test_build_cci_teaser_assets.py
- Create: paper/generated/cci_teaser_provenance.json
- Create: paper/figures/cci_teaser/10429_source.jpg
- Create: paper/figures/cci_teaser/10429_mask_overlay.png
- Create: paper/figures/cci_teaser/10429_bld.jpg
- Create: paper/figures/cci_teaser/10429_ours.jpg
- Create: paper/figures/cci_teaser/10429_bld_crop.png
- Create: paper/figures/cci_teaser/10429_ours_crop.png
- Create: corresponding six files for samples 18004 and 28408

**Interfaces:**
- Consumes: the experiment root, CelebA-HQ image root, sample IDs, and paired fixed-mask outputs
- Produces: build_teaser_assets(experiment_root: Path, image_root: Path, sample_ids: Sequence[int], output_dir: Path, provenance_path: Path) -> dict[str, object]

- [ ] **Step 1: Write failing unit tests using synthetic images**

Create tests/test_build_cci_teaser_assets.py. Build a temporary 64 by 64 RGB
source, binary mouth mask, BLD output, and adaptive output under the same
directory layout as the experiment. Assert:

    crop_box = square_crop_box(mask_image, min_side=48, scale=4.0)
    assert crop_box == (8, 8, 56, 56)

    payload = build_teaser_assets(
        experiment_root,
        image_root,
        [1],
        output_dir,
        provenance_path,
    )
    assert payload["samples"][0]["sample_id"] == 1
    assert Image.open(output_dir / "00001_source.jpg").size == (512, 512)
    assert Image.open(output_dir / "00001_mask_overlay.png").size == (512, 512)
    assert Image.open(output_dir / "00001_bld_crop.png").size == (512, 512)

Test square_crop_box directly with a centered 12 by 8 mask on a 64 by 64
canvas and require a clipped square region with side
max(48, 4 * max(mask_width, mask_height)). For build_teaser_assets, assert the
recorded box uses normalized 512 by 512 coordinates. Assert the overlay differs
from the source inside the mask and is identical outside it.
Add rejection tests for an empty mask, missing paired output, non-identical A0
and A11 semantic masks, duplicate sample IDs, and missing source image.

- [ ] **Step 2: Run the asset tests and verify they fail**

Run:

    .venv-ml/bin/python -m pytest -q tests/test_build_cci_teaser_assets.py

Expected: collection fails because scripts.build_cci_teaser_assets does not
exist.

- [ ] **Step 3: Implement the deterministic asset builder**

Create scripts/build_cci_teaser_assets.py with this control flow and these
exact helper interfaces:

    @dataclass(frozen=True)
    class SamplePaths:
        sample_id: int
        source: Path
        mask_a0: Path
        mask_a11: Path
        bld: Path
        adaptive: Path

    def build_teaser_assets(
        experiment_root: Path,
        image_root: Path,
        sample_ids: Sequence[int],
        output_dir: Path,
        provenance_path: Path,
    ) -> dict[str, object]:
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError("sample IDs must be non-empty and unique")
        output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for sample_id in sample_ids:
            paths = resolve_sample_paths(
                experiment_root, image_root, sample_id
            )
            records.append(build_sample_assets(paths, output_dir))
        payload = {
            "settings": {
                "image_size": 512,
                "overlay_alpha": 0.45,
                "crop_scale": 1.75,
                "minimum_crop_side": 128,
            },
            "samples": records,
        }
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return payload

    def resolve_sample_paths(
        experiment_root: Path, image_root: Path, sample_id: int
    ) -> SamplePaths
    def square_crop_box(
        mask: Image.Image, min_side: int, scale: float
    ) -> tuple[int, int, int, int]
    def mask_overlay(
        source: Image.Image,
        mask: Image.Image,
        color: tuple[int, int, int] = (220, 32, 96),
        alpha: float = 0.45,
    ) -> Image.Image
    def build_sample_assets(
        paths: SamplePaths, output_dir: Path
    ) -> dict[str, object]

Implementation requirements:

1. Resolve sample directories with five-digit IDs under
   experiment_root/smile/ID.
2. Load corrected end-to-end outputs from
   A0/candidates/d8/sd2_bld_grid_corrected.png and the corresponding A11 path.
3. Require A0 and A11 semantic masks to be byte-identical.
4. Normalize source and outputs to 512 by 512 RGB using LANCZOS; normalize masks
   to 512 by 512 single-channel using NEAREST.
5. Produce a magenta mask overlay with alpha 0.45 only inside the binary mask.
6. Derive a square crop centered on the mask bounding box with side
   max(128, 1.75 * max(width, height)), clipped to the image, then resize the crop
   to 512 by 512 using LANCZOS.
7. Save deterministic, paper-owned source, overlay, full-output, and crop files
   using five-digit sample IDs.
8. Record source paths, SHA-256 hashes, crop boxes, output paths, and builder
   settings in JSON with allow_nan=False.
9. The CLI accepts --experiment_root, --image_root, --sample_ids,
   --output_dir, and --provenance_out.

- [ ] **Step 4: Run the asset tests and verify they pass**

Run:

    .venv-ml/bin/python -m pytest -q tests/test_build_cci_teaser_assets.py

Expected: all tests pass.

- [ ] **Step 5: Generate the real paper-owned assets**

Run:

    .venv-ml/bin/python scripts/build_cci_teaser_assets.py \
      --experiment_root outputs/attacked_a0_a11_smile300_seed42/mouth \
      --image_root data/CelebAMask-HQ/CelebA-HQ-img \
      --sample_ids 10429 18004 28408 \
      --output_dir paper/figures/cci_teaser \
      --provenance_out paper/generated/cci_teaser_provenance.json

Expected: 18 image assets plus one provenance JSON. Visually inspect the
10429 and 18004 crops to confirm teeth/neutral-mouth differences remain clear.

- [ ] **Step 6: Commit the deterministic figure assets**

    git add scripts/build_cci_teaser_assets.py tests/test_build_cci_teaser_assets.py
    git add -f paper/figures/cci_teaser paper/generated/cci_teaser_provenance.json
    git commit -m "docs: add reproducible CCI qualitative teaser"

### Task 3: Rewrite the Manuscript Around Completed Evidence

**Files:**
- Modify: paper/cci_trust_region.tex
- Modify: paper/cci_trust_region.pdf

**Interfaces:**
- Consumes: paper/generated/cci_end_to_end_metrics.tex and paper/figures/cci_teaser/*
- Produces: a five-section result-led conference manuscript with Figure 1 qualitative evidence, Figure 2 method flow, and one complete end-to-end table

- [ ] **Step 1: Replace the generated-metric input and result-led abstract**

Replace the old metric input with:

    \input{generated/cci_end_to_end_metrics.tex}

Write the abstract so it contains this exact scope phrase:

    on the full paired end-to-end evaluation cohort

The quantitative result sentence must use generated macros and follow this
meaning:

    Adaptive trust-region CCI reaches FR 81.0%, COUT 0.115, and CD 2.889
    while retaining FVA 100.0%, FS 0.9958, FID 17.43, and symmetric FID
    72.39. Relative to BLD under the same mask and attack protocol, it gains
    16.7 percentage points in FR and 0.213 in COUT and reduces CD by 1.8%.

Do not mention the old clean result, Target@0.8, fixed-weight result, sample
count, or pending studies in the abstract.

- [ ] **Step 2: Add the full-width qualitative teaser as Figure 1**

Place Figure 1 after the Introduction’s opening motivation. Use a four-column
primary row:

    10429_source.jpg | 10429_mask_overlay.png | 10429_bld.jpg | 10429_ours.jpg

Label the columns Source, Semantic intervention mask, BLD, and Ours. Add a
second band of enlarged paired mouth crops:

    10429_bld_crop.png | 10429_ours_crop.png
    18004_bld_crop.png | 18004_ours_crop.png
    28408_bld_crop.png | 28408_ours_crop.png

Use LaTeX labels rather than rasterized text. The caption must state that all
examples use paired inputs, masks, seeds, diffusion settings, and the same
localized attack protocol; the masks come from the fixed mouth-region
evaluation; and aggregate claims come from the quantitative table.

- [ ] **Step 3: Keep the connected method and compact Figure 2**

Retain Methodology subsections:

    Preliminaries
    Overall Framework
    Classifier-Causal Mask Discovery
    Localized Trust-Region CCI

Shorten the existing TikZ framework to seven nodes:

    discovery images → Grad-CAM++ proposals → paired interventions
    → verified graph → semantic support → trust-region CCI → counterfactual

Keep the dashed graph-to-support connection and causal-scope note. Renumber it
as Figure 2 by placing it after the qualitative teaser. Preserve the essential
intervention, target residual, predicted-clean, lexicographic subproblem,
epsilon mapping, BLD, and final-restoration equations.

- [ ] **Step 4: Replace the experiment and result sections**

Under Evaluation Protocol, state exactly once:

    We evaluate \(N=300\) eligible CelebAMask-HQ sources.

Do not use the numeral 300 elsewhere in the TeX source. Describe the paired
mouth-only support, seed 42, SD2.1/DDIM settings, and common localized
smooth-boundary post-attack. Define symmetric FID as the mean of two
deterministic cross-split FIDs. State the same-classifier provenance of COUT,
FR, MNAC, and CD.

Create one table with rows BLD and Adaptive trust-region CCI (Ours), populated
only by generated macros:

    Method | FID ↓ | sym. FID ↓ | FVA ↑ | FS ↑ |
             MNAC ↓ | CD ↓ | COUT ↑ | FR (%) ↑

Bold only the best verified value in each column; FVA is a tie. The result
narrative must accurately say that ours improves FR, COUT, CD, and symmetric
FID, while FID, FVA, and FS remain closely matched. Leave MNAC visible in the
table without claiming it improves.

Add separate Quantitative Results, Qualitative Results, and Discussion
paragraphs. Remove the old clean table, restoration table, discovery
placeholder, both pending attacked tables, and all “planned” result prose.

- [ ] **Step 5: Rewrite the conclusion around the completed end-to-end result**

Summarize the connected method and use generated deltas without stating the
sample count. Retain a concise causal boundary and biometric-data ethics
sentence. Do not mention pending evidence or the incomplete region experiment.

- [ ] **Step 6: Run manuscript-policy checks**

Run:

    test "$(rg -o '\b300\b' paper/cci_trust_region.tex | wc -l | tr -d ' ')" -eq 1
    rg -n 'full paired end-to-end evaluation cohort' paper/cci_trust_region.tex
    rg -n 'symmetric FID|semantic intervention mask' paper/cci_trust_region.tex
    if rg -n 'spatial FID|Pending|planned end-to-end|200 unique|Target@0\.8|A0|A11' \
      paper/cci_trust_region.tex; then exit 1; fi
    git diff --check

Expected: the count is one; required phrases are present; the forbidden scan
produces no output.

- [ ] **Step 7: Commit the manuscript rewrite**

    git add -f paper/cci_trust_region.tex
    git commit -m "docs: make CCI paper result-led and visual"

### Task 4: Compile and Validate the Conference Paper

**Files:**
- Modify: paper/cci_trust_region.pdf
- Verify: paper/cci_trust_region.log
- Verify: paper/generated/cci_end_to_end_metrics.json
- Verify: paper/generated/cci_teaser_provenance.json

**Interfaces:**
- Consumes: completed Tasks 1–3
- Produces: a valid, visually inspected PDF with reproducible metrics and figures

- [ ] **Step 1: Run the focused and regression test suite**

Run:

    .venv-ml/bin/python -m pytest -q \
      tests/test_build_cci_conference_metrics.py \
      tests/test_build_cci_teaser_assets.py \
      tests/test_build_cci_paper_metrics.py \
      tests/test_trust_region_solver.py \
      tests/test_trust_region_controller.py \
      tests/test_sd2_clean_cci.py

Expected: all tests pass.

- [ ] **Step 2: Rebuild both generated artifact groups and require a clean diff**

Run the Task 1 Step 5 and Task 2 Step 5 commands again, then:

    git diff --check
    git status --short

Expected: deterministic generation introduces no unexpected changes.

- [ ] **Step 3: Compile the paper**

From paper/ run:

    tectonic --keep-logs --keep-intermediates cci_trust_region.tex

Expected: exit code 0 and a regenerated cci_trust_region.pdf.

- [ ] **Step 4: Validate LaTeX and PDF structure**

Run:

    if rg -n 'Overfull|Undefined|Citation.*undefined|Reference.*undefined' \
      paper/cci_trust_region.log; then exit 1; fi
    qpdf --check paper/cci_trust_region.pdf
    test "$(qpdf --show-npages paper/cci_trust_region.pdf)" -ge 6

Expected: no layout/reference diagnostics and a valid PDF.

- [ ] **Step 5: Inspect Figure 1 and Figure 2 at final scale**

Split the PDF into pages and render the pages containing Figures 1 and 2:

    mkdir -p /tmp/cci-conference-pages
    qpdf --split-pages=1 paper/cci_trust_region.pdf \
      /tmp/cci-conference-pages/page-%d.pdf
    qlmanage -t -s 1800 -o /tmp/cci-conference-pages \
      /tmp/cci-conference-pages/page-1.pdf \
      /tmp/cci-conference-pages/page-2.pdf \
      /tmp/cci-conference-pages/page-3.pdf

Inspect the rendered PNGs. Require legible column labels, visible mouth
differences, no stretched faces, no clipped captions, and a readable compact
method flow.

- [ ] **Step 6: Run final evidence and policy verification**

Run:

    .venv-ml/bin/python -m pytest -q \
      tests/test_build_cci_conference_metrics.py \
      tests/test_build_cci_teaser_assets.py \
      tests/test_build_cci_paper_metrics.py \
      tests/test_trust_region_solver.py \
      tests/test_trust_region_controller.py \
      tests/test_sd2_clean_cci.py
    test "$(rg -o '\b300\b' paper/cci_trust_region.tex | wc -l | tr -d ' ')" -eq 1
    if rg -n 'spatial FID|Pending|200 unique|Target@0\.8|A0|A11' \
      paper/cci_trust_region.tex; then exit 1; fi
    git diff --check

Expected: all tests and manuscript invariants pass.

- [ ] **Step 7: Commit the compiled paper**

    git add -f paper/cci_trust_region.pdf
    git commit -m "docs: publish result-led CCI conference paper"
