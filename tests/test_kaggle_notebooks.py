import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = REPO_ROOT / "notebooks"


def notebook_source(name: str) -> str:
    payload = json.loads((NOTEBOOK_ROOT / name).read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert payload["metadata"]["kernelspec"]["language"] == "python"
    return "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"]
    )


def test_global_discovery_notebook_is_two_task_resumable_and_max_four():
    source = notebook_source("01_global_graph_discovery.ipynb")

    assert "Path('/kaggle/input/cci-sd2-assets')" in source
    assert "DIFFUSION_ASSET_ROOT / 'stable-diffusion-2-1'" in source
    assert "https://github.com/lokissdo/cci-diff.git" in source
    assert "GIT_REF = 'main'" in source
    assert "'git', 'clone'" in source
    assert "rglob('CelebA-HQ-img')" in source
    assert "rglob('CelebAMask-HQ-mask-anno')" in source
    assert "/kaggle/input/cci-assets" in source
    assert "/kaggle/input/cci-sd2-assets" in source
    assert "--allow_model_download" not in source
    assert "SAMPLE_COUNT = 300" in source
    assert "MAX_SELECTED_REGIONS = 4" in source
    assert "STOP_FLIP_RATE = 0.96" in source
    assert "'smile'" in source
    assert "'hair': {" not in source
    assert "runpy.run_path" in source
    assert "PYTHONUNBUFFERED" in source
    assert "screen_counterfactual_regions.py" in source
    assert "run_counterfactual_region_interventions.py" in source
    assert "discover_counterfactual_graph.py" in source
    assert "discovery_ids.json" in source
    assert "cuda" in source


def test_full_cci_notebook_is_standalone_with_assumed_region_policy():
    source = notebook_source("02_full_cci_fixed_vs_adaptive.ipynb")

    assert "Path('/kaggle/input/cci-sd2-assets')" in source
    assert "DIFFUSION_ASSET_ROOT / 'stable-diffusion-2-1'" in source
    assert "https://github.com/lokissdo/cci-diff.git" in source
    assert "GIT_REF = 'main'" in source
    assert "'git', 'clone'" in source
    assert "rglob('CelebA-HQ-img')" in source
    assert "rglob('CelebAMask-HQ-mask-anno')" in source
    assert "/kaggle/input/cci-sd2-assets" in source
    assert "--allow_model_download" not in source
    assert "ASSUMED_REGIONS" in source
    assert "discovery_ids.json" not in source
    assert "SAMPLE_COUNT = 300" in source
    assert "'--features', 'smile'" in source
    assert "'--features', 'smile', 'hair'" not in source
    assert "runpy.run_path" in source
    assert "PYTHONUNBUFFERED" in source
    assert "--controller_modes" in source
    assert "'disabled'" in source
    assert "'fixed_equal'" in source
    assert "'feedback'" in source
    assert "--exclude_ids_json" not in source
    assert "'mouth', 'upper_lip', 'lower_lip'" in source
    assert "'hair'" not in source
    assert "pilot_results.csv" in source
    assert "'A0': 'raw_bld'" in source
    assert "cuda" in source
