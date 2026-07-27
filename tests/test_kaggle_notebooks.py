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

    assert "SAMPLE_COUNT = 300" in source
    assert "MAX_SELECTED_REGIONS = 4" in source
    assert "STOP_FLIP_RATE = 0.96" in source
    assert "'smile'" in source
    assert "'hair'" in source
    assert "screen_counterfactual_regions.py" in source
    assert "run_counterfactual_region_interventions.py" in source
    assert "discover_counterfactual_graph.py" in source
    assert "discovery_ids.json" in source
    assert "cuda" in source


def test_full_cci_notebook_compares_fixed_and_feedback_on_disjoint_ids():
    source = notebook_source("02_full_cci_fixed_vs_adaptive.ipynb")

    assert "SAMPLE_COUNT = 300" in source
    assert "--controller_modes" in source
    assert "'disabled'" in source
    assert "'fixed_equal'" in source
    assert "'feedback'" in source
    assert "--exclude_ids_json" in source
    assert "discovery_ids.json" in source
    assert "'mouth', 'upper_lip', 'lower_lip'" in source
    assert "'hair'" in source
    assert "pilot_results.csv" in source
    assert "'A0': 'raw_bld'" in source
    assert "cuda" in source
