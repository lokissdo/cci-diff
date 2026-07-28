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


def assert_standalone_command_contract(source: str, *, mode: str) -> None:
    assert "run_kaggle_smile.py" in source
    assert "'-u'" in source
    assert f"'--mode', '{mode}'" in source
    assert "subprocess.run(command, check=True)" in source
    assert "'-q'" not in source
    assert "runtime_packages" not in source
    assert "'--max_workers', str(MAX_WORKERS)" in source
    assert "torch.cuda.device_count()" in source
    assert "torch.backends.mps.is_available()" in source
    assert "'--classifier_path', CLASSIFIER_PATH" in source
    assert "'--identity_model_path', IDENTITY_MODEL_PATH" in source


def test_global_discovery_notebook_is_two_task_resumable_and_max_four():
    source = notebook_source("01_global_graph_discovery.ipynb")

    assert_standalone_command_contract(source, mode="discovery")
    assert "MODEL_PATH = 'sd2-community/stable-diffusion-2-1'" in source
    assert "https://github.com/lokissdo/cci-diff.git" in source
    assert "GIT_REF = 'main'" in source
    assert "'git', 'clone'" in source
    assert "/kaggle/input/cci-sd2-assets" not in source
    assert "SAMPLE_COUNT = 300" in source
    assert "MAX_SELECTED_REGIONS = 4" in source
    assert "STOP_FLIP_RATE = 0.96" in source
    assert "'smile'" in source
    assert "'hair': {" not in source
    assert "PYTHONUNBUFFERED" in source
    assert "cuda" in source


def test_full_cci_notebook_is_standalone_with_assumed_region_policy():
    source = notebook_source("02_full_cci_fixed_vs_adaptive.ipynb")

    assert_standalone_command_contract(source, mode="evaluation")
    assert "MODEL_PATH = 'sd2-community/stable-diffusion-2-1'" in source
    assert "https://github.com/lokissdo/cci-diff.git" in source
    assert "GIT_REF = 'main'" in source
    assert "'git', 'clone'" in source
    assert "/kaggle/input/cci-sd2-assets" not in source
    assert "SAMPLE_COUNT = 300" in source
    assert "PYTHONUNBUFFERED" in source
    assert "cuda" in source
