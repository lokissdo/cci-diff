import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_smile_graph_individual_100.sh"


def test_experiment_wrapper_help_describes_all_three_stages():
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "discovery interventions" in completed.stdout
    assert "frozen influence graph" in completed.stdout
    assert "held-out individual-region CCI" in completed.stdout


def test_experiment_wrapper_discovers_regions_before_interventions():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "screen_counterfactual_regions.py" in source
    assert "--select_top_k" not in source
    assert "--max_selected_regions" in source
    assert "MAX_SELECTED_REGIONS" in source
    assert "--stop_flip_rate" in source
    assert "MINIMUM_CAPTURED_SALIENCY" in source
    assert "--minimum_captured_saliency" in source
    assert "--dry_run" in source
    assert '["region_sets"]' in source
    assert "--candidate_regions mouth upper_lip lower_lip" not in source
