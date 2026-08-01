from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "cci_trust_region.tex"


def _latex_environment_with_label(text: str, environment: str, label: str) -> str:
    marker = rf"\label{{{label}}}"
    marker_index = text.find(marker)
    assert marker_index >= 0, f"missing {marker}"
    begin = rf"\begin{{{environment}}}"
    end = rf"\end{{{environment}}}"
    begin_index = text.rfind(begin, 0, marker_index)
    end_index = text.find(end, marker_index)
    assert begin_index >= 0 and end_index >= 0
    return text[begin_index : end_index + len(end)]


def test_figure_one_is_the_complete_method_without_bld() -> None:
    text = PAPER.read_text(encoding="utf-8")
    overview = _latex_environment_with_label(text, "figure*", "fig:overview")

    assert "BLD" not in overview
    assert "Classifier-Causal Mask Discovery" in overview
    assert "Localized Trust-Region Counterfactual Generation" in overview
    assert "10429_source.jpg" in overview
    assert "10429_mask_overlay.png" in overview
    assert "10429_ours.jpg" in overview


def test_published_celebahq_smile_table_has_all_methods() -> None:
    text = PAPER.read_text(encoding="utf-8")
    table = _latex_environment_with_label(
        text, "table*", "tab:published-celebahq"
    )

    for method in (
        "DiVE",
        "STEEX",
        "DiME",
        r"ACE $\ell_1$",
        r"ACE $\ell_2$",
        "LDCE-txt",
        "TiME",
        "RCSB",
        "MaskDiME",
    ):
        assert method in table
    assert r"\EndToEndAdaptiveFID" in table
    assert r"\EndToEndAdaptiveFRPct" in table


def test_cross_protocol_metric_boundary_is_explicit() -> None:
    text = PAPER.read_text(encoding="utf-8")

    assert "ten-repeat split-FID" in text
    assert "deterministic two-direction split-FID" in text
    assert "cross-protocol context" in text
