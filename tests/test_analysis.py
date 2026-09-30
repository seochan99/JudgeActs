"""Synthetic unit fixtures only; these are never manuscript observations."""
import json
import pytest

from analysis.analyze import build_rows
from analysis.empirical_figures import _interval, pending, STYLE
from src.common import save_jsonl


def group():
    candidates = []
    for identity, utility, stereotype in [("a", .25, 0), ("b", .75, 1), ("c", .75, 0)]:
        candidates.append({"id": identity, "utility": utility,
            "stereotype": stereotype, "missing_explicit": 0,
            "missing_implicit": 0, "image_quality": 1, "overall": utility})
    return {"prompt_id": "fixture", "country": "Fixture", "category": "Test",
            "candidates": candidates}


def record(model, permutation, selected=None):
    return {"prompt_id": "fixture", "model": model, "permutation": permutation,
            "parse_ok": selected is not None, "selected_id": selected, "seconds": 1}


def test_policy_identity_ties_and_exact_random(tmp_path):
    qwen = tmp_path / "qwen_main.jsonl"
    smol = tmp_path / "smol_main.jsonl"
    save_jsonl(qwen, [record("qwen", k, "b") for k in range(3)])
    save_jsonl(smol, [record("smol", k, "b") for k in range(3)])
    df, order, validation = build_rows([group()], [qwen, smol])
    rows = df.set_index("policy")
    assert rows.loc["Random", "utility"] == pytest.approx(7 / 12)
    assert rows.loc["Random", "hsr"] == pytest.approx(1 / 3)
    assert rows.loc["Oracle", "stereotype"] == .5  # average alignment-maximizing ties
    assert rows.loc["Qwen", "regret"] == 0
    assert rows.loc["Qwen unanimous", "selected_id"] == "b"
    assert rows.loc["Cross-model", "selected_id"] == "b"
    assert order["agreement"].tolist() == [1, 1]
    assert validation["qwen"]["calls"] == 3


def test_invalid_call_is_not_silently_imputed(tmp_path):
    qwen = tmp_path / "qwen_main.jsonl"
    save_jsonl(qwen, [record("qwen", 0), record("qwen", 1, "b"), record("qwen", 2, "b")])
    df, order, validation = build_rows([group()], [qwen])
    assert set(df.policy) == {"Random", "Oracle"}
    assert order.empty
    assert validation["qwen"]["invalid"] == 1


def test_percentile_interval_is_not_clipped_to_mean():
    assert _interval({"regret": .1, "regret_ci": [.11, .12]}) == (.1, .11, .12)
    with pytest.raises(ValueError):
        _interval({"regret": .1, "regret_ci": [.2, .1]})


def test_pending_figures_are_vector_and_contain_no_main_values(tmp_path):
    import matplotlib.pyplot as plt
    import pymupdf
    import xml.etree.ElementTree as ET
    with plt.rc_context(STYLE):
        pending(tmp_path)
    for name in ["main_regret", "country_regret", "coverage_risk"]:
        with pymupdf.open(tmp_path / f"{name}.pdf") as doc:
            assert doc[0].rect.width == pytest.approx(3.35 * 72)
            assert "No empirical observations displayed" in doc[0].get_text()
        svg = ET.parse(tmp_path / f"{name}.svg")
        assert svg.findall(".//{http://www.w3.org/2000/svg}text")
