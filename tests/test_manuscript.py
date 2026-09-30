"""Text integrity tests without introducing manuscript observations."""
import pytest
from analysis.manuscript_text import pending
from analysis import render_paper


def test_pending_prose_does_not_claim_effects():
    fragments = pending()
    assert "not submission-ready" in fragments["status"]
    assert "findings remain pending" in fragments["abstract"]
    assert all(chr(8212) not in s and "---" not in s for s in fragments.values())


def test_generator_rejects_em_dash(monkeypatch, tmp_path):
    monkeypatch.setattr(render_paper, "GEN", tmp_path)
    for content in ["a" + chr(8212) + "b", "a---b", r"a\textemdash b"]:
        with pytest.raises(ValueError):
            render_paper.write("fixture", content)
    render_paper.write("fixture", "A clear sentence. Another sentence.")
    assert (tmp_path / "fixture.tex").read_text().startswith("A clear sentence.")
