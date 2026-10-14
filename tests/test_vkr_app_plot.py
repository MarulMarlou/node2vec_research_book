"""Plotly visualization check without launching a Streamlit server."""
import importlib
from pathlib import Path
import sys
import types
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("plotly")


def test_plot_draws_only_supplied_evidence(monkeypatch):
    fake_st = types.ModuleType("streamlit")
    fake_st.cache_resource = lambda **kwargs: lambda f: f
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    mod = importlib.import_module("src.vkr_app")
    payload = {"nodes": [
        {"id": "book:1", "label": "One", "type": "book", "x": 0, "y": 0},
        {"id": "author:a", "label": "Writer", "type": "author", "x": 1, "y": 0.55},
        {"id": "book:2", "label": "Two", "type": "book", "x": 2, "y": 0}],
        "edges": [{"from_id": "book:1", "to_id": "author:a", "relation": "written_by"},
                  {"from_id": "book:2", "to_id": "author:a", "relation": "written_by"}]}
    chart = mod.draw_evidence_path(payload)
    assert len(chart.data) == 4
    assert sum(trace.mode == "lines" for trace in chart.data) == 2
    assert len(chart.layout.annotations) == 2
    assert all(a.text == "written_by" for a in chart.layout.annotations)
