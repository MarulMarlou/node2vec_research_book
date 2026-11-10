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



def test_pca_plot_hides_user_ids_in_hover(monkeypatch):
    import pandas as pd
    from src.vkr_embedding_projection import Projection2D

    fake_st = types.ModuleType("streamlit")
    fake_st.cache_resource = lambda **kwargs: lambda f: f
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    mod = importlib.import_module("src.vkr_app")
    points = pd.DataFrame([
        {"id": "book:001", "label": "Book", "type": "book",
         "x": 0.0, "y": 0.0, "highlighted": True},
        {"id": "user:secret123", "label": "User secret123", "type": "user",
         "x": 1.0, "y": 0.1, "highlighted": False},
    ])
    result = Projection2D(points, (0.6, 0.3), 2, 2)
    chart = mod.draw_embedding_map(result)
    all_hover = " ".join(str(trace.customdata) for trace in chart.data)
    assert "book:001" in all_hover
    assert "user:secret123" not in all_hover
    assert "User secret123" not in all_hover
    assert "ID читателя скрыт" in all_hover



def test_evidence_path_plot_hides_identifying_user_ids(monkeypatch):
    fake_st = types.ModuleType("streamlit")
    fake_st.cache_resource = lambda **kwargs: lambda f: f
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    mod = importlib.import_module("src.vkr_app")
    evidence = {"nodes": [
        {"id": "book:b1", "label": "Novel", "type": "book", "x": 0, "y": 0},
        {"id": "user:999999", "label": "User 999999", "type": "user", "x": 1, "y": 0.5},
        {"id": "book:b2", "label": "Other novel", "type": "book", "x": 2, "y": 0},
    ], "edges": [
        {"from_id": "user:999999", "to_id": "book:b1", "relation": "rated"},
        {"from_id": "user:999999", "to_id": "book:b2", "relation": "rated"},
    ]}
    figure = mod.draw_evidence_path(evidence)
    exposed = " ".join(str(trace.customdata) for trace in figure.data)
    assert "999999" not in exposed
    assert any("Читатель" in str(trace.text) for trace in figure.data)
