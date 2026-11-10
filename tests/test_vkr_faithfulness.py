"""Faithfulness tests only use synthetic KG/rating data, never Book-Crossing."""
import json

import networkx as nx
import pandas as pd
import pytest

from src.vkr_explain import find_explanations
from src.vkr_faithfulness import (
    ablate_explanation_path, ablate_matched_control,
    ablate_degree_matched_control, run_faithfulness)
from src.vkr_pipeline import run_pipeline


def tiny_graph():
    g = nx.MultiDiGraph()
    for node, kind in (("book:001", "book"), ("book:002", "book"),
                       ("book:003", "book"), ("book:004", "book"),
                       ("author:a", "author"), ("author:b", "author")):
        g.add_node(node, entity_type=kind, label=node)
    for i, (b, a) in enumerate((("book:001", "author:a"),
                                 ("book:002", "author:a"),
                                 ("book:003", "author:b"),
                                 ("book:004", "author:b")), 2):
        g.add_edge(b, a, relation="written_by",
                   source_file="BX-Books.csv", source_row=str(i), source_url="")
    return g


def test_ablation_removes_only_verified_path_facts():
    graph = tiny_graph()
    found = find_explanations(graph, "book:001", "book:002",
                              method="shortest", max_depth=2)
    path = found["paths"][0]
    assert path["length"] == 2
    ablated = ablate_explanation_path(graph, path)
    assert ablated.number_of_edges() == graph.number_of_edges() - 2
    assert graph.has_edge("book:001", "author:a")
    assert graph.has_edge("book:002", "author:a")
    assert not ablated.has_edge("book:001", "author:a")
    assert not ablated.has_edge("book:002", "author:a")
    assert ablated.has_edge("book:003", "author:b")
    assert ablated.has_edge("book:004", "author:b")
    altered = dict(path)
    altered["edges"] = [dict(x) for x in path["edges"]]
    altered["edges"][0]["source_row"] = "fabricated"
    with pytest.raises(ValueError, match="provenance"):
        ablate_explanation_path(graph, altered)


def test_matched_controls_use_only_unrelated_edges_same_relation():
    graph = tiny_graph()
    path = find_explanations(graph, "book:001", "book:002",
                             method="shortest", max_depth=2)["paths"][0]
    control = ablate_matched_control(graph, path, seed=2)
    assert control is not None
    assert control.number_of_edges() == graph.number_of_edges() - 2
    assert control.has_edge("book:001", "author:a")
    assert control.has_edge("book:002", "author:a")
    assert not control.has_edge("book:003", "author:b")
    assert not control.has_edge("book:004", "author:b")


def test_faithfulness_full_synthetic_run(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([
        {"ISBN": f"00{i}", "Book-Title": f"Book {i}",
         "Book-Author": f"Author {i%2}", "Publisher": "Press"}
        for i in range(1, 7)
    ]).to_csv(raw / "BX-Books.csv", sep=";", encoding="latin-1", index=False)
    pd.DataFrame([
        {"User-ID": str(user), "ISBN": f"00{i}", "Book-Rating": 9}
        for user in (1, 2, 3) for i in range(1, 7)
    ]).to_csv(raw / "BX-Book-Ratings.csv", sep=";", encoding="latin-1", index=False)
    processed = tmp_path / "processed"
    results = tmp_path / "results"
    run_pipeline(raw, processed, results, max_users=None, max_books=None,
                 dimensions=8, walk_length=6, num_walks=2, epochs=1)
    study = run_faithfulness(processed, results, tmp_path / "faithfulness",
                             source_book="book:001", max_pairs=1, controls=1)
    assert study["status"] == "pilot_completed"
    assert study["evaluated_pairs"] == 1
    assert len(study["records"]) == 1
    row = study["records"][0]
    assert row["score_baseline_refit"] is not None
    assert row["score_baseline_saved"] is not None
    assert row["path_length"] == 2
    assert row["controls_requested"] == 1
    assert len(row["null_control_score_drops"]) == row["controls_valid"]
    if row["controls_valid"]:
        assert row["null_control_degree_mismatch_mean"] is not None
        assert row["explanation_minus_control_score_drop"] is not None
    assert abs(row["score_baseline_saved"] - row["score_baseline_refit"]) < 0.0001
    assert (tmp_path / "faithfulness" / "faithfulness_pairs.csv").is_file()
    persisted = json.loads((tmp_path / "faithfulness" / "faithfulness_report.json").read_text())
    assert persisted["original_graph_edges"] > 0
    assert all("user:" not in str(key) for key in persisted["records"][0].keys())



def test_degree_matched_controls_preserve_explanation_edges_and_report_mismatch():
    graph = tiny_graph()
    path = find_explanations(graph, "book:001", "book:002",
                             method="shortest", max_depth=2)["paths"][0]
    a, mismatch = ablate_degree_matched_control(graph, path, seed=11)
    b, again = ablate_degree_matched_control(graph, path, seed=11)
    assert a is not None and b is not None
    assert mismatch == again == 0.0
    assert set(a.edges(keys=True)) == set(b.edges(keys=True))
    assert a.number_of_edges() == graph.number_of_edges() - 2
    assert a.has_edge("book:001", "author:a")
    assert a.has_edge("book:002", "author:a")
    assert not a.has_edge("book:003", "author:b")
    assert not a.has_edge("book:004", "author:b")
    assert graph.number_of_edges() == 4


def test_degree_matched_control_rejects_unverified_explanation():
    graph = tiny_graph()
    path = find_explanations(graph, "book:001", "book:002",
                             method="shortest", max_depth=2)["paths"][0]
    altered = dict(path)
    altered["edges"] = [dict(e) for e in path["edges"]]
    altered["edges"][0]["source_file"] = "fabricated.csv"
    with pytest.raises(ValueError, match="provenance"):
        ablate_degree_matched_control(graph, altered, seed=1)
