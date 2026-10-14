from pathlib import Path
import sys

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vkr_graph_audit import graph_audit
from src.vkr_prepare import prepare


def make_data(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([
        {"ISBN": "01", "Book-Title": "A", "Book-Author": "Writer", "Publisher": "Press"},
        {"ISBN": "02", "Book-Title": "B", "Book-Author": "Writer", "Publisher": "Press"},
        {"ISBN": "03", "Book-Title": "C", "Book-Author": "Writer2", "Publisher": "Press"},
    ]).to_csv(raw / "BX-Books.csv", sep=";", index=False, encoding="latin-1")
    pd.DataFrame([
        {"User-ID": "1", "ISBN": "01", "Book-Rating": 8},
        {"User-ID": "1", "ISBN": "02", "Book-Rating": 9},
        {"User-ID": "2", "ISBN": "02", "Book-Rating": 5},
        {"User-ID": "2", "ISBN": "03", "Book-Rating": 8},
        {"User-ID": "2", "ISBN": "01", "Book-Rating": 9},
    ]).to_csv(raw / "BX-Book-Ratings.csv", sep=";", index=False, encoding="latin-1")
    processed = tmp_path / "processed"
    prepare(raw, processed, max_users=None, max_books=None)
    return processed


def test_graph_audit_structure_and_no_leakage(tmp_path):
    processed = make_data(tmp_path)
    audit = graph_audit(processed)
    assert audit["audit_passed"]
    assert audit["test_rating_edges_in_graph"] == 0
    assert audit["train_test_overlapping_pairs"] == 0
    assert audit["graph_train_rating_mismatch"]["missing"] == 0
    assert audit["missing_provenance_file_edges"] == 0
    assert audit["largest_weak_component_fraction"] > 0


def test_graph_audit_detects_added_test_edge(tmp_path):
    processed = make_data(tmp_path)
    test = pd.read_csv(processed / "vkr_test_ratings.csv", dtype=str)
    assert not test.empty
    user, isbn = test.loc[0, ["User-ID", "ISBN"]]
    graph_path = processed / "vkr_knowledge_graph.graphml"
    graph = nx.read_graphml(graph_path, force_multigraph=True)
    graph.add_edge("user:" + user, "book:" + isbn, relation="rated",
                   source_file="BX-Book-Ratings.csv", weight=1.0)
    nx.write_graphml(graph, graph_path)
    audit = graph_audit(processed)
    assert not audit["audit_passed"]
    assert audit["test_rating_edges_in_graph"] >= 1
