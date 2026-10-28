from pathlib import Path
import sys
import networkx as nx
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vkr_prepare import (load_book_crossing, split_ratings, restrict_from_training,
                             build_knowledge_graph, node2vec_projection, prepare)
from src.vkr_metrics import ranking_metrics

def fixture_data():
    books = pd.DataFrame([
        {"ISBN": "001", "Book-Title": "Book A", "Book-Author": "A. Writer", "Publisher": "Pub", "source_row": "2"},
        {"ISBN": "002", "Book-Title": "Book B", "Book-Author": "A. Writer", "Publisher": "Pub", "source_row": "3"},
        {"ISBN": "003", "Book-Title": "Book C", "Book-Author": "B. Writer", "Publisher": "Else", "source_row": "4"},
    ])
    ratings = pd.DataFrame([
        {"User-ID": "7", "ISBN": "001", "Book-Rating": 9, "source_row": "2"},
        {"User-ID": "7", "ISBN": "002", "Book-Rating": 8, "source_row": "3"},
        {"User-ID": "7", "ISBN": "003", "Book-Rating": 2, "source_row": "4"},
        {"User-ID": "8", "ISBN": "001", "Book-Rating": 10, "source_row": "5"},
    ])
    return books, ratings

def test_split_is_deterministic_and_disjoint():
    _, ratings = fixture_data()
    train, test = split_ratings(ratings, test_ratio=0.5, seed=42)
    assert len(test) == 1
    assert len(train) == 3
    assert test.iloc[0]["User-ID"] == "7"
    assert test.iloc[0]["Book-Rating"] >= 7
    assert set(zip(train["User-ID"], train["ISBN"])).isdisjoint(
        set(zip(test["User-ID"], test["ISBN"])))
    shuffled = ratings.sample(frac=1, random_state=5).reset_index(drop=True)
    _, shuffled_test = split_ratings(shuffled, test_ratio=0.5, seed=42)
    assert test["ISBN"].tolist() == shuffled_test["ISBN"].tolist()

def test_train_only_selection():
    _, ratings = fixture_data()
    train, test = split_ratings(ratings, test_ratio=0.5)
    filtered_train, filtered_test = restrict_from_training(train, test, max_users=1, max_books=1)
    assert set(filtered_train["User-ID"]) == {"7"}
    assert set(filtered_test["User-ID"]).issubset({"7"})
    assert set(filtered_test["ISBN"]).issubset(set(filtered_train["ISBN"]))

def test_typed_graph_provenance_and_no_test_edge(tmp_path):
    books, ratings = fixture_data()
    train, test = split_ratings(ratings, test_ratio=0.5)
    genres = pd.DataFrame([{"isbn": "001", "genre": "Novel", "source_url": "https://example.org/fact/1"}])
    G = build_knowledge_graph(books, train, genres)
    assert isinstance(G, nx.MultiDiGraph)
    relations = {e["relation"] for _, _, e in G.edges(data=True)}
    assert {"rated", "written_by", "published_by", "has_genre"}.issubset(relations)
    assert any(e["source_url"] == "https://example.org/fact/1" for _, _, e in G.edges(data=True))
    for r in test.itertuples(index=False):
        assert not any(e.get("relation") == "rated"
                       for e in G.get_edge_data(f"user:{r[0]}", f"book:{r[1]}", default={}).values())
    path = tmp_path / "g.graphml"
    nx.write_graphml(G, path)
    restored = nx.read_graphml(path)
    assert restored.number_of_edges() == G.number_of_edges()
    projection = node2vec_projection(G)
    assert isinstance(projection, nx.Graph)
    assert projection.number_of_nodes() == G.number_of_nodes()

def test_metric_denominators_and_duplicates():
    m = ranking_metrics(["book:1", "book:2", "book:3"], {"book:2", "book:4"}, (2, 5))
    assert m["p@2"] == 0.5
    assert m["r@2"] == 0.5
    assert m["rr"] == 0.5
    assert 0 <= m["ndcg@2"] <= 1
    assert ranking_metrics([], set(), (5,))["ndcg@5"] == 0
    with pytest.raises(ValueError):
        ranking_metrics(["x", "x"], {"x"})

def test_end_to_end_tiny_dataset(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    books, ratings = fixture_data()
    books.drop(columns=["source_row"]).to_csv(raw / "BX-Books.csv", sep=";", encoding="latin-1", index=False)
    ratings.drop(columns=["source_row"]).to_csv(raw / "BX-Book-Ratings.csv", sep=";", encoding="latin-1", index=False)
    result = prepare(raw, tmp_path / "processed", max_users=None, max_books=None)
    assert result["train_rows"] == 3
    assert result["test_rows"] == 1
    assert result["relation_types"]["rated"] == 3
    assert (tmp_path / "processed" / "vkr_knowledge_graph.graphml").is_file()
    assert (tmp_path / "processed" / "vkr_preparation_report.json").is_file()

def test_zero_ratings_not_treated_as_negative(tmp_path):
    books, ratings = fixture_data()
    ratings.loc[len(ratings)] = ["8", "002", 0, "6"]
    raw = tmp_path / "raw"
    raw.mkdir()
    books.drop(columns=["source_row"]).to_csv(raw / "BX-Books.csv", sep=";", encoding="latin-1", index=False)
    ratings.drop(columns=["source_row"]).to_csv(raw / "BX-Book-Ratings.csv", sep=";", encoding="latin-1", index=False)
    _, cleaned = load_book_crossing(raw)
    assert 0 not in set(cleaned["Book-Rating"])
