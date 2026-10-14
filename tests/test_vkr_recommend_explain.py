from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vkr_prepare import build_knowledge_graph, node2vec_projection
from src.vkr_recommend import (biased_walk, train_node2vec, cosine_ranking,
                               evaluate_user_holdout, metadata_book_ranking,
                               recommend_similar_books, save_embeddings, load_embeddings)
from src.vkr_explain import find_explanations, explain_and_rerank


def graph_fixture():
    books = pd.DataFrame([
        dict(ISBN="1", **{"Book-Title": "Alpha", "Book-Author": "Writer A", "Publisher": "Pub", "source_row": "2"}),
        dict(ISBN="2", **{"Book-Title": "Beta", "Book-Author": "Writer A", "Publisher": "Pub", "source_row": "3"}),
        dict(ISBN="3", **{"Book-Title": "Gamma", "Book-Author": "Writer B", "Publisher": "Other", "source_row": "4"}),
        dict(ISBN="4", **{"Book-Title": "Delta", "Book-Author": "Writer C", "Publisher": "Other", "source_row": "5"}),
    ])
    train = pd.DataFrame([
        {"User-ID": "7", "ISBN": "1", "Book-Rating": 9, "source_row": "2"},
        {"User-ID": "7", "ISBN": "2", "Book-Rating": 8, "source_row": "3"},
        {"User-ID": "8", "ISBN": "3", "Book-Rating": 9, "source_row": "4"},
        {"User-ID": "8", "ISBN": "4", "Book-Rating": 2, "source_row": "5"},
    ])
    genres = pd.DataFrame([
        {"isbn": "1", "genre": "Mystery", "source_url": "https://books.test/1"},
        {"isbn": "3", "genre": "Mystery", "source_url": "https://books.test/3"},
    ])
    return build_knowledge_graph(books, train, genres)


def test_weighted_second_order_walk_and_validation():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=5)
    graph.add_edge("b", "c", weight=1)
    import random
    assert biased_walk(graph, "a", length=5, p=1, q=1, rng=random.Random(3)) == biased_walk(
        graph, "a", length=5, p=1, q=1, rng=random.Random(3))
    with pytest.raises(ValueError):
        biased_walk(graph, "a", length=5, p=0, q=1, rng=random.Random(3))


def test_node2vec_trains_on_projection_and_roundtrip(tmp_path):
    graph = node2vec_projection(graph_fixture())
    e1 = train_node2vec(graph, dimensions=8, walk_length=6, num_walks=3, epochs=2, seed=12)
    e2 = train_node2vec(graph, dimensions=8, walk_length=6, num_walks=3, epochs=2, seed=12)
    assert len(e1) == graph.number_of_nodes()
    assert e1.shape[1] == 8
    np.testing.assert_allclose(e1.values, e2.values, rtol=0, atol=0)
    save_embeddings(e1, tmp_path)
    np.testing.assert_allclose(load_embeddings(tmp_path).values, e1.values)


def test_book_similarity_does_not_include_query():
    emb = pd.DataFrame([[1, 0], [0.9, 0.1], [0, 1]], index=["book:1","book:2","book:3"])
    rec = recommend_similar_books(emb, "book:1", top_k=2)
    assert [b for b,_ in rec] == ["book:2","book:3"]
    assert not cosine_ranking(emb, "missing", emb.index.tolist())


def test_user_holdout_filters_training_interactions():
    emb = pd.DataFrame([[1,0], [1,0], [0.95,0.05], [0,1]],
                       index=["user:1", "book:A", "book:B", "book:C"])
    train = pd.DataFrame([{"User-ID":"1","ISBN":"A","Book-Rating":10}])
    test = pd.DataFrame([{"User-ID":"1","ISBN":"B","Book-Rating":9}])
    report = evaluate_user_holdout(emb, train, test, ks=(1,2))
    assert report["p@1"] == 1.0, report
    assert report["r@1"] == 1.0
    assert report["mrr_at_max_k"] == 1.0
    assert report["user_coverage"] == 1.0


def test_user_holdout_uncovered_user_is_not_hidden():
    emb = pd.DataFrame([[1,0], [0,1]], index=["book:A","book:B"])
    train = pd.DataFrame([{"User-ID":"1","ISBN":"A","Book-Rating":10}])
    test = pd.DataFrame([{"User-ID":"1","ISBN":"B","Book-Rating":9}])
    report = evaluate_user_holdout(emb, train, test, ks=(1,))
    assert report["evaluated_users"] == 0
    assert report["p@1"] == 0
    assert report["user_coverage"] == 0


def test_metadata_baseline_does_not_use_readers():
    baseline = dict(metadata_book_ranking(graph_fixture(), "book:1", top_k=3))
    assert baseline["book:2"] > baseline["book:3"]


def test_shortest_uses_real_graph_facts_and_author():
    graph = graph_fixture()
    result = find_explanations(graph, "book:1", "book:2", method="shortest")
    assert result["paths"][0]["length"] == 2
    assert "Writer A" in result["paths"][0]["text"]
    assert all(f["source_file"] == "BX-Books.csv" for f in result["paths"][0]["edges"])
    assert set(result["paths"][0]["nodes"]) <= set(graph)


def test_informative_genre_fact_and_no_fabrication():
    graph = graph_fixture()
    result = find_explanations(graph, "book:1", "book:3", method="informative", top_n=5)
    assert any("Mystery" in p["text"] for p in result["paths"])
    genre_path = next(p for p in result["paths"] if "Mystery" in p["text"])
    assert all(f["source_url"].startswith("https://books.test") for f in genre_path["edges"])


def test_low_ratings_do_not_yield_strong_reader_explanations():
    result = find_explanations(graph_fixture(), "book:3", "book:4",
                               method="shortest", max_depth=2)
    assert not any("читател" in path["text"] for path in result["paths"])


def test_multihop_paths_and_bounded_search():
    graph = graph_fixture()
    result = find_explanations(graph, "book:1", "book:4", method="shortest", max_depth=4)
    assert result["paths"] and result["paths"][0]["length"] == 4
    assert "→" in result["paths"][0]["text"]
    bounded = find_explanations(graph, "book:1", "book:4", method="informative",
                                max_depth=4, max_expansions=1)
    assert bounded["truncated"]


def test_posthoc_keeps_order_rerank_changes_it():
    graph = graph_fixture()
    candidates = [("book:4", 0.8), ("book:2", 0.4)]
    posthoc = explain_and_rerank(graph, candidates, "book:1", mode="posthoc", max_depth=2)
    assert [r["book_id"] for r in posthoc] == ["book:4", "book:2"]
    assert posthoc[0]["explanation"] is None
    ranked = explain_and_rerank(graph, candidates, "book:1", mode="rerank",
                                 alpha=1.0, max_depth=2)
    assert ranked[0]["book_id"] == "book:2"
    assert ranked[0]["rank_score"] != ranked[0]["base_score"]


def test_invalid_options_rejected():
    graph = graph_fixture()
    with pytest.raises(ValueError):
        find_explanations(graph, "book:1", "book:2", method="unknown")
    with pytest.raises(ValueError):
        explain_and_rerank(graph, [], "book:1", alpha=1.5)
