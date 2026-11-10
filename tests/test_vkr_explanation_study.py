"""Synthetic validation/test isolation for explanation-aware recommendation study."""
import json

import numpy as np
import pandas as pd
import pytest
import networkx as nx

from src.vkr_explanation_study import (
    _cohort_evaluation, _prepared_candidates, _rank,
    explanation_configs, run_explanation_study, paired_bootstrap,
)


def sample_bx(raw):
    raw.mkdir()
    pd.DataFrame([
        {"ISBN": f"{i:03d}", "Book-Title": f"Book {i}",
         "Book-Author": f"Author {i % 4}",
         "Publisher": f"Publisher {i % 3}"}
        for i in range(1, 21)
    ]).to_csv(raw / "BX-Books.csv", sep=";", encoding="latin-1", index=False)
    pd.DataFrame([
        {"User-ID": str(user), "ISBN": f"{i:03d}",
         "Book-Rating": 7 + (i % 3)}
        for user in range(1, 7) for i in range(1, 21)
    ]).to_csv(raw / "BX-Book-Ratings.csv", sep=";", encoding="latin-1", index=False)


def test_eight_configs_and_reranking_logic():
    configs = explanation_configs()
    assert len(configs) == 8
    assert len({(x["method"], x["mode"], x["alpha"]) for x in configs}) == 8
    rows = [
        {"book_id": "book:a", "base_score": 0.9, "informativeness": 0.0},
        {"book_id": "book:b", "base_score": 0.5, "informativeness": 100.0},
    ]
    assert _rank(rows, mode="posthoc", alpha=0.0)[0]["book_id"] == "book:a"
    assert _rank(rows, mode="rerank", alpha=0.8)[0]["book_id"] == "book:b"
    with pytest.raises(ValueError):
        _rank(rows, mode="rerank", alpha=2)


def test_max_cosine_uses_train_liked_anchor_and_excludes_seen():
    ids = ["book:a", "book:b", "book:c"]
    vectors = pd.DataFrame([[1., 0.], [0.8, 0.2], [-1., 0.]], index=ids)
    ranked = _prepared_candidates(vectors, {"book:a"}, ["book:a"], candidate_pool=20)
    assert [r["book_id"] for r in ranked] == ["book:b", "book:c"]
    assert all(r["anchor"] == "book:a" for r in ranked)


def test_explanation_study_chooses_on_validation_and_holds_test(tmp_path):
    raw = tmp_path / "raw"
    sample_bx(raw)
    report = run_explanation_study(
        raw, tmp_path / "results", max_users=6, max_books=20,
        max_evaluated_users=3, candidate_pool=20, max_depth=2,
        split_seed=2042, model_params={
            "dimensions": 8, "walk_length": 6, "num_walks": 2,
            "epochs": 1, "p": 1., "q": 1., "window": 3, "seed": 43})
    assert report["train_rows"] > 0
    assert report["validation_rows"] > 0 and report["test_rows"] > 0
    assert len(report["validation"]["results"]) == 8
    best = report["selected"]
    assert best["ndcg@10"] == max(x["ndcg@10"] for x in report["validation"]["results"])
    assert report["test_selected"]["method"] == best["method"]
    assert report["test_selected"]["mode"] == best["mode"]
    assert report["test_selected"]["alpha"] == best["alpha"]
    assert report["test_posthoc_control"]["mode"] == "posthoc"
    assert report["best_rerank_validation"]["mode"] == "rerank"
    assert report["test_best_rerank"]["mode"] == "rerank"
    assert report["test_best_rerank"]["method"] == report["best_rerank_validation"]["method"]
    assert report["test_best_rerank"]["alpha"] == report["best_rerank_validation"]["alpha"]
    assert "ndcg@10" in report["test_best_rerank"]
    assert report["test_cohort_size"] <= 3
    assert report["test_selected"]["users_evaluated"] <= 3
    assert 0 <= report["test_selected"]["explanation_coverage"] <= 1
    assert 0 <= report["test_selected"]["ndcg@10"] <= 1
    assert "mrr@5" in report["test_selected"] and "mrr@10" in report["test_selected"]
    assert report["validation"]["max_paths"] == 50
    assert report["validation"]["max_expansions"] == 400
    assert report["max_paths"] == 50 and report["max_expansions"] == 400
    assert (tmp_path / "results" / "explanation_validation_grid.csv").is_file()
    assert (tmp_path / "results" / "explanation_study_report.json").is_file()
    assert (tmp_path / "results" / "explanation_test_paired_deltas.csv").is_file()
    assert report["paired_rerank_vs_posthoc"]["users"] == report["test_cohort_size"]
    assert abs(report["paired_rerank_vs_posthoc"]["ndcg@10"]["mean_delta"] -
               (report["test_best_rerank"]["ndcg@10"] -
                report["test_posthoc_control"]["ndcg@10"])) < 1e-12
    anonymized = pd.read_csv(tmp_path / "results" / "explanation_test_paired_deltas.csv")
    assert len(anonymized) == report["test_cohort_size"]
    assert not any("user-id" in col.lower() or col.lower() == "userid"
                   for col in anonymized.columns)
    assert (tmp_path / "results" / "train_only_node2vec" / "vkr_n2v.npy").is_file()
    assert len(json.loads(
        (tmp_path / "results" / "explanation_study_report.json").read_text())[
            "source_sha256"]["BX-Books.csv"]) == 64


def test_only_requested_explanation_configs_scored_on_test(tmp_path):
    raw = tmp_path / "raw"
    sample_bx(raw)
    from src.vkr_prepare import load_book_crossing, build_knowledge_graph, node2vec_projection
    from src.vkr_embedding_study import three_way_split
    from src.vkr_recommend import train_node2vec
    books, ratings = load_book_crossing(raw)
    train, val, test = three_way_split(ratings, max_users=6, max_books=20, seed=2042)
    graph = build_knowledge_graph(books, train)
    embeddings = train_node2vec(
        node2vec_projection(graph), dimensions=8, walk_length=6,
        num_walks=2, epochs=1, seed=42)
    target_config = {"method": "shortest", "mode": "rerank", "alpha": 0.5}
    result = _cohort_evaluation(
        graph, embeddings, train, test, max_evaluated_users=2,
        candidate_pool=20, seed=77, configs=[
            {"method": "shortest", "mode": "posthoc", "alpha": 0.0},
            target_config])
    assert len(result["results"]) == 2
    assert {x["mode"] for x in result["results"]} == {"posthoc", "rerank"}
    assert {x["method"] for x in result["results"]} == {"shortest"}



def test_paired_bootstrap_is_deterministic_and_no_user_ids_are_exported():
    baseline = [
        {"ndcg@10": 0.1, "p@10": 0.0, "mrr@10": 0.0, "explanation_coverage": 0.8},
        {"ndcg@10": 0.2, "p@10": 0.1, "mrr@10": 0.5, "explanation_coverage": 0.9},
        {"ndcg@10": 0.2, "p@10": 0.1, "mrr@10": 0.0, "explanation_coverage": 0.9},
    ]
    reranker = [dict(r, **{"ndcg@10": r["ndcg@10"] + v})
                for r, v in zip(baseline, [0.03, 0.01, -0.01])]
    first, summary = paired_bootstrap(reranker, baseline, seed=17, replicates=300)
    second, other = paired_bootstrap(reranker, baseline, seed=17, replicates=300)
    pd.testing.assert_frame_equal(first, second)
    assert summary == other
    assert abs(summary["ndcg@10"]["mean_delta"] - 0.01) < 1e-12
    assert summary["ndcg@10"]["users_improved"] == 2
    assert summary["ndcg@10"]["users_declined"] == 1
    assert summary["ndcg@10"]["users_tied"] == 0
    assert summary["ndcg@10"]["bootstrap_ci95"][0] <= 0.01 <= summary["ndcg@10"]["bootstrap_ci95"][1]
    assert "User-ID" not in first.columns
    assert not first.isnull().any().any()
    with pytest.raises(ValueError, match="same nonempty cohort"):
        paired_bootstrap(reranker[:2], baseline, seed=17)
