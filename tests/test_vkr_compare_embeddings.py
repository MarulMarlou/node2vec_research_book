"""Methodology regression for fair Node2Vec/TransE comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd

from src.vkr_compare_embeddings import (
    _auc_explicit, _signed_split, comparison_grid, run_comparison
)


def _make_synthetic_bx(raw: Path, *, users: int = 5, books: int = 15) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"ISBN": f"{i:04d}", "Book-Title": f"Book {i}",
         "Book-Author": f"Author {i % 4}", "Publisher": f"Pub {i % 3}"}
        for i in range(1, books + 1)
    ]).to_csv(raw / "BX-Books.csv", index=False, encoding="latin-1", sep=";")
    pd.DataFrame([
        {"User-ID": str(u), "ISBN": f"{i:04d}",
         "Book-Rating": 8 if (i + u) % 3 else 3}
        for u in range(1, users + 1) for i in range(1, books + 1)
    ]).to_csv(raw / "BX-Book-Ratings.csv", index=False, encoding="latin-1", sep=";")


def test_roc_auc_only_compares_explicit_heldout_positives_and_negatives():
    assert _auc_explicit([0.8, 0.2], [0.1, 0.4]) == 0.75
    assert _auc_explicit([0.5], [0.5]) == 0.5
    assert _auc_explicit([], [0.4]) is None
    assert _auc_explicit([0.3], []) is None


def test_signed_split_heldout_lows_never_enter_training(tmp_path):
    raw = tmp_path / "raw"
    _make_synthetic_bx(raw)
    from src.vkr_prepare import load_book_crossing
    _, ratings = load_book_crossing(raw)
    a = _signed_split(ratings, max_users=5, max_books=15, seed=2028)
    b = _signed_split(ratings.sample(frac=1, random_state=10),
                      max_users=5, max_books=15, seed=2028)
    pa = [set(zip(x["User-ID"], x["ISBN"])) for x in a]
    pb = [set(zip(x["User-ID"], x["ISBN"])) for x in b]
    assert pa == pb
    assert not pa[0] & pa[1] and not pa[0] & pa[2] and not pa[1] & pa[2]
    assert (a[1]["Book-Rating"] < 7).any()
    assert (a[2]["Book-Rating"] < 7).any()


def test_realistic_comparison_pipeline_with_explicit_auc(tmp_path):
    raw = tmp_path / "raw"
    _make_synthetic_bx(raw)
    out = tmp_path / "study"
    grid = {
        "Node2Vec": [{"dimensions": 8, "walk_length": 8, "num_walks": 2,
                      "epochs": 2, "p": 1.0, "q": 1.0, "window": 3}],
        "TransE": [{"dimensions": 8, "epochs": 3, "batch_size": 16,
                    "learning_rate": 0.05, "margin": 1.0}],
    }
    report = run_comparison(raw, out, max_users=5, max_books=15,
                            seeds=(42,), grid=grid)
    assert report["experiments"] == 2
    assert report["experiments_by_method"] == {"Node2Vec": 1, "TransE": 1}
    assert report["rows"]["train"] > 0
    assert report["rows"]["validation"] > 0
    assert report["rows"]["test"] > 0
    for method in ("Node2Vec", "TransE"):
        result = report["selected"][method]
        assert 0 <= result["test"]["ndcg@10"] <= 1
        assert 0 <= result["test"]["f1@10"] <= 1
        assert result["test"]["auc_eligible_users"] > 0
        assert 0 <= result["test"]["auc_roc_explicit"] <= 1
        assert result["test"]["auc_protocol"].startswith("observed held-out ratings")
    assert len(pd.read_csv(out / "comparison_experiments.csv")) == 2
    assert (out / "selected_transe_relations.csv").is_file()
    assert (out / "selected_node2vec" / "vkr_n2v.npy").is_file()
    assert (out / "comparison_report.json").is_file()


def test_full_grid_contains_at_least_30_both_algorithms():
    grid = comparison_grid("full")
    assert len(grid["Node2Vec"]) * 2 + len(grid["TransE"]) * 2 == 42
    assert len(grid["TransE"]) > 0
