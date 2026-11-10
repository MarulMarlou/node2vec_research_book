"""Synthetic, paired candidate-pool study tests; never calls network or real data."""
import json

import pandas as pd
import pytest

from src.vkr_candidate_pool_study import (
    _only_config, run_candidate_pool_study,
)


def sample_bx(raw, n_books=20, n_users=6):
    raw.mkdir()
    pd.DataFrame([
        {"ISBN": f"{i:03d}", "Book-Title": f"Book {i}",
         "Book-Author": f"Author {i % 4}",
         "Publisher": f"Publisher {i % 3}"}
        for i in range(1, n_books + 1)
    ]).to_csv(raw / "BX-Books.csv", sep=";", encoding="latin-1", index=False)
    pd.DataFrame([
        {"User-ID": str(user), "ISBN": f"{i:03d}",
         "Book-Rating": 7 + (i % 3)}
        for user in range(1, n_users + 1)
        for i in range(1, n_books + 1)
    ]).to_csv(raw / "BX-Book-Ratings.csv", sep=";", encoding="latin-1", index=False)


def test_comparison_uses_one_split_and_model_for_both_pool_sizes(tmp_path):
    raw = tmp_path / "raw"
    sample_bx(raw)
    output = tmp_path / "pool_comparison"
    report = run_candidate_pool_study(
        raw, output, split_seed=2102, max_users=6, max_books=20,
        max_evaluated_users=3, max_depth=2, max_paths=25,
        max_expansions=50, model_params={
            "dimensions": 8, "walk_length": 6, "num_walks": 2,
            "epochs": 1, "p": 1., "q": 1., "window": 3, "seed": 43})
    assert report["train_rows"] > 0
    assert report["validation_rows"] > 0
    assert report["test_rows"] > 0
    assert report["test_cohort"] <= 3
    val = pd.read_csv(output / "candidate_pool_validation_grid.csv")
    assert len(val) == 16
    assert set(val["candidate_pool"]) == {25, 100}
    for size in (25, 100):
        scoped = val[val["candidate_pool"] == size].sort_values(
            ["ndcg@10", "p@10", "config_id"],
            ascending=[False, False, True]).iloc[0]
        chosen = report[f"validation_selected_pool{size}"]
        assert chosen["method"] == scoped["method"]
        assert chosen["mode"] == scoped["mode"]
        assert chosen["alpha"] == scoped["alpha"]
    assert report["test_pool25_selected"]["users_evaluated"] == report["test_cohort"]
    assert report["test_pool100_selected"]["users_evaluated"] == report["test_cohort"]
    assert report["test_pool100_fixed25_configuration"]["users_evaluated"] == report["test_cohort"]
    assert _only_config(report["test_pool100_fixed25_configuration"]) == _only_config(
        report["validation_selected_pool25"])
    assert _only_config(report["test_pool100_selected"]) == _only_config(
        report["validation_selected_pool100"])

    for name in (
        "candidate_pool_test_fixed_configuration_deltas.csv",
        "candidate_pool_test_selected_pipeline_deltas.csv",
    ):
        df = pd.read_csv(output / name)
        assert len(df) == report["test_cohort"]
        assert df["anonymous_user_index"].tolist() == list(range(report["test_cohort"]))
        assert df.isna().sum().sum() == 0
        assert not any("User-ID" in x for x in df.columns)
    assert (output / "candidate_pool_study_report.json").is_file()
    assert (output / "train_only_node2vec" / "vkr_n2v.npy").is_file()

    # With only twenty books, there cannot be more than 25 candidates,
    # so increasing the pool cap to 100 changes nothing.
    assert abs(report["paired_pool100_fixed_config_minus_pool25"]
               ["ndcg@10"]["mean_delta"]) < 1e-12
    assert report["test_candidate_ceiling_pool25"]["chosen_pool_has_positive_users"] == (
        report["test_candidate_ceiling_pool100"]["chosen_pool_has_positive_users"])
    assert report["paired_pool100_selected_minus_pool25_selected"]["users"] == (
        report["test_cohort"])


def test_rejects_empty_study_data(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(FileNotFoundError):
        run_candidate_pool_study(raw, tmp_path / "out")



def test_expanded_pool_really_contains_more_candidates_and_trains_once(
        tmp_path, monkeypatch):
    import src.vkr_candidate_pool_study as subject

    raw = tmp_path / "raw"
    raw.mkdir()
    n_books = 70
    pd.DataFrame([
        {"ISBN": f"{i:03d}", "Book-Title": f"Book {i}",
         "Book-Author": f"Author {i % 5}",
         "Publisher": f"Publisher {i % 3}"}
        for i in range(1, n_books + 1)
    ]).to_csv(raw / "BX-Books.csv", sep=";", encoding="latin-1", index=False)
    # Each user knows a small part of the catalogue, leaving >25 candidate
    # books for both methods when pool limits really differ.
    pd.DataFrame([
        {"User-ID": str(user), "ISBN": f"{((user - 1) * 9 + i) % n_books + 1:03d}",
         "Book-Rating": 8}
        for user in range(1, 9)
        for i in range(18)
    ]).to_csv(raw / "BX-Book-Ratings.csv", sep=";", encoding="latin-1", index=False)

    original = subject.train_node2vec
    fits = []

    def counted_train(graph, **params):
        fits.append(params.copy())
        return original(graph, **params)

    monkeypatch.setattr(subject, "train_node2vec", counted_train)
    result = subject.run_candidate_pool_study(
        raw, tmp_path / "comparison", split_seed=2102,
        max_users=8, max_books=70, max_evaluated_users=4,
        max_depth=2, max_paths=20, max_expansions=50,
        model_params={
            "dimensions": 8, "walk_length": 6, "num_walks": 2,
            "epochs": 1, "p": 1., "q": 1., "window": 3, "seed": 43})
    assert len(fits) == 1, "Both pools must share one exact model fit"
    assert result["test_pool25_selected"]["actual_pool_candidates"] <= 25
    assert result["test_pool100_fixed25_configuration"]["actual_pool_candidates"] > 25
    assert result["test_pool100_fixed25_configuration"]["actual_pool_candidates"] > (
        result["test_pool25_selected"]["actual_pool_candidates"])
    assert (result["test_pool100_fixed25_configuration"]["total_pool_explanation_time_s"] >= 0)
    assert (result["test_candidate_ceiling_pool100"]["chosen_pool_has_positive_users"] >=
            result["test_candidate_ceiling_pool25"]["chosen_pool_has_positive_users"])
