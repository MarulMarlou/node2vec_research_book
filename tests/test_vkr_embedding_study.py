from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vkr_embedding_study import configs, three_way_split, run_study


def test_experiment_grid_contains_36_full_runs():
    assert len(configs("quick")) == 3
    assert len(configs("full")) == 12
    with pytest.raises(ValueError):
        configs("incorrect")


def test_three_splits_are_disjoint_and_order_independent():
    df = pd.DataFrame([{"User-ID": str(user), "ISBN": f"{i:03d}", "Book-Rating": 8}
                       for user in range(1, 6) for i in range(1, 10)])
    a = three_way_split(df, max_users=5, max_books=9)
    b = three_way_split(df.sample(frac=1, random_state=24), max_users=5, max_books=9)
    sets = [[set(zip(part["User-ID"], part["ISBN"])) for part in triple] for triple in (a, b)]
    assert sets[0] == sets[1]
    for train, val, test in sets:
        assert train and val and test
        assert not train & val and not train & test and not val & test


def test_study_fits_on_train_and_reports_validation_selection(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([
        {"ISBN": f"{i:03d}", "Book-Title": f"Book {i}",
         "Book-Author": f"Author {i%3}", "Publisher": "Press"}
        for i in range(1, 12)
    ]).to_csv(raw / "BX-Books.csv", sep=";", encoding="latin-1", index=False)
    pd.DataFrame([
        {"User-ID": str(u), "ISBN": f"{i:03d}", "Book-Rating": 8 + i % 2}
        for u in range(1, 6) for i in range(1, 12)
    ]).to_csv(raw / "BX-Book-Ratings.csv", sep=";", encoding="latin-1", index=False)
    out = tmp_path / "study"
    report = run_study(raw, out, mode="quick", max_users=5, max_books=11, seeds=(42,))
    assert report["experiments"] == 3
    assert report["rows"]["validation"] > 0 and report["rows"]["test"] > 0
    assert report["selected"]["algorithm"] == "Node2Vec"
    assert 0 <= report["selected_test"]["methods"]["node2vec_max_cosine"]["ndcg@10"] <= 1
    assert len(pd.read_csv(out / "embedding_experiments.csv")) == 3
    assert (out / "selected_embeddings" / "vkr_n2v.npy").is_file()


def test_study_requires_holdout_positives(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([
        {"ISBN": "001", "Book-Title": "Book", "Book-Author": "Author", "Publisher": "Press"}
    ]).to_csv(raw / "BX-Books.csv", sep=";", encoding="latin-1", index=False)
    pd.DataFrame([
        {"User-ID": "1", "ISBN": "001", "Book-Rating": 9}
    ]).to_csv(raw / "BX-Book-Ratings.csv", sep=";", encoding="latin-1", index=False)
    with pytest.raises(ValueError, match="validation"):
        run_study(raw, tmp_path / "study", max_users=1, max_books=1)
