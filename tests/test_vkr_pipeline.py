from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vkr_pipeline import run_pipeline


def test_end_to_end_pipeline_on_synthetic_books(tmp_path):
    import pandas as pd
    raw = tmp_path / "raw"
    raw.mkdir()
    books = pd.DataFrame([
        {"ISBN":f"00{i}", "Book-Title":f"Book {i}",
         "Book-Author":f"Author {i%2}", "Publisher":"Press"}
        for i in range(1,6)
    ])
    ratings = pd.DataFrame([
        {"User-ID":str(user), "ISBN":f"00{book}", "Book-Rating":8 + book%2}
        for user in (1,2,3) for book in range(1,6)
    ])
    books.to_csv(raw / "BX-Books.csv", sep=";", index=False, encoding="latin-1")
    ratings.to_csv(raw / "BX-Book-Ratings.csv", sep=";", index=False, encoding="latin-1")
    report = run_pipeline(raw, tmp_path/"processed", tmp_path/"results",
                          max_users=None, max_books=None, dimensions=8,
                          walk_length=6, num_walks=2, epochs=1,
                          example_book="book:001")
    assert report["preparation"]["test_rows"] == 3
    assert report["user_to_book_holdout"]["users_with_test_positives"] == 3
    assert report["user_to_book_holdout"]["user_coverage"] == 1
    assert report["book_to_book_demonstration_no_ground_truth"]["node2vec_top10"]
    assert (tmp_path/"results"/"metrics"/"vkr_pipeline_report.json").is_file()
    assert (tmp_path/"results"/"embeddings"/"vkr_n2v.npy").is_file()
    json.dumps(report)
