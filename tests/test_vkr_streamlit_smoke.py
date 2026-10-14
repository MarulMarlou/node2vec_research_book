"""Headless Streamlit AppTest: execute the actual page using synthetic model data."""
from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from src.vkr_pipeline import run_pipeline


def test_streamlit_page_loads_and_recommends(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    books = pd.DataFrame([
        {"ISBN": f"00{i}", "Book-Title": f"Book {i}",
         "Book-Author": f"Writer {i % 2}", "Publisher": "Press"}
        for i in range(1, 7)
    ])
    ratings = pd.DataFrame([
        {"User-ID": str(user), "ISBN": f"00{book}", "Book-Rating": 9}
        for user in (1, 2, 3) for book in range(1, 7)
    ])
    books.to_csv(raw / "BX-Books.csv", sep=";", encoding="latin-1", index=False)
    ratings.to_csv(raw / "BX-Book-Ratings.csv", sep=";", encoding="latin-1", index=False)
    processed, results = tmp_path / "processed", tmp_path / "results"
    run_pipeline(raw, processed, results, max_users=None, max_books=None,
                 dimensions=8, walk_length=6, num_walks=2, epochs=1)

    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root / "src" / "vkr_app.py"), default_timeout=20).run()
    assert not at.exception
    assert len(at.sidebar.text_input) >= 2
    at.sidebar.text_input[0].set_value(str(processed))
    at.sidebar.text_input[1].set_value(str(results))
    at.run(timeout=20)
    assert not at.exception, [str(e.message) for e in at.exception]
    assert len(at.metric) == 3
    assert len(at.tabs) == 4
    assert any("Book" in element.value for element in at.header)
