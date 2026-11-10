"""Tests use local fake Open Library JSON responses: no live HTTP calls."""
import json

import pandas as pd
import pytest

from src.vkr_genre_enrichment import (
    BASE, CachedOpenLibraryClient, enrich_training_catalog,
    extract_genres_for_isbn, identify_genres, isbn_valid, normalize_isbn,
)


ISBN = "0306406152"  # A valid ISBN-10
WORK = "/works/OL123W"
EDITION = "/books/OL456M"


def fake_fetch(url):
    if url == f"{BASE}/isbn/{ISBN}.json":
        return {"key": EDITION, "isbn_10": [ISBN], "works": [{"key": WORK}],
                "subjects": ["Poetry"]}
    if url == f"{BASE}{WORK}.json":
        return {"key": WORK, "subjects": ["Science fiction", "History of printing",
                                          "Fantasy", "Fiction -- 20th century"]}
    raise AssertionError(f"Unexpected URL {url}")


def test_exact_isbn_matching_and_subject_taxonomy_only():
    assert isbn_valid(ISBN)
    assert isbn_valid("9780306406157")
    assert normalize_isbn("978-0-306-40615-7") == "9780306406157"
    assert not isbn_valid("0306406151")
    assert not isbn_valid("not_an_isbn")
    assert identify_genres(["Science fiction", "fantasy", "Science", "Fantasy",
                            "Biography of scientists"]) == ["Fantasy", "Science fiction"]
    facts = extract_genres_for_isbn(ISBN, fake_fetch)
    assert {(r["genre"], r["source_url"]) for r in facts} == {
        ("Fantasy", f"{BASE}{WORK}.json"),
        ("Science fiction", f"{BASE}{WORK}.json"),
        ("Poetry", f"{BASE}{EDITION}.json"),
    }
    assert {r["isbn"] for r in facts} == {ISBN}


def test_rejects_mismatched_isbn_and_untrusted_work_key():
    def mismatched(url):
        return {"isbn_10": ["0596007973"], "subjects": ["Fantasy"],
                "key": EDITION}
    assert extract_genres_for_isbn(ISBN, mismatched) == []

    def hostile(url):
        return {"isbn_10": [ISBN], "key": EDITION,
                "works": [{"key": "https://attacker.invalid/secret"}],
                "subjects": ["History of printing"]}
    assert extract_genres_for_isbn(ISBN, hostile) == []


def test_small_training_only_batch_resumes_and_never_invents_missing_facts(tmp_path):
    train = tmp_path / "train.csv"
    pd.DataFrame([{"ISBN": ISBN, "User-ID": "1"}, {"ISBN": "9780306406157", "User-ID": "1"},
                  {"ISBN": "invalid", "User-ID": "2"}]).to_csv(train, index=False)
    work_count = []

    def fetch(url):
        work_count.append(url)
        if url == f"{BASE}/isbn/9780306406157.json":
            return {"key": "/books/OL999M", "isbn_13": ["9780306406157"],
                    "subjects": ["Art history"]}
        return fake_fetch(url)

    csv = tmp_path / "book_genres.csv"
    report = enrich_training_catalog(train, csv, max_isbns=2,
                                     contact="research@example.org", fetch=fetch)
    assert report["attempted_this_batch"] == 2
    assert report["facts_added_this_batch"] == 3
    assert report["covered_training_isbns"] == 1
    assert len(pd.read_csv(csv, dtype=str)) == 3
    assert json.loads(csv.with_suffix(".attempts.json").read_text())[ISBN] == 3
    assert json.loads(csv.with_suffix(".attempts.json").read_text())["9780306406157"] == 0
    again = enrich_training_catalog(train, csv, max_isbns=2,
                                    contact="research@example.org", fetch=fetch)
    assert again["attempted_this_batch"] == 0
    assert len(pd.read_csv(csv, dtype=str)) == 3
    assert len(work_count) == 3


def test_small_network_batches_enforced(tmp_path):
    train = tmp_path / "train.csv"
    pd.DataFrame([{"ISBN": ISBN}]).to_csv(train, index=False)
    with pytest.raises(ValueError, match="1 and 25"):
        enrich_training_catalog(train, tmp_path / "out.csv", contact="a@b.org",
                                max_isbns=100)


def test_cache_only_open_library_json_and_preserves_response(tmp_path):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            return b'{"key":"/works/OL123W","subjects":["Fantasy"]}'

    def opener(request, timeout):
        calls.append(request)
        assert "research@example.org" in request.get_header("User-agent")
        return Response()

    client = CachedOpenLibraryClient(tmp_path / "cache", contact="research@example.org",
                                     opener=opener, sleep=lambda seconds: None)
    url = f"{BASE}{WORK}.json"
    assert client(url)["subjects"] == ["Fantasy"]
    assert client(url)["subjects"] == ["Fantasy"]
    assert len(calls) == 1
    assert client.requests == 1 and client.cache_hits == 1
    with pytest.raises(ValueError, match="Only Open Library"):
        client("https://attacker.invalid/secret.json")
