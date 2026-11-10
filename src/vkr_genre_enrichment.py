"""Small, rate-limited ISBN enrichment with verifiable Open Library subjects.

This is NOT a bulk harvesting tool. Open Library explicitly recommends monthly
dumps for bulk use. The resulting genre.csv is a conservative mapping of known
subject headings, not ground truth manually verified by an expert.

Usage:
 python -m src.vkr_genre_enrichment --contact researcher@example.org

Requires a prior quick/main run so that only training-supported ISBNs are
selected. The user must explicitly opt in to network calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


# Exact subject labels only. No substring matching or guessed genres from
# titles: "War", "Fiction -- 20th century", etc. are not silently mapped.
SUBJECT_GENRES = {
    "science fiction": "Science fiction",
    "science fiction stories": "Science fiction",
    "fantasy": "Fantasy",
    "fantasy fiction": "Fantasy",
    "fantasy stories": "Fantasy",
    "mystery fiction": "Mystery",
    "detective and mystery stories": "Mystery",
    "detective stories": "Mystery",
    "romance fiction": "Romance",
    "romance": "Romance",
    "love stories": "Romance",
    "historical fiction": "Historical fiction",
    "historical stories": "Historical fiction",
    "horror fiction": "Horror",
    "horror stories": "Horror",
    "thrillers (fiction)": "Thriller",
    "thrillers": "Thriller",
    "biography": "Biography",
    "biographies": "Biography",
    "autobiography": "Autobiography",
    "history": "History",
    "poetry": "Poetry",
    "drama": "Drama",
    "plays": "Drama",
    "fairy tales": "Fairy tales",
    "adventure stories": "Adventure",
    "adventure fiction": "Adventure",
    "juvenile fiction": "Children's fiction",
    "young adult fiction": "Young adult fiction",
    "humorous fiction": "Humor",
}
_ISBN = re.compile(r"(?:[0-9]{9}[0-9X]|[0-9]{13})\Z")
_WORK = re.compile(r"/works/OL[0-9]+W\Z")
_EDITION = re.compile(r"/books/OL[0-9]+M\Z")
BASE = "https://openlibrary.org"


def normalize_isbn(value: str) -> str:
    return re.sub(r"[\s-]", "", str(value)).upper()


def isbn_valid(value: str) -> bool:
    isbn = normalize_isbn(value)
    if not _ISBN.fullmatch(isbn):
        return False
    if len(isbn) == 10:
        total = sum((10-i) * (10 if ch == "X" else int(ch))
                    for i, ch in enumerate(isbn))
        return total % 11 == 0
    return sum(int(c) * (1 if i % 2 == 0 else 3)
               for i, c in enumerate(isbn[:-1])) % 10 == (-int(isbn[-1])) % 10


def identify_genres(subjects: object) -> list[str]:
    """Translate only explicitly whitelisted controlled subject terms."""
    if not isinstance(subjects, list):
        return []
    out = set()
    for subject in subjects:
        if not isinstance(subject, str):
            continue
        normalized = " ".join(subject.casefold().split())
        if normalized in SUBJECT_GENRES:
            out.add(SUBJECT_GENRES[normalized])
    return sorted(out)


def isbn_matches_edition(edition: dict, isbn: str) -> bool:
    aliases = []
    for field in ("isbn_10", "isbn_13"):
        listed = edition.get(field, [])
        if isinstance(listed, list):
            aliases.extend(item for item in listed if isinstance(item, str))
    return any(normalize_isbn(a) == isbn for a in aliases)


def extract_genres_for_isbn(isbn: str, fetch: Callable[[str], dict]) -> list[dict[str, str]]:
    """Never accept a title-only match or follow untrusted arbitrary URLs."""
    code = normalize_isbn(isbn)
    if not isbn_valid(code):
        return []
    edition = fetch(f"{BASE}/isbn/{code}.json")
    if not isinstance(edition, dict) or not isbn_matches_edition(edition, code):
        return []
    sources = []
    edition_key = edition.get("key")
    if isinstance(edition_key, str) and _EDITION.fullmatch(edition_key):
        sources.append((edition.get("subjects", []), f"{BASE}{edition_key}.json"))
    works = edition.get("works", [])
    if not isinstance(works, list):
        works = []
    for work in works:
        if not isinstance(work, dict):
            continue
        key = work.get("key")
        if isinstance(key, str) and _WORK.fullmatch(key):
            payload = fetch(f"{BASE}{key}.json")
            if isinstance(payload, dict) and payload.get("key") == key:
                sources.append((payload.get("subjects", []), f"{BASE}{key}.json"))
    facts = {(genre, url) for subjects, url in sources
             for genre in identify_genres(subjects)}
    return [{"isbn": str(isbn), "genre": genre, "source_url": url}
            for genre, url in sorted(facts)]


class CachedOpenLibraryClient:
    """Offline cached lookups; deliberate 1 request/s cap (anonymous rate)."""

    def __init__(self, cache_dir: Path, *, contact: str, timeout: float = 12,
                 wait_seconds: float = 1.1, opener=None, sleep=None):
        if not contact or "@" not in contact:
            raise ValueError("--contact must contain a valid contact email")
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.wait_seconds = wait_seconds
        self.opener = opener or urlopen
        self.sleep = sleep or time.sleep
        self.contact = contact
        self.last_request = None
        self.requests = 0
        self.cache_hits = 0

    def __call__(self, url: str) -> dict:
        from hashlib import sha256
        if not url.startswith(f"{BASE}/") or not url.endswith(".json"):
            raise ValueError("Only Open Library JSON URLs may be requested")
        key = sha256(url.encode("utf-8")).hexdigest()
        target = self.cache_dir / f"{key}.json"
        if target.is_file():
            self.cache_hits += 1
            return json.loads(target.read_text(encoding="utf-8"))
        now = time.monotonic()
        if self.last_request is not None:
            rest = self.wait_seconds - (now - self.last_request)
            if rest > 0:
                self.sleep(rest)
        req = Request(url, headers={
            "User-Agent": f"Node2VecVKR-Research/1.0 ({self.contact})",
            "Accept": "application/json",
        })
        self.last_request = time.monotonic()
        try:
            with self.opener(req, timeout=self.timeout) as response:
                payload = response.read(2_000_001)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            # No fabricated empty subject record: preserve the failure.
            raise RuntimeError(f"Open Library request failed for {url}: {exc}") from exc
        if len(payload) > 2_000_000:
            raise ValueError("Open Library JSON payload exceeded size limit")
        obj = json.loads(payload.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("Open Library response must be a JSON object")
        target.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        self.requests += 1
        return obj


def enrich_training_catalog(train_csv: Path, output_csv: Path, *, contact: str,
                            max_isbns: int = 12, cache_dir: Path | None = None,
                            fetch=None) -> dict:
    """A bounded batch; never query entire 1k/5k book graph from public API."""
    if not 1 <= max_isbns <= 25:
        raise ValueError("max_isbns must be between 1 and 25; use offline dumps for bulk access")
    if not train_csv.is_file():
        raise FileNotFoundError(f"Training ratings not found: {train_csv}")
    frame = pd.read_csv(train_csv, dtype={"ISBN": str, "User-ID": str})
    if "ISBN" not in frame:
        raise ValueError("Training CSV lacks ISBN")
    requested = sorted(frame["ISBN"].dropna().astype(str).unique())
    if not requested:
        raise ValueError("Training catalog is empty")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Persist attempts, including ISBNs with no metadata: no repeated
    # requests for the same missing work when extending the batch.
    attempts_path = output_csv.with_suffix(".attempts.json")
    if attempts_path.exists():
        status = json.loads(attempts_path.read_text(encoding="utf-8"))
        if not isinstance(status, dict):
            raise ValueError("Attempt ledger must be a JSON object")
    else:
        status = {}
    existing = pd.read_csv(output_csv, dtype=str).fillna("") if output_csv.is_file() else pd.DataFrame(columns=["isbn", "genre", "source_url"])
    if not {"isbn", "genre", "source_url"} <= set(existing.columns):
        raise ValueError("Existing enrichment CSV lacks isbn,genre,source_url")
    client = fetch or CachedOpenLibraryClient(cache_dir or output_csv.parent / "openlibrary_cache", contact=contact)
    todo = [isbn for isbn in requested
            if isbn not in status and isbn_valid(isbn)][:max_isbns]
    successes = []
    failures = {}
    for isbn in todo:
        try:
            facts = extract_genres_for_isbn(isbn, client)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            failures[isbn] = str(exc)
            # Do not mark failures as complete: they can be retried.
            continue
        successes.extend(facts)
        status[isbn] = len(facts)
        # Save after each ISBN to make partial enrichment resumable.
        appended = pd.concat([existing, pd.DataFrame(successes)], ignore_index=True)
        appended = appended.drop_duplicates(subset=["isbn", "genre", "source_url"])
        appended.sort_values(["isbn", "genre", "source_url"]).to_csv(output_csv, index=False)
        attempts_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "attempted_this_batch": len(todo),
        "matched_this_batch": sum(1 for isbn in todo if status.get(isbn, 0) > 0),
        "facts_added_this_batch": len(successes),
        "completed_unique_isbns": len(status),
        "training_catalog_unique_isbns": len(requested),
        "covered_training_isbns": len(set(existing["isbn"]) | {x["isbn"] for x in successes}),
        "unmatched_isbns": sum(val == 0 for val in status.values()),
        "failed_requests": failures,
        "network_requests": getattr(client, "requests", None),
        "cache_hits": getattr(client, "cache_hits", None),
        "caveat": "Exact Open Library subject headings mapped to a fixed genre taxonomy; not manual expert verification.",
        "source": "Open Library ISBN/Works JSON API",
    }
    report_path = output_csv.with_suffix(".coverage.json")
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Small opt-in, provenance-preserving Open Library subject/genre enrichment")
    p.add_argument("--train-csv", type=Path, default=Path("data/processed/vkr_train_ratings.csv"))
    p.add_argument("--output-csv", type=Path, default=Path("data/raw/book_genres.csv"))
    p.add_argument("--contact", required=True, help="Your real contact email for Open Library User-Agent")
    p.add_argument("--max-isbns", type=int, default=12, help="1..25; API is not for bulk harvest")
    args = p.parse_args()
    print(json.dumps(enrich_training_catalog(args.train_csv, args.output_csv,
                                             contact=args.contact, max_isbns=args.max_isbns),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
