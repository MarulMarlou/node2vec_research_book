"""Reproducible Book-Crossing preparation for the explainable book recommender.

Run with ``python -m src.vkr_prepare`` from the repository root. This is an
independent VКR pipeline; the legacy experiments are intentionally unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import networkx as nx
import pandas as pd


REQUIRED_BOOK_COLUMNS = {"ISBN", "Book-Title", "Book-Author", "Publisher"}
REQUIRED_RATING_COLUMNS = {"User-ID", "ISBN", "Book-Rating"}


def _read_bx(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Book-Crossing file: {path}")
    # The original BX distribution is semicolon-delimited and latin-1 encoded.
    df = pd.read_csv(path, sep=";", encoding="latin-1", dtype=str,
                     on_bad_lines="warn", low_memory=False)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    df["source_row"] = [str(i) for i in range(2, len(df) + 2)]
    return df


def _clean_id(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def load_book_crossing(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    books = _read_bx(raw_dir / "BX-Books.csv", REQUIRED_BOOK_COLUMNS)
    ratings = _read_bx(raw_dir / "BX-Book-Ratings.csv", REQUIRED_RATING_COLUMNS)
    for df in (books, ratings):
        df["ISBN"] = _clean_id(df["ISBN"])
        df.drop(df[df["ISBN"] == ""].index, inplace=True)
    books["Book-Title"] = _clean_id(books["Book-Title"])
    books.drop(books[books["Book-Title"] == ""].index, inplace=True)
    books = books.drop_duplicates(subset="ISBN", keep="first").reset_index(drop=True)

    ratings["User-ID"] = _clean_id(ratings["User-ID"])
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    ratings = ratings[(ratings["User-ID"] != "") &
                      ratings["Book-Rating"].between(1, 10) &
                      (ratings["Book-Rating"] % 1 == 0)]
    # Zero means implicit feedback in BX; it is not an explicit negative rating.
    ratings = ratings[ratings["ISBN"].isin(set(books["ISBN"]))]
    ratings = ratings.drop_duplicates(["User-ID", "ISBN"], keep="last")
    ratings = ratings.reset_index(drop=True)
    ratings["Book-Rating"] = ratings["Book-Rating"].astype(int)
    return books, ratings


def _stable_order_key(seed: int, user_id: str, isbn: str) -> str:
    return hashlib.sha256(f"{seed}|{user_id}|{isbn}".encode("utf-8")).hexdigest()


def split_ratings(ratings: pd.DataFrame, *, test_ratio: float = 0.2,
                  min_positive_rating: int = 7, seed: int = 42
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out positives *before* building/training a graph.

    For each user with at least two positive ratings, hold out at least one
    positive, retaining at least one for training. Unrated and non-positive
    interactions are not test positives and remain in the train split.
    """
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    if not 1 <= min_positive_rating <= 10:
        raise ValueError("min_positive_rating must be in [1, 10]")
    required = REQUIRED_RATING_COLUMNS
    if required - set(ratings.columns):
        raise ValueError("ratings lack required columns")
    ratings = ratings.reset_index(drop=True)
    heldout: set[int] = set()
    positive = ratings[ratings["Book-Rating"] >= min_positive_rating]
    for user_id, part in positive.groupby("User-ID", sort=True):
        if len(part) < 2:
            continue
        ranked = sorted(part.index.tolist(), key=lambda i: (
            _stable_order_key(seed, str(user_id), str(ratings.at[i, "ISBN"])),
            str(ratings.at[i, "ISBN"])))
        n_test = min(len(ranked) - 1, max(1, math.floor(len(ranked) * test_ratio)))
        heldout.update(ranked[:n_test])
    test = ratings.loc[sorted(heldout)].copy().reset_index(drop=True)
    train = ratings.drop(index=list(heldout)).reset_index(drop=True)
    return train, test


def _select_by_train_count(df: pd.DataFrame, column: str, limit: int | None) -> set[str]:
    counts = df[column].value_counts()
    ranked = sorted(counts.index.tolist(), key=lambda value: (-int(counts[value]), str(value)))
    if limit is not None:
        if limit <= 0:
            raise ValueError("max_users/max_books must be positive")
        ranked = ranked[:limit]
    return set(ranked)


def restrict_from_training(train: pd.DataFrame, test: pd.DataFrame,
                           max_users: int | None, max_books: int | None
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select supported users/books using only the training interactions."""
    users = _select_by_train_count(train, "User-ID", max_users)
    train = train[train["User-ID"].isin(users)]
    test = test[test["User-ID"].isin(users)]
    isbns = _select_by_train_count(train, "ISBN", max_books)
    train = train[train["ISBN"].isin(isbns)].reset_index(drop=True)
    test = test[test["ISBN"].isin(isbns)].reset_index(drop=True)
    return train, test


def _key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def load_genres(path: Path | None) -> pd.DataFrame:
    """Optional enrichment with columns isbn,genre,source_url."""
    columns = ["isbn", "genre", "source_url"]
    if path is None:
        return pd.DataFrame(columns=columns)
    if not path.is_file():
        raise FileNotFoundError(f"Genre enrichment file not found: {path}")
    result = pd.read_csv(path, dtype=str).fillna("")
    if set(columns) - set(result.columns):
        raise ValueError("Genre CSV must have isbn,genre,source_url columns")
    for col in columns:
        result[col] = result[col].str.strip()
    result = result[(result["isbn"] != "") & (result["genre"] != "")]
    if (result["source_url"] == "").any():
        raise ValueError("Each genre fact needs a source_url for verification")
    return result[columns].drop_duplicates().reset_index(drop=True)


def build_knowledge_graph(books: pd.DataFrame, train: pd.DataFrame,
                          genres: pd.DataFrame | None = None) -> nx.MultiDiGraph:
    """Store typed *directed* facts and provenance, without held-out ratings."""
    graph = nx.MultiDiGraph()
    supported = set(train["ISBN"])
    for _, b in books.iterrows():
        isbn = str(b["ISBN"])
        if isbn not in supported:
            continue
        book_id = f"book:{isbn}"
        graph.add_node(book_id, entity_type="book", label=str(b["Book-Title"]), isbn=isbn)
        for column, kind, relation in (("Book-Author", "author", "written_by"),
                                        ("Publisher", "publisher", "published_by")):
            label = str(b.get(column, "") or "").strip()
            if not label or label.lower() == "nan":
                continue
            node_id = f"{kind}:{_key(label)}"
            graph.add_node(node_id, entity_type=kind, label=label)
            graph.add_edge(book_id, node_id, relation=relation, weight=1.0,
                           source_file="BX-Books.csv", source_row=str(b["source_row"]), source_url="")

    for _, r in train.iterrows():
        uid, isbn = str(r["User-ID"]), str(r["ISBN"])
        user_id = f"user:{uid}"
        book_id = f"book:{isbn}"
        graph.add_node(user_id, entity_type="user", label=f"User {uid}")
        graph.add_edge(user_id, book_id, relation="rated", rating=int(r["Book-Rating"]),
                       weight=float(r["Book-Rating"]) / 10.0,
                       source_file="BX-Book-Ratings.csv", source_row=str(r["source_row"]), source_url="")

    if genres is not None:
        for record in genres.itertuples(index=False):
            isbn, label, source_url = str(record.isbn), str(record.genre), str(record.source_url)
            book_id = f"book:{isbn}"
            if not graph.has_node(book_id):
                continue
            genre_id = f"genre:{_key(label)}"
            graph.add_node(genre_id, entity_type="genre", label=label)
            graph.add_edge(book_id, genre_id, relation="has_genre", weight=1.0,
                           source_file="book_genres.csv", source_row="", source_url=source_url)
    return graph


def node2vec_projection(graph: nx.MultiDiGraph) -> nx.Graph:
    """An undirected simple projection for Node2Vec; retain KG for explanations."""
    projection = nx.Graph()
    projection.add_nodes_from(graph.nodes(data=True))
    for source, target, edge in graph.edges(data=True):
        if not projection.has_edge(source, target):
            projection.add_edge(source, target, weight=float(edge.get("weight", 1.0)))
        else:
            projection[source][target]["weight"] += float(edge.get("weight", 1.0))
    return projection


def prepare(raw_dir: Path, processed_dir: Path, *, genre_csv: Path | None = None,
            max_users: int | None = 1000, max_books: int | None = 5000,
            seed: int = 42, test_ratio: float = 0.2, positive_threshold: int = 7) -> dict:
    books, ratings = load_book_crossing(raw_dir)
    train, test = split_ratings(ratings, seed=seed, test_ratio=test_ratio,
                                min_positive_rating=positive_threshold)
    initial_train, initial_test = len(train), len(test)
    train, test = restrict_from_training(train, test, max_users, max_books)
    genres = load_genres(genre_csv)
    graph = build_knowledge_graph(books, train, genres)
    projection = node2vec_projection(graph)
    processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_dir / "vkr_train_ratings.csv", index=False)
    test.to_csv(processed_dir / "vkr_test_ratings.csv", index=False)
    nx.write_graphml(graph, processed_dir / "vkr_knowledge_graph.graphml")
    nx.write_graphml(projection, processed_dir / "vkr_node2vec_graph.graphml")
    stats = {
        "seed": seed, "test_ratio": test_ratio, "positive_threshold": positive_threshold,
        "max_users": max_users, "max_books": max_books,
        "validated_rating_rows": len(ratings), "train_rows_before_limit": initial_train,
        "test_rows_before_limit": initial_test, "train_rows": len(train),
        "test_rows": len(test), "kg_nodes": graph.number_of_nodes(),
        "kg_edges": graph.number_of_edges(),
        "entity_types": {str(k): int(v) for k, v in pd.Series(nx.get_node_attributes(graph, "entity_type")).value_counts().items()},
        "relation_types": {str(k): int(v) for k, v in pd.Series([e["relation"] for _, _, e in graph.edges(data=True)]).value_counts().items()},
        "genre_source": str(genre_csv) if genre_csv else None,
        "warning": "Missing genre enrichment; genre requirements remain incomplete" if genres.empty else None,
    }
    (processed_dir / "vkr_preparation_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Build a leakage-safe typed book knowledge graph")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--genre-csv", type=Path, default=None)
    p.add_argument("--max-users", type=int, default=1000)
    p.add_argument("--max-books", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--positive-threshold", type=int, default=7)
    args = p.parse_args()
    stats = prepare(args.raw_dir, args.processed_dir, genre_csv=args.genre_csv,
                    max_users=args.max_users, max_books=args.max_books, seed=args.seed,
                    test_ratio=args.test_ratio, positive_threshold=args.positive_threshold)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
