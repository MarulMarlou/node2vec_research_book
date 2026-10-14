"""Validation-selected Node2Vec study; do not use test data for tuning.

Quick: 3 configurations, 1 seed. Full: 12 configurations, 3 seeds = 36 runs.
TransE is a separate pending comparison, not silently replaced by Jaccard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import networkx as nx
import pandas as pd
from threadpoolctl import threadpool_limits

from src.vkr_prepare import (load_book_crossing, split_ratings, restrict_from_training,
                             build_knowledge_graph, node2vec_projection, load_genres)
from src.vkr_book_eval import evaluate_book_retrieval_proxy
from src.vkr_recommend import train_node2vec, save_embeddings


def configs(mode: str) -> list[dict]:
    if mode == "quick":
        return [{"dimensions": 32, "walk_length": 12, "num_walks": 2, "epochs": 2,
                 "p": p, "q": q, "window": 5}
                for p, q in [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)]]
    if mode == "full":
        return [{"dimensions": dimensions, "walk_length": length,
                 "num_walks": 4, "epochs": 3, "p": p, "q": q, "window": 5}
                for dimensions in (64, 128) for length in (20, 40)
                for p, q in [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)]]
    raise ValueError("mode must be quick or full")


def three_way_split(ratings: pd.DataFrame, *, max_users: int,
                    max_books: int, seed: int = 2027,
                    positive_threshold: int = 7
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = split_ratings(ratings, test_ratio=0.2, seed=seed,
                                    min_positive_rating=positive_threshold)
    train, validation = split_ratings(train_val, test_ratio=0.2, seed=seed + 1,
                                      min_positive_rating=positive_threshold)
    train, validation = restrict_from_training(train, validation, max_users, max_books)
    users, books = set(train["User-ID"]), set(train["ISBN"])
    test = test[test["User-ID"].isin(users) & test["ISBN"].isin(books)].reset_index(drop=True)
    pairs = [set(zip(frame["User-ID"], frame["ISBN"])) for frame in (train, validation, test)]
    if pairs[0] & pairs[1] or pairs[0] & pairs[2] or pairs[1] & pairs[2]:
        raise ValueError("Train/validation/test overlap")
    return train, validation, test


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_study(raw_dir: Path, output_dir: Path, *, mode: str = "quick",
              max_users: int = 150, max_books: int = 1000,
              seed_split: int = 2027, seeds: tuple[int, ...] | None = None,
              positive_threshold: int = 7, genre_csv: Path | None = None) -> dict:
    if not isinstance(max_users, int) or max_users < 1 or not isinstance(max_books, int) or max_books < 1:
        raise ValueError("max_users/max_books must be positive")
    if seeds is None:
        seeds = (42,) if mode == "quick" else (42, 43, 44)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be nonempty and distinct")
    variants = configs(mode)
    books, ratings = load_book_crossing(raw_dir)
    train, val, test = three_way_split(ratings, max_users=max_users, max_books=max_books,
                                      seed=seed_split, positive_threshold=positive_threshold)
    if train.empty or val.empty:
        raise ValueError("Training and validation must not be empty")
    graph = build_knowledge_graph(books, train, load_genres(genre_csv))
    projection = node2vec_projection(graph)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "validation.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    nx.write_graphml(graph, output_dir / "training_knowledge_graph.graphml")

    experiments = []
    best_key = None
    best_row = None
    best_embeddings = None
    best_validation = None
    for config_id, params in enumerate(variants, start=1):
        for seed in seeds:
            started = time.perf_counter()
            embeddings = train_node2vec(projection, seed=seed, **params)
            seconds = time.perf_counter() - started
            with threadpool_limits(limits=1, user_api="blas"):
                proxy = evaluate_book_retrieval_proxy(graph, embeddings, train, val,
                                                      positive_threshold=positive_threshold)
            node2vec = proxy.get("methods", {}).get("node2vec_max_cosine", {})
            jaccard = proxy.get("methods", {}).get("metadata_max_jaccard", {})
            if not node2vec:
                raise ValueError("No valid validation evaluation examples")
            row = {
                "experiment": f"n2v_{config_id:02d}_seed{seed}",
                "algorithm": "Node2Vec", "seed": seed, **params,
                "train_seconds": round(seconds, 4),
                "validation_p@10": node2vec["p@10"],
                "validation_ndcg@10": node2vec["ndcg@10"],
                "validation_mrr@20": node2vec["mrr_at_max_k"],
                "baseline_jaccard_validation_ndcg@10": jaccard["ndcg@10"],
                "validation_eligible_users": proxy["eligible_users"],
                "validation_coverage": proxy["coverage"],
            }
            experiments.append(row)
            # Validation metrics select; deterministic ID/seed break exact ties.
            # Wall-clock timing must never determine the selected model.
            key = (row["validation_ndcg@10"], row["validation_p@10"],
                   -config_id, -seed)
            if best_key is None or key > best_key:
                best_key, best_row, best_embeddings, best_validation = key, row, embeddings, proxy

    # An independent held-out test is evaluated once after selection.
    with threadpool_limits(limits=1, user_api="blas"):
        final_test = evaluate_book_retrieval_proxy(
            graph, best_embeddings, train, test, positive_threshold=positive_threshold)
    pd.DataFrame(experiments).to_csv(output_dir / "embedding_experiments.csv", index=False)
    save_embeddings(best_embeddings, output_dir / "selected_embeddings")
    report = {
        "protocol": "train/validation/test; select Node2Vec by validation NDCG@10; test once",
        "caveats": [
            "Book-book relevance is a held-out user preference proxy, not expert annotation.",
            "Selected embeddings are fitted on train only, not refitted on train+validation.",
            "Jaccard is not a graph-embedding algorithm. TransE comparison remains pending.",
        ],
        "mode": mode, "seed_split": seed_split, "training_seeds": list(seeds),
        "source_sha256": {name: sha256(raw_dir / name)
                          for name in ("BX-Books.csv", "BX-Book-Ratings.csv")},
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(),
        "experiments": len(experiments), "selected": best_row,
        "selected_validation": best_validation, "selected_test": final_test,
        "results_csv": "embedding_experiments.csv",
    }
    (output_dir / "study_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validated Node2Vec study for VKR")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/study"))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--max-users", type=int, default=150)
    parser.add_argument("--max-books", type=int, default=1000)
    parser.add_argument("--seed-split", type=int, default=2027)
    parser.add_argument("--genre-csv", type=Path, default=None)
    args = parser.parse_args()
    result = run_study(args.raw_dir, args.output_dir, mode=args.mode,
                       max_users=args.max_users, max_books=args.max_books,
                       seed_split=args.seed_split, genre_csv=args.genre_csv)
    print(json.dumps({"experiments": result["experiments"], "selected": result["selected"],
                      "test": result["selected_test"].get("methods", {})},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
