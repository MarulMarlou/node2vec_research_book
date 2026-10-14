"""Executable train-only VКR prototype with separate evaluation and demonstration.

This reports user-book evaluation separately from book-book UI recommendations:
book-book relevance currently has no independent labeled ground truth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import networkx as nx
import pandas as pd
from src.vkr_prepare import prepare
from src.vkr_recommend import (train_node2vec, save_embeddings,
                               evaluate_user_holdout, recommend_similar_books,
                               metadata_book_ranking)
from src.vkr_explain import explain_and_rerank


def run_pipeline(raw_dir: Path, processed_dir: Path, results_dir: Path, *,
                 genre_csv: Path | None = None, max_users: int | None = 1000,
                 max_books: int | None = 5000, seed: int = 42,
                 dimensions: int = 64, walk_length: int = 20,
                 num_walks: int = 4, epochs: int = 3,
                 example_book: str | None = None) -> dict:
    preparation = prepare(raw_dir, processed_dir, genre_csv=genre_csv,
                          max_users=max_users, max_books=max_books, seed=seed)
    # GraphML auto-downgrades MultiDiGraph to DiGraph when no parallel edges exist.
    graph = nx.read_graphml(processed_dir / "vkr_knowledge_graph.graphml",
                            force_multigraph=True)
    projection = nx.read_graphml(processed_dir / "vkr_node2vec_graph.graphml")
    train = pd.read_csv(processed_dir / "vkr_train_ratings.csv",
                        dtype={"User-ID": str, "ISBN": str})
    test = pd.read_csv(processed_dir / "vkr_test_ratings.csv",
                       dtype={"User-ID": str, "ISBN": str})
    started = time.perf_counter()
    embeddings = train_node2vec(projection, dimensions=dimensions,
                                walk_length=walk_length, num_walks=num_walks,
                                epochs=epochs, seed=seed)
    train_seconds = time.perf_counter() - started
    save_embeddings(embeddings, results_dir / "embeddings")
    user_metrics = evaluate_user_holdout(embeddings, train, test)

    books = sorted(n for n, attrs in graph.nodes(data=True)
                   if attrs.get("entity_type") == "book")
    chosen = example_book or (books[0] if books else None)
    if chosen is not None and chosen not in books:
        raise ValueError(f"Example book {chosen} is not in training catalog")
    demonstration = {}
    if chosen is not None:
        candidates = recommend_similar_books(embeddings, chosen, top_k=20)
        demonstration = {
            "source_book": chosen,
            "node2vec_top10": [{"book_id": b, "cosine": s} for b, s in candidates[:10]],
            "metadata_control_top10": [
                {"book_id": b, "jaccard": s}
                for b, s in metadata_book_ranking(graph, chosen, top_k=10)
            ],
            "shortest_posthoc": explain_and_rerank(
                graph, candidates, chosen, method="shortest", mode="posthoc")[:10],
            "informative_posthoc": explain_and_rerank(
                graph, candidates, chosen, method="informative", mode="posthoc")[:10],
            "informative_rerank_alpha_0_2": explain_and_rerank(
                graph, candidates, chosen, method="informative", mode="rerank",
                alpha=0.2)[:10],
        }
    report = {
        "status": "unverified_on_real_Book_Crossing",
        "preparation": preparation,
        "train_seconds": train_seconds,
        "parameters": {
            "model": "Node2Vec", "seed": seed, "dimensions": dimensions,
            "walk_length": walk_length, "num_walks": num_walks, "epochs": epochs,
        },
        "user_to_book_holdout": user_metrics,
        "book_to_book_demonstration_no_ground_truth": demonstration,
    }
    destination = results_dir / "metrics" / "vkr_pipeline_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test ВКР research prototype")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--genre-csv", type=Path, default=None)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--max-books", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--num-walks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--example-book", type=str, default=None)
    args = parser.parse_args()
    report = run_pipeline(args.raw_dir, args.processed_dir, args.results_dir,
                          genre_csv=args.genre_csv, max_users=args.max_users,
                          max_books=args.max_books, seed=args.seed,
                          dimensions=args.dimensions, walk_length=args.walk_length,
                          num_walks=args.num_walks, epochs=args.epochs,
                          example_book=args.example_book)
    print(json.dumps({
        "report": str(args.results_dir / "metrics" / "vkr_pipeline_report.json"),
        "test_metrics": report["user_to_book_holdout"],
        "train_seconds": report["train_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
