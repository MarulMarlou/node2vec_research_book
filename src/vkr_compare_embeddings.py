"""Head-to-head Node2Vec/TransE on one train/validation/test protocol.

Train-only graph; validation selects one configuration per method; test evaluated
only for the selected models. Explicit 1..6 ratings are separately held out for
ROC-AUC; unrated items are NOT mislabeled as known negatives.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.vkr_embedding_study import configs as n2v_configs, sha256, three_way_split
from src.vkr_prepare import load_book_crossing, restrict_from_training, load_genres, build_knowledge_graph, node2vec_projection
from src.vkr_recommend import train_node2vec, cosine_ranking, save_embeddings
from src.vkr_transe import train_transe, transe_rank_user_books, TransEResult
from src.vkr_metrics import ranking_metrics


def comparison_grid(mode: str) -> dict[str, list[dict]]:
    if mode == "quick":
        return {
            "Node2Vec": n2v_configs("quick")[:2],
            "TransE": [
                {"dimensions": 32, "epochs": 4, "batch_size": 128,
                 "learning_rate": lr, "margin": 1.0}
                for lr in (0.025, 0.05)],
        }
    if mode == "full":
        return {
            "Node2Vec": n2v_configs("full"),  # 12 × 2 seeds = 24 runs
            "TransE": [
                {"dimensions": dim, "epochs": 8, "batch_size": 256,
                 "learning_rate": lr, "margin": 1.0}
                for dim in (32, 64, 128) for lr in (0.025, 0.05, 0.1)
            ],  # 9 × 2 seeds = 18 runs; total 42
        }
    raise ValueError("mode must be quick or full")


def _signed_split(ratings: pd.DataFrame, *, max_users: int, max_books: int,
                  seed: int, threshold: int = 7
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extend the positive-stratified split with some explicit low ratings.

    Original split holds out positives only, leaving all ratings 1..6 in train.
    This method moves some of those low ratings into val and test, *before*
    selecting a train-only catalog, so ROC-AUC can use genuine observed low
    ratings rather than invented negative/unobserved labels.
    """
    train, validation, test = three_way_split(
        ratings, max_users=max_users, max_books=max_books,
        seed=seed, positive_threshold=threshold)
    # Important: selection above is preliminary. Re-select from the reduced
    # train after removing explicit lows to avoid selection leakage.
    low = train[train["Book-Rating"] < threshold]
    val_ids: list[int] = []
    test_ids: list[int] = []
    for user, frame in low.groupby("User-ID", sort=True):
        if len(frame) < 3:
            continue
        ordered = sorted(frame.index, key=lambda i: (
            hashlib.sha256(f"low|{seed}|{user}|{train.at[i, 'ISBN']}".encode()).hexdigest(),
            str(train.at[i, "ISBN"])))
        n_test = min(len(ordered) - 2, max(1, int(len(ordered) * 0.2)))
        n_val = min(len(ordered) - n_test - 1,
                    max(1, int((len(ordered) - n_test) * 0.2)))
        test_ids.extend(ordered[:n_test])
        val_ids.extend(ordered[n_test:n_test + n_val])
    if test_ids:
        test = pd.concat([test, train.loc[test_ids]], ignore_index=True)
    if val_ids:
        validation = pd.concat([validation, train.loc[val_ids]], ignore_index=True)
    train = train.drop(index=val_ids + test_ids).reset_index(drop=True)
    # All three folds are finally restricted to the catalog/entities selected
    # from training only (some may be removed after moving low ratings).
    train, validation = restrict_from_training(train, validation, max_users, max_books)
    users, isbns = set(train["User-ID"]), set(train["ISBN"])
    test = test[test["User-ID"].isin(users) &
                test["ISBN"].isin(isbns)].reset_index(drop=True)
    p = [set(zip(s["User-ID"], s["ISBN"])) for s in (train, validation, test)]
    if p[0] & p[1] or p[0] & p[2] or p[1] & p[2]:
        raise ValueError("Train/validation/test contain overlapping user-book pairs")
    return train, validation, test


def _auc_explicit(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney ROC-AUC; ties score half a win."""
    if not pos or not neg:
        return None
    a = np.asarray(pos, dtype=float)[:, None]
    b = np.asarray(neg, dtype=float)[None, :]
    return float(np.mean((a > b) + 0.5 * (a == b)))


def evaluate_user_ranking(
    embeddings: pd.DataFrame | TransEResult, train: pd.DataFrame,
    holdout: pd.DataFrame, *, method: str, threshold: int = 7,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """Same candidate sets/labels for both algorithms; relation-specific scores for TransE."""
    if method not in {"Node2Vec", "TransE"}:
        raise ValueError("unknown method")
    matrix = embeddings if method == "Node2Vec" else embeddings.entities
    books = sorted(str(i) for i in matrix.index if str(i).startswith("book:"))
    book_set = set(books)
    seen: dict[str, set[str]] = defaultdict(set)
    for user, isbn in train[["User-ID", "ISBN"]].itertuples(index=False, name=None):
        seen[f"user:{user}"].add(f"book:{isbn}")
    metrics = []
    auc = []
    positive_users = 0
    explicit_pair_users = 0
    eligible_items = 0
    for user, frame in holdout.groupby("User-ID", sort=True):
        user_id = f"user:{user}"
        positives = {f"book:{b}" for b in frame.loc[frame["Book-Rating"] >= threshold, "ISBN"]}
        negatives = {f"book:{b}" for b in frame.loc[frame["Book-Rating"] < threshold, "ISBN"]}
        if not positives:
            continue
        positive_users += 1
        candidates = [book for book in books if book not in seen[user_id]]
        if method == "Node2Vec":
            ranking = cosine_ranking(embeddings, user_id, candidates)
        else:
            ranking = transe_rank_user_books(embeddings, user_id, candidates)
        rankings = [book for book, _ in ranking]
        score_by_book = dict(ranking)
        metric_row = ranking_metrics(rankings, positives, ks=ks)
        # Report MRR separately for k=5,10,20, not only at max(k).
        for k in ks:
            first = next((i + 1 for i, book in enumerate(rankings[:k])
                          if book in positives), None)
            metric_row[f"mrr@{k}"] = 1.0 / first if first is not None else 0.0
        metrics.append(metric_row)
        pos_values = [score_by_book[b] for b in sorted(positives & book_set) if b in score_by_book]
        neg_values = [score_by_book[b] for b in sorted(negatives & book_set) if b in score_by_book]
        user_auc = _auc_explicit(pos_values, neg_values)
        if user_auc is not None:
            explicit_pair_users += 1
            eligible_items += len(pos_values) + len(neg_values)
            auc.append(user_auc)
    if not metrics:
        raise ValueError("No holdout users with positive ratings")
    summary = {("rr_at_max_k" if key == "rr" else key):
               float(np.mean([row[key] for row in metrics]))
               for key in metrics[0]}
    summary.update({
        "positive_holdout_users": positive_users,
        "auc_roc_explicit": float(np.mean(auc)) if auc else None,
        "auc_eligible_users": explicit_pair_users,
        "auc_explicit_rated_items": eligible_items,
        "auc_coverage": explicit_pair_users / positive_users,
        "auc_protocol": "observed held-out ratings 7..10 positive vs 1..6 negative; macro per-user",
        "candidate_protocol": "all train-supported books without any train rating by user",
    })
    return summary


def run_comparison(
    raw_dir: Path, output_dir: Path, *, mode: str = "quick",
    max_users: int = 150, max_books: int = 1000, seed_split: int = 2028,
    seeds: tuple[int, ...] | None = None,
    grid: dict[str, list[dict]] | None = None,
    genre_csv: Path | None = None,
) -> dict:
    if max_users <= 0 or max_books <= 0:
        raise ValueError("max_users/max_books must be positive")
    if seeds is None:
        seeds = (42,) if mode == "quick" else (42, 43)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("at least one unique training seed is required")
    if grid is None:
        grid = comparison_grid(mode)
    if set(grid) != {"Node2Vec", "TransE"} or not all(grid.values()):
        raise ValueError("comparison requires both Node2Vec and TransE configurations")
    books, ratings = load_book_crossing(raw_dir)
    train, val, test = _signed_split(
        ratings, max_users=max_users, max_books=max_books, seed=seed_split)
    if train.empty or val.empty or test.empty:
        raise ValueError("Train, validation and test must all be nonempty")
    graph = build_knowledge_graph(books, train, load_genres(genre_csv))
    projection = node2vec_projection(graph)

    output_dir.mkdir(parents=True, exist_ok=True)
    for label, frame in (("train", train), ("validation", val), ("test", test)):
        frame.to_csv(output_dir / f"{label}.csv", index=False)
    nx.write_graphml(graph, output_dir / "train_knowledge_graph.graphml")
    rows = []
    selected = {}
    models = {}
    for algorithm in ("Node2Vec", "TransE"):
        for config_id, params in enumerate(grid[algorithm], 1):
            for seed in seeds:
                start = time.perf_counter()
                if algorithm == "Node2Vec":
                    model = train_node2vec(projection, seed=seed, **params)
                else:
                    model = train_transe(graph, seed=seed, **params)
                duration = time.perf_counter() - start
                with threadpool_limits(limits=1, user_api="blas"):
                    met = evaluate_user_ranking(model, train, val, method=algorithm)
                row = {
                    "algorithm": algorithm,
                    "experiment": f"{algorithm.lower()}_{config_id:02d}_seed{seed}",
                    "config_id": config_id, "seed": seed, **params,
                    "train_seconds": round(duration, 5),
                    "validation_ndcg@10": met["ndcg@10"],
                    "validation_p@10": met["p@10"],
                    "validation_f1@10": met["f1@10"],
                    "validation_mrr@20": met["mrr@20"],
                    "validation_auc_roc_explicit": met["auc_roc_explicit"],
                    "validation_auc_coverage": met["auc_coverage"],
                }
                rows.append(row)
                key = (row["validation_ndcg@10"], row["validation_p@10"],
                       -config_id, -seed)
                if algorithm not in selected or key > selected[algorithm]["key"]:
                    selected[algorithm] = {"key": key, "row": row, "validation": met}
                    models[algorithm] = model

    # One final comparison on held-out test, after all hyperparameters selected.
    test_metrics = {}
    with threadpool_limits(limits=1, user_api="blas"):
        for algorithm in ("Node2Vec", "TransE"):
            test_metrics[algorithm] = evaluate_user_ranking(
                models[algorithm], train, test, method=algorithm)
    csv_file = output_dir / "comparison_experiments.csv"
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    # Save only models selected via validation; never log raw held-out IDs in report.
    save_embeddings(models["Node2Vec"], output_dir / "selected_node2vec")
    transe = models["TransE"]
    transe.entities.to_csv(output_dir / "selected_transe_entities.csv")
    transe.relations.to_csv(output_dir / "selected_transe_relations.csv")

    report = {
        "protocol": "train/validation/test with explicit low-rating holdout; select by validation NDCG@10; evaluate test once per model",
        "source_sha256": {name: sha256(raw_dir / name) for name in
                          ("BX-Books.csv", "BX-Book-Ratings.csv")},
        "mode": mode, "seed_split": seed_split, "training_seeds": list(seeds),
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "graph": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
                  "relations": sorted({str(edge["relation"]) for _, _, edge in graph.edges(data=True)})},
        "experiments": len(rows),
        "experiments_by_method": {name: len(grid[name]) * len(seeds)
                                 for name in ("Node2Vec", "TransE")},
        "selected": {name: {"parameters": selected[name]["row"],
                            "validation": selected[name]["validation"],
                            "test": test_metrics[name]}
                     for name in ("Node2Vec", "TransE")},
        "limitations": [
            "Random per-user holdout is not a chronological evaluation.",
            "Unknown ratings remain unknown; top-k relevance is incomplete.",
            "AUC uses only held-out observed explicit ratings >=7 vs 1..6, not unrated items.",
            "AUC is undefined for users without both observed positive and negative holdouts; see auc_coverage.",
            "Node2Vec and TransE use different scoring geometries but identical candidate catalogs and user folds.",
            "Training follows the original graph rated edges, including explicit lower ratings; compare a positive-only ablation later.",
            "Hyperparameters and algorithms are selected on validation; no test optimization.",
        ],
    }
    (output_dir / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Node2Vec vs TransE controlled study")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/comparison"))
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--max-users", type=int, default=150)
    parser.add_argument("--max-books", type=int, default=1000)
    parser.add_argument("--seed-split", type=int, default=2028)
    parser.add_argument("--genre-csv", type=Path, default=None)
    args = parser.parse_args()
    report = run_comparison(args.raw_dir, args.output_dir, mode=args.mode,
                            max_users=args.max_users, max_books=args.max_books,
                            seed_split=args.seed_split, genre_csv=args.genre_csv)
    print(json.dumps({"experiments": report["experiments"],
                      "test": {name: entry["test"] for name, entry in
                               report["selected"].items()},
                      "report": str(args.output_dir / "comparison_report.json")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
