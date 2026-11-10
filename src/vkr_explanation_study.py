"""Validation-only selection of explanatory paths / reranking for the VKR.

This is a separate book-to-book *preference proxy*, not expert annotations
of semantic similarity. The evaluator scores the SAME candidate pool for
shortest/informative and posthoc/rerank. It selects explanation method and
alpha on validation, evaluates the selected configuration ONCE on test.
No held-out interactions become graph edges or model inputs.
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

from src.vkr_embedding_study import three_way_split
from src.vkr_explain import find_explanations
from src.vkr_metrics import ranking_metrics
from src.vkr_prepare import (build_knowledge_graph, load_book_crossing,
                             load_genres, node2vec_projection)
from src.vkr_recommend import train_node2vec, save_embeddings


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


MODEL_PARAMS = {
    "dimensions": 64, "walk_length": 20, "num_walks": 4,
    "epochs": 3, "p": 1.0, "q": 1.0, "window": 3, "seed": 43,
}
METHODS = ("shortest", "informative")
ALPHAS = (0.2, 0.5, 0.8)
KS = (5, 10, 20)


def explanation_configs() -> list[dict]:
    return ([{"method": method, "mode": "posthoc", "alpha": 0.0}
             for method in METHODS] +
            [{"method": method, "mode": "rerank", "alpha": alpha}
             for method in METHODS for alpha in ALPHAS])


def _eligible_users(holdout: pd.DataFrame, *, threshold: int,
                    max_users: int, seed: int) -> tuple[list[str], int]:
    positive = holdout[holdout["Book-Rating"] >= threshold]
    users = sorted(str(u) for u in positive["User-ID"].unique())
    if len(users) > max_users:
        rng = np.random.default_rng(seed)
        picked = rng.choice(len(users), size=max_users, replace=False)
        users = sorted(users[int(i)] for i in picked)
    return users, positive["User-ID"].nunique()


def _train_books(train: pd.DataFrame, threshold: int
                 ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    seen = defaultdict(set)
    liked = defaultdict(set)
    for user, isbn, rating in train[["User-ID", "ISBN", "Book-Rating"]].itertuples(
            index=False, name=None):
        uid = str(user)
        book = f"book:{isbn}"
        seen[uid].add(book)
        if int(rating) >= threshold:
            liked[uid].add(book)
    return dict(seen), dict(liked)


def _prepared_candidates(embeddings: pd.DataFrame, seen: set[str],
                         anchors: list[str], *, candidate_pool: int
                         ) -> list[dict]:
    """Rank once by max cosine to a positive train book, before any explanation."""
    catalog = sorted(n for n in embeddings.index if str(n).startswith("book:"))
    candidates = [n for n in catalog if n not in seen]
    anchors = sorted(a for a in anchors if a in embeddings.index)
    if not anchors or not candidates:
        return []
    norm = embeddings.loc[catalog].to_numpy(dtype=float)
    length = np.linalg.norm(norm, axis=1, keepdims=True)
    norm = np.divide(norm, length, out=np.zeros_like(norm), where=length > 0)
    by_id = {book: i for i, book in enumerate(catalog)}
    base = norm[[by_id[b] for b in candidates]] @ norm[[by_id[a] for a in anchors]].T
    max_i = np.argmax(base, axis=1)
    prepared = [
        {"book_id": book,
         "anchor": anchors[int(max_i[i])],
         "base_score": float(base[i, int(max_i[i])])}
        for i, book in enumerate(candidates)
    ]
    return sorted(prepared, key=lambda row: (-row["base_score"], row["book_id"]))[:candidate_pool]


def _collect_explanations(graph: nx.MultiDiGraph, pool: list[dict], *,
                          method: str, max_depth: int,
                          max_paths: int = 50,
                          max_expansions: int = 400) -> list[dict]:
    rows = []
    for item in pool:
        started = time.perf_counter()
        res = find_explanations(
            graph, item["anchor"], item["book_id"], method=method,
            max_depth=max_depth, top_n=1, max_paths=max_paths,
            max_expansions=max_expansions)
        seconds = time.perf_counter() - started
        path = res["paths"][0] if res["paths"] else None
        rows.append({
            **item,
            "path_length": int(path["length"]) if path else None,
            "informativeness": float(path["informativeness"]) if path else 0.0,
            "has_explanation": path is not None,
            "truncated": bool(res["truncated"]),
            "explanation_time_s": seconds,
        })
    return rows


def _rank(rows: list[dict], *, mode: str, alpha: float) -> list[dict]:
    if mode == "posthoc":
        return list(rows)
    if mode != "rerank" or not 0 <= alpha <= 1:
        raise ValueError("mode/alpha invalid")
    return sorted(rows, key=lambda row: (
        -(1 - alpha) * ((row["base_score"] + 1) / 2) -
        alpha * (row["informativeness"] / (1 + row["informativeness"])),
        row["book_id"]))


def _cohort_evaluation(
    graph: nx.MultiDiGraph, embeddings: pd.DataFrame,
    train: pd.DataFrame, holdout: pd.DataFrame, *,
    max_evaluated_users: int, candidate_pool: int, seed: int,
    threshold: int = 7, max_depth: int = 2,
    configs: list[dict] | None = None,
) -> dict:
    if not 1 <= max_evaluated_users <= 500 or candidate_pool < max(KS):
        raise ValueError("max_evaluated_users must be 1..500, candidate_pool >= 20")
    if max_depth not in (2, 4):
        raise ValueError("max_depth must be 2 or 4")
    users, total_users = _eligible_users(
        holdout, threshold=threshold, max_users=max_evaluated_users, seed=seed)
    if not users:
        raise ValueError("No holdout positives to evaluate")
    seen, liked = _train_books(train, threshold)
    heldout_positive = holdout[holdout["Book-Rating"] >= threshold]
    truth = {str(uid): {f"book:{v}" for v in part["ISBN"]}
             for uid, part in heldout_positive.groupby("User-ID", sort=True)}
    active_configs = explanation_configs() if configs is None else configs
    if not active_configs:
        raise ValueError("At least one explanation configuration required")
    methods = tuple(sorted(set(c["method"] for c in active_configs)))
    by_config: dict[tuple[str, str, float], list[dict]] = {
        (c["method"], c["mode"], c["alpha"]): [] for c in active_configs
    }
    for user in users:
        anchors = sorted(liked.get(user, set()))[:3]
        pool = _prepared_candidates(
            embeddings, seen.get(user, set()), anchors,
            candidate_pool=candidate_pool)
        for method in methods:
            rows = _collect_explanations(graph, pool, method=method,
                                         max_depth=max_depth)
            for config in active_configs:
                if config["method"] != method:
                    continue
                ranked = _rank(rows, mode=config["mode"], alpha=config["alpha"])
                ranking_ids = [r["book_id"] for r in ranked]
                relevance = truth.get(user, set())
                metric = ranking_metrics(ranking_ids, relevance, ks=KS)
                for k in KS:
                    first = next((i + 1 for i, book in enumerate(ranking_ids[:k], 1)
                                  if book in relevance), None)
                    metric[f"mrr@{k}"] = (1.0 / first) if first else 0.0
                first_ten = ranked[:10]
                explanation_lengths = [
                    r["path_length"] for r in first_ten if r["has_explanation"]]
                metric.update({
                    "covered_user": bool(pool),
                    "explanation_coverage": (
                        sum(r["has_explanation"] for r in first_ten) / len(first_ten)
                        if first_ten else 0.0),
                    "mean_path_length": (
                        sum(explanation_lengths) / len(explanation_lengths)
                        if explanation_lengths else 0.0),
                    "explained_items": len(explanation_lengths),
                    "truncated_fraction": (
                        sum(r["truncated"] for r in first_ten) / len(first_ten)
                        if first_ten else 0.0),
                    "mean_explanation_time_s": (
                        sum(r["explanation_time_s"] for r in first_ten) / len(first_ten)
                        if first_ten else 0.0),
                })
                by_config[(method, config["mode"], config["alpha"])].append(metric)
    results = []
    for i, config in enumerate(active_configs):
        rows = by_config[(config["method"], config["mode"], config["alpha"])]
        if len(rows) != len(users):
            raise ValueError("Inconsistent eligible users between evaluation methods")
        numerical_keys = [
            k for k in rows[0]
            if isinstance(rows[0][k], (float, int, bool))
        ]
        aggregate = {
            key if key != "rr" else "rr_at_max_k": float(np.mean([r[key] for r in rows]))
            for key in numerical_keys
        }
        results.append({**config, "config_id": i + 1, "users_evaluated": len(users),
                        **aggregate})
    return {"cohort_users": len(users), "eligible_users_total": int(total_users),
            "evaluated_users_fraction": len(users) / total_users,
            "candidate_pool": candidate_pool,
            "results": results}


def run_explanation_study(raw_dir: Path, output_dir: Path, *,
                          max_users: int = 150, max_books: int = 1000,
                          max_evaluated_users: int = 20,
                          candidate_pool: int = 25, max_depth: int = 2,
                          split_seed: int = 2042,
                          genre_csv: Path | None = None,
                          model_params: dict | None = None) -> dict:
    """Prepare a new independent split; optimize alpha on validation ONLY."""
    if max_users <= 0 or max_books <= 0:
        raise ValueError("max_users/max_books must be positive")
    params = dict(MODEL_PARAMS if model_params is None else model_params)
    books, ratings = load_book_crossing(raw_dir)
    train, val, test = three_way_split(
        ratings, max_users=max_users, max_books=max_books, seed=split_seed)
    if train.empty or val.empty or test.empty:
        raise ValueError("Need nonempty train/validation/test")
    graph = build_knowledge_graph(books, train, load_genres(genre_csv))
    with threadpool_limits(limits=1, user_api="blas"):
        embeddings = train_node2vec(node2vec_projection(graph), **params)
        val_report = _cohort_evaluation(
            graph, embeddings, train, val, max_evaluated_users=max_evaluated_users,
            candidate_pool=candidate_pool, seed=split_seed + 101, max_depth=max_depth)
        scored = val_report["results"]
        best = max(scored, key=lambda row: (
            row["ndcg@10"], row["p@10"], -int(row["config_id"])))
        # Test contains only the validation-selected configuration and a
        # predeclared alpha-free posthoc reference. No unused alpha is scored.
        test_configs = [{"method": best["method"], "mode": "posthoc", "alpha": 0.0}]
        if best["mode"] != "posthoc":
            test_configs.append({"method": best["method"], "mode": best["mode"],
                                 "alpha": best["alpha"]})
        test_report = _cohort_evaluation(
            graph, embeddings, train, test, max_evaluated_users=max_evaluated_users,
            candidate_pool=candidate_pool, seed=split_seed + 202,
            max_depth=max_depth, configs=test_configs)
    def choose(rows, method, mode, alpha):
        return next(r for r in rows
                    if r["method"] == method and r["mode"] == mode
                    and r["alpha"] == alpha)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scored).to_csv(output_dir / "explanation_validation_grid.csv", index=False)
    save_embeddings(embeddings, output_dir / "train_only_node2vec")
    result = {
        "protocol": ("Train-only Node2Vec; same book candidate pools, method/alpha "
                     "selected on validation; fixed posthoc baseline and selected "
                     "config evaluated once on independent test"),
        "split_seed": split_seed, "model_params": params,
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(),
        "genre_csv": str(genre_csv) if genre_csv else None,
        "max_depth": max_depth,
        "candidate_pool": candidate_pool,
        "validation": val_report,
        "selected": best,
        "test_selected": choose(test_report["results"], best["method"], best["mode"], best["alpha"]),
        "test_posthoc_control": choose(test_report["results"], best["method"], "posthoc", 0.0),
        "test_cohort_size": test_report["cohort_users"],
        "test_eligible_user_total": test_report["eligible_users_total"],
        "source_sha256": {
            name: _file_sha256(raw_dir / name)
            for name in ("BX-Books.csv", "BX-Book-Ratings.csv")
        },
        "limitations": [
            "Book-to-book relevance is held-out user-preference proxy, not expert semantic similarity.",
            "Method and alpha were selected on validation only, not on the test.",
            "Only a bounded sample of test users and top candidate pool was evaluated.",
            "The same fixed training graph is used for all explanation comparisons.",
            "Graph paths verify facts, not their causal influence on Node2Vec score.",
            "Informative search may be truncated; report truncation rate and search budgets.",
            "Explanation time includes Python search and depends on the machine.",
            "Random per-user splits are not chronological.",
        ],
    }
    (output_dir / "explanation_study_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="VKR explanation methods and reranking study")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/explanation_study"))
    parser.add_argument("--max-users", type=int, default=150)
    parser.add_argument("--max-books", type=int, default=1000)
    parser.add_argument("--max-evaluated-users", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=25)
    parser.add_argument("--max-depth", type=int, choices=(2, 4), default=2)
    parser.add_argument("--split-seed", type=int, default=2042)
    parser.add_argument("--genre-csv", type=Path, default=None)
    args = parser.parse_args()
    report = run_explanation_study(
        args.raw_dir, args.output_dir, max_users=args.max_users,
        max_books=args.max_books, max_evaluated_users=args.max_evaluated_users,
        candidate_pool=args.candidate_pool, max_depth=args.max_depth,
        split_seed=args.split_seed, genre_csv=args.genre_csv)
    print(json.dumps({"validation_selected": report["selected"],
                      "test_selected": report["test_selected"],
                      "test_posthoc_control": report["test_posthoc_control"],
                      "report": str(args.output_dir / "explanation_study_report.json")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
