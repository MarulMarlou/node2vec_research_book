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
    max_paths: int = 50, max_expansions: int = 400,
    configs: list[dict] | None = None,
    collect_per_user: bool = False,
) -> dict:
    if not 1 <= max_evaluated_users <= 500 or candidate_pool < max(KS):
        raise ValueError("max_evaluated_users must be 1..500, candidate_pool >= 20")
    if max_depth not in (2, 4):
        raise ValueError("max_depth must be 2 or 4")
    if not 1 <= max_paths <= 500 or not 1 <= max_expansions <= 20000:
        raise ValueError("Search budgets must be max_paths=1..500 and max_expansions=1..20000")
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
    candidate_ceiling_rows: list[dict] = []
    # All candidate-recall diagnostics are descriptive: holdout labels are
    # NEVER used to choose the pool, model, alpha or path explanation.
    model_catalog = {str(node) for node in embeddings.index
                     if str(node).startswith("book:")}
    for user in users:
        anchors = sorted(liked.get(user, set()))[:3]
        # One deterministic cosine candidate order; evaluating a top-100
        # ceiling does not imply explaining or reranking those 100 books.
        base_order = _prepared_candidates(
            embeddings, seen.get(user, set()), anchors,
            candidate_pool=max(100, candidate_pool))
        pool = base_order[:candidate_pool]
        positive = truth.get(user, set())
        source_pool = {r["book_id"] for r in pool}
        top25 = {r["book_id"] for r in base_order[:25]}
        top100 = {r["book_id"] for r in base_order[:100]}
        candidates_available = positive & model_catalog
        denominator = len(positive)
        candidate_ceiling_rows.append({
            "model_catalog_has_positive": bool(candidates_available),
            "base_top25_has_positive": bool(positive & top25),
            "base_top100_has_positive": bool(positive & top100),
            "chosen_pool_has_positive": bool(positive & source_pool),
            "model_catalog_positive_recall": (
                len(candidates_available) / denominator if denominator else 0.0),
            "base_top25_positive_recall": (
                len(positive & top25) / denominator if denominator else 0.0),
            "base_top100_positive_recall": (
                len(positive & top100) / denominator if denominator else 0.0),
            "chosen_pool_positive_recall": (
                len(positive & source_pool) / denominator if denominator else 0.0),
        })
        for method in methods:
            rows = _collect_explanations(graph, pool, method=method,
                                         max_depth=max_depth, max_paths=max_paths,
                                         max_expansions=max_expansions)
            for config in active_configs:
                if config["method"] != method:
                    continue
                ranked = _rank(rows, mode=config["mode"], alpha=config["alpha"])
                ranking_ids = [r["book_id"] for r in ranked]
                relevance = truth.get(user, set())
                metric = ranking_metrics(ranking_ids, relevance, ks=KS)
                for k in KS:
                    first = next((i for i, book in enumerate(ranking_ids[:k], 1)
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
    diagnostics = {
        "users": len(candidate_ceiling_rows),
        "catalog_has_positive_users": sum(
            r["model_catalog_has_positive"] for r in candidate_ceiling_rows),
        "top25_has_positive_users": sum(
            r["base_top25_has_positive"] for r in candidate_ceiling_rows),
        "top100_has_positive_users": sum(
            r["base_top100_has_positive"] for r in candidate_ceiling_rows),
        "chosen_pool_has_positive_users": sum(
            r["chosen_pool_has_positive"] for r in candidate_ceiling_rows),
    }
    for field in ("model_catalog_positive_recall", "base_top25_positive_recall",
                  "base_top100_positive_recall", "chosen_pool_positive_recall"):
        diagnostics["mean_" + field] = float(np.mean(
            [r[field] for r in candidate_ceiling_rows]))
    summary = {"cohort_users": len(users), "eligible_users_total": int(total_users),
               "evaluated_users_fraction": len(users) / total_users,
               "candidate_pool": candidate_pool, "max_paths": max_paths,
               "max_expansions": max_expansions, "results": results,
               "candidate_ceiling": diagnostics}
    if collect_per_user:
        # Users retain the same stable order for each method; IDs are not
        # emitted. This enables strictly paired uncertainty calculations.
        summary["per_user_results"] = by_config
    return summary


def paired_bootstrap(
    reranker_rows: list[dict], posthoc_rows: list[dict], *,
    seed: int, replicates: int = 3000,
) -> tuple[pd.DataFrame, dict]:
    """Paired, cohort-conditional uncertainty for validation-selected rerank.

    Both lists are generated for the same ordered users, candidates and test
    labels, without exposing user IDs or reselecting alpha by test metrics.
    This confidence interval does not account for a new train/test split,
    graph/model re-fit, or variation of the selected model.
    """
    if len(reranker_rows) != len(posthoc_rows) or not reranker_rows:
        raise ValueError("Both test models must have the same nonempty cohort")
    if not 100 <= replicates <= 100000:
        raise ValueError("replicates must be between 100 and 100000")
    keys = ("ndcg@10", "p@10", "mrr@10", "explanation_coverage")
    records = []
    for i, (reranker, baseline) in enumerate(zip(reranker_rows, posthoc_rows)):
        row = {"anonymous_user_index": i}
        for metric in keys:
            selected = float(reranker[metric])
            control = float(baseline[metric])
            if not np.isfinite([selected, control]).all():
                raise ValueError("Non-finite paired user metric")
            row[f"{metric}_rerank"] = selected
            row[f"{metric}_posthoc"] = control
            row[f"{metric}_delta"] = selected - control
        records.append(row)
    results = pd.DataFrame(records)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(records), size=(replicates, len(records)))
    summary = {"users": len(records), "bootstrap_replicates": replicates,
               "bootstrap_seed": seed, "ci_level": 0.95,
               "interpretation": "conditional on fixed split, model and evaluated user cohort"}
    for metric in keys:
        effects = results[f"{metric}_delta"].to_numpy(dtype=float)
        means = effects[indices].mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        summary[metric] = {"mean_delta": float(effects.mean()),
                           "bootstrap_ci95": [float(lo), float(hi)],
                           "users_improved": int(np.count_nonzero(effects > 0)),
                           "users_declined": int(np.count_nonzero(effects < 0)),
                           "users_tied": int(np.count_nonzero(effects == 0))}
    return results, summary


def run_explanation_study(raw_dir: Path, output_dir: Path, *,
                          max_users: int = 150, max_books: int = 1000,
                          max_evaluated_users: int = 20,
                          candidate_pool: int = 25, max_depth: int = 2,
                          max_paths: int = 50, max_expansions: int = 400,
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
            candidate_pool=candidate_pool, seed=split_seed + 101, max_depth=max_depth,
            max_paths=max_paths, max_expansions=max_expansions)
        scored = val_report["results"]
        best = max(scored, key=lambda row: (
            row["ndcg@10"], row["p@10"], -int(row["config_id"])))
        # A priori comparator: pick the strongest reranker on validation,
        # even when posthoc wins overall. Compare its held-out effect exactly
        # once against an alpha-free posthoc control, never optimize on test.
        best_rerank = max(
            (row for row in scored if row["mode"] == "rerank"),
            key=lambda row: (row["ndcg@10"], row["p@10"], -int(row["config_id"])))
        test_configs = [
            {"method": best["method"], "mode": "posthoc", "alpha": 0.0},
            {"method": best_rerank["method"], "mode": "rerank",
             "alpha": best_rerank["alpha"]},
        ]
        if not any(c["method"] == best["method"] and c["mode"] == best["mode"]
                   and c["alpha"] == best["alpha"] for c in test_configs):
            test_configs.append({"method": best["method"], "mode": best["mode"],
                                 "alpha": best["alpha"]})
        test_report = _cohort_evaluation(
            graph, embeddings, train, test, max_evaluated_users=max_evaluated_users,
            candidate_pool=candidate_pool, seed=split_seed + 202,
            max_depth=max_depth, max_paths=max_paths,
            max_expansions=max_expansions, configs=test_configs,
            collect_per_user=True)
    def choose(rows, method, mode, alpha):
        return next(r for r in rows
                    if r["method"] == method and r["mode"] == mode
                    and r["alpha"] == alpha)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scored).to_csv(output_dir / "explanation_validation_grid.csv", index=False)
    # A priori paired effect on the same test users and candidate pools,
    # comparing validation-selected reranker with its matched posthoc control.
    compare_key = (best_rerank["method"], "rerank", best_rerank["alpha"])
    base_key = (best_rerank["method"], "posthoc", 0.0)
    user_effects, paired_uncertainty = paired_bootstrap(
        test_report["per_user_results"][compare_key],
        test_report["per_user_results"][base_key],
        seed=split_seed + 303,
    )
    user_effects.to_csv(output_dir / "explanation_test_paired_deltas.csv", index=False)
    save_embeddings(embeddings, output_dir / "train_only_node2vec")
    result = {
        "protocol": ("Train-only Node2Vec; same book candidate pools, method/alpha "
                     "selected on validation; fixed posthoc, selected config, and "
                     "validation-selected reranker evaluated once on independent test"),
        "split_seed": split_seed, "model_params": params,
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(),
        "genre_csv": str(genre_csv) if genre_csv else None,
        "max_depth": max_depth,
        "max_paths": max_paths,
        "max_expansions": max_expansions,
        "candidate_pool": candidate_pool,
        "validation": val_report,
        "candidate_ceiling_validation": val_report["candidate_ceiling"],
        "selected": best,
        "best_rerank_validation": best_rerank,
        "test_selected": choose(test_report["results"], best["method"], best["mode"], best["alpha"]),
        "test_posthoc_control": choose(test_report["results"], best["method"], "posthoc", 0.0),
        "test_best_rerank": choose(test_report["results"], best_rerank["method"], "rerank",
                                   best_rerank["alpha"]),
        "test_cohort_size": test_report["cohort_users"],
        "test_eligible_user_total": test_report["eligible_users_total"],
        "candidate_ceiling_test": test_report["candidate_ceiling"],
        "paired_rerank_vs_posthoc": paired_uncertainty,
        "paired_results_csv": "explanation_test_paired_deltas.csv",
        "source_sha256": {
            name: _file_sha256(raw_dir / name)
            for name in ("BX-Books.csv", "BX-Book-Ratings.csv")
        },
        "limitations": [
            "Book-to-book relevance is held-out user-preference proxy, not expert semantic similarity.",
            "Method and alpha were selected on validation only, not on the test.",
            "The rerank-vs-posthoc test comparison is preselected on validation; no post-test alpha optimization.",
            "Only a bounded sample of test users and top candidate pool was evaluated.",
            "The same fixed training graph is used for all explanation comparisons.",
            "Graph paths verify facts, not their causal influence on Node2Vec score.",
            "Informative search may be truncated; report truncation rate and search budgets.",
            "Explanation time includes Python search and depends on the machine.",
            "Random per-user splits are not chronological.",
            "Candidate ceilings are descriptive diagnostics from held-out labels, not tuning criteria; model catalog and top25/top100 recall are bounded by the fixed sampled graph.",
            "Paired bootstrap CI is conditional on this one fixed train/test split, trained model, and selected user cohort; it does not quantify model/split uncertainty.",
            "User-level deltas are anonymized by positional index, with no Book-Crossing user identifiers saved.",
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
    parser.add_argument("--max-paths", type=int, default=50)
    parser.add_argument("--max-expansions", type=int, default=400)
    parser.add_argument("--genre-csv", type=Path, default=None)
    args = parser.parse_args()
    report = run_explanation_study(
        args.raw_dir, args.output_dir, max_users=args.max_users,
        max_books=args.max_books, max_evaluated_users=args.max_evaluated_users,
        candidate_pool=args.candidate_pool, max_depth=args.max_depth,
        max_paths=args.max_paths, max_expansions=args.max_expansions,
        split_seed=args.split_seed, genre_csv=args.genre_csv)
    print(json.dumps({"validation_selected": report["selected"],
                      "test_selected": report["test_selected"],
                      "test_posthoc_control": report["test_posthoc_control"],
                      "test_best_rerank": report["test_best_rerank"],
                      "report": str(args.output_dir / "explanation_study_report.json")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
