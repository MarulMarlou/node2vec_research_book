"""Fair candidate-pool 25 vs 100 comparison on one new train/validation/test.

Run ONCE on a fresh split. Both pool sizes share exactly the same trained
Node2Vec, input graph, user cohort, and original cosine score ordering.
Method/alpha are selected separately by validation only. Independent test is
evaluated for predefined comparisons without picking winners on test.

The "fixed_25_config" experiment isolates the effect of pool size while
holding method and alpha fixed. "selected_pipeline" compares separately
validation-tuned pipelines and is not a pure pool-size effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from src.vkr_embedding_study import three_way_split
from src.vkr_explanation_study import (
    MODEL_PARAMS, _cohort_evaluation, _file_sha256, paired_bootstrap,
)
from src.vkr_prepare import (
    build_knowledge_graph, load_book_crossing, load_genres,
    node2vec_projection,
)
from src.vkr_recommend import train_node2vec, save_embeddings


def _only_config(row: dict) -> dict:
    return {key: row[key] for key in ("method", "mode", "alpha")}


def _key(row: dict) -> tuple[str, str, float]:
    return row["method"], row["mode"], row["alpha"]


def _selected(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: (
        r["ndcg@10"], r["p@10"], -int(r["config_id"])))


def run_candidate_pool_study(raw_dir: Path, output_dir: Path, *,
                             split_seed: int = 2102,
                             max_users: int = 150,
                             max_books: int = 1000,
                             max_evaluated_users: int = 100,
                             max_depth: int = 4,
                             max_paths: int = 200,
                             max_expansions: int = 5000,
                             genre_csv: Path | None = None,
                             model_params: dict | None = None) -> dict:
    """Compare two pool sizes with one trained model, and paired test users."""
    params = dict(MODEL_PARAMS if model_params is None else model_params)
    books, ratings = load_book_crossing(raw_dir)
    train, val, test = three_way_split(
        ratings, max_users=max_users, max_books=max_books, seed=split_seed)
    if train.empty or val.empty or test.empty:
        raise ValueError("Train, validation and test must all be nonempty")
    graph = build_knowledge_graph(books, train, load_genres(genre_csv))
    output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1, user_api="blas"):
        vectors = train_node2vec(node2vec_projection(graph), **params)
        validation = {}
        selected = {}
        for pool_size in (25, 100):
            result = _cohort_evaluation(
                graph, vectors, train, val,
                max_evaluated_users=max_evaluated_users,
                candidate_pool=pool_size, seed=split_seed + 101,
                max_depth=max_depth, max_paths=max_paths,
                max_expansions=max_expansions)
            validation[pool_size] = result
            selected[pool_size] = _selected(result["results"])

        # Precommit to three test outputs BEFORE using any test labels:
        # (A) selected25@pool25, (B) same config@pool100,
        # (C) selected100@pool100. (A)-(B) isolates pool size.
        fixed = _only_config(selected[25])
        tuned = _only_config(selected[100])
        pool100_configs = [fixed]
        if _key(tuned) != _key(fixed):
            pool100_configs.append(tuned)
        common = dict(max_evaluated_users=max_evaluated_users,
                      seed=split_seed + 202, max_depth=max_depth,
                      max_paths=max_paths, max_expansions=max_expansions,
                      collect_per_user=True)
        test25 = _cohort_evaluation(
            graph, vectors, train, test, candidate_pool=25,
            configs=[fixed], **common)
        test100 = _cohort_evaluation(
            graph, vectors, train, test, candidate_pool=100,
            configs=pool100_configs, **common)

    if (test25["cohort_users"] != test100["cohort_users"] or
            test25["eligible_users_total"] != test100["eligible_users_total"]):
        raise AssertionError("Different test cohorts; paired comparison invalid")

    base = test25["per_user_results"][_key(fixed)]
    expanded_fixed = test100["per_user_results"][_key(fixed)]
    expanded_tuned = test100["per_user_results"][_key(tuned)]
    fixed_delta, fixed_ci = paired_bootstrap(
        expanded_fixed, base, seed=split_seed + 303)
    pipeline_delta, pipeline_ci = paired_bootstrap(
        expanded_tuned, base, seed=split_seed + 404)
    fixed_delta.to_csv(
        output_dir / "candidate_pool_test_fixed_configuration_deltas.csv",
        index=False)
    pipeline_delta.to_csv(
        output_dir / "candidate_pool_test_selected_pipeline_deltas.csv",
        index=False)
    validation_csv = pd.DataFrame([
        {"candidate_pool": size, **r}
        for size in (25, 100) for r in validation[size]["results"]
    ])
    validation_csv.to_csv(
        output_dir / "candidate_pool_validation_grid.csv", index=False)
    save_embeddings(vectors, output_dir / "train_only_node2vec")

    def evaluated(result: dict, config: dict) -> dict:
        return next(row for row in result["results"]
                    if _key(row) == _key(config))

    report = {
        "protocol": (
            "Same train-only Node2Vec, same validation/test users and cosine "
            "order for pool25 vs pool100; method/alpha selected on validation "
            "for each pool. Fixed25-configuration comparison isolates pool size; "
            "selected-pipeline comparison includes separately tuned methods. "
            "Both test comparisons predeclared before test evaluation."
        ),
        "split_seed": split_seed,
        "train_rows": len(train), "validation_rows": len(val),
        "test_rows": len(test),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "model_params": params,
        "max_users": max_users, "max_books": max_books,
        "max_depth": max_depth, "max_paths": max_paths,
        "max_expansions": max_expansions,
        "genre_csv": str(genre_csv) if genre_csv else None,
        "validation_cohort": validation[25]["cohort_users"],
        "test_cohort": test25["cohort_users"],
        "test_eligible_users_total": test25["eligible_users_total"],
        "validation_selected_pool25": selected[25],
        "validation_selected_pool100": selected[100],
        "validation_candidate_ceiling_pool25": validation[25]["candidate_ceiling"],
        "validation_candidate_ceiling_pool100": validation[100]["candidate_ceiling"],
        "test_candidate_ceiling_pool25": test25["candidate_ceiling"],
        "test_candidate_ceiling_pool100": test100["candidate_ceiling"],
        "test_pool25_selected": evaluated(test25, fixed),
        "test_pool100_fixed25_configuration": evaluated(test100, fixed),
        "test_pool100_selected": evaluated(test100, tuned),
        "paired_pool100_fixed_config_minus_pool25": fixed_ci,
        "paired_pool100_selected_minus_pool25_selected": pipeline_ci,
        "result_files": [
            "candidate_pool_validation_grid.csv",
            "candidate_pool_test_fixed_configuration_deltas.csv",
            "candidate_pool_test_selected_pipeline_deltas.csv",
        ],
        "source_sha256": {
            name: _file_sha256(raw_dir / name)
            for name in ("BX-Books.csv", "BX-Book-Ratings.csv")
        },
        "limitations": [
            "Book-to-book user preference proxy, not semantic gold-standard.",
            "Enlarging candidate pool changes computational expense; report total_pool_explanation_time_s, not just top10 latency.",
            "Candidate100 may improve recall ceiling without improving top10 NDCG.",
            "Selected-pipeline comparison can change method/alpha AND pool size; fixed-configuration comparison isolates pool size.",
            "Confidence intervals condition on a single split/model and preselected user cohort.",
            "All hyperparameters and configurations selected solely using validation.",
            "No source user IDs or held-out edges enter the learned graph or output CSV.",
        ],
    }
    (output_dir / "candidate_pool_study_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="VKR paired candidate pool 25 vs 100")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/candidate_pool_study"))
    p.add_argument("--split-seed", type=int, default=2102)
    p.add_argument("--max-users", type=int, default=150)
    p.add_argument("--max-books", type=int, default=1000)
    p.add_argument("--max-evaluated-users", type=int, default=100)
    p.add_argument("--max-depth", type=int, choices=(2, 4), default=4)
    p.add_argument("--max-paths", type=int, default=200)
    p.add_argument("--max-expansions", type=int, default=5000)
    p.add_argument("--genre-csv", type=Path, default=None)
    args = p.parse_args()
    report = run_candidate_pool_study(
        args.raw_dir, args.output_dir, split_seed=args.split_seed,
        max_users=args.max_users, max_books=args.max_books,
        max_evaluated_users=args.max_evaluated_users,
        max_depth=args.max_depth, max_paths=args.max_paths,
        max_expansions=args.max_expansions, genre_csv=args.genre_csv)
    print(json.dumps({
        "validation_selected_pool25": _only_config(report["validation_selected_pool25"]),
        "validation_selected_pool100": _only_config(report["validation_selected_pool100"]),
        "test_pool25_ndcg10": report["test_pool25_selected"]["ndcg@10"],
        "test_pool100_fixed_config_ndcg10":
            report["test_pool100_fixed25_configuration"]["ndcg@10"],
        "test_pool100_selected_ndcg10": report["test_pool100_selected"]["ndcg@10"],
        "report": str(args.output_dir / "candidate_pool_study_report.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
