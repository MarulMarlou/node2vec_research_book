"""Book-to-book proxy evaluation from train-positive books to held-out likes.

This is not an independently annotated similarity benchmark or a temporal test.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.vkr_metrics import ranking_metrics


def _fact_features(graph: nx.MultiDiGraph) -> dict[str, set[str]]:
    supported = {"author", "publisher", "genre"}
    result: dict[str, set[str]] = {}
    for node, props in graph.nodes(data=True):
        if props.get("entity_type") == "book":
            result[str(node)] = {str(nbr) for nbr in graph.successors(node)
                                 if graph.nodes[nbr].get("entity_type") in supported}
    return result


def _per_user_book_maps(ratings: pd.DataFrame, threshold: int) -> dict[str, set[str]]:
    relevant = ratings[ratings["Book-Rating"] >= threshold]
    result: dict[str, set[str]] = defaultdict(set)
    for user, isbn in relevant[["User-ID", "ISBN"]].itertuples(index=False, name=None):
        result[f"user:{user}"].add(f"book:{isbn}")
    return dict(result)


def evaluate_book_retrieval_proxy(
    graph: nx.MultiDiGraph, embeddings: pd.DataFrame,
    train: pd.DataFrame, test: pd.DataFrame, *,
    ks: tuple[int, ...] = (5, 10, 20), positive_threshold: int = 7,
    max_anchors: int = 3,
) -> dict[str, Any]:
    """Fair Node2Vec versus metadata Jaccard on the same book candidates.

    Anchors are positive train books; targets are held-out positive user ratings.
    Already rated train books are excluded. Candidate score is the MAX across
    anchors. Uncovered test users are included with zero scores.
    """
    if not 1 <= positive_threshold <= 10:
        raise ValueError("positive_threshold must be in [1,10]")
    if max_anchors < 1 or not ks or any(k <= 0 for k in ks):
        raise ValueError("max_anchors and ks must be positive")
    train_positive = _per_user_book_maps(train, positive_threshold)
    test_positive = _per_user_book_maps(test, positive_threshold)
    book_ids = sorted(n for n in embeddings.index if str(n).startswith("book:"))
    index = {b: i for i, b in enumerate(book_ids)}
    if not test_positive or not book_ids:
        return {"eligible_users": len(test_positive), "users_with_anchors": 0,
                "methods": {}, "reason": "No test positives or no training book embeddings"}

    matrix = embeddings.loc[book_ids].to_numpy(dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    features = _fact_features(graph)
    all_seen: dict[str, set[str]] = defaultdict(set)
    for user, isbn in train[["User-ID", "ISBN"]].itertuples(index=False, name=None):
        all_seen[f"user:{user}"].add(f"book:{isbn}")

    metrics: dict[str, list[dict[str, float]]] = {"node2vec_max_cosine": [],
                                                  "metadata_max_jaccard": []}
    covered, users_with_rankable_targets = 0, 0
    for user in sorted(test_positive):
        relevant = test_positive[user]
        anchors = sorted(train_positive.get(user, set()) & set(index))[:max_anchors]
        candidates = [book for book in book_ids if book not in all_seen[user]]
        if any(b in candidates for b in relevant):
            users_with_rankable_targets += 1
        if anchors and candidates:
            covered += 1
            positions = [index[b] for b in anchors]
            cosine_scores = np.max(normalized @ normalized[positions].T, axis=1)
            cosine = sorted(candidates, key=lambda b: (-float(cosine_scores[index[b]]), b))
            baseline: dict[str, float] = {}
            for b in candidates:
                b_features = features.get(b, set())
                sim = 0.0
                for anchor in anchors:
                    a_features = features.get(anchor, set())
                    union = a_features | b_features
                    if union:
                        sim = max(sim, len(a_features & b_features) / len(union))
                baseline[b] = sim
            metadata = sorted(candidates, key=lambda b: (-baseline[b], b))
        else:
            cosine = []
            metadata = []
        metrics["node2vec_max_cosine"].append(ranking_metrics(cosine, relevant, ks=ks))
        metrics["metadata_max_jaccard"].append(ranking_metrics(metadata, relevant, ks=ks))

    count = len(test_positive)
    report: dict[str, Any] = {
        "protocol": "book_book_via_user_train_likes_to_heldout_likes",
        "ground_truth_proxy": "Held-out positive user ratings, not independent book similarity judgments",
        "split_type": "random per-user holdout (not chronological)",
        "positive_threshold": positive_threshold,
        "anchor_combination": "maximum similarity over up to max_anchors sorted train-positive ISBNs",
        "max_anchors": max_anchors,
        "eligible_users": count,
        "users_with_anchors": covered,
        "users_with_rankable_targets": users_with_rankable_targets,
        "coverage": covered / count,
        "methods": {},
    }
    for method, rows in metrics.items():
        report["methods"][method] = {
            key if key != "rr" else "mrr_at_max_k": sum(row[key] for row in rows) / count
            for key in rows[0]
        }
    return report
