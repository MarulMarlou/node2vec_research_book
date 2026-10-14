"""Train-only Node2Vec and clean, explicitly separated recommendation protocols."""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
import json
import math
import random
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.vkr_metrics import ranking_metrics


def biased_walk(graph: nx.Graph, start: str, *, length: int, p: float, q: float,
                rng: random.Random) -> list[str]:
    """A weighted, second-order Node2Vec walk; the graph is train-only."""
    if length < 1 or p <= 0 or q <= 0:
        raise ValueError("length >= 1 and p,q > 0 are required")
    if start not in graph:
        raise KeyError(start)
    walk = [start]
    while len(walk) < length:
        current = walk[-1]
        neighbors = sorted(graph.neighbors(current))
        if not neighbors:
            break
        previous = walk[-2] if len(walk) >= 2 else None
        weights = []
        for nxt in neighbors:
            weight = float(graph[current][nxt].get("weight", 1.0))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("all graph edge weights must be positive and finite")
            if previous is not None:
                if nxt == previous:
                    weight /= p
                elif not graph.has_edge(previous, nxt):
                    weight /= q
            weights.append(weight)
        walk.append(rng.choices(neighbors, weights=weights, k=1)[0])
    return walk


def _cached_walks(graph: nx.Graph, nodes: list[str], *, walk_length: int,
                  num_walks: int, p: float, q: float, seed: int) -> list[list[str]]:
    """Avoid O(degree) recomputation on every step of long Node2Vec walks.

    Cache the cumulative transition weights for each directed (previous,current)
    state, and use O(log degree) binary search per subsequent random step.
    Each cache entry is computed only when its state is visited. This retains
    the original second-order weighted transition probabilities.
    """
    if p <= 0 or q <= 0 or not all(math.isfinite(x) for x in (p, q)):
        raise ValueError("p and q must be positive and finite")
    neighbors = {n: tuple(sorted(graph.neighbors(n))) for n in nodes}
    neighbor_sets = {n: set(adj) for n, adj in neighbors.items()}
    edge_weights = {}
    for current, adjacent in neighbors.items():
        weights = np.fromiter((float(graph[current][n].get("weight", 1.0))
                               for n in adjacent), dtype=np.float64, count=len(adjacent))
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("all graph edge weights must be positive and finite")
        edge_weights[current] = weights
    rng = random.Random(seed)
    transition_cdf: dict[tuple[str | None, str], np.ndarray] = {}

    def advance(previous: str | None, current: str) -> str | None:
        adjacent = neighbors[current]
        if not adjacent:
            return None
        key = (previous, current)
        cumulative = transition_cdf.get(key)
        if cumulative is None:
            weights = edge_weights[current].copy()
            if previous is not None:
                prior_set = neighbor_sets[previous]
                for i, next_node in enumerate(adjacent):
                    if next_node == previous:
                        weights[i] /= p
                    elif next_node not in prior_set:
                        weights[i] /= q
            cumulative = np.cumsum(weights)
            transition_cdf[key] = cumulative
        position = bisect_left(cumulative, rng.random() * float(cumulative[-1]))
        return adjacent[min(position, len(adjacent) - 1)]

    walks: list[list[str]] = []
    for _ in range(num_walks):
        shuffled = nodes.copy()
        rng.shuffle(shuffled)
        for start in shuffled:
            walk = [start]
            while len(walk) < walk_length:
                previous = walk[-2] if len(walk) > 1 else None
                next_node = advance(previous, walk[-1])
                if next_node is None:
                    break
                walk.append(next_node)
            if len(walk) > 1:
                walks.append(walk)
    return walks


def train_node2vec(graph: nx.Graph, *, dimensions: int = 64,
                   walk_length: int = 30, num_walks: int = 8,
                   p: float = 1.0, q: float = 1.0, seed: int = 42,
                   epochs: int = 5, window: int = 5) -> pd.DataFrame:
    """Deterministic single-worker Word2Vec; never receives test interactions."""
    from gensim.models import Word2Vec

    if not graph or dimensions <= 0 or num_walks <= 0 or epochs <= 0 or window <= 0:
        raise ValueError("graph and training parameters must be nonempty/positive")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("train_node2vec expects the simple undirected train projection")
    nodes = sorted(map(str, graph.nodes()))
    if len(nodes) != len(graph):
        raise ValueError("node IDs must be unique string identifiers")
    walks = _cached_walks(graph, nodes, walk_length=walk_length,
                          num_walks=num_walks, p=p, q=q, seed=seed)
    if not walks:
        raise ValueError("graph contains no usable edges for training")
    model = Word2Vec(sentences=walks, vector_size=dimensions, window=window,
                     min_count=1, sg=1, negative=5, workers=1,
                     seed=seed, epochs=epochs, sorted_vocab=1)
    return pd.DataFrame.from_dict(
        {node: model.wv[node] for node in nodes if node in model.wv}, orient="index"
    ).sort_index()


def save_embeddings(embeddings: pd.DataFrame, out_dir: Path, name: str = "vkr_n2v") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", embeddings.to_numpy())
    (out_dir / f"{name}_ids.json").write_text(
        json.dumps(list(embeddings.index), ensure_ascii=False), encoding="utf-8")


def load_embeddings(out_dir: Path, name: str = "vkr_n2v") -> pd.DataFrame:
    ids = json.loads((out_dir / f"{name}_ids.json").read_text(encoding="utf-8"))
    array = np.load(out_dir / f"{name}.npy", allow_pickle=False)
    if array.shape[0] != len(ids):
        raise ValueError("embedding row count differs from ID count")
    return pd.DataFrame(array, index=ids)


def cosine_ranking(embeddings: pd.DataFrame, query_id: str, candidate_ids: list[str],
                   *, excluded: set[str] | None = None) -> list[tuple[str, float]]:
    """Stable descending ranking over explicit candidates (not the test labels)."""
    if query_id not in embeddings.index:
        return []
    excluded = excluded or set()
    candidate_ids = sorted({n for n in candidate_ids
                            if n != query_id and n in embeddings.index and n not in excluded})
    if not candidate_ids:
        return []
    query = embeddings.loc[query_id].to_numpy(dtype=float)
    matrix = embeddings.loc[candidate_ids].to_numpy(dtype=float)
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    scores = np.divide(matrix @ query, denom, out=np.zeros(len(candidate_ids)), where=denom > 0)
    return sorted(zip(candidate_ids, map(float, scores)), key=lambda item: (-item[1], item[0]))


def recommend_similar_books(embeddings: pd.DataFrame, book_id: str,
                            top_k: int = 10) -> list[tuple[str, float]]:
    """Book-to-book retrieval for the UI; not evaluated by user-book holdout metrics."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    books = [n for n in embeddings.index if str(n).startswith("book:")]
    return cosine_ranking(embeddings, book_id, books)[:top_k]


def metadata_book_ranking(graph: nx.MultiDiGraph, source_book: str,
                          *, top_k: int = 10) -> list[tuple[str, float]]:
    """Control baseline: Jaccard of shared author/publisher/genre facts only."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if source_book not in graph:
        return []
    meta_types = {"author", "publisher", "genre"}
    features: dict[str, set[str]] = defaultdict(set)
    for node, attrs in graph.nodes(data=True):
        if attrs.get("entity_type") != "book":
            continue
        for nbr in graph.successors(node):
            if graph.nodes[nbr].get("entity_type") in meta_types:
                features[node].add(nbr)
    anchor = features.get(source_book, set())
    rows = []
    for book, feature_set in features.items():
        if book == source_book:
            continue
        union = anchor | feature_set
        score = len(anchor & feature_set) / len(union) if union else 0.0
        rows.append((book, float(score)))
    return sorted(rows, key=lambda item: (-item[1], item[0]))[:top_k]


def evaluate_user_holdout(embeddings: pd.DataFrame, train: pd.DataFrame,
                          test: pd.DataFrame, *, ks: tuple[int, ...] = (5, 10, 20),
                          positive_threshold: int = 7) -> dict[str, Any]:
    """Macro user-to-book test metrics; all training-rated books are excluded.

    Missing embeddings/users are counted as uncovered, never silently counted as
    successful. Users with test positives but no rankable candidates are counted
    as failed evaluations (zero metrics). AUC is intentionally not fabricated.
    """
    if not 1 <= positive_threshold <= 10:
        raise ValueError("positive_threshold must be between 1 and 10")
    positives = test[test["Book-Rating"] >= positive_threshold]
    truth = {f"user:{user}": {f"book:{b}" for b in group["ISBN"]}
             for user, group in positives.groupby("User-ID", sort=True)}
    if not truth:
        return {"users_with_test_positives": 0, "evaluated_users": 0,
                "user_coverage": 0.0, "reason": "No held-out positive ratings"}
    observed: dict[str, set[str]] = defaultdict(set)
    for user, book in train[["User-ID", "ISBN"]].itertuples(index=False, name=None):
        observed[f"user:{user}"].add(f"book:{book}")
    catalog = [n for n in embeddings.index if str(n).startswith("book:")]
    accumulated: list[dict[str, float]] = []
    covered = 0
    for user in sorted(truth):
        if user in embeddings.index and catalog:
            covered += 1
            ranking = cosine_ranking(embeddings, user, catalog, excluded=observed[user])
            rank_ids = [book for book, _ in ranking]
        else:
            rank_ids = []
        accumulated.append(ranking_metrics(rank_ids, truth[user], ks=ks))
    report: dict[str, Any] = {
        "users_with_test_positives": len(truth), "evaluated_users": covered,
        "user_coverage": covered / len(truth),
        "positive_threshold": positive_threshold,
        "ranking_protocol": "user_to_book/train_rated_excluded/heldout_positives",
    }
    for key in accumulated[0]:
        report[key] = sum(row[key] for row in accumulated) / len(accumulated)
    report["mrr_at_max_k"] = report.pop("rr")
    return report
