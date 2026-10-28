"""A reproducible, typed-negative-sampling TransE baseline for the VKR knowledge graph.

This implementation uses only NumPy/NetworkX/Pandas, not the legacy PyKEEN
runner or its incompatible graph IDs. It trains on the provided graph alone;
held-out user ratings must never be passed as graph edges.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import time

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class TransEResult:
    entities: pd.DataFrame
    relations: pd.DataFrame
    train_seconds: float
    epoch_losses: list[float]
    triples: int
    skipped_negative_samples: int


def extract_triples(graph: nx.MultiDiGraph) -> list[tuple[str, str, str]]:
    """Remove only exact duplicate (head, relation, tail) facts."""
    if not graph.is_directed() or not graph.is_multigraph():
        raise ValueError("TransE expects the directed, typed MultiDiGraph")
    triples = set()
    for h, t, attrs in graph.edges(data=True):
        relation = str(attrs.get("relation", "")).strip()
        if not relation:
            raise ValueError("All graph edges must contain relation")
        if not graph.nodes[h].get("entity_type") or not graph.nodes[t].get("entity_type"):
            raise ValueError("All graph nodes must contain entity_type")
        triples.add((str(h), relation, str(t)))
    if not triples:
        raise ValueError("TransE requires at least one triple")
    return sorted(triples)


def _typed_negative(
    head: int, relation: int, tail: int, *,
    heads_by_relation: dict[int, np.ndarray],
    tails_by_relation: dict[int, np.ndarray],
    seen: set[tuple[int, int, int]], rng: np.random.Generator,
) -> tuple[int, int] | None:
    """Corrupt head/tail within the observed relation signature, never known positives."""
    for attempt in range(48):
        corrupt_head = bool(rng.integers(0, 2))
        if corrupt_head:
            choices = heads_by_relation[relation]
            candidate = int(choices[int(rng.integers(len(choices)))])
            negative = (candidate, relation, tail)
            if candidate != head and negative not in seen:
                return candidate, tail
        else:
            choices = tails_by_relation[relation]
            candidate = int(choices[int(rng.integers(len(choices)))])
            negative = (head, relation, candidate)
            if candidate != tail and negative not in seen:
                return head, candidate
    # Rare dense relation: deterministic exhaustive fallback.
    for candidate in heads_by_relation[relation]:
        candidate = int(candidate)
        if candidate != head and (candidate, relation, tail) not in seen:
            return candidate, tail
    for candidate in tails_by_relation[relation]:
        candidate = int(candidate)
        if candidate != tail and (head, relation, candidate) not in seen:
            return head, candidate
    return None


def train_transe(
    graph: nx.MultiDiGraph, *, dimensions: int = 32, epochs: int = 10,
    batch_size: int = 256, learning_rate: float = 0.05, margin: float = 1.0,
    seed: int = 42,
) -> TransEResult:
    """Train TransE with typed filtered negative sampling and margin-ranking loss.

    Score: -||entity(head) + relation - entity(tail)||_2.
    Each batch accumulates and normalizes per-entity gradients before SGD
    updates to prevent high-degree entities from causing exploding updates.
    """
    if (dimensions < 1 or epochs < 1 or batch_size < 1 or
        not math.isfinite(learning_rate) or learning_rate <= 0 or
        not math.isfinite(margin) or margin <= 0):
        raise ValueError("dimensions/epochs/batch_size and learning_rate/margin must be positive")
    triples = extract_triples(graph)
    nodes = sorted(map(str, graph.nodes))
    relation_names = sorted({r for _, r, _ in triples})
    ni = {node: i for i, node in enumerate(nodes)}
    ri = {rel: i for i, rel in enumerate(relation_names)}
    triples_i = np.asarray([(ni[h], ri[r], ni[t]) for h, r, t in triples], dtype=np.int64)
    seen = {tuple(map(int, row)) for row in triples_i}
    types = {str(node): str(attrs["entity_type"]) for node, attrs in graph.nodes(data=True)}
    # Use relation-specific domain/range entity types, never arbitrary types
    # as negatives (e.g. do not replace a rated book with an author).
    domains: dict[int, set[str]] = defaultdict(set)
    ranges: dict[int, set[str]] = defaultdict(set)
    for h, r, t in triples:
        domains[ri[r]].add(types[h])
        ranges[ri[r]].add(types[t])
    heads = {r: np.array([ni[n] for n in nodes if types[n] in domain], dtype=np.int64)
             for r, domain in domains.items()}
    tails = {r: np.array([ni[n] for n in nodes if types[n] in ranges[r]], dtype=np.int64)
             for r in ranges}

    rng = np.random.default_rng(seed)
    entities = rng.normal(0, 1, size=(len(nodes), dimensions))
    entities /= np.maximum(np.linalg.norm(entities, axis=1, keepdims=True), 1e-12)
    relations = rng.normal(0, 0.1, size=(len(relation_names), dimensions))
    epoch_losses: list[float] = []
    skipped = 0
    started = time.perf_counter()

    for _ in range(epochs):
        order = rng.permutation(len(triples_i))
        loss_sum, valid_sum = 0.0, 0
        for offset in range(0, len(order), batch_size):
            triples_batch = triples_i[order[offset:offset + batch_size]]
            positive = []
            negative = []
            for h, r, t in triples_batch:
                neg = _typed_negative(int(h), int(r), int(t),
                                      heads_by_relation=heads,
                                      tails_by_relation=tails,
                                      seen=seen, rng=rng)
                if neg is None:
                    skipped += 1
                    continue
                positive.append((int(h), int(r), int(t)))
                negative.append((neg[0], int(r), neg[1]))
            if not positive:
                continue
            pos = np.asarray(positive, dtype=np.int64)
            neg = np.asarray(negative, dtype=np.int64)
            ph, pr, pt = pos.T
            nh, nr, nt = neg.T
            pvec = entities[ph] + relations[pr] - entities[pt]
            nvec = entities[nh] + relations[nr] - entities[nt]
            pdist = np.linalg.norm(pvec, axis=1)
            ndist = np.linalg.norm(nvec, axis=1)
            losses = margin + pdist - ndist
            active = losses > 0
            loss_sum += float(np.maximum(losses, 0).sum())
            valid_sum += len(pos)
            if not np.any(active):
                continue
            ph, pr, pt = ph[active], pr[active], pt[active]
            nh, nr, nt = nh[active], nr[active], nt[active]
            gp = pvec[active] / np.maximum(pdist[active, None], 1e-12)
            gn = nvec[active] / np.maximum(ndist[active, None], 1e-12)
            entity_grad = np.zeros_like(entities)
            entity_count = np.zeros(len(nodes), dtype=np.int64)
            for ids, grads in ((ph, gp), (pt, -gp), (nh, -gn), (nt, gn)):
                np.add.at(entity_grad, ids, grads)
                np.add.at(entity_count, ids, 1)
            changed = np.flatnonzero(entity_count)
            entity_grad[changed] /= entity_count[changed, None]
            entities[changed] -= learning_rate * entity_grad[changed]
            # Standard TransE constraint: entity vectors within unit L2 ball.
            norm = np.linalg.norm(entities[changed], axis=1)
            entities[changed] /= np.maximum(norm[:, None], 1.0)
            relation_grad = np.zeros_like(relations)
            relation_count = np.zeros(len(relation_names), dtype=np.int64)
            np.add.at(relation_grad, pr, gp - gn)
            np.add.at(relation_count, pr, 1)
            changed_rel = np.flatnonzero(relation_count)
            relations[changed_rel] -= learning_rate * (
                relation_grad[changed_rel] / relation_count[changed_rel, None])
        if valid_sum == 0:
            raise ValueError("No valid filtered typed negatives: the graph may be saturated")
        epoch_losses.append(loss_sum / valid_sum)

    return TransEResult(
        entities=pd.DataFrame(entities.astype(np.float32), index=nodes),
        relations=pd.DataFrame(relations.astype(np.float32), index=relation_names),
        train_seconds=time.perf_counter() - started,
        epoch_losses=epoch_losses,
        triples=len(triples),
        skipped_negative_samples=skipped,
    )


def transe_user_book_scores(model: TransEResult, user_id: str,
                            candidate_books: list[str], relation: str = "rated"
                            ) -> dict[str, float]:
    """Relation-aware scores for the same candidate books as Node2Vec."""
    if user_id not in model.entities.index or relation not in model.relations.index:
        return {}
    candidates = sorted(set(candidate_books) & set(model.entities.index))
    if not candidates:
        return {}
    origin = (model.entities.loc[user_id].to_numpy(dtype=float) +
              model.relations.loc[relation].to_numpy(dtype=float))
    target = model.entities.loc[candidates].to_numpy(dtype=float)
    distances = np.linalg.norm(origin[None, :] - target, axis=1)
    return dict(zip(candidates, map(float, -distances)))


def transe_rank_user_books(model: TransEResult, user_id: str,
                           candidate_books: list[str],
                           excluded: set[str] | None = None) -> list[tuple[str, float]]:
    scores = transe_user_book_scores(model, user_id, candidate_books)
    excluded = excluded or set()
    return sorted(((book, score) for book, score in scores.items() if book not in excluded),
                  key=lambda item: (-item[1], item[0]))
