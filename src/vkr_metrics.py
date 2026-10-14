"""Ranking metrics for held-out recommendations; query relevance is supplied externally."""
from __future__ import annotations
import math
from collections.abc import Iterable

def ranking_metrics(ranked_ids: Iterable[str], relevant_ids: Iterable[str],
                    ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    ks = tuple(ks)
    if not ks or any(not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    ranked = list(ranked_ids)
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranked_ids must contain unique items")
    relevant = set(relevant_ids)
    outcome: dict[str, float] = {}
    for k in ks:
        topk = ranked[:k]
        hits = sum(book in relevant for book in topk)
        precision = hits / k
        recall = hits / len(relevant) if relevant else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        dcg = sum(1 / math.log2(i + 2) for i, book in enumerate(topk) if book in relevant)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        outcome.update({f"p@{k}": precision, f"r@{k}": recall,
                        f"f1@{k}": f1, f"ndcg@{k}": dcg / idcg if idcg else 0.0})
    first = next((i + 1 for i, book in enumerate(ranked[:max(ks)]) if book in relevant), None)
    outcome["rr"] = 1.0 / first if first is not None else 0.0
    return outcome
