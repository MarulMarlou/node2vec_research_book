"""Faithful, bounded KG-path explanations and optional explanation-aware reranking."""
from __future__ import annotations

import math
import time
from typing import Any
import networkx as nx


HUB_TYPES = {"author", "publisher", "genre", "user"}
HUB_WEIGHT = {"author": 1.2, "genre": 1.1, "publisher": 0.8, "user": 0.5}
REL_LABELS = {"written_by": "автор", "published_by": "издательство",
              "has_genre": "жанр", "rated": "читатель"}


def _fact(graph: nx.MultiDiGraph, a: str, b: str) -> dict[str, Any]:
    """Retrieve a real directed fact, regardless of traversal direction."""
    for source, target in ((a, b), (b, a)):
        edge_map = graph.get_edge_data(source, target, default={})
        if edge_map:
            attrs = edge_map[sorted(edge_map, key=str)[0]]
            return {"source": source, "target": target,
                    "relation": attrs["relation"],
                    "source_file": attrs.get("source_file", ""),
                    "source_row": attrs.get("source_row", ""),
                    "source_url": attrs.get("source_url", ""),
                    **({"rating": attrs["rating"]} if "rating" in attrs else {})}
    raise ValueError(f"No graph fact linking {a!r} and {b!r}")


def _valid_path(graph: nx.MultiDiGraph, nodes: list[str]) -> bool:
    """Only book–feature–book (repeated) chains with supported strong ratings."""
    if len(nodes) not in (3, 5) or len(set(nodes)) != len(nodes):
        return False
    types = [graph.nodes[n].get("entity_type") for n in nodes]
    if not all(t == "book" for t in types[::2]):
        return False
    if not all(t in HUB_TYPES for t in types[1::2]):
        return False
    for i in range(len(nodes) - 1):
        e = _fact(graph, nodes[i], nodes[i + 1])
        actual_type = types[i] if i % 2 else types[i + 1]
        expected = {"author": "written_by", "publisher": "published_by",
                    "genre": "has_genre", "user": "rated"}[actual_type]
        if e["relation"] != expected:
            return False
        if actual_type == "user" and float(e.get("rating", 0)) < 7:
            return False
    return True


def _rank_path(graph: nx.MultiDiGraph, nodes: list[str]) -> float:
    n_books = sum(a.get("entity_type") == "book" for _, a in graph.nodes(data=True))
    features = []
    for feature in nodes[1::2]:
        kind = graph.nodes[feature]["entity_type"]
        degree_books = sum(graph.nodes[n].get("entity_type") == "book"
                           for n in set(graph.predecessors(feature)) | set(graph.successors(feature)))
        idf = math.log((1 + n_books) / (1 + degree_books))
        features.append(HUB_WEIGHT[kind] * max(0.0, idf))
    return max(0.0, sum(features) / len(features) - 0.3 * (len(nodes) - 3))


def _render(graph: nx.MultiDiGraph, nodes: list[str]) -> str:
    if len(nodes) == 3:
        hub = nodes[1]
        label = str(graph.nodes[hub].get("label", hub))
        typ = graph.nodes[hub]["entity_type"]
        if typ == "author":
            return f"Обе книги написаны автором «{label}»."
        if typ == "publisher":
            return f"Обе книги изданы издательством «{label}»."
        if typ == "genre":
            return f"Обе книги относятся к жанру «{label}»."
        return "Обе книги получили высокие оценки от одного и того же читателя."
    labels = []
    for i, n in enumerate(nodes):
        kind = graph.nodes[n]["entity_type"]
        label = str(graph.nodes[n].get("label", n))
        if kind == "user":
            labels.append("читатель с высокими оценками")
        elif kind == "book":
            labels.append(f"книга «{label}»")
        else:
            relation = _fact(graph, nodes[i - 1], n)["relation"]
            labels.append(f"{REL_LABELS[relation]} «{label}»")
    return "Связь в графе: " + " → ".join(labels) + "."


def _present(graph: nx.MultiDiGraph, nodes: list[str]) -> dict[str, Any]:
    return {"nodes": nodes,
            "edges": [_fact(graph, a, b) for a, b in zip(nodes, nodes[1:])],
            "length": len(nodes) - 1,
            "informativeness": _rank_path(graph, nodes),
            "text": _render(graph, nodes)}


def find_explanations(graph: nx.MultiDiGraph, source_book: str, target_book: str,
                      *, method: str = "shortest", max_depth: int = 4,
                      top_n: int = 3, max_paths: int = 200,
                      max_expansions: int = 5000) -> dict[str, Any]:
    """Find existing, semantically admissible KG paths with search budgets.

    Truncation means stronger paths might have been missed.
    """
    if method not in ("shortest", "informative"):
        raise ValueError("method must be 'shortest' or 'informative'")
    if max_depth not in (2, 4) or min(top_n, max_paths, max_expansions) <= 0:
        raise ValueError("max_depth must be 2 or 4; limits must be positive")
    if source_book not in graph or target_book not in graph or source_book == target_book:
        return {"paths": [], "truncated": False, "searched": 0}
    if graph.nodes[source_book].get("entity_type") != "book" or graph.nodes[target_book].get("entity_type") != "book":
        return {"paths": [], "truncated": False, "searched": 0}

    undirected = graph.to_undirected(as_view=True)
    paths: list[list[str]] = []
    truncated = False
    shared = sorted(set(undirected.neighbors(source_book)) & set(undirected.neighbors(target_book)))
    for hub in shared:
        candidate = [source_book, hub, target_book]
        if _valid_path(graph, candidate):
            if len(paths) >= max_paths:
                truncated = True
                break
            paths.append(candidate)

    if max_depth == 4 and (method == "informative" or not paths):
        expansions = 0
        def explore(nodes: list[str]) -> None:
            nonlocal expansions, truncated
            if truncated:
                return
            if len(nodes) == 5:
                if nodes[-1] == target_book and _valid_path(graph, nodes):
                    if len(paths) >= max_paths:
                        truncated = True
                    else:
                        paths.append(nodes.copy())
                return
            for neighbor in sorted(undirected.neighbors(nodes[-1])):
                expansions += 1
                if expansions > max_expansions:
                    truncated = True
                    return
                if neighbor in nodes:
                    continue
                level = len(nodes)
                kind = graph.nodes[neighbor].get("entity_type")
                if level % 2 == 0 and kind != "book":
                    continue
                if level % 2 == 1 and kind not in HUB_TYPES:
                    continue
                if neighbor == target_book and level != 4:
                    continue
                if level == 4 and neighbor != target_book:
                    continue
                prefix = nodes + [neighbor]
                if len(prefix) == 3 and not _valid_path(graph, prefix):
                    continue
                explore(prefix)
                if truncated:
                    return
        explore([source_book])

    ranked = [_present(graph, p) for p in paths]
    if method == "shortest":
        ranked.sort(key=lambda p: (p["length"], -p["informativeness"], p["nodes"]))
    else:
        ranked.sort(key=lambda p: (-p["informativeness"], p["length"], p["nodes"]))
    return {"paths": ranked[:top_n], "truncated": truncated, "searched": len(paths)}


def explain_and_rerank(graph: nx.MultiDiGraph,
                       recommendations: list[tuple[str, float]],
                       source_book: str, *, method: str = "informative",
                       mode: str = "posthoc", alpha: float = 0.2,
                       max_depth: int = 4) -> list[dict[str, Any]]:
    """Posthoc keeps original order; rerank blends cosine and KG explanation."""
    if mode not in ("posthoc", "rerank") or not 0 <= alpha <= 1:
        raise ValueError("mode must be posthoc or rerank; alpha in [0,1]")
    seen: set[str] = set()
    rows = []
    for book, cosine in recommendations:
        if book in seen:
            raise ValueError("duplicate candidates")
        seen.add(book)
        if not math.isfinite(float(cosine)) or not -1.00001 <= float(cosine) <= 1.00001:
            raise ValueError("base score must be a finite cosine similarity in [-1,1]")
        started = time.perf_counter()
        result = find_explanations(graph, source_book, book, method=method, max_depth=max_depth,
                                   top_n=1)
        duration = time.perf_counter() - started
        path = result["paths"][0] if result["paths"] else None
        strength = path["informativeness"] if path else 0.0
        scaled_strength = strength / (1 + strength)
        rank_score = (1 - alpha) * ((float(cosine) + 1) / 2) + alpha * scaled_strength
        rows.append({"book_id": book, "base_score": float(cosine),
                     "rank_score": rank_score if mode == "rerank" else float(cosine),
                     "explanation": path["text"] if path else None,
                     "path": path, "informativeness": strength,
                     "explanation_time_s": duration, "truncated": result["truncated"]})
    if mode == "rerank":
        rows.sort(key=lambda row: (-row["rank_score"], row["book_id"]))
    return rows
