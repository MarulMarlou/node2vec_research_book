"""Testable non-Streamlit backend for the VKR demonstration dashboard."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import json
from typing import Any

import networkx as nx
import pandas as pd

from src.vkr_explain import explain_and_rerank
from src.vkr_recommend import load_embeddings, metadata_book_ranking, recommend_similar_books


@dataclass
class DashboardAssets:
    graph: nx.MultiDiGraph
    embeddings: pd.DataFrame
    report: dict[str, Any]


class ArtifactsError(RuntimeError):
    """Artifacts missing, incompatible, or incomplete."""


def load_dashboard_assets(processed_dir: Path, results_dir: Path) -> DashboardAssets:
    """Load exactly the artifacts from a completed VKR pipeline and verify hashes."""
    graph_file = processed_dir / "vkr_knowledge_graph.graphml"
    embedding_dir = results_dir / "embeddings"
    metrics_file = results_dir / "metrics" / "vkr_pipeline_report.json"
    required = [graph_file, embedding_dir / "vkr_n2v.npy",
                embedding_dir / "vkr_n2v_ids.json", metrics_file]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ArtifactsError("Missing artifacts; run python -m src.vkr_pipeline: " + ", ".join(missing))
    try:
        graph = nx.read_graphml(graph_file, force_multigraph=True)
        embeddings = load_embeddings(embedding_dir)
        report = json.loads(metrics_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ArtifactsError(f"Unable to load VKR artifacts: {exc}") from exc
    if not isinstance(report, dict) or not report.get("preparation"):
        raise ArtifactsError("Missing preparation metadata in the pipeline report")
    if not isinstance(graph, nx.MultiDiGraph):
        raise ArtifactsError("The graph must be a directed MultiDiGraph")
    graph_nodes = set(map(str, graph.nodes()))
    embedding_nodes = set(map(str, embeddings.index))
    if embedding_nodes != graph_nodes:
        raise ArtifactsError("Model/graph node ID sets do not match; regenerate artifacts together")
    if not embedding_nodes or embeddings.index.has_duplicates:
        raise ArtifactsError("Embeddings are empty or contain duplicate IDs")
    if report["preparation"].get("kg_nodes") != graph.number_of_nodes():
        raise ArtifactsError("Graph size differs from the pipeline report; files may be stale")
    if not isinstance(report.get("parameters"), dict) or report["parameters"].get("model") != "Node2Vec":
        raise ArtifactsError("The report does not identify a Node2Vec training run")
    digests = report.get("artifact_sha256")
    if not isinstance(digests, dict):
        raise ArtifactsError("Artifact SHA-256 manifest is missing")
    expected_paths = {
        "knowledge_graph": graph_file,
        "embeddings": embedding_dir / "vkr_n2v.npy",
        "embedding_ids": embedding_dir / "vkr_n2v_ids.json",
        "train_ratings": processed_dir / "vkr_train_ratings.csv",
        "test_ratings": processed_dir / "vkr_test_ratings.csv",
    }
    for key, path in expected_paths.items():
        if not path.is_file():
            raise ArtifactsError(f"Missing research artifact: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digests.get(key) != digest.hexdigest():
            raise ArtifactsError(f"SHA-256 mismatch for {key}; regenerate all artifacts together")
    return DashboardAssets(graph=graph, embeddings=embeddings, report=report)


def list_books(graph: nx.MultiDiGraph, query: str = "", limit: int = 100) -> list[dict[str, str]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    query = query.casefold().strip()
    matches = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("entity_type") != "book":
            continue
        title = str(attrs.get("label", node_id))
        isbn = str(attrs.get("isbn", node_id.removeprefix("book:")))
        if query and query not in title.casefold() and query not in isbn.casefold():
            continue
        matches.append({"id": str(node_id), "title": title, "isbn": isbn})
    matches.sort(key=lambda item: (item["title"].casefold(), item["isbn"], item["id"]))
    return matches[:limit]


def recommend_for_display(assets: DashboardAssets, source_book: str, *,
                          method: str = "informative", mode: str = "posthoc",
                          alpha: float = 0.2, top_k: int = 10,
                          candidate_pool: int = 25, max_depth: int = 4
                          ) -> dict[str, Any]:
    """Book-to-book scores and graph-backed path evidence."""
    if top_k <= 0 or candidate_pool < top_k:
        raise ValueError("top_k must be positive and candidate_pool >= top_k")
    if source_book not in assets.graph or assets.graph.nodes[source_book].get("entity_type") != "book":
        raise ValueError("Selected source is not a known book")
    candidates = recommend_similar_books(assets.embeddings, source_book, top_k=candidate_pool)
    detailed = explain_and_rerank(assets.graph, candidates, source_book, method=method,
                                   mode=mode, alpha=alpha, max_depth=max_depth)
    displayed = detailed[:top_k]
    response_rows = []
    for i, row in enumerate(displayed, start=1):
        item = dict(row)
        item["rank"] = i
        item["title"] = str(assets.graph.nodes[row["book_id"]].get("label", row["book_id"]))
        item["isbn"] = str(assets.graph.nodes[row["book_id"]].get("isbn", ""))
        response_rows.append(item)
    return {
        "source_book": source_book, "method": method, "mode": mode,
        "alpha": float(alpha), "candidate_count": len(candidates),
        "results": response_rows,
        "explanation_coverage": sum(row["path"] is not None for row in displayed) / len(displayed)
                                if displayed else 0.0,
        "truncated_count": sum(bool(row["truncated"]) for row in displayed),
        "average_explanation_time_s": sum(float(row["explanation_time_s"]) for row in displayed)
                                      / len(displayed) if displayed else 0.0,
        "explanation_caveat": "A graph path verifies a factual connection; it does not prove that this path caused Node2Vec's score.",
    }


def compare_metadata_baseline(graph: nx.MultiDiGraph, source_book: str,
                              top_k: int = 10) -> list[dict[str, Any]]:
    rows = []
    for rank, (book_id, jaccard) in enumerate(metadata_book_ranking(graph, source_book, top_k=top_k), start=1):
        rows.append({"rank": rank, "book_id": book_id,
                     "title": str(graph.nodes[book_id].get("label", book_id)),
                     "jaccard": jaccard})
    return rows


def path_chart_data(graph: nx.MultiDiGraph, path: dict[str, Any]) -> dict[str, Any]:
    """Build a Plotly-ready structure containing only actual verified edges."""
    if not path or not path.get("nodes") or not path.get("edges"):
        raise ValueError("An explanation path is required")
    nodes = path["nodes"]
    edges = path["edges"]
    if len(edges) != len(nodes) - 1:
        raise ValueError("Path node/edge counts do not match")
    output_nodes = []
    for position, node_id in enumerate(nodes):
        if node_id not in graph:
            raise ValueError("Path contains a missing graph node")
        attrs = graph.nodes[node_id]
        output_nodes.append({"id": node_id, "label": str(attrs.get("label", node_id)),
                             "type": str(attrs.get("entity_type", "unknown")),
                             "x": position, "y": 0.55 if position % 2 else 0.0})
    output_edges = []
    for i, evidence in enumerate(edges):
        source, target = evidence["source"], evidence["target"]
        if {source, target} != {nodes[i], nodes[i + 1]}:
            raise ValueError("Path evidence has mismatched endpoints")
        if not any(attrs.get("relation") == evidence["relation"] for attrs in
                   graph.get_edge_data(source, target, default={}).values()):
            raise ValueError("Path evidence is not present in the graph")
        output_edges.append({"from_id": source, "to_id": target,
                             "relation": evidence["relation"],
                             "source_file": evidence.get("source_file", ""),
                             "source_row": evidence.get("source_row", ""),
                             "source_url": evidence.get("source_url", "")})
    return {"nodes": output_nodes, "edges": output_edges}
