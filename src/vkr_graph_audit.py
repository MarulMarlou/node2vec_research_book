"""Audit the real knowledge graph: composition, connectivity and leakage."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def graph_audit(processed_dir: Path) -> dict:
    graph = nx.read_graphml(processed_dir / "vkr_knowledge_graph.graphml", force_multigraph=True)
    train = pd.read_csv(processed_dir / "vkr_train_ratings.csv", dtype={"User-ID": str, "ISBN": str})
    test = pd.read_csv(processed_dir / "vkr_test_ratings.csv", dtype={"User-ID": str, "ISBN": str})
    types = Counter(str(attrs.get("entity_type", "unknown")) for _, attrs in graph.nodes(data=True))
    relations = Counter(str(data.get("relation", "unknown")) for _, _, data in graph.edges(data=True))
    degrees = np.fromiter((d for _, d in graph.degree()), dtype=int)
    components = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
    by_type: dict[str, list[int]] = defaultdict(list)
    for node, attrs in graph.nodes(data=True):
        by_type[str(attrs.get("entity_type", "unknown"))].append(int(graph.degree(node)))
    train_pairs = set(zip(train["User-ID"], train["ISBN"]))
    test_pairs = set(zip(test["User-ID"], test["ISBN"]))
    rated_pairs = {(source.removeprefix("user:"), target.removeprefix("book:"))
                   for source, target, edge in graph.edges(data=True)
                   if edge.get("relation") == "rated"}
    collisions = len(train_pairs & test_pairs)
    leaked = len(test_pairs & rated_pairs)
    missing = len(train_pairs - rated_pairs)
    unexpected = len(rated_pairs - train_pairs)
    no_source = sum(not attrs.get("source_file") for _, _, attrs in graph.edges(data=True))
    count = graph.number_of_nodes()
    errors = []
    if collisions or leaked:
        errors.append("test ratings leaked into training graph")
    if missing or unexpected:
        errors.append("rated graph edges differ from training ratings")
    if no_source:
        errors.append("graph edges lack source_file provenance")
    return {
        "graph_type": type(graph).__name__,
        "nodes": count, "edges": graph.number_of_edges(),
        "nodes_by_type": dict(sorted(types.items())),
        "relations_by_type": dict(sorted(relations.items())),
        "train_ratings": len(train), "test_ratings": len(test),
        "train_test_overlapping_pairs": collisions,
        "test_rating_edges_in_graph": leaked,
        "graph_train_rating_mismatch": {"missing": missing, "unexpected": unexpected},
        "weak_components": len(components),
        "largest_weak_component_nodes": components[0] if components else 0,
        "largest_weak_component_fraction": components[0] / count if count else 0,
        "isolated_nodes": nx.number_of_isolates(graph),
        "degree": {
            "min": int(degrees.min()) if len(degrees) else 0,
            "mean": float(degrees.mean()) if len(degrees) else 0.0,
            "median": float(np.median(degrees)) if len(degrees) else 0.0,
            "p90": float(np.percentile(degrees, 90)) if len(degrees) else 0.0,
            "p99": float(np.percentile(degrees, 99)) if len(degrees) else 0.0,
            "max": int(degrees.max()) if len(degrees) else 0,
        },
        "degree_by_type": {name: {"min": min(values),
                                  "median": float(np.median(values)),
                                  "mean": float(np.mean(values)),
                                  "max": max(values)}
                           for name, values in sorted(by_type.items())},
        "missing_provenance_file_edges": no_source,
        "genre_enrichment_present": types.get("genre", 0) > 0,
        "audit_passed": not errors, "errors": errors,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="VKR KG structural and leakage audit")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("results/metrics/vkr_graph_audit.json"))
    args = p.parse_args()
    result = graph_audit(args.processed_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
