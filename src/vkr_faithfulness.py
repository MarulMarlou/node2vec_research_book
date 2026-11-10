"""Bounded graph-evidence ablation study with SAME-seed Node2Vec retraining.

This evaluates score/rank sensitivity to *actual* KG path facts. It is a
controlled pilot, not a causal proof: changing the graph changes stochastic
walks and can affect the entire embedding space. Relation-matched random
removals provide a basic null control; future work needs stronger controls,
multiple splits/seeds and statistical uncertainty intervals.

Usage:
 python -m src.vkr_faithfulness --max-pairs 2 --controls 2

Requires completed vkr_pipeline artifacts, which are hash-checked by the
dashboard loader. It NEVER trains on test interaction edges.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any

import networkx as nx
import pandas as pd
from threadpoolctl import threadpool_limits

from src.vkr_dashboard import load_dashboard_assets
from src.vkr_explain import find_explanations
from src.vkr_prepare import node2vec_projection
from src.vkr_recommend import (cosine_ranking, recommend_similar_books,
                               train_node2vec)


def _edge_key_for_evidence(graph: nx.MultiDiGraph, fact: dict) -> Any:
    """Find one edge matching both relation and provenance, never a guessed edge."""
    source, target = fact["source"], fact["target"]
    edges = graph.get_edge_data(source, target, default={})
    candidates = sorted((
        key for key, attrs in edges.items()
        if (attrs.get("relation") == fact.get("relation") and
            str(attrs.get("source_file", "")) == str(fact.get("source_file", "")) and
            str(attrs.get("source_row", "")) == str(fact.get("source_row", "")) and
            str(attrs.get("source_url", "")) == str(fact.get("source_url", "")))
    ), key=str)
    if not candidates:
        raise ValueError("Explanation evidence is not present with matching provenance")
    return candidates[0]


def ablate_explanation_path(graph: nx.MultiDiGraph, path: dict) -> nx.MultiDiGraph:
    """Delete exactly one confirmed edge for each fact of a selected path."""
    if not path or not path.get("nodes") or not path.get("edges"):
        raise ValueError("Cannot ablate an empty path")
    if len(path["edges"]) != len(path["nodes"]) - 1:
        raise ValueError("Path/fact lengths differ")
    reduced = graph.copy()
    for i, fact in enumerate(path["edges"]):
        if {fact["source"], fact["target"]} != {path["nodes"][i], path["nodes"][i + 1]}:
            raise ValueError("Path evidence endpoints mismatch")
        key = _edge_key_for_evidence(reduced, fact)
        reduced.remove_edge(fact["source"], fact["target"], key=key)
    return reduced


def ablate_matched_control(graph: nx.MultiDiGraph, path: dict, seed: int
                           ) -> nx.MultiDiGraph | None:
    """Random removal of equal number and relation types, away from path nodes.

    These controls do not match endpoint degree; report that limitation.
    """
    rng = random.Random(seed)
    protected = set(path["nodes"])
    reduced = graph.copy()
    for fact in path["edges"]:
        relation = fact["relation"]
        options = sorted([
            (str(s), str(t), k)
            for s, t, k, attrs in reduced.edges(keys=True, data=True)
            if (attrs.get("relation") == relation and
                s not in protected and t not in protected)
        ], key=lambda row: (row[0], row[1], str(row[2])))
        if not options:
            return None
        s, t, k = options[rng.randrange(len(options))]
        reduced.remove_edge(s, t, key=k)
    return reduced


def _ranking(embeddings: pd.DataFrame, source: str, catalog: list[str],
             target: str) -> dict[str, Any]:
    """Fixed original catalog; record missing embeddings rather than imputing."""
    missing = sorted(set(catalog) - set(embeddings.index))
    if source not in embeddings.index or target not in embeddings.index:
        return {"score": None, "rank": None, "uncovered_books": len(missing),
                "reason": "source or target embedding absent after ablation"}
    pairs = cosine_ranking(embeddings, source, catalog)
    found = next(((rank, score) for rank, (book, score) in enumerate(pairs, 1)
                  if book == target), None)
    if found is None:
        return {"score": None, "rank": None, "uncovered_books": len(missing),
                "reason": "target absent from fixed catalog"}
    # Ranks require identical candidate universes; score can still be
    # observed if other books lost embeddings, but the new rank is not fair.
    return {"score": float(found[1]),
            "rank": int(found[0]) if not missing else None,
            "uncovered_books": len(missing),
            "reason": ("candidate embeddings missing; rank not comparable"
                       if missing else None)}


def _find_pairs(graph: nx.MultiDiGraph, embeddings: pd.DataFrame, *,
                source: str | None, max_pairs: int, method: str
                ) -> tuple[str | None, list[tuple[str, dict]]]:
    if source is not None and (source not in graph or graph.nodes[source].get("entity_type") != "book"):
        raise ValueError("source must be a book in the verified graph")
    anchors = ([source] if source else
               sorted(n for n, a in graph.nodes(data=True)
                      if a.get("entity_type") == "book" and n in embeddings.index)[:30])
    best: tuple[str | None, list[tuple[str, dict]]] = (None, [])
    for anchor in anchors:
        candidates = recommend_similar_books(embeddings, anchor, top_k=20)
        pairs: list[tuple[str, dict]] = []
        for target, _ in candidates:
            path_data = find_explanations(
                graph, anchor, target, method=method, max_depth=2,
                max_paths=60, max_expansions=1200)
            if path_data["paths"]:
                pairs.append((target, path_data["paths"][0]))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) > len(best[1]):
            best = (anchor, pairs)
        if len(pairs) >= max_pairs:
            return anchor, pairs
    return best


def run_faithfulness(processed_dir: Path, results_dir: Path, output_dir: Path, *,
                     source_book: str | None = None,
                     max_pairs: int = 2, controls: int = 2,
                     method: str = "shortest", random_seed: int = 71) -> dict:
    if not 1 <= max_pairs <= 8 or not 1 <= controls <= 5:
        raise ValueError("max_pairs must be 1..8 and controls 1..5")
    if method not in ("shortest", "informative"):
        raise ValueError("method must be shortest or informative")
    assets = load_dashboard_assets(processed_dir, results_dir)
    graph = assets.graph
    saved = assets.embeddings
    model_params = assets.report.get("parameters", {})
    if model_params.get("model") != "Node2Vec":
        raise ValueError("Current faithfulness experiment expects trained Node2Vec")
    params = {
        "dimensions": int(model_params["dimensions"]),
        "walk_length": int(model_params["walk_length"]),
        "num_walks": int(model_params["num_walks"]),
        "epochs": int(model_params["epochs"]),
        "p": 1.0, "q": 1.0, "window": 5,
        "seed": int(model_params["seed"]),
    }
    anchor, pairs = _find_pairs(graph, saved, source=source_book,
                                max_pairs=max_pairs, method=method)
    catalog = sorted(n for n, a in graph.nodes(data=True)
                     if a.get("entity_type") == "book" and n in saved.index)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not pairs:
        report = {"status": "no_explainable_pairs", "source_book": anchor,
                  "evaluated_pairs": 0, "message": "No length-2 factual paths in inspected recommendations"}
        (output_dir / "faithfulness_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def fit(g: nx.MultiDiGraph) -> pd.DataFrame:
        with threadpool_limits(limits=1, user_api="blas"):
            return train_node2vec(node2vec_projection(g), **params)

    # Same-process baseline refit: do not attribute environment drift or
    # out-of-date binaries to any removed explanatory fact.
    refit = fit(graph)
    records = []
    for idx, (target, path) in enumerate(pairs, 1):
        before = _ranking(refit, anchor, catalog, target)
        saved_baseline = _ranking(saved, anchor, catalog, target)
        removed = ablate_explanation_path(graph, path)
        after = _ranking(fit(removed), anchor, catalog, target)
        controls_data = []
        for control_idx in range(controls):
            control_graph = ablate_matched_control(
                graph, path, seed=random_seed + idx * 1000 + control_idx)
            if control_graph is not None:
                controls_data.append(_ranking(fit(control_graph), anchor, catalog, target))
        valid_controls = [row for row in controls_data if row["score"] is not None]
        drop = before["score"] - after["score"] if (before["score"] is not None and
                                                   after["score"] is not None) else None
        null_mean = (sum(before["score"] - row["score"] for row in valid_controls)
                     / len(valid_controls)) if before["score"] is not None and valid_controls else None
        records.append({
            "source_book": anchor, "recommended_book": target,
            "explanation_method": method, "path_length": path["length"],
            "removed_relation_counts": dict(Counter(edge["relation"] for edge in path["edges"])),
            "score_baseline_saved": saved_baseline["score"],
            "score_baseline_refit": before["score"],
            "score_ablated": after["score"],
            "cosine_score_drop": drop,
            "rank_baseline": before["rank"], "rank_ablated": after["rank"],
            "rank_change": after["rank"] - before["rank"] if (before["rank"] is not None and
                                                             after["rank"] is not None) else None,
            "null_control_score_drop_mean": null_mean,
            "controls_requested": controls, "controls_valid": len(valid_controls),
            "ablation_uncovered_books": after["uncovered_books"],
            "ablation_reason": after["reason"],
        })
    pd.DataFrame(records).to_csv(output_dir / "faithfulness_pairs.csv", index=False)
    report = {
        "status": "pilot_completed",
        "source_book": anchor,
        "evaluated_pairs": len(records),
        "params": params, "control_seed": random_seed,
        "original_graph_nodes": graph.number_of_nodes(),
        "original_graph_edges": graph.number_of_edges(),
        "model_artifact_sha256": assets.report["artifact_sha256"]["embeddings"],
        "results_csv": "faithfulness_pairs.csv",
        "records": records,
        "limitations": [
            "A valid factual path is not proof of causation for Node2Vec cosine similarity.",
            "Deleting path facts can perturb random walks and the whole representation.",
            "Control removals match relation types and number of edges, not node degree.",
            "A single training seed and limited number of source recommendations cannot establish statistical significance.",
            "Missing source/target embeddings after removal are reported as unavailable, not scored zero.",
            "Only train graph facts are removed; validation/test interactions never enter graph training.",
            "Scores refer to user-selected current pipeline parameters (p=q=1, window=5).",
        ],
    }
    (output_dir / "faithfulness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Bounded faithfulness ablation with Node2Vec retraining")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--output-dir", type=Path, default=Path("results/faithfulness"))
    p.add_argument("--source-book", default=None, help="Optional book:<ISBN> anchor")
    p.add_argument("--max-pairs", type=int, default=2)
    p.add_argument("--controls", type=int, default=2)
    p.add_argument("--method", choices=("shortest", "informative"), default="shortest")
    args = p.parse_args()
    outcome = run_faithfulness(args.processed_dir, args.results_dir,
                               args.output_dir, source_book=args.source_book,
                               max_pairs=args.max_pairs, controls=args.controls,
                               method=args.method)
    print(json.dumps({"status": outcome["status"],
                      "evaluated_pairs": outcome["evaluated_pairs"],
                      "report": str(args.output_dir / "faithfulness_report.json")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
