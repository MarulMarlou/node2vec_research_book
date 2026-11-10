"""Deterministic 2D PCA view of the CURRENT Node2Vec embedding artifacts.

Visualization is exploratory: proximity on a 2D projection is not a retrieval
metric and should not be confused with verified KG paths or embedding cosine.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class Projection2D:
    points: pd.DataFrame
    explained_variance_axes: tuple[float, float]
    sampled_nodes: int
    total_nodes: int
    method: str = "PCA"
    seed: int = 42


def project_embeddings_pca(
    embeddings: pd.DataFrame, graph: nx.MultiDiGraph, *,
    max_nodes: int = 750, seed: int = 42,
    include: tuple[str, ...] = (),
) -> Projection2D:
    """Reproducible PCA on a capped, seeded sample including selected book IDs.

    Centers input embeddings, computes a true 2D principal-component projection
    via SVD, and records the explained variance proportion. No graph geometry
    or artificial links are added. This is a view of a sample, not the full
    embedding space when the graph exceeds max_nodes.
    """
    if max_nodes < 2:
        raise ValueError("max_nodes must be >= 2")
    if embeddings.empty or embeddings.shape[1] < 1:
        raise ValueError("Embeddings are empty")
    if embeddings.index.has_duplicates:
        raise ValueError("Duplicate embedding IDs")
    available = sorted(str(n) for n in embeddings.index if n in graph)
    if len(available) < 2:
        raise ValueError("Need at least two nodes with embeddings in the graph")
    selected = sorted(set(include) & set(available))
    if len(selected) > max_nodes:
        raise ValueError("max_nodes too small for requested highlighted nodes")
    remaining = [n for n in available if n not in set(selected)]
    slots = max_nodes - len(selected)
    if len(remaining) > slots:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(remaining), size=slots, replace=False)
        remaining = sorted(remaining[int(i)] for i in chosen)
    ids = sorted(set(selected + remaining))
    matrix = embeddings.loc[ids].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding coordinates contain NaN or infinity")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * singular[None, :2]
    if coords.shape[1] == 1:
        coords = np.column_stack((coords[:, 0], np.zeros(len(coords))))
    for axis in range(2):
        # PCA signs are mathematically arbitrary: fix them deterministically.
        if axis < len(vh) and vh[axis, int(np.argmax(abs(vh[axis])))] < 0:
            coords[:, axis] *= -1.0
    total_variance = float(np.square(singular).sum())
    var = tuple((float(np.square(singular[i]) / total_variance)
                 if i < len(singular) and total_variance > 0 else 0.0)
                for i in range(2))
    highlighted = set(selected)
    data = [{
        "id": node,
        "label": str(graph.nodes[node].get("label", node)),
        "type": str(graph.nodes[node].get("entity_type", "unknown")),
        "x": float(coords[i, 0]),
        "y": float(coords[i, 1]),
        "highlighted": node in highlighted,
    } for i, node in enumerate(ids)]
    return Projection2D(pd.DataFrame(data), var, len(ids), len(available), seed=seed)
