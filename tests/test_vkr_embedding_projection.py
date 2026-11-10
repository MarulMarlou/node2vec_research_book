"""PCA plot is deterministic, sampled, honest and uses only model coordinates."""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.vkr_embedding_projection import project_embeddings_pca


def fixtures():
    g = nx.MultiDiGraph()
    ids = [f"book:{i:03d}" for i in range(12)]
    for index, n in enumerate(ids):
        g.add_node(n, label=f"Book {index}", entity_type="book")
    model = pd.DataFrame(
        np.random.default_rng(32).normal(size=(len(ids), 6)),
        index=ids)
    return g, model


def test_projection_deterministic_and_includes_original_and_recommendations():
    g, embs = fixtures()
    a = project_embeddings_pca(embs, g, max_nodes=5, seed=41,
                               include=("book:000", "book:001"))
    b = project_embeddings_pca(embs, g, max_nodes=5, seed=41,
                               include=("book:001", "book:000"))
    assert a.sampled_nodes == 5 and a.total_nodes == 12
    assert set(a.points.loc[a.points["highlighted"], "id"]) == {"book:000", "book:001"}
    pd.testing.assert_frame_equal(a.points, b.points)
    assert a.method == "PCA"
    assert a.explained_variance_axes[0] >= a.explained_variance_axes[1] >= 0
    assert sum(a.explained_variance_axes) <= 1.00001


def test_principal_component_2d_and_no_graph_edge_invention():
    g, embs = fixtures()
    a = project_embeddings_pca(embs, g, max_nodes=20)
    assert len(a.points) == 12
    assert list(a.points.columns) == ["id", "label", "type", "x", "y", "highlighted"]
    assert np.isfinite(a.points[["x", "y"]].to_numpy()).all()
    assert "edge" not in a.points.columns
    assert set(a.points["type"]) == {"book"}


def test_constant_and_degenerate_data_yield_valid_zero_variance():
    g, embs = fixtures()
    const = pd.DataFrame(np.ones((12, 1)), index=embs.index)
    a = project_embeddings_pca(const, g, max_nodes=20)
    assert a.explained_variance_axes == (0., 0.)
    assert np.allclose(a.points["y"].to_numpy(), 0)


def test_projection_rejects_bad_vectors_and_limits():
    g, embs = fixtures()
    with pytest.raises(ValueError, match="max_nodes"):
        project_embeddings_pca(embs, g, max_nodes=1)
    with pytest.raises(ValueError, match="highlighted"):
        project_embeddings_pca(embs, g, max_nodes=2,
                               include=("book:000", "book:001", "book:002"))
    embs.iloc[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        project_embeddings_pca(embs, g)
