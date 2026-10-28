"""Regressions for new, train-only TransE on the typed VKR graph."""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.vkr_transe import (extract_triples, train_transe,
                            transe_rank_user_books, transe_user_book_scores,
                            _typed_negative)


def graph_fixture():
    g = nx.MultiDiGraph()
    for node, kind in (("user:u1", "user"), ("user:u2", "user"),
                       ("book:b1", "book"), ("book:b2", "book"),
                       ("book:b3", "book"), ("author:a1", "author"),
                       ("author:a2", "author"), ("publisher:p", "publisher")):
        g.add_node(node, entity_type=kind)
    for u, b in (("user:u1", "book:b1"), ("user:u1", "book:b2"),
                 ("user:u2", "book:b2"), ("user:u2", "book:b3")):
        g.add_edge(u, b, relation="rated", rating=8)
    for b, a in (("book:b1", "author:a1"), ("book:b2", "author:a1"),
                 ("book:b3", "author:a2")):
        g.add_edge(b, a, relation="written_by")
        g.add_edge(b, "publisher:p", relation="published_by")
    return g


def test_transe_reproducible_and_relation_aware():
    graph = graph_fixture()
    first = train_transe(graph, dimensions=8, epochs=8, batch_size=3,
                         learning_rate=0.04, seed=42)
    second = train_transe(graph, dimensions=8, epochs=8, batch_size=3,
                          learning_rate=0.04, seed=42)
    assert first.triples == graph.number_of_edges()
    assert first.skipped_negative_samples == 0
    assert np.array_equal(first.entities.to_numpy(), second.entities.to_numpy())
    assert np.array_equal(first.relations.to_numpy(), second.relations.to_numpy())
    assert first.epoch_losses == second.epoch_losses
    assert first.train_seconds >= 0
    assert len(first.epoch_losses) == 8
    assert np.isfinite(first.entities.to_numpy()).all()
    assert max(np.linalg.norm(first.entities.to_numpy(), axis=1)) <= 1.00001
    books = ["book:b1", "book:b2", "book:b3"]
    scores = transe_user_book_scores(first, "user:u1", books)
    assert set(scores) == set(books)
    ranked = transe_rank_user_books(first, "user:u1", books, excluded={"book:b1"})
    assert len(ranked) == 2
    assert {r[0] for r in ranked} == {"book:b2", "book:b3"}
    assert all(np.isfinite(r[1]) for r in ranked)


def test_negative_sampling_filters_all_observed_triples_and_entity_type():
    import numpy as np
    rng = np.random.default_rng(77)
    # domain IDs user:0,user:1; range IDs book:2,book:3
    seen = {(0, 0, 2), (1, 0, 3)}
    for _ in range(40):
        neg = _typed_negative(0, 0, 2,
                              heads_by_relation={0: np.array([0, 1])},
                              tails_by_relation={0: np.array([2, 3])},
                              seen=seen, rng=rng)
        assert neg is not None
        assert (neg[0], 0, neg[1]) not in seen
        assert neg[0] in (0, 1) and neg[1] in (2, 3)


def test_invalid_graph_and_config_fail_closed():
    g = graph_fixture()
    with pytest.raises(ValueError, match="positive"):
        train_transe(g, dimensions=0)
    with pytest.raises(ValueError, match="MultiDiGraph"):
        extract_triples(nx.Graph())
    g.nodes["book:b1"].pop("entity_type")
    with pytest.raises(ValueError, match="entity_type"):
        train_transe(g)


def test_does_not_read_holdout_edges():
    g = graph_fixture()
    assert ("user:u1", "rated", "book:b3") not in extract_triples(g)
    model = train_transe(g, dimensions=5, epochs=2, seed=10)
    assert model.triples == len(extract_triples(g))
