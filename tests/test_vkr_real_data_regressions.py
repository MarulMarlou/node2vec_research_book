"""Regression checks for real-world Book-Crossing quoting and Node2Vec scaling."""
from pathlib import Path
import sys

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vkr_prepare import load_book_crossing
from src.vkr_recommend import _cached_walks


def test_bx_backslash_escaped_quotes_and_embedded_semicolon(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'BX-Books.csv').write_text(
        '"ISBN";"Book-Title";"Book-Author";"Publisher"\n'
        '"0001";"A \\"quoted\\"; title";"Author";"Publisher"\n',
        encoding='latin-1')
    (raw / 'BX-Book-Ratings.csv').write_text(
        '"User-ID";"ISBN";"Book-Rating"\n"7";"0001";"9"\n',
        encoding='latin-1')
    books, ratings = load_book_crossing(raw)
    assert books.iloc[0]['Book-Title'] == 'A "quoted"; title'
    assert len(books) == len(ratings) == 1
    assert books.iloc[0]['ISBN'] == '0001'


def test_cached_walk_distribution_matches_second_order_p_q():
    graph = nx.Graph()
    graph.add_edge('a', 'b', weight=1.0)
    graph.add_edge('b', 'c', weight=3.0)
    # Conditioned on a->b: weight back to a=1/p=0.5;
    # weight forward to c=3/q=6. Thus P(c)=6/6.5.
    walks = _cached_walks(graph, sorted(graph.nodes()), walk_length=3,
                          num_walks=2000, p=2.0, q=0.5, seed=42)
    from_a = [walk for walk in walks if walk[0] == 'a']
    probability = sum(walk[2] == 'c' for walk in from_a) / len(from_a)
    assert abs(probability - 6 / 6.5) < 0.025


def test_cached_walk_is_reproducible_and_rejects_nonpositive_weights():
    graph = nx.Graph()
    graph.add_edge('a', 'b', weight=1)
    runs = [_cached_walks(graph, sorted(graph.nodes()), walk_length=8,
                          num_walks=3, p=1, q=1, seed=17) for _ in range(2)]
    assert runs[0] == runs[1]
    graph['a']['b']['weight'] = -0.2
    with pytest.raises(ValueError, match='positive'):
        _cached_walks(graph, sorted(graph.nodes()), walk_length=8,
                      num_walks=3, p=1, q=1, seed=17)
