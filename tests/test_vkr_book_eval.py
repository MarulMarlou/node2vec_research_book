from pathlib import Path
import sys
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vkr_book_eval import evaluate_book_retrieval_proxy
from src.vkr_prepare import build_knowledge_graph


def synthetic_case():
    books = pd.DataFrame([
        {"ISBN": code, "Book-Title": title, "Book-Author": author,
         "Publisher": "Press", "source_row": str(i)}
        for i, (code, title, author) in enumerate([
            ("A", "Anchor", "X"), ("B", "Target", "X"),
            ("C", "Previously read", "Y"), ("D", "Negative candidate", "Y")], start=2)
    ])
    train = pd.DataFrame([
        {"User-ID": "1", "ISBN": "A", "Book-Rating": 9, "source_row": "2"},
        {"User-ID": "1", "ISBN": "C", "Book-Rating": 3, "source_row": "3"},
        {"User-ID": "2", "ISBN": "B", "Book-Rating": 9, "source_row": "4"},
        {"User-ID": "2", "ISBN": "D", "Book-Rating": 7, "source_row": "5"},
    ])
    test = pd.DataFrame([{"User-ID": "1", "ISBN": "B", "Book-Rating": 10, "source_row": "6"}])
    G = build_knowledge_graph(books, train)
    embeddings = pd.DataFrame([[1., 0], [0.96, 0.04], [1., 0], [0., 1.]],
                              index=["book:A", "book:B", "book:C", "book:D"])
    return G, embeddings, train, test


def test_book_retrieval_proxy_excludes_train_and_finds_test():
    report = evaluate_book_retrieval_proxy(*synthetic_case(), ks=(1,2))
    assert report["eligible_users"] == 1
    assert report["coverage"] == 1
    assert report["methods"]["node2vec_max_cosine"]["p@1"] == 1
    assert report["methods"]["metadata_max_jaccard"]["p@1"] == 1
    assert report["ground_truth_proxy"].startswith("Held-out")


def test_missing_training_anchor_counts_as_zero():
    graph, emb, train, test = synthetic_case()
    train = train[train["User-ID"] == "2"]
    report = evaluate_book_retrieval_proxy(graph, emb, train, test, ks=(1,))
    assert report["eligible_users"] == 1
    assert report["users_with_anchors"] == 0
    assert report["methods"]["node2vec_max_cosine"]["p@1"] == 0


def test_empty_test_and_invalid_parameters():
    graph, emb, train, test = synthetic_case()
    empty = evaluate_book_retrieval_proxy(graph, emb, train, test.iloc[:0])
    assert not empty["methods"]
    with pytest.raises(ValueError):
        evaluate_book_retrieval_proxy(graph, emb, train, test, max_anchors=0)
    with pytest.raises(ValueError):
        evaluate_book_retrieval_proxy(graph, emb, train, test, positive_threshold=0)


def test_no_hidden_positive_in_train_graph():
    graph, emb, train, test = synthetic_case()
    assert not graph.has_edge("user:1", "book:B")
    result = evaluate_book_retrieval_proxy(graph, emb, train, test, ks=(1,))
    assert result["methods"]["node2vec_max_cosine"]["r@1"] == 1
