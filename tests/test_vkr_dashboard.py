from pathlib import Path
import sys
import json
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vkr_dashboard import (ArtifactsError, load_dashboard_assets, list_books,
                               recommend_for_display, compare_metadata_baseline,
                               path_chart_data)
from src.vkr_pipeline import run_pipeline


@pytest.fixture
def dashboard(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    books = pd.DataFrame([
        {"ISBN": f"00{i}", "Book-Title": "Same Title" if i in (1,2) else f"Book {i}",
         "Book-Author": f"Author {i % 2}", "Publisher": "Press"}
        for i in range(1, 7)
    ])
    ratings = pd.DataFrame([
        {"User-ID": str(user), "ISBN": f"00{book}", "Book-Rating": 9}
        for user in (1,2,3) for book in range(1,7)
    ])
    books.to_csv(raw / "BX-Books.csv", sep=";", index=False, encoding="latin-1")
    ratings.to_csv(raw / "BX-Book-Ratings.csv", sep=";", index=False, encoding="latin-1")
    processed = tmp_path / "processed"
    results = tmp_path / "results"
    run_pipeline(raw, processed, results, max_users=None, max_books=None,
                 dimensions=8, walk_length=5, num_walks=2, epochs=1)
    return processed, results


def test_loader_validates_report_graph_and_embedding_ids(dashboard):
    processed, results = dashboard
    assets = load_dashboard_assets(processed, results)
    assert assets.graph.number_of_nodes() == len(assets.embeddings)
    assert assets.report["preparation"]["train_rows"] > 0
    ids_path = results / "embeddings" / "vkr_n2v_ids.json"
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    ids[0] = "different:ID"
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    with pytest.raises(ArtifactsError, match="sets do not match"):
        load_dashboard_assets(processed, results)


def test_missing_artifacts_fails_clear(dashboard):
    processed, results = dashboard
    (results / "embeddings" / "vkr_n2v.npy").unlink()
    with pytest.raises(ArtifactsError, match="Missing artifacts"):
        load_dashboard_assets(processed, results)


def test_search_uses_title_or_isbn_and_keeps_edition_ids(dashboard):
    processed, results = dashboard
    assets = load_dashboard_assets(processed, results)
    matches = list_books(assets.graph, "same title")
    assert len(matches) == 2
    assert {m["isbn"] for m in matches} == {"001", "002"}
    assert list_books(assets.graph, "005")[0]["id"] == "book:005"
    with pytest.raises(ValueError):
        list_books(assets.graph, limit=0)


def test_posthoc_and_rerank_preserve_evidence(dashboard):
    processed, results = dashboard
    assets = load_dashboard_assets(processed, results)
    posthoc = recommend_for_display(assets, "book:001", method="shortest", mode="posthoc",
                                     top_k=3, candidate_pool=5)
    rerank = recommend_for_display(assets, "book:001", method="informative", mode="rerank",
                                    alpha=0.5, top_k=3, candidate_pool=5)
    assert len(posthoc["results"]) == 3
    assert posthoc["method"] == "shortest"
    assert rerank["mode"] == "rerank"
    assert 0 <= posthoc["explanation_coverage"] <= 1
    assert "caused" in posthoc["explanation_caveat"]
    assert all(row["book_id"] != "book:001" for row in posthoc["results"])
    assert all("rank_score" in row and "base_score" in row for row in rerank["results"])
    assert all(row["path"] is None or row["explanation"] for row in posthoc["results"])
    assert len(compare_metadata_baseline(assets.graph, "book:001", top_k=3)) == 3


def test_plot_data_contains_only_real_evidence(dashboard):
    processed, results = dashboard
    assets = load_dashboard_assets(processed, results)
    response = recommend_for_display(assets, "book:001", top_k=5, candidate_pool=5,
                                      method="shortest", max_depth=2)
    explained = next(row for row in response["results"] if row["path"])
    plot = path_chart_data(assets.graph, explained["path"])
    assert len(plot["nodes"]) == 3
    assert len(plot["edges"]) == 2
    assert all(row["source_file"] for row in plot["edges"])
    broken = dict(explained["path"])
    broken["edges"] = [dict(e) for e in broken["edges"]]
    broken["edges"][0]["relation"] = "fabricated"
    with pytest.raises(ValueError, match="not present"):
        path_chart_data(assets.graph, broken)


def test_parameters_rejected(dashboard):
    processed, results = dashboard
    assets = load_dashboard_assets(processed, results)
    with pytest.raises(ValueError, match="candidate_pool"):
        recommend_for_display(assets, "book:001", top_k=10, candidate_pool=2)
    with pytest.raises(ValueError, match="not a known book"):
        recommend_for_display(assets, "author:author 1")


def test_hash_manifest_catches_graph_change_with_same_nodes(dashboard):
    processed, results = dashboard
    graph_path = processed / "vkr_knowledge_graph.graphml"
    original = graph_path.read_bytes()
    graph_path.write_bytes(original + b"\n<!--tampered-->\n")
    with pytest.raises(ArtifactsError, match="SHA-256 mismatch"):
        load_dashboard_assets(processed, results)


def test_manifest_records_original_input_sha256(dashboard):
    processed, results = dashboard
    assets = load_dashboard_assets(processed, results)
    for filename in ("BX-Books.csv", "BX-Book-Ratings.csv"):
        assert len(assets.report["input_sha256"][filename]) == 64
