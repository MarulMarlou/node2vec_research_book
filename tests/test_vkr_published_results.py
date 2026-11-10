"""Prevent accidental corruption of the published 42-run research results."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "docs" / "results"


def _csv(name):
    with (ROOT / name).open(newline="", encoding="utf-8-sig") as src:
        return list(csv.DictReader(src))


def test_published_full_study_contains_all_distinct_fits():
    report = json.loads((ROOT / "stage2_2026-11-10_comparison_report.json").read_text(encoding="utf-8"))
    experiments = _csv("stage2_2026-11-10_experiments.csv")
    assert report["mode"] == "full" and report["seed_split"] == 2030
    assert report["experiments"] == len(experiments) == 42
    assert report["experiments_by_method"] == {"Node2Vec": 24, "TransE": 18}
    keys = {(row["algorithm"], row["config_id"], row["seed"]) for row in experiments}
    assert len(keys) == 42
    assert set(row["seed"] for row in experiments) == {"42", "43"}
    for row in experiments:
        for field in ("train_seconds", "validation_ndcg@10", "validation_p@10"):
            value = float(row[field])
            assert math.isfinite(value) and value >= 0
    for algorithm in ("Node2Vec", "TransE"):
        candidates = [row for row in experiments if row["algorithm"] == algorithm]
        selected = max(candidates, key=lambda row: (
            float(row["validation_ndcg@10"]), float(row["validation_p@10"]),
            -int(row["config_id"]), -int(row["seed"])))
        assert selected["experiment"] == report["selected"][algorithm]["parameters"]["experiment"]


def test_published_aggregate_matches_seed_level_values():
    experiments = _csv("stage2_2026-11-10_experiments.csv")
    configs = _csv("stage2_2026-11-10_config_summary.csv")
    assert len(configs) == 21
    for config in configs:
        matches = [
            row for row in experiments
            if row["algorithm"] == config["algorithm"] and row["config_id"] == config["config_id"]
        ]
        assert len(matches) == int(config["runs"]) == 2
        for summary_name, source_name in [
            ("validation_ndcg10", "validation_ndcg@10"),
            ("validation_p10", "validation_p@10"),
            ("train_seconds", "train_seconds"),
        ]:
            nums = [float(row[source_name]) for row in matches]
            assert math.isclose(statistics.mean(nums), float(config[summary_name + "_mean"]), abs_tol=1e-12)
            assert math.isclose(statistics.stdev(nums), float(config[summary_name + "_std"]), abs_tol=1e-12)
