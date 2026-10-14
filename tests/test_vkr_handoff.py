"""Customer handoff regression: links, launcher modes and private outputs."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    "README.md",
    "QUICK_START.md",
    "CODE_DESCRIPTION.md",
    "docs/VKR_WINDOWS.md",
    "docs/VKR_MILESTONES_1_2.md",
    "docs/VKR_STAGE1.md",
    "docs/VKR_STAGE2.md",
    "docs/VKR_STAGE3.md",
    "docs/LEGACY_PROJECT.md",
]


def test_delivery_documentation_has_no_broken_relative_markdown_links():
    for name in DOCS:
        doc = ROOT / name
        assert doc.is_file(), name
        content = doc.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            relative = target.split("#", 1)[0]
            if not relative:
                continue
            assert (doc.parent / relative).is_file(), (name, target)


def test_windows_launcher_includes_all_advertised_entry_points():
    script = ROOT / "scripts" / "run_vkr_windows.ps1"
    text = script.read_text(encoding="utf-8")
    for mode in ("quick", "main", "study-quick", "study-full", "tests", "app"):
        assert f'"{mode}"' in text
    assert "py -3.11" in text
    assert "src.vkr_embedding_study" in text
    assert "src.vkr_graph_audit" in text
    assert "requirements-vkr-app.txt" in text


def test_venv_and_book_crossing_outputs_are_ignored():
    patterns = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for expected in (".venv*/", "/data/raw/", "/data/processed/",
                     "/results/study/", "/results/embeddings/",
                     "/results/metrics/vkr_*.json"):
        assert expected in patterns
    assert "experiments_updated.py" not in (ROOT / "QUICK_START.md").read_text(encoding="utf-8")
