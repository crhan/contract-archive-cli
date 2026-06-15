from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = "mi" + "neru"
SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "archive",
    "build",
    "dist",
    "htmlcov",
    "input",
    "output.legacy",
}
SKIP_PATHS = {
    Path(".claude") / "worktrees",
}


def _should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in SKIP_PARTS for part in rel.parts):
        return True
    return any(rel == skip or skip in rel.parents for skip in SKIP_PATHS)


def test_repository_has_no_legacy_ocr_engine_name() -> None:
    hits: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or _should_skip(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if FORBIDDEN in text.lower():
            hits.append(str(path.relative_to(ROOT)))

    assert hits == []


def test_legacy_ocr_pipeline_module_is_removed() -> None:
    legacy_module = "contract_archive.pipelines." + "mi" + "neru_pipeline"
    assert importlib.util.find_spec(legacy_module) is None
