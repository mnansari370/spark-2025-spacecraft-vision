from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """
    Find repository root by walking up until we find a marker structure.
    """
    start = start or Path(__file__).resolve()
    for d in [start] + list(start.parents):
        if (d / "README.md").exists() and (d / "models").exists() and (d / "run").exists():
            return d
    # fallback: reports/scripts/common/paths.py -> go up 3 -> repo root
    return Path(__file__).resolve().parents[3]


def report_out_latest() -> tuple[Path, Path]:
    """
    Canonical output dirs for the report.
    Returns: (figures_dir, tables_dir)
    """
    base = repo_root() / "reports" / "latest"
    fig = base / "figures"
    tab = base / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    return fig, tab
