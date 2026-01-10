from __future__ import annotations
from pathlib import Path

def repo_root() -> Path:
    # reports/scripts/common/paths.py -> common -> scripts -> reports -> repo
    return Path(__file__).resolve().parents[3]

def report_out_latest() -> tuple[Path, Path]:
    repo = repo_root()
    fig = repo / "reports" / "outputs" / "latest" / "figures"
    tab = repo / "reports" / "outputs" / "latest" / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    return fig, tab
