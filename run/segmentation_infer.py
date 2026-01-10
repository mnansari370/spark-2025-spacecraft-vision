from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main():
    repo_root = get_repo_root()

    cmd = [
        sys.executable,
        str(repo_root / "scripts/segmentation/infer_segmentation.py"),
        "--split", "test",
    ] + sys.argv[1:]

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
