from __future__ import annotations

import json
import shutil
import argparse
from pathlib import Path
import subprocess
import sys


def find_repo_root(start: Path) -> Path:
    for d in [start] + list(start.parents):
        if (d / "README.md").exists() and (d / "models").exists() and (d / "scripts").exists():
            return d
    for d in [start] + list(start.parents):
        if d.name == "spark_project":
            return d
    raise RuntimeError(f"Could not find repo root from {start}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_gt", default="data/annotations/spark_val.json")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--score_thr", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", default="inference_results/detection/val_predictions.json")
    args = ap.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve().parent)
    coco_path = repo_root / args.coco_gt
    out_json = repo_root / args.out_json

    coco = json.loads(coco_path.read_text())
    file_names = [im["file_name"] for im in coco["images"]]
    if args.limit and args.limit > 0:
        file_names = file_names[: args.limit]

    src_root = repo_root / "data" / "spark-2024-train-val" / "images"
    tmp_root = repo_root / "inference_results" / "detection" / "_tmp_val_images"

    # Clean + recreate
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # copy/symlink with SAME relative path as COCO file_name
    copied = 0
    for rel in file_names:
        src = src_root / rel
        dst = tmp_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            raise FileNotFoundError(f"Missing source image: {src}")
        # symlink is fastest on same filesystem; fallback to copy
        try:
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
        copied += 1

    print(f"Prepared {copied} val images in: {tmp_root}")

    cmd = [
        sys.executable, str(repo_root / "run" / "detection_infer.py"),
        "--images_dir", str(tmp_root),
        "--out_json", str(out_json),
        "--limit", str(args.limit),
        "--score_thr", str(args.score_thr),
        "--device", str(args.device),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
