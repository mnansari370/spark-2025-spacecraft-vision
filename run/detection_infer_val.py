#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """
    Find the project root (spark_project).
    Heuristics:
      1) Contains README.md + models/ + scripts/
      2) Folder name is spark_project
    """
    for d in [start] + list(start.parents):
        if (d / "README.md").exists() and (d / "models").exists() and (d / "scripts").exists():
            return d
    for d in [start] + list(start.parents):
        if d.name == "spark_project":
            return d
    raise RuntimeError(f"Could not find repo root from {start}")


def main():
    ap = argparse.ArgumentParser(description="Run detection inference on COCO val split by staging images into a tmp folder.")
    ap.add_argument("--coco_gt", default="data/annotations/spark_val.json", help="COCO val annotations JSON")
    ap.add_argument("--images_root", default="data/spark-2024-train-val/images", help="Root of train/val images")
    ap.add_argument("--tmp_dir", default="inference_results/detection/_tmp_val_images", help="Temporary folder to stage val images")
    ap.add_argument("--out_json", default="inference_results/detection/val_predictions.json", help="Output predictions JSON")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images, else only first N from COCO")
    ap.add_argument("--score_thr", type=float, default=0.0, help="Score threshold passed to detection_infer.py")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve().parent)
    coco_path = (repo_root / args.coco_gt).resolve()
    images_root = (repo_root / args.images_root).resolve()
    tmp_root = (repo_root / args.tmp_dir).resolve()
    out_json = (repo_root / args.out_json).resolve()

    if not coco_path.exists():
        raise FileNotFoundError(f"Missing COCO GT json: {coco_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Missing images root: {images_root}")

    coco = json.loads(coco_path.read_text())
    file_names = [im["file_name"] for im in coco.get("images", [])]
    if not file_names:
        raise RuntimeError(f"No images found in COCO JSON: {coco_path}")

    if args.limit and args.limit > 0:
        file_names = file_names[: args.limit]

    # Clean tmp folder
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Stage images (symlink if possible)
    copied = 0
    for rel in file_names:
        rel = rel.replace("\\", "/")
        src = images_root / rel
        dst = tmp_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            raise FileNotFoundError(f"Missing source image: {src}")

        try:
            # If symlink exists already, remove it
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)

        copied += 1

    print(f"[OK] Prepared {copied} val images in: {tmp_root}")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str((repo_root / "run" / "detection_infer.py").resolve()),
        "--images_dir", str(tmp_root),
        "--out_json", str(out_json),
        "--limit", str(args.limit),
        "--score_thr", str(args.score_thr),
        "--device", str(args.device),
    ]

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[DONE] Wrote predictions to: {out_json}")


if __name__ == "__main__":
    main()
