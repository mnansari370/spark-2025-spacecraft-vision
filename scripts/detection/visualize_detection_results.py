#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import json
import random
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# repo root: scripts/detection/ -> scripts/ -> repo
REPO_ROOT = Path(__file__).resolve().parents[2]

CLASSES = [
    "Cheops",
    "LisaPathfinder",
    "ObservationSat1",
    "Proba2",
    "Proba3",
    "Proba3ocs",
    "Smart1",
    "Soho",
    "VenusExpress",
    "XMM Newton",
]


def parse_image_id(v) -> int:
    """Parse numeric id from strings like 'test_00012_img.jpg' or '00012'."""
    if isinstance(v, int):
        return v

    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)

        m = re.search(r"test_(\d{5})", s)
        if m:
            return int(m.group(1))

        m = re.search(r"(\d{5})", s)
        if m:
            return int(m.group(1))

    raise ValueError(f"Cannot parse image_id: {v!r}")


def find_image_path(images_dir: Path, image_id: int) -> Path:
    """
    Search recursively for a file starting with 'test_00000' under images_dir.
    Works with nested folders too.
    """
    prefix = f"test_{image_id:05d}"
    hits = sorted(images_dir.rglob(prefix + "*"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No file found for prefix={prefix} under: {images_dir}")


def load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def cxcywh_norm_to_xyxy_px(bbox, W: int, H: int):
    cx, cy, bw, bh = bbox
    x1 = (cx - bw / 2.0) * W
    y1 = (cy - bh / 2.0) * H
    x2 = (cx + bw / 2.0) * W
    y2 = (cy + bh / 2.0) * H
    return x1, y1, x2, y2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_bbox_and_label(img: Image.Image, label: str, score: float, bbox, line_w: int = 6):
    draw = ImageDraw.Draw(img)
    W, H = img.size

    x1, y1, x2, y2 = cxcywh_norm_to_xyxy_px(bbox, W, H)
    x1, y1, x2, y2 = map(int, [
        clamp(x1, 0, W - 1),
        clamp(y1, 0, H - 1),
        clamp(x2, 0, W - 1),
        clamp(y2, 0, H - 1),
    ])

    # bbox rectangle (orange)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 165, 0), width=line_w)

    text = f"{label} {score:.2f}"
    font = load_font(28)

    # text size robustly
    try:
        bx0, by0, bx1, by1 = draw.textbbox((0, 0), text, font=font)
        tw, th = bx1 - bx0, by1 - by0
    except Exception:
        tw, th = draw.textsize(text, font=font)

    pad = 6
    tx = x1
    ty = y1 - (th + 2 * pad)
    if ty < 0:
        ty = y1 + 2

    bg = [tx, ty, tx + tw + 2 * pad, ty + th + 2 * pad]
    bg[0] = clamp(bg[0], 0, W - 1)
    bg[1] = clamp(bg[1], 0, H - 1)
    bg[2] = clamp(bg[2], 0, W - 1)
    bg[3] = clamp(bg[3], 0, H - 1)

    draw.rectangle(bg, fill=(0, 0, 0))
    draw.text((bg[0] + pad, bg[1] + pad), text, fill=(255, 255, 255), font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred_json",
        default="inference_results/detection/detection_test_predictions.json",
        help="Path to prediction JSON",
    )
    ap.add_argument(
        "--images_dir",
        default="data/spark-2024-detection-test/images",
        help="Folder containing test images",
    )
    ap.add_argument(
        "--out_dir",
        default="inference_results/detection/visuals",
        help="Output folder for visualization images",
    )
    ap.add_argument("--n_per_class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_score", type=float, default=0.0)
    args = ap.parse_args()

    random.seed(args.seed)

    pred_path = (REPO_ROOT / args.pred_json).expanduser().resolve()
    images_dir = (REPO_ROOT / args.images_dir).expanduser().resolve()
    out_dir = (REPO_ROOT / args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"pred_json not found: {pred_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    data = json.loads(pred_path.read_text())

    per_class = {c: [] for c in CLASSES}

    for item in data:
        raw_id = item.get("image_id")
        dets = item.get("detections", [])
        if not dets:
            continue

        top = max(dets, key=lambda d: float(d.get("score", 0.0)))
        cat = top.get("category")
        score = float(top.get("score", 0.0))
        bbox = top.get("bbox")

        if cat not in per_class:
            continue
        if bbox is None or score < args.min_score:
            continue

        img_id = parse_image_id(raw_id)
        per_class[cat].append((img_id, score, bbox))

    for cls in CLASSES:
        candidates = per_class[cls]
        if not candidates:
            print(f"[WARN] {cls}: no detections above min_score={args.min_score}")
            continue

        random.shuffle(candidates)
        chosen = candidates[: args.n_per_class]

        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for img_id, score, bbox in chosen:
            try:
                img_path = find_image_path(images_dir, img_id)
                img = Image.open(img_path).convert("RGB")
                img = draw_bbox_and_label(img, cls, score, bbox, line_w=6)

                out_name = f"{cls}__test_{img_id:05d}__{score:.2f}.png"
                img.save(cls_dir / out_name, quality=95)
                saved += 1
            except Exception as e:
                print(f"[WARN] {cls}: image_id={img_id} failed: {e}")

        print(f"[OK] {cls}: saved {saved} images -> {cls_dir}")

    print(f"\nDone.\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
