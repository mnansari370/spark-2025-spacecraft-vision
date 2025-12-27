python -c "import sys; sys.path.insert(0,'models/detection/rtdetrv2'); from src.core import YAMLConfig; print('OK')"
#!/usr/bin/env python3
import os
import re
import json
import random
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# The 10 SPARK satellite classes (keep this fixed)
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

    raise ValueError(f"Can't parse image_id: {v!r}")


def find_image_path(images_dir: Path, image_id: int) -> Path:
    """
    Your test images are NOT always named 'test_00000.png'.
    Often they look like: 'test_00000_img.jpg'
    So we just look for ANY file starting with 'test_00000'.
    """
    prefix = f"test_{image_id:05d}"

    # Most robust: glob prefix + anything
    hits = sorted(images_dir.glob(prefix + "*"))
    if hits:
        # pick the first match (stable ordering)
        return hits[0]

    raise FileNotFoundError(f"Could not find files starting with {prefix} in {images_dir}")


def load_font(size: int) -> ImageFont.FreeTypeFont:
    # Use a common font if available, otherwise fallback.
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def cxcywh_norm_to_xyxy_px(bbox, W, H):
    # bbox format in your JSON: [cx, cy, w, h] normalized (0..1)
    cx, cy, bw, bh = bbox
    x1 = (cx - bw / 2.0) * W
    y1 = (cy - bh / 2.0) * H
    x2 = (cx + bw / 2.0) * W
    y2 = (cy + bh / 2.0) * H
    return x1, y1, x2, y2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_bbox_and_label(img: Image.Image, label: str, score: float, bbox,
                        color=(255, 165, 0), line_w=6):
    """
    Draw bbox in orange, and put text just above the bbox.
    If bbox is too close to the top edge, put text inside the bbox.
    """
    draw = ImageDraw.Draw(img)
    W, H = img.size

    x1, y1, x2, y2 = cxcywh_norm_to_xyxy_px(bbox, W, H)
    x1, y1, x2, y2 = map(int, [clamp(x1, 0, W - 1), clamp(y1, 0, H - 1),
                               clamp(x2, 0, W - 1), clamp(y2, 0, H - 1)])

    # bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

    # label text
    text = f"{label} {score:.2f}"
    font = load_font(28)

    # text size (safe across PIL versions)
    try:
        bx0, by0, bx1, by1 = draw.textbbox((0, 0), text, font=font)
        tw, th = bx1 - bx0, by1 - by0
    except Exception:
        tw, th = draw.textsize(text, font=font)

    pad = 6
    tx = x1
    ty = y1 - (th + 2 * pad)

    # If too close to top, push inside the bbox
    if ty < 0:
        ty = y1 + 2

    # black background box
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
    ap.add_argument("--pred_json", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_score", type=float, default=0.0)
    args = ap.parse_args()

    random.seed(args.seed)

    pred_path = Path(os.path.expanduser(args.pred_json))
    images_dir = Path(os.path.expanduser(args.images_dir))
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.load(open(pred_path, "r"))

    # Collect top-1 detection per image, grouped by class
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

    # Save visuals, one folder per class
    for cls in CLASSES:
        candidates = per_class[cls]
        if not candidates:
            print(f"[WARN] {cls}: no images found in predictions")
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

                img = draw_bbox_and_label(
                    img, cls, score, bbox,
                    color=(255, 165, 0),   # orange
                    line_w=6              # thicker box
                )

                # Keep filename readable
                out_name = f"{cls}__test_{img_id:05d}__{score:.2f}.png"
                img.save(cls_dir / out_name, quality=95)
                saved += 1
            except Exception as e:
                print(f"[WARN] {cls}: failed on image_id={img_id}: {e}")

        print(f"[OK] {cls}: saved {saved} images -> {cls_dir}")

    print(f"\nDone.\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
