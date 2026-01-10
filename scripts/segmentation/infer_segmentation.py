from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# repo root: scripts/segmentation/ -> scripts/ -> repo
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.segmentation.model_factory import build_model, load_state_dict  # noqa: E402


def collect_images(images_root: Path, split: str) -> list[Path]:
    """
    Returns image paths relative to images_root.

    Train/Val structure:
      images/<Mission>/<split>/image_XXXXX_img.jpg

    Test structure:
      stream-1-test/test_XXXXX_img.jpg (flat or nested)
    """
    if split in ("train", "val"):
        rels: list[Path] = []
        for p in images_root.rglob("*_img.jpg"):
            rel = p.relative_to(images_root)
            parts = rel.parts
            if len(parts) >= 3 and parts[-2] == split:
                rels.append(rel)
        return sorted(rels)

    return sorted([p.relative_to(images_root) for p in images_root.rglob("test_*_img.jpg")])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val", "train"], default="test")
    ap.add_argument("--images_root", default="", help="Root folder of images")
    ap.add_argument("--out_dir", default="", help="Output folder for PNG masks")
    ap.add_argument(
        "--ckpt",
        default=str(REPO_ROOT / "checkpoints" / "segmentation" / "segmentation_model" / "best.pth"),
        help="Segmentation checkpoint path",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_tta", action="store_true", help="Disable flip-TTA")
    ap.add_argument("--device", default="", help="cuda or cpu (default auto)")
    args = ap.parse_args()

    # defaults by split
    if args.split == "test":
        images_root = Path(args.images_root) if args.images_root else (
            REPO_ROOT / "data" / "spark-2024-segmentation-test" / "stream-1-test"
        )
        out_dir = Path(args.out_dir) if args.out_dir else (
            REPO_ROOT / "inference_results" / "segmentation" / "predicted_masks"
        )
    else:
        images_root = Path(args.images_root) if args.images_root else (
            REPO_ROOT / "data" / "spark-2024-train-val" / "images"
        )
        out_dir = Path(args.out_dir) if args.out_dir else (
            REPO_ROOT / "inference_results" / "segmentation" / f"{args.split}_predicted_masks"
        )

    ckpt_path = Path(args.ckpt)

    if not images_root.exists():
        raise FileNotFoundError(f"Missing images_root: {images_root}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = (device.type == "cuda")

    print("REPO_ROOT:", REPO_ROOT)
    print("SPLIT:", args.split)
    print("IMAGES_ROOT:", images_root)
    print("OUT_DIR:", out_dir)
    print("CKPT:", ckpt_path)
    print("Device:", device)

    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(device)
    state = load_state_dict(ckpt_path)
    model.load_state_dict(state, strict=True)
    model.eval()

    rels = collect_images(images_root, args.split)
    if args.limit and args.limit > 0:
        rels = rels[: args.limit]
        print(f"Quick-test mode: running only {len(rels)} images")

    print(f"Found images: {len(rels)}")

    for rel in tqdm(rels, desc=f"Seg infer ({args.split})"):
        img_path = images_root / rel
        out_name = rel.name.replace("_img.jpg", "_layer.png")

        # for train/val, keep mission/split structure
        if args.split in ("train", "val") and len(rel.parts) >= 3:
            mission = rel.parts[0]
            split_name = rel.parts[1]
            out_subdir = out_dir / mission / split_name
            out_subdir.mkdir(parents=True, exist_ok=True)
            out_path = out_subdir / out_name
        else:
            out_path = out_dir / out_name

        im = Image.open(img_path).convert("RGB")
        arr = np.array(im, dtype=np.uint8)

        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        x = x.unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits1 = model(x)
            if args.no_tta:
                logits = logits1
            else:
                x_flip = torch.flip(x, dims=[3])
                logits2 = model(x_flip)
                logits2 = torch.flip(logits2, dims=[3])
                logits = 0.5 * (logits1 + logits2)

        pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        Image.fromarray(pred).save(out_path)

    print("Saved masks to:", out_dir)


if __name__ == "__main__":
    main()
