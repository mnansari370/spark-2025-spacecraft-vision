# run/segmentation_infer_val.py
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

THIS_FILE = Path(__file__).resolve()

def find_repo_root(start: Path) -> Path:
    for d in [start] + list(start.parents):
        if (d / "README.md").exists() and (d / "scripts").exists() and (d / "models").exists():
            return d
    for d in [start] + list(start.parents):
        if d.name == "spark_project":
            return d
    raise RuntimeError(f"Could not find repo root from {start}")

REPO_ROOT = find_repo_root(THIS_FILE.parent)
sys.path.insert(0, str(REPO_ROOT))

from models.segmentation.model_factory import build_model, load_state_dict  # noqa: E402


def collect_val_images(images_root: Path):
    """
    Returns list of Path relative to images_root:
      <Mission>/val/image_XXXXX_img.jpg
    """
    rels = []
    # images_root structure: images/<Mission>/(train|val)/image_XXXXX_img.jpg
    for p in images_root.rglob("*_img.jpg"):
        rel = p.relative_to(images_root)
        parts = rel.parts
        if len(parts) >= 3 and parts[-2] == "val":
            rels.append(rel)
    return sorted(rels)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", default=str(REPO_ROOT / "data" / "spark-2024-train-val" / "images"))
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "inference_results" / "segmentation" / "val_predicted_masks"))
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "checkpoints" / "segmentation" / "segmentation_model" / "best.pth"))
    ap.add_argument("--limit", type=int, default=0, help="If >0, run only first N val images")
    ap.add_argument("--no_tta", action="store_true", help="Disable flip-TTA")
    ap.add_argument("--device", default="", help="cuda or cpu (default auto)")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    out_root = Path(args.out_dir)
    ckpt_path = Path(args.ckpt)

    if not images_root.exists():
        raise FileNotFoundError(f"Missing images_root: {images_root}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing ckpt: {ckpt_path}")

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("REPO_ROOT:", REPO_ROOT)
    print("IMAGES_ROOT:", images_root)
    print("OUT_DIR:", out_root)
    print("CKPT:", ckpt_path)
    print("Device:", device)

    # load model (same as your working test inference)
    model = build_model(device)
    state = load_state_dict(ckpt_path)
    model.load_state_dict(state, strict=True)
    model.eval()

    rels = collect_val_images(images_root)
    if args.limit and args.limit > 0:
        rels = rels[: args.limit]
        print(f"Quick-test mode: running only {len(rels)} images")

    print(f"Found VAL images: {len(rels)}")

    use_amp = (device.type == "cuda")

    for rel in tqdm(rels, desc="Seg VAL infer"):
        img_path = images_root / rel

        # rel: <Mission>/val/image_XXXXX_img.jpg
        mission = rel.parts[0]
        out_subdir = out_root / mission / "val"
        out_subdir.mkdir(parents=True, exist_ok=True)

        out_name = rel.name.replace("_img.jpg", "_layer.png")
        out_path = out_subdir / out_name

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

        pred = logits.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(pred).save(out_path)

    print("Saved VAL predicted masks to:", out_root)


if __name__ == "__main__":
    main()
