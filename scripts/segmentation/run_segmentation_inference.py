from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable so `import models...` always works
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from models.segmentation.model_factory import build_model, load_state_dict


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="If >0, run only first N images")
    ap.add_argument("--no_tta", action="store_true", help="Disable flip-TTA")
    ap.add_argument("--device", default="", help="cuda or cpu (default: auto)")
    args = ap.parse_args()

    test_dir = REPO_ROOT / "data" / "spark-2024-segmentation-test" / "stream-1-test"
    out_png_dir = REPO_ROOT / "inference_results" / "segmentation" / "predicted_masks"
    out_png_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = REPO_ROOT / "checkpoints" / "segmentation" / "segmentation_model" / "best.pth"

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("REPO_ROOT:", REPO_ROOT)
    print("Device:", device)
    print("CKPT:", ckpt_path)
    print("TEST_DIR:", test_dir)
    print("OUT:", out_png_dir)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {test_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(device)
    state = load_state_dict(ckpt_path)
    model.load_state_dict(state, strict=True)
    model.eval()

    imgs = sorted(test_dir.glob("test_*_img.jpg"))
    if args.limit and args.limit > 0:
        imgs = imgs[: args.limit]
        print(f"Quick-test mode: running only {len(imgs)} images")

    use_amp = (device.type == "cuda")

    for p in tqdm(imgs, desc="Seg infer"):
        im = Image.open(p).convert("RGB")
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
        out_name = p.name.replace("_img.jpg", "_layer.png")
        Image.fromarray(pred).save(out_png_dir / out_name)

    print("Saved PNG masks to:", out_png_dir)


if __name__ == "__main__":
    main()
