import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import segmentation_models_pytorch as smp


# ---------------------------------------------------------
# Purpose:
#   Run segmentation inference on the SPARK test set (4000 imgs)
#   using flip-TTA and save label PNGs (0=bg, 1=body, 2=panels).
#
# Output:
#   inference_results/segmentation/predicted_masks/test_XXXXX_layer.png
# ---------------------------------------------------------


PROJECT_ROOT = find_repo_root(Path(__file__).resolve())

TEST_DIR = PROJECT_ROOT / "data" / "spark-2024-segmentation-test" / "stream-1-test"
OUT_PNG_DIR = PROJECT_ROOT / "inference_results" / "segmentation" / "predicted_masks"
OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = PROJECT_ROOT / "checkpoints" / "segmentation" / "segmentation_model" / "best.pth"


def build_model(device: torch.device) -> torch.nn.Module:
    # DeepLabV3+ with ResNet50 backbone
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,   # weights come from checkpoint
        in_channels=3,
        classes=3,
    ).to(device)
    return model


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    # model(x) returns logits [B,3,H,W]
    return model(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model + checkpoint
    model = build_model(device)
    print("Loading ckpt:", CKPT_PATH)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    # Support both {"model": state_dict} and direct state_dict formats
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    imgs = sorted(TEST_DIR.glob("test_*_img.jpg"))
    print("Found test images:", len(imgs))

    use_amp = (device.type == "cuda")

    for p in tqdm(imgs, desc="Infer TTA"):
        im = Image.open(p).convert("RGB")
        arr = np.array(im, dtype=np.uint8)  # (H,W,3)
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        x = x.unsqueeze(0).to(device)

        # ---- TTA: original + horizontal flip, average logits
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits1 = predict_logits(model, x)

            x_flip = torch.flip(x, dims=[3])
            logits2 = predict_logits(model, x_flip)
            logits2 = torch.flip(logits2, dims=[3])

            logits = 0.5 * (logits1 + logits2)

        pred = logits.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.uint8)  # (H,W) 0/1/2

        out_name = p.name.replace("_img.jpg", "_layer.png")
        Image.fromarray(pred).save(OUT_PNG_DIR / out_name)

    print("Saved PNG masks to:", OUT_PNG_DIR)


if __name__ == "__main__":
    main()
