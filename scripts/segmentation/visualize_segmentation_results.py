#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image



# background = black
# body       = red
# panels     = blue
BG  = (0, 0, 0)
RED = (255, 0, 0)
BLU = (0, 0, 255)


def load_mask_from_npz(npz_path: Path) -> np.ndarray:
    """
    Load a segmentation mask from an .npz file.
    """
    z = np.load(npz_path)

    if "layer" in z:
        lab = z["layer"].astype(np.uint8)
        return lab

    if "data" in z:
        data = z["data"].astype(bool)
        if data.ndim != 3 or data.shape[2] < 3:
            raise ValueError(f"Unexpected 'data' shape in {npz_path.name}: {data.shape}")

        body = data[..., 0]
        panels = data[..., 2]

        lab = np.zeros(body.shape, dtype=np.uint8)
        lab[body] = 1
        lab[panels] = 2
        return lab

    raise ValueError(f"Unknown keys in {npz_path.name}: {list(z.keys())}")


def label_to_color_image(lab: np.ndarray) -> np.ndarray:
    """
    Convert label mask (0/1/2) -> RGB image with:
      bg=black, body=red, panels=blue
    Panel priority: panels overwrite body if any overlap ever happens.
    """
    h, w = lab.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)  # starts black

    body = (lab == 1)
    panels = (lab == 2)

    # paint body first
    out[body] = RED
    # paint panels last (priority)
    out[panels] = BLU

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--npz_dir",
        default="inference_results/segmentation/npz_tmp",
        help="Folder containing extracted test_XXXXX_layer.npz files",
    )
    ap.add_argument(
        "--out_dir",
        default="inference_results/segmentation/visuals_per_class",
        help="Where to write visualization PNGs (flat folder, no subfolders)",
    )
    ap.add_argument("--n", type=int, default=40, help="How many images to export (top-N by fg pixels)")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]  # .../spark_project
    npz_dir = (project_root / args.npz_dir).resolve()
    out_dir = (project_root / args.out_dir).resolve()

    if not npz_dir.exists():
        raise FileNotFoundError(f"npz_dir not found: {npz_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(npz_dir.glob("test_*_layer.npz"))
    print(f"NPZ files found: {len(npz_files)} in {npz_dir}")

    if len(npz_files) == 0:
        print("Nothing to visualize (npz_tmp is empty). Did you extract the zip?")
        return

    # Score each file by number of foreground pixels (body + panels)
    scored = []
    for f in npz_files:
        try:
            lab = load_mask_from_npz(f)
            fg = int((lab == 1).sum() + (lab == 2).sum())
            scored.append((fg, f))
        except Exception as e:
            print(f"[WARN] skip {f.name}: {e}")

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[: min(args.n, len(scored))]

    print(f"Exporting top {len(chosen)} masks by foreground size -> {out_dir}")

    saved = 0
    for fg, f in chosen:
        try:
            lab = load_mask_from_npz(f)
            rgb = label_to_color_image(lab)
            img = Image.fromarray(rgb, mode="RGB")

            # Keep naming super clean
            out_name = f"{f.stem.replace('_layer','')}_seg.png"  # test_00000_seg.png
            img.save(out_dir / out_name)
            saved += 1
        except Exception as e:
            print(f"[WARN] failed {f.name}: {e}")

    print(f"[OK] Saved {saved}/{len(chosen)} visuals.")
    print(f"Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()
