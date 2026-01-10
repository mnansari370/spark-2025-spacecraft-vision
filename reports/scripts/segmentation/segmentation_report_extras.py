import argparse
import math
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def find_repo_root(start: Path) -> Path:
    for d in [start] + list(start.parents):
        if (d / "README.md").exists() and (d / "models").exists() and (d / "scripts").exists():
            return d
    raise RuntimeError(f"Could not find repo root from {start}")

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_mask(p: Path):
    try:
        m = np.array(Image.open(p))
        if m.ndim != 2:
            m = m[..., 0]
        return m.astype(np.uint8)
    except Exception:
        return None

def overlay_mask_on_image(img: np.ndarray, mask: np.ndarray, alpha=0.45):
    """
    img: H,W,3 uint8
    mask: H,W uint8 with {0,1,2}
    """
    out = img.astype(np.float32).copy()

    # simple overlay colors (don’t overthink; report just needs clarity)
    # class 1 (body): add to R channel
    # class 2 (panels): add to G channel
    body = (mask == 1)
    panels = (mask == 2)

    out[body, 0] = (1 - alpha) * out[body, 0] + alpha * 255
    out[panels, 1] = (1 - alpha) * out[panels, 1] + alpha * 255

    return np.clip(out, 0, 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000, help="Use first N masks for stats (0=all)")
    ap.add_argument("--vis_n", type=int, default=20, help="Number of overlay samples")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    args = ap.parse_args()

    repo = find_repo_root(Path(__file__).resolve().parent)
    fig_dir = repo / "reports" / "figures"
    tab_dir = repo / "reports" / "tables"
    safe_mkdir(fig_dir); safe_mkdir(tab_dir)

    mask_dir = repo / "inference_results" / "segmentation" / "predicted_masks"
    img_dir  = repo / "data" / "spark-2024-segmentation-test" / "stream-1-test"

    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask dir: {mask_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing test images dir: {img_dir}")

    masks = sorted(mask_dir.glob("test_*_layer.png"))
    if not masks:
        raise RuntimeError("No masks found. Run segmentation inference first.")

    if args.limit and args.limit > 0:
        masks = masks[: args.limit]

    print("Using masks:", len(masks))

    # ---- Compute per-image ratios ----
    body_ratios = []
    panels_ratios = []
    bad = 0

    # top lists: (ratio, mask_path)
    top_body = []
    top_panels = []

    for mp in masks:
        m = load_mask(mp)
        if m is None:
            bad += 1
            continue
        H, W = m.shape
        total = float(H * W)

        br = float(np.sum(m == 1)) / total
        pr = float(np.sum(m == 2)) / total
        body_ratios.append(br)
        panels_ratios.append(pr)

        top_body.append((br, mp.name))
        top_panels.append((pr, mp.name))

    body_ratios = np.array(body_ratios, dtype=np.float32)
    panels_ratios = np.array(panels_ratios, dtype=np.float32)

    # ---- Histograms ----
    plt.figure()
    plt.hist(body_ratios, bins=40)
    plt.xlabel("Body area ratio")
    plt.ylabel("Count")
    plt.title("Segmentation: Body area ratio distribution")
    outp = fig_dir / "segmentation_body_ratio_hist.png"
    plt.tight_layout()
    plt.savefig(outp, dpi=200)
    plt.close()
    print("Wrote:", outp)

    plt.figure()
    plt.hist(panels_ratios, bins=40)
    plt.xlabel("Panels area ratio")
    plt.ylabel("Count")
    plt.title("Segmentation: Panels area ratio distribution")
    outp = fig_dir / "segmentation_panels_ratio_hist.png"
    plt.tight_layout()
    plt.savefig(outp, dpi=200)
    plt.close()
    print("Wrote:", outp)

    # ---- Tables: top coverage ----
    top_body.sort(reverse=True, key=lambda x: x[0])
    top_panels.sort(reverse=True, key=lambda x: x[0])

    out_csv = tab_dir / "segmentation_top_body_coverage.csv"
    with out_csv.open("w") as f:
        f.write("rank,mask_name,body_ratio\n")
        for i, (r, name) in enumerate(top_body[:50], 1):
            f.write(f"{i},{name},{r:.6f}\n")
    print("Wrote:", out_csv)

    out_csv = tab_dir / "segmentation_top_panels_coverage.csv"
    with out_csv.open("w") as f:
        f.write("rank,mask_name,panels_ratio\n")
        for i, (r, name) in enumerate(top_panels[:50], 1):
            f.write(f"{i},{name},{r:.6f}\n")
    print("Wrote:", out_csv)

    # ---- Overlay visualization grid ----
    random.seed(args.seed)
    sample = masks.copy()
    random.shuffle(sample)
    sample = sample[: args.vis_n]

    vis_imgs = []
    for mp in sample:
        mask = load_mask(mp)
        if mask is None:
            continue
        # get matching image name
        img_name = mp.name.replace("_layer.png", "_img.jpg")
        ip = img_dir / img_name
        if not ip.exists():
            continue
        img = np.array(Image.open(ip).convert("RGB"), dtype=np.uint8)
        ov = overlay_mask_on_image(img, mask, alpha=args.alpha)
        vis_imgs.append(Image.fromarray(ov))

    if vis_imgs:
        n = len(vis_imgs)
        cols = min(5, n)
        rows = math.ceil(n / cols)
        thumb_w = 480
        thumb_h = int(thumb_w * 0.75)

        grid = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (255, 255, 255))
        for i, im in enumerate(vis_imgs):
            im = im.resize((thumb_w, thumb_h))
            r = i // cols
            c = i % cols
            grid.paste(im, (c * thumb_w, r * thumb_h))

        outp = fig_dir / "segmentation_overlay_grid.png"
        grid.save(outp)
        print("Wrote:", outp)
    else:
        print("No overlay grid created (missing matching images?)")

    # ---- Summary ----
    out_txt = tab_dir / "segmentation_extras_summary.txt"
    with out_txt.open("w") as f:
        f.write("Segmentation extras summary\n")
        f.write(f"Masks processed: {len(masks)}\n")
        f.write(f"Bad masks skipped: {bad}\n")
        if len(body_ratios) > 0:
            f.write(f"Body ratio mean: {float(np.mean(body_ratios)):.6f}\n")
            f.write(f"Body ratio median: {float(np.median(body_ratios)):.6f}\n")
            f.write(f"Panels ratio mean: {float(np.mean(panels_ratios)):.6f}\n")
            f.write(f"Panels ratio median: {float(np.median(panels_ratios)):.6f}\n")
    print("Wrote:", out_txt)

if __name__ == "__main__":
    main()
