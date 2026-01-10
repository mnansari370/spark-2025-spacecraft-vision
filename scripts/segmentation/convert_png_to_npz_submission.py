import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def png_to_bool3(label: np.ndarray) -> np.ndarray:
    # label: (H,W) uint8 with values {0,1,2}
    body = (label == 1)
    panels = (label == 2)

    out = np.zeros((label.shape[0], label.shape[1], 3), dtype=bool)
    out[..., 0] = body
    out[..., 2] = panels
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with *_layer.png")
    ap.add_argument("--output_dir", required=True, help="Folder to write *_layer.npz")
    ap.add_argument("--zip_path", required=True, help="Final zip path (flat, no folders inside)")
    args = ap.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pngs = sorted(inp.glob("test_*_layer.png"))
    print("PNG count:", len(pngs))

    converted = 0
    skipped = 0

    for p in pngs:
        # skip 0-byte files
        if p.stat().st_size == 0:
            skipped += 1
            continue

        name = p.name.replace(".png", ".npz")

        try:
            lab = np.array(Image.open(p))
        except Exception as e:
            print("SKIP (bad png):", p, "|", e)
            skipped += 1
            continue

        if lab.ndim == 3:
            lab = lab[..., 0]
        if lab.ndim != 2:
            print("SKIP (unexpected mask dims):", p, "shape=", lab.shape)
            skipped += 1
            continue

        data = png_to_bool3(lab.astype(np.uint8))
        np.savez_compressed(out / name, data=data)
        converted += 1

    npzs = sorted(out.glob("test_*_layer.npz"))
    print("NPZ count:", len(npzs))
    print("Converted:", converted, "| Skipped:", skipped)

    pat = re.compile(r"test_\d{5}_layer\.npz$")
    ok = sum(1 for f in npzs if pat.match(f.name))
    print("NPZ name check:", ok, "/", len(npzs))

    zip_path = Path(args.zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in npzs:
            z.write(f, arcname=f.name)

    print("Wrote zip:", zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        has_folders = any("/" in n for n in names)
        print("ZIP entries:", len(names), "| folders:", has_folders)
        if names:
            print("min:", min(names), "max:", max(names))


if __name__ == "__main__":
    main()
