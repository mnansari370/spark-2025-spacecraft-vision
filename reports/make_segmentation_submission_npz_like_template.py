import argparse
from pathlib import Path
import re
import numpy as np
from PIL import Image

PAT = re.compile(r"test_(\d+)_layer\.png$")

def parse_idx(p: Path) -> int:
    m = PAT.search(p.name)
    return int(m.group(1)) if m else 10**18

def load_label_png(p: Path) -> np.ndarray:
    arr = np.array(Image.open(p))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template_npz", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--num_classes", type=int, default=3)
    args = ap.parse_args()

    template_npz = Path(args.template_npz)
    pred_dir = Path(args.pred_dir)
    out_npz = Path(args.out_npz)

    if not template_npz.exists():
        raise SystemExit(f"Template NPZ not found: {template_npz}")
    if not pred_dir.exists():
        raise SystemExit(f"pred_dir not found: {pred_dir}")

    tz = np.load(template_npz, allow_pickle=True)
    keys = list(tz.keys())
    if len(keys) != 1:
        raise SystemExit(f"Template NPZ should have exactly 1 key, found keys={keys}")
    key = keys[0]
    tmpl = tz[key]

    print("Template:", template_npz)
    print("Template key:", key)
    print("Template arr dtype:", getattr(tmpl, "dtype", None), "shape:", getattr(tmpl, "shape", None))

    # Expect template like (H, W, C) boolean
    if tmpl.ndim != 3:
        raise SystemExit(f"Unexpected template ndim={tmpl.ndim}. Expected 3 (H,W,C). Shape={tmpl.shape}")
    H, W, C = tmpl.shape
    if C != args.num_classes:
        raise SystemExit(f"Template C={C} but --num_classes={args.num_classes}. Use --num_classes {C}")

    files = sorted(pred_dir.glob("test_*_layer.png"), key=parse_idx)
    if not files:
        raise SystemExit(f"No test_*_layer.png found in: {pred_dir}")

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    N = len(files)
    out = np.zeros((N, H, W, C), dtype=bool)

    for i, p in enumerate(files):
        lab = load_label_png(p)
        if lab.shape != (H, W):
            raise SystemExit(f"Shape mismatch at {p}: got {lab.shape}, expected {(H,W)}")
        if lab.max() >= C:
            raise SystemExit(f"Label values out of range in {p}: max={int(lab.max())}, expected < {C}")

        # one-hot boolean: out[i,:,:,k] = (lab == k)
        for k in range(C):
            out[i, :, :, k] = (lab == k)

        if (i + 1) % 200 == 0 or (i + 1) == N:
            print(f"[{i+1}/{N}] done")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **{key: out})

    print("Saved:", out_npz)
    print("Key:", key)
    print("Array shape:", out.shape, "dtype:", out.dtype)
    print("First/Last:", files[0].name, files[-1].name)

if __name__ == "__main__":
    main()
