import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Folder with predicted *_layer.png")
    ap.add_argument("--out_npz", required=True, help="Output .npz path")
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--w", type=int, default=1024)
    ap.add_argument("--num_classes", type=int, default=3)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(pred_dir.glob("test_*_layer.png"))
    if len(files) == 0:
        raise RuntimeError(f"No test_*_layer.png found in: {pred_dir}")

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    N = len(files)
    H, W, C = args.h, args.w, args.num_classes

    # IMPORTANT: this is big (~12.6GB for 4000 x 1024 x 1024 x 3 bool)
    data = np.zeros((N, H, W, C), dtype=bool)

    first_name, last_name = files[0].name, files[-1].name
    print("Found:", N, "masks")
    print("First/Last:", first_name, last_name)
    print("Alloc data:", data.shape, data.dtype)

    for i, p in enumerate(files):
        arr = np.array(Image.open(p))
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.shape != (H, W):
            raise RuntimeError(f"Bad shape {arr.shape} for {p.name}, expected {(H,W)}")

        # label map values must be 0..C-1
        mx = int(arr.max())
        mn = int(arr.min())
        if mn < 0 or mx >= C:
            raise RuntimeError(f"Bad label range in {p.name}: min={mn}, max={mx}, expected [0..{C-1}]")

        # one-hot -> bool (H,W,C)
        # data[i,:,:,k] = (arr==k)
        for k in range(C):
            data[i, :, :, k] = (arr == k)

        if (i+1) % 200 == 0 or (i+1) == N:
            print(f"[{i+1}/{N}]")

    # Save in the SAME key name as template: "data"
    np.savez(out_npz, data=data)
    print("Saved:", out_npz)
    print("Key: data")
    print("Shape:", data.shape, "dtype:", data.dtype)

    # light sanity (sample a few images only - no huge sums over all)
    idxs = [0, N//2, N-1] if N >= 3 else list(range(N))
    for j in idxs:
        s = data[j].sum(axis=-1)
        print(f"Sample {j}: one-hot sums min/max =", int(s.min()), int(s.max()))

if __name__ == "__main__":
    main()
