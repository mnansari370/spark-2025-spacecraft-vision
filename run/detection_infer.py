from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as F


def find_repo_root(start: Path) -> Path:
    for d in [start] + list(start.parents):
        if (d / "README.md").exists() and (d / "models").exists() and (d / "scripts").exists():
            return d
    for d in [start] + list(start.parents):
        if d.name == "spark_project":
            return d
    raise RuntimeError(f"Could not find repo root from {start}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="", help="Folder containing .jpg images (can have subfolders)")
    ap.add_argument("--out_json", default="", help="Output json path")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score_thr", type=float, default=0.01)
    ap.add_argument("--device", default="", help="cuda or cpu (default auto)")
    args = ap.parse_args()

    this_file = Path(__file__).resolve()
    repo_root = find_repo_root(this_file.parent)

    # Defaults
    images_dir = repo_root / "data" / "spark-2024-detection-test" / "images"
    out_json = repo_root / "inference_results" / "detection" / "detection_test_predictions.json"

    if args.images_dir:
        images_dir = Path(args.images_dir)
    if args.out_json:
        out_json = Path(args.out_json)

    # Ensure RT-DETRv2 code is importable
    import sys
    sys.path.insert(0, str(repo_root / "models" / "detection" / "rtdetrv2"))
    from src.core import YAMLConfig  # after sys.path insert

    config_path = repo_root / "models" / "detection" / "rtdetrv2" / "configs" / "rtdetrv2" / "rtdetrv2_r50vd_spark_40ep.yml"
    weights_path = repo_root / "checkpoints" / "detection" / "detection_model" / "best.pth"

    print("REPO_ROOT:", repo_root)
    print("CONFIG:", config_path)
    print("WEIGHTS:", weights_path)
    print("IMAGES_DIR:", images_dir)
    print("OUT_JSON:", out_json)

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images_dir: {images_dir}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load model
    cfg = YAMLConfig(str(config_path))
    model = cfg.model.to(device)

    print("Loading weights...")
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Classes (match COCO category names)
    classes = [
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

    #  IMPORTANT: recurse + preserve relative path as image_id
    files = sorted(images_dir.rglob("*.jpg"))
    if args.limit and args.limit > 0:
        files = files[: args.limit]
        print(f"Quick-test mode: running only {len(files)} images")

    print(f"Found {len(files)} images.")

    results = []
    for i, img_path in enumerate(files, 1):
        rel_id = str(img_path.relative_to(images_dir)).replace("\\", "/")  # for safety
        img = Image.open(img_path).convert("RGB")

        img_resized = img.resize((640, 640))
        img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        logits = outputs["pred_logits"][0]  # [N, C]
        boxes = outputs["pred_boxes"][0]    # [N, 4] cxcywh normalized

        probs = logits.sigmoid().max(dim=-1)
        scores = probs.values
        labels = probs.indices

        dets = []
        for s, lab, box in zip(scores, labels, boxes):
            s_val = float(s.item())
            if s_val < args.score_thr:
                continue
            lab = int(lab.item())
            cx, cy, w, h = [float(x) for x in box.tolist()]
            dets.append({"category": classes[lab], "score": s_val, "bbox": [cx, cy, w, h]})

        results.append({"image_id": rel_id, "detections": dets})

        if i % 200 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] done")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results))
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
