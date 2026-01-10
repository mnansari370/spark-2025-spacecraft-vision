from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

def main():
    # call your existing script, but point it to VAL images + VAL output
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "segmentation" / "run_segmentation_inference.py"),
        "--images_root", str(REPO_ROOT / "data" / "spark-2024-train-val" / "images"),
        "--split", "val",
        "--out_dir", str(REPO_ROOT / "inference_results" / "segmentation" / "val_predicted_masks"),
    ] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
