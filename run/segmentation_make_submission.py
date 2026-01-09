from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

def main():
    inp = REPO_ROOT/"inference_results/segmentation/predicted_masks"
    out = REPO_ROOT/"inference_results/segmentation/npz_tmp"
    zip_path = REPO_ROOT/"inference_results/segmentation/final/submission_segmentation.zip"

    cmd = [
        sys.executable, str(REPO_ROOT/"scripts/segmentation/convert_png_to_npz_submission.py"),
        "--input_dir", str(inp),
        "--output_dir", str(out),
        "--zip_path", str(zip_path),
    ]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
