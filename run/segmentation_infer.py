from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

def main():
    # pass through args to the real script
    cmd = [sys.executable, str(REPO_ROOT/"scripts/segmentation/run_segmentation_inference.py")] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
