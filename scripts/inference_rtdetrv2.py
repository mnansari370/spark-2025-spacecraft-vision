import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

from src.core import YAMLConfig

"""
Single-image inference script for RT-DETRv2 on the SPARK-2024 dataset.
"""

# Make sure Python can see the RT-DETRv2 code inside rtdetr/rtdetrv2_pytorch
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RTDETR_DIR = os.path.join(CURRENT_DIR, "..", "rtdetr", "rtdetrv2_pytorch")
sys.path.append(RTDETR_DIR)

from src.core import YAMLConfig

CONFIG_PATH = "configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml"
WEIGHTS_PATH = "/home/users/nmo/spark_project/models/rtdetrv2_spark/best.pth"

# Example image from the validation set
IMAGE_PATH = "/home/users/nmo/spark_project/data/spark-2024-train-val/images/Cheops/val/image_00007_img.jpg"

OUTPUT_DIR = "/home/users/nmo/spark_project/inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Build model from YAML config
cfg = YAMLConfig(CONFIG_PATH)
model = cfg.model
model.to(device)

print("Loading weights...")
checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
state_dict = checkpoint.get("model", checkpoint)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(missing)}, unexpected: {len(unexpected)}")

model.eval()

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"IMAGE_PATH does not exist: {IMAGE_PATH}")

img = Image.open(IMAGE_PATH).convert("RGB")
orig_w, orig_h = img.size
print(f"Original image size: {orig_w} x {orig_h}")

# Resize to the same size used during training
target_size = (640, 640)
img_resized = img.resize(target_size)
img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img_tensor)

if isinstance(outputs, (list, tuple)):
    outputs = outputs[0]

if not isinstance(outputs, dict):
    raise TypeError(f"Unexpected model output type: {type(outputs)}")

if "pred_logits" not in outputs or "pred_boxes" not in outputs:
    raise KeyError(f"Unexpected output keys: {outputs.keys()}")

logits = outputs["pred_logits"][0]
boxes = outputs["pred_boxes"][0]

# Convert logits to scores and predicted labels
probs = logits.sigmoid().max(dim=-1)
scores = probs.values
labels = probs.indices

plt.figure(figsize=(8, 8))
plt.imshow(img_resized)
ax = plt.gca()

W, H = target_size
score_thresh = 0.5
num_shown = 0

for score, label, box in zip(scores, labels, boxes):
    score = float(score.item())
    if score < score_thresh:
        continue

    label_id = int(label.item())
    if 0 <= label_id < len(CLASS_NAMES):
        class_name = CLASS_NAMES[label_id]
    else:
        class_name = f"class_{label_id}"

    cx, cy, w, h = box.tolist()

    x0 = (cx - w / 2.0) * W
    y0 = (cy - h / 2.0) * H
    x1 = (cx + w / 2.0) * W
    y1 = (cy + h / 2.0) * H

    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2)
    ax.add_patch(rect)
    ax.text(
        x0,
        y0,
        f"{class_name} {score:.2f}",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.5),
        color="white",
    )
    num_shown += 1

plt.axis("off")
save_path = os.path.join(OUTPUT_DIR, "prediction_single.jpg")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
plt.close()

print(f"Saved visualization with {num_shown} boxes to: {save_path}")

