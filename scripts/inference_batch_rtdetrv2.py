import os
import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RTDETR_DIR = os.path.join(CURRENT_DIR, "..", "rtdetr", "rtdetrv2_pytorch")
sys.path.append(RTDETR_DIR)

# RT-DETRv2 YAML config loader
from src.core import YAMLConfig

# ======== CONFIG ==========
CONFIG_PATH = "configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml"
WEIGHTS_PATH = "/home/users/nmo/spark_project/models/rtdetrv2_spark/best.pth"

# Folder with class images (SPARK-2024 train-val images)
BASE_DIR = "/home/users/nmo/spark_project/data/spark-2024-train-val/images"

# Output folder for random batch predictions
OUT_DIR = "/home/users/nmo/spark_project/inference_results/batch_random_10"
os.makedirs(OUT_DIR, exist_ok=True)

# 10 SPARK CLASS NAMES
CLASSES = [
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

# ======== LOAD MODEL ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

cfg = YAMLConfig(CONFIG_PATH)
model = cfg.model
model.to(device)

print("Loading weights...")
checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
state_dict = checkpoint.get("model", checkpoint)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

model.eval()


# ======== INFERENCE ON ONE IMAGE ==========
def run_inference(image_path, save_path):
    img = Image.open(image_path).convert("RGB")

    # Resize to training size
    img_resized = img.resize((640, 640))
    img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)

    # RT-DETR sometimes returns (outputs, ) as tuple
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]

    if not isinstance(outputs, dict):
        raise TypeError(f"Unexpected model output type: {type(outputs)}")

    if "pred_logits" not in outputs or "pred_boxes" not in outputs:
        raise KeyError(f"Unexpected output keys: {outputs.keys()}")

    logits = outputs["pred_logits"][0]  # [num_queries, num_classes]
    boxes = outputs["pred_boxes"][0]    # [num_queries, 4] in cx,cy,w,h (normalized)

    # Convert logits to probabilities and labels
    probs = logits.sigmoid().max(dim=-1)
    scores = probs.values
    labels = probs.indices

    plt.figure(figsize=(8, 8))
    plt.imshow(img_resized)
    ax = plt.gca()

    W, H = 640, 640
    shown = 0
    score_thresh = 0.5

    for score, label, box in zip(scores, labels, boxes):
        score = float(score.item())
        if score < score_thresh:
            continue

        label_id = int(label.item())
        if 0 <= label_id < len(CLASSES):
            class_name = CLASSES[label_id]
        else:
            class_name = f"class_{label_id}"

        cx, cy, w, h = box.tolist()

        x0 = (cx - w / 2.0) * W
        y0 = (cy - h / 2.0) * H
        x1 = (cx + w / 2.0) * W
        y1 = (cy + h / 2.0) * H

        rect = plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            linewidth=2,
            edgecolor="lime",
        )
        ax.add_patch(rect)
        ax.text(
            x0,
            y0,
            f"{class_name} {score:.2f}",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.5),
            color="white",
        )
        shown += 1

    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()

    return shown


# ======== MAIN LOOP: 10 RANDOM IMAGES PER CLASS ==========
print("\n==== Running inference on 10 RANDOM images per class ====\n")

for class_name in CLASSES:
    class_val_dir = os.path.join(BASE_DIR, class_name, "val")

    if not os.path.isdir(class_val_dir):
        print(f"[!] Missing class folder: {class_val_dir}")
        continue

    # All .jpg files in this class/val folder
    files = [f for f in os.listdir(class_val_dir) if f.lower().endswith(".jpg")]
    num_files = len(files)

    if num_files == 0:
        print(f"[!] No .jpg files found for class {class_name} in {class_val_dir}")
        continue

    if num_files < 10:
        print(f"[!] Only {num_files} images found for {class_name}, using all of them.")
        selected = files
    else:
        selected = random.sample(files, 10)

    # Output subfolder per class
    save_dir_class = os.path.join(OUT_DIR, class_name)
    os.makedirs(save_dir_class, exist_ok=True)

    print(f"[+] {class_name}: running on {len(selected)} images...")

    for img_name in selected:
        image_path = os.path.join(class_val_dir, img_name)
        save_name = img_name.replace(".jpg", "_pred.jpg")
        save_path = os.path.join(save_dir_class, save_name)

        boxes = run_inference(image_path, save_path)
        print(f"    Saved: {save_path} (boxes: {boxes})")

print("\n==== DONE! Check your output folder ====\n")
print(f"Output saved in: {OUT_DIR}")

