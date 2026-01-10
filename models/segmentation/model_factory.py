from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import torch
import segmentation_models_pytorch as smp

# 0=background, 1=body, 2=panels
NUM_CLASSES = 3


def build_model(device: torch.device) -> torch.nn.Module:
    """
    Create the segmentation network.
    Using DeepLabV3+ with ResNet50 encoder.
    """
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model.to(device)


def load_state_dict(ckpt_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Loads checkpoint saved either as:
      - raw state_dict
      - dict with key "model"
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not hasattr(state, "keys"):
        raise ValueError(f"Unexpected checkpoint format: {ckpt_path}")

    return state
