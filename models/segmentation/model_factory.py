from __future__ import annotations
from pathlib import Path
from typing import Dict, Union

import torch
import segmentation_models_pytorch as smp

NUM_CLASSES = 3  # 0=bg, 1=body, 2=panels


def build_model(device: torch.device) -> torch.nn.Module:
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model.to(device)


def load_state_dict(ckpt_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # supports {"epoch":..., "model": state_dict} OR raw state_dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    if not hasattr(state, "keys"):
        raise ValueError(f"Unexpected checkpoint format: {ckpt_path}")

    return state
