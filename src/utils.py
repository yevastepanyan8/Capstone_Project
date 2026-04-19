"""
Shared utilities for the artwork anomaly detection pipeline.

Provides device selection and model loading used by multiple modules.
"""

import torch
from torch import nn
from torchvision import models


def get_device() -> torch.device:
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_resnet50(device: torch.device) -> nn.Module:
    """Load ResNet-50 with final FC removed → 2048-dim feature vector."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone.eval().to(device)
    return backbone
