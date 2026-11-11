"""
Model Factory for Cellular Traffic Vision Transformer Benchmark.
Provides unified instantiation of:
- 'dayvit': Compact Day-Tokenized ViT (58.8K params, input 7x24)
- 'vit_tiny_pretrained': Standard ViT-Tiny with ImageNet-1k weights (5.5M params, input 224x224x3)
- 'vit_tiny_scratch': Standard ViT-Tiny with random weight initialization (5.5M params, input 224x224x3)
"""
from typing import Optional, Dict, Any
import torch.nn as nn
from .day_vit import DayViT
from .vit_tiny import ViTTiny


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Returns the total number of parameters in a PyTorch model regardless of requires_grad."""
    return sum(p.numel() for p in model.parameters())


def build_model(
    name: str = "dayvit",
    num_classes: int = 4,
    pretrained: bool = False,
    dropout: Optional[float] = None,
    **kwargs: Any
) -> nn.Module:
    """
    Factory function for instantiating model architectures.
    """
    name_clean = name.lower().strip()

    if name_clean in ["dayvit", "custom_vit", "compact_vit"]:
        dp = dropout if dropout is not None else 0.05
        return DayViT(
            num_classes=num_classes,
            embed_dim=kwargs.get("embed_dim", 48),
            num_heads=kwargs.get("num_heads", 4),
            num_layers=kwargs.get("num_layers", 3),
            mlp_ratio=kwargs.get("mlp_ratio", 2.0),
            dropout=dp,
        )

    elif name_clean in ["vit_tiny_pretrained", "vit_tiny_pt", "vit_tiny_lpft"]:
        dp = dropout if dropout is not None else 0.1
        return ViTTiny(num_classes=num_classes, pretrained=True, dropout=dp)

    elif name_clean in ["vit_tiny_scratch", "vit_tiny_random", "vit_tiny"]:
        dp = dropout if dropout is not None else 0.1
        # If pretrained explicitly passed as True, respect it, otherwise default to False
        is_pt = pretrained if "pretrained" in kwargs else False
        return ViTTiny(num_classes=num_classes, pretrained=is_pt, dropout=dp)

    else:
        valid_models = ["dayvit", "vit_tiny_pretrained", "vit_tiny_scratch"]
        raise ValueError(f"Unknown model name: '{name}'. Supported architectures: {valid_models}")
