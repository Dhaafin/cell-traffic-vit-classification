"""
ViT-Tiny Architecture Wrapper (Pretrained Transfer Learning & From-Scratch).
Implements the standard Vision Transformer Tiny (DeiT-Ti / vit-tiny-patch16-224):
- Embedding Dimension D = 192
- Depth L = 12 Encoder Blocks
- Attention Heads h = 3 (head dimension d_k = 64)
- MLP Expansion Ratio r = 4.0 (intermediate_size = 768)
- Patch Size P = 16x16 on 224x224x3 images (196 patch tokens + 1 CLS token = 197 tokens)
- Total Parameters: 5,525,188 (4-class)
"""
from typing import Optional, Dict
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification

DEFAULT_MODEL_ID = "WinKawaks/vit-tiny-patch16-224"
CLASS_MAP: Dict[int, str] = {0: "UU", 1: "UD", 2: "CU", 3: "CD"}
INV_CLASS_MAP: Dict[str, int] = {v: k for k, v in CLASS_MAP.items()}


class ViTTiny(nn.Module):
    """
    Standard Vision Transformer Tiny for 4-Class Cellular KPI Classification.
    Supports:
      1. Pretrained mode (pretrained=True): Loads ImageNet-1k weights with re-initialized 4-class head.
      2. Scratch mode (pretrained=False): Initializes all 12 blocks and patch projections from random weights.
    """
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.1,
        model_id: str = DEFAULT_MODEL_ID,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model_id = model_id

        if pretrained:
            config = ViTConfig.from_pretrained(model_id)
            config.num_labels = num_classes
            config.hidden_dropout_prob = dropout
            config.attention_probs_dropout_prob = dropout
            config.id2label = CLASS_MAP
            config.label2id = INV_CLASS_MAP

            self.vit = ViTForImageClassification.from_pretrained(
                model_id,
                config=config,
                ignore_mismatched_sizes=True
            )
        else:
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                hidden_size=192,
                num_hidden_layers=12,
                num_attention_heads=3,
                intermediate_size=768,
                hidden_act="gelu",
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                qkv_bias=True,
                num_labels=num_classes,
                id2label=CLASS_MAP,
                label2id=INV_CLASS_MAP
            )
            self.vit = ViTForImageClassification(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch_Size, 3, 224, 224)
        Returns:
            logits: Tensor of shape (Batch_Size, num_classes)
        """
        outputs = self.vit(x)
        return outputs.logits

    def set_trainable_mode(self, mode: str = "all", unfreeze_last_n: int = 4):
        """
        Configures parameter grad states for LP-FT or Scratch training:
          - 'all': All layers trainable (for From Scratch or Full Fine-Tuning).
          - 'linear_probe': Freeze 100% backbone, only classifier is trainable.
          - 'partial_finetune': Classifier + final LayerNorm + last N blocks trainable.
        """
        if mode == "all":
            for p in self.parameters():
                p.requires_grad = True
            return

        # Freeze everything first
        for p in self.parameters():
            p.requires_grad = False

        if mode == "linear_probe":
            for p in self.vit.classifier.parameters():
                p.requires_grad = True
        elif mode == "partial_finetune":
            # Unfreeze classifier and final layernorm
            for p in self.vit.classifier.parameters():
                p.requires_grad = True
            for p in self.vit.vit.layernorm.parameters():
                p.requires_grad = True

            encoder_layers = self.vit.vit.encoder.layer
            total_blocks = len(encoder_layers)
            start_idx = max(0, total_blocks - unfreeze_last_n)
            for i in range(start_idx, total_blocks):
                for p in encoder_layers[i].parameters():
                    p.requires_grad = True
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'all', 'linear_probe', 'partial_finetune'.")
