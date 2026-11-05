"""
DayViT: Day-Tokenized Compact Vision Transformer Architecture
Custom designed for native 7x24 spatio-temporal cellular throughput matrices.

Key Features:
- Day-as-Token Tokenization: Projects each 24-hour diurnal profile directly to an embedding vector.
- Ultra-compact parameter footprint (~56.4K - 58.8K trainable parameters).
- Pre-LayerNorm (norm_first=True) for stable and rapid gradient convergence.
- Multi-Head Self-Attention capturing inter-day temporal dependencies.
"""
import torch
import torch.nn as nn


class DayViT(nn.Module):
    """
    DayViT Architecture for 7x24 Spatio-Temporal Matrix Classification.
    """
    def __init__(
        self,
        num_classes: int = 4,
        embed_dim: int = 48,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_ratio: float = 2.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 1. Day-wise Linear Projection: (B, 7, 24) -> (B, 7, embed_dim)
        self.day_proj = nn.Linear(24, embed_dim)

        # 2. [CLS] Token & 1D Positional Embedding (7 Days + 1 CLS = 8 Tokens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 7 + 1, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Transformer Encoder Blocks (Pre-LN, GELU, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Final LayerNorm & Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.day_proj.weight)
        if self.day_proj.bias is not None:
            nn.init.zeros_(self.day_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch_Size, 7, 24)
        Returns:
            logits: Tensor of shape (Batch_Size, num_classes)
        """
        B = x.shape[0]

        # Tokenize days: (B, 7, 24) -> (B, 7, embed_dim)
        tokens = self.day_proj(x)

        # Prepend [CLS] Token: (B, 8, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_tok = torch.cat((cls_tokens, tokens), dim=1)

        # Add Positional Embedding
        x_tok = self.pos_drop(x_tok + self.pos_embed)

        # Transformer Encoder
        out = self.transformer(x_tok)  # (B, 8, embed_dim)

        # Extract [CLS] Token Representation
        cls_rep = self.norm(out[:, 0])

        # Classification Head
        return self.head(cls_rep)


# Alias for backwards compatibility
CustomTrafficViT = DayViT


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DayViT()
    dummy_input = torch.randn(16, 7, 24)
    output = model(dummy_input)
    print(f"Model: DayViT instantiated successfully.")
    print(f"Input Shape: {dummy_input.shape} -> Output Logits: {output.shape}")
    print(f"Trainable Parameters: {count_parameters(model):,}")
