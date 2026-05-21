from __future__ import annotations

import torch
from torch import nn


HEAD_TYPES = ("linear", "mlp")


def build_head(embed_dim: int, num_classes: int, head_type: str) -> nn.Module:
    if head_type == "linear":
        return nn.Linear(embed_dim, num_classes)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    raise ValueError(f"Unknown head_type {head_type!r}. Expected one of: {', '.join(HEAD_TYPES)}")


class DinoClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int | None = None,
        num_classes: int = 2,
        head_type: str = "mlp",
    ):
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        embed_dim = hidden_dim or getattr(getattr(encoder, "config", None), "hidden_size", None)
        if embed_dim is None:
            raise ValueError("Could not infer DINO hidden size; pass hidden_dim explicitly")

        self.head_type = head_type
        self.head = build_head(int(embed_dim), num_classes, head_type)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.encoder(pixel_values=pixel_values)
            feats = outputs.pooler_output.float()
        return self.head(feats)
