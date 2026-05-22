from __future__ import annotations

import torch
from torch import nn


HEAD_TYPES = ("linear", "mlp")


def build_head(
    embed_dim: int,
    num_classes: int,
    head_type: str,
    mlp_layers: int = 2,
) -> nn.Module:
    if head_type == "linear":
        return nn.Linear(embed_dim, num_classes)
    if head_type == "mlp":
        if mlp_layers < 2:
            raise ValueError("MLP heads need at least 2 linear layers; use head_type='linear' for 1 layer")
        layers: list[nn.Module] = []
        in_dim = embed_dim
        for _ in range(mlp_layers - 1):
            layers.extend([nn.Linear(in_dim, 256), nn.ReLU()])
            in_dim = 256
        layers.append(nn.Linear(in_dim, num_classes))
        return nn.Sequential(*layers)
    raise ValueError(f"Unknown head_type {head_type!r}. Expected one of: {', '.join(HEAD_TYPES)}")


class DinoClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int | None = None,
        num_classes: int = 2,
        head_type: str = "mlp",
        mlp_layers: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        embed_dim = hidden_dim or getattr(getattr(encoder, "config", None), "hidden_size", None)
        if embed_dim is None:
            raise ValueError("Could not infer DINO hidden size; pass hidden_dim explicitly")

        self.head_type = head_type
        self.mlp_layers = mlp_layers
        self.head = build_head(int(embed_dim), num_classes, head_type, mlp_layers=mlp_layers)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.encoder(pixel_values=pixel_values)
            feats = outputs.pooler_output.float()
        return self.head(feats)
