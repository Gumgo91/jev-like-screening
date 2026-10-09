"""Output heads: scalar hit logit a + auxiliary docking-score estimate s."""
from __future__ import annotations

import torch
import torch.nn as nn


class HitScoreHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2)
        )

    def forward(self, feats: torch.Tensor) -> dict:
        out = self.mlp(feats)
        return {"logit": out[:, 0], "score": out[:, 1]}
