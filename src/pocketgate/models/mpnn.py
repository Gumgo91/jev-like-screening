"""Shared edge-aware message passing layer (native PyTorch).

Native PyTorch implementation: index_add_ aggregation of the edge messages
with a GRUCell update. Not claimed as equivalent to any specific PyG layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeMPNNLayer(nn.Module):
    def __init__(self, dim: int, edge_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(dim + edge_dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        m = F.relu(self.msg(torch.cat([h[src], edge_attr], dim=-1)))
        agg = torch.zeros_like(h).index_add_(0, dst, m)
        return self.gru(agg, h)


class GraphEncoder(nn.Module):
    """Stack of EdgeMPNNLayer over flat batched graphs."""

    def __init__(self, in_dim: int, edge_dim: int, dim: int = 128,
                 layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList(
            [EdgeMPNNLayer(dim, edge_dim) for _ in range(layers)]
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h = self.drop(F.relu(self.inp(x)))
        for layer in self.layers:
            h = self.drop(layer(h, edge_index, edge_attr))
        return h


def masked_mean(h_padded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,N,d),(B,N) -> (B,d). Errors on fully-masked rows."""
    if (~mask).all(dim=1).any():
        raise ValueError("masked_mean: at least one row is fully masked")
    m = mask.unsqueeze(-1).to(h_padded.dtype)
    return (h_padded * m).sum(1) / m.sum(1)
