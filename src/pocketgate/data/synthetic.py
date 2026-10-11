"""Constant synthetic pocket graph that replaces the real pocket of every target in the constant-pocket networks."""
from __future__ import annotations

import hashlib

import numpy as np
import torch


def make_synthetic_pocket(pockets_train: dict, seed: int = 11) -> dict:
    """One fixed graph for every target: 256 nodes with the per-feature
    mean of train-pocket node features, ring-lattice edges k=16,
    constant RBF edge features. Deterministic; hash recorded."""
    xs = np.concatenate([g["x"].numpy() if torch.is_tensor(g["x"])
                         else g["x"] for g in pockets_train.values()])
    mean_x = xs.mean(0).astype(np.float32)
    n = 256
    x = np.repeat(mean_x[None, :], n, axis=0)
    src, dst = [], []
    for i in range(n):
        for k in range(1, 17):
            src.append(i)
            dst.append((i + k) % n)
    edge_index = np.asarray([src, dst], dtype=np.int64)
    from pocketgate.data.features import _rbf
    edge_attr = _rbf(np.full(len(src), 2.0)).astype(np.float32)
    g = {"x": torch.as_tensor(x), "edge_index": torch.as_tensor(edge_index),
         "edge_attr": torch.as_tensor(edge_attr)}
    g["hash"] = hashlib.sha256(
        x.tobytes() + edge_index.tobytes() + edge_attr.tobytes()
    ).hexdigest()
    return g
