"""Batching of ligand / pocket graphs into flat torch tensors."""
from __future__ import annotations

import numpy as np
import torch


def collate_ligands(graphs: list[dict]) -> dict:
    """-> flat batch: x(sumA,F), edge_index(2,sumE), edge_attr, batch(sumA),
    n_atoms(B,)."""
    xs, eis, eas, batch, n_atoms = [], [], [], [], []
    off = 0
    for i, g in enumerate(graphs):
        n = g["x"].shape[0]
        xs.append(np.asarray(g["x"]))
        eis.append(np.asarray(g["edge_index"]) + off)
        eas.append(np.asarray(g["edge_attr"]))
        batch.append(np.full(n, i, dtype=np.int64))
        n_atoms.append(n)
        off += n
    total_e = sum(e.shape[0] for e in eas)
    edge_attr = (np.concatenate(eas) if total_e else
                 np.zeros((0, eas[0].shape[1] if eas else 0), np.float32))
    return {
        "x": torch.from_numpy(np.concatenate(xs)),
        "edge_index": torch.from_numpy(np.concatenate(eis, axis=1)),
        "edge_attr": torch.from_numpy(edge_attr.astype(np.float32)),
        "batch": torch.from_numpy(np.concatenate(batch)),
        "n_atoms": torch.tensor(n_atoms, dtype=torch.long),
        "n_graphs": len(graphs),
    }


def collate_pockets(graphs: list[dict]) -> dict:
    xs, eis, eas, batch, n_res = [], [], [], [], []
    off = 0
    for i, g in enumerate(graphs):
        n = g["x"].shape[0]
        xs.append(np.asarray(g["x"]))
        eis.append(np.asarray(g["edge_index"]) + off)
        eas.append(np.asarray(g["edge_attr"]))
        batch.append(np.full(n, i, dtype=np.int64))
        n_res.append(n)
        off += n
    return {
        "x": torch.from_numpy(np.concatenate(xs)),
        "edge_index": torch.from_numpy(np.concatenate(eis, axis=1)),
        "edge_attr": torch.from_numpy(np.concatenate(eas).astype(np.float32)),
        "batch": torch.from_numpy(np.concatenate(batch)),
        "n_res": torch.tensor(n_res, dtype=torch.long),
        "n_graphs": len(graphs),
    }


def pad_flat(h: torch.Tensor, batch: torch.Tensor,
             n_items: torch.Tensor, n_graphs: int):
    """flat (sumN,d) + batch vector -> padded (B,Nmax,d), mask (B,Nmax).

    Assumes nodes of each graph are contiguous and ordered.
    """
    d = h.shape[1]
    nmax = int(n_items.max().item())
    out = h.new_zeros(n_graphs, nmax, d)
    mask = torch.zeros(n_graphs, nmax, dtype=torch.bool, device=h.device)
    starts = torch.cumsum(n_items, 0) - n_items
    pos = torch.arange(h.shape[0], device=h.device) - starts[batch]
    out[batch, pos] = h
    mask[batch, pos] = True
    return out, mask
