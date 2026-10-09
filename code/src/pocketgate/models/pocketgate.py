"""PocketGate model (M) and neural baselines (B3, B4, B5).

All share LigandEncoder/PocketEncoder modules for a controlled
comparison. Each model exposes:

  encode_pocket(pocket_graph_batch) -> ctx dict (H_P pad, mask, kv, pool)
  forward(ctx, ligand_batch) -> {logit, score}   (B,) each

ctx caches everything ligand-independent; `forward` never touches
pocket inputs again -> cache/recompute equivalence is testable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pocketgate.data.graphbatch import pad_flat
from pocketgate.data.features import ATOM_DIM, BOND_DIM

from .encoders import LigandEncoder, PocketEncoder
from .heads import HitScoreHead
from .interaction import InteractionStack
from .mpnn import masked_mean


def _ligand_pool(H_L_flat, lig_batch):
    """flat atom embeddings -> per-graph padded pool."""
    H_pad, mask = pad_flat(H_L_flat, lig_batch["batch"],
                           lig_batch["n_atoms"], lig_batch["n_graphs"])
    return H_pad, mask, masked_mean(H_pad, mask)


class PocketGate(nn.Module):
    """M: pocket-conditioned cross-attention model."""

    def __init__(self, dim=128, pocket_layers=3, ligand_layers=3,
                 interaction_layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.pocket_encoder = PocketEncoder(dim, pocket_layers, dropout)
        self.ligand_encoder = LigandEncoder(dim, ligand_layers, dropout)
        self.interaction = InteractionStack(dim, interaction_layers, heads,
                                            dropout)
        self.head = HitScoreHead(3 * dim, dim)

    def encode_pocket(self, pb) -> dict:
        H_P = self.pocket_encoder(pb["x"], pb["edge_index"], pb["edge_attr"])
        H_pad, mask = pad_flat(H_P, pb["batch"], pb["n_res"], pb["n_graphs"])
        return {
            "H_P": H_pad,                 # (Bp,M,d)
            "pocket_mask": mask,          # (Bp,M)
            "kv": self.interaction.kv_all(H_pad),
            "pocket_pool": masked_mean(H_pad, mask),  # (Bp,d)
        }

    def forward(self, ctx, lb, target_idx=None) -> dict:
        H_L = self.ligand_encoder(lb["x"], lb["edge_index"], lb["edge_attr"])
        H_pad, mask, _ = _ligand_pool(H_L, lb)
        kv = ctx["kv"]
        if ctx["H_P"].shape[0] == 1 and H_pad.shape[0] > 1:
            # single pocket broadcast over the whole ligand batch
            KVs = [(K.squeeze(0), V.squeeze(0)) for K, V in kv]
            H_pad = self.interaction(
                H_pad, KVs,
                pocket_mask=ctx["pocket_mask"].squeeze(0),
                ligand_mask=mask,
            )
            pocket_pool = ctx["pocket_pool"].expand(H_pad.shape[0], -1)
        else:
            H_pad = self.interaction(
                H_pad, kv, pocket_mask=ctx["pocket_mask"], ligand_mask=mask)
            pocket_pool = ctx["pocket_pool"]
        inter_pool = masked_mean(H_pad, mask)
        lig_pool = masked_mean(H_pad, mask)  # post-interaction pooled ligand
        feats = torch.cat([lig_pool, pocket_pool, inter_pool], dim=-1)
        return self.head(feats)


class LigandOnlyNet(nn.Module):
    """B3: ligand encoder + optional target-id embedding -> head."""

    def __init__(self, dim=128, ligand_layers=3, dropout=0.1,
                 n_targets: int = 0, target_emb_dim: int = 32):
        super().__init__()
        self.ligand_encoder = LigandEncoder(dim, ligand_layers, dropout)
        self.target_emb = (nn.Embedding(n_targets, target_emb_dim)
                           if n_targets else None)
        self.head = HitScoreHead(dim + (target_emb_dim if n_targets else 0),
                                 dim)

    def encode_pocket(self, pb):
        return None

    def forward(self, ctx, lb, target_idx=None) -> dict:
        H_L = self.ligand_encoder(lb["x"], lb["edge_index"], lb["edge_attr"])
        _, _, pool = _ligand_pool(H_L, lb)
        if self.target_emb is not None:
            if target_idx is None:
                raise ValueError("B3 conditioned: target_idx required")
            pool = torch.cat([pool, self.target_emb(target_idx)], dim=-1)
        return self.head(pool)


class PooledConcatNet(nn.Module):
    """B4: same encoders, pooled concatenation -> MLP (no atom-level
    interaction)."""

    def __init__(self, dim=128, pocket_layers=3, ligand_layers=3,
                 dropout=0.1):
        super().__init__()
        self.pocket_encoder = PocketEncoder(dim, pocket_layers, dropout)
        self.ligand_encoder = LigandEncoder(dim, ligand_layers, dropout)
        self.head = HitScoreHead(2 * dim, dim)

    def encode_pocket(self, pb):
        H_P = self.pocket_encoder(pb["x"], pb["edge_index"], pb["edge_attr"])
        H_pad, mask = pad_flat(H_P, pb["batch"], pb["n_res"], pb["n_graphs"])
        return {"pocket_pool": masked_mean(H_pad, mask)}

    def forward(self, ctx, lb, target_idx=None) -> dict:
        H_L = self.ligand_encoder(lb["x"], lb["edge_index"], lb["edge_attr"])
        _, _, lig_pool = _ligand_pool(H_L, lb)
        pp = ctx["pocket_pool"]
        if pp.shape[0] == 1 and lig_pool.shape[0] > 1:
            pp = pp.expand(lig_pool.shape[0], -1)
        return self.head(torch.cat([lig_pool, pp], dim=-1))


class DualEncoderNet(nn.Module):
    """B5: dot-product / bilinear dual encoder (fast-retrieval family)."""

    def __init__(self, dim=128, pocket_layers=3, ligand_layers=3,
                 dropout=0.1):
        super().__init__()
        self.pocket_encoder = PocketEncoder(dim, pocket_layers, dropout)
        self.ligand_encoder = LigandEncoder(dim, ligand_layers, dropout)
        self.W_l = nn.Linear(dim, dim)
        self.W_p = nn.Linear(dim, dim)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        self.score_head = nn.Linear(dim, 1)

    def encode_pocket(self, pb):
        H_P = self.pocket_encoder(pb["x"], pb["edge_index"], pb["edge_attr"])
        H_pad, mask = pad_flat(H_P, pb["batch"], pb["n_res"], pb["n_graphs"])
        pool = masked_mean(H_pad, mask)
        return {"pocket_vec": self.W_p(pool)}

    def forward(self, ctx, lb, target_idx=None) -> dict:
        H_L = self.ligand_encoder(lb["x"], lb["edge_index"], lb["edge_attr"])
        _, _, lig_pool = _ligand_pool(H_L, lb)
        lv = self.W_l(lig_pool)
        pv = ctx["pocket_vec"]
        if pv.shape[0] == 1 and lv.shape[0] > 1:
            pv = pv.expand(lv.shape[0], -1)
        logit = (lv * pv).sum(-1) / self.logit_scale.clamp(min=1e-3)
        score = self.score_head(lig_pool * pv).squeeze(-1)
        return {"logit": logit, "score": score}
