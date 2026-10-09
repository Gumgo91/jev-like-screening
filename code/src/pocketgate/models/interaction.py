"""Ligand->pocket cross-attention with cacheable protein-side K/V.

Per plan 6.4: protein memory (K_P, V_P per layer) does not depend on the
ligand and can be cached per target. Ligand-side representation is
updated through residual + LayerNorm + FFN; protein state is never
updated as a function of ligands (no bidirectional recompute, no
ligand-ligand attention).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnLayer(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dh = dim // heads
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.W_o = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.drop = nn.Dropout(dropout)

    def kv(self, H_P: torch.Tensor):
        """H_P (B,M,d) or (M,d) -> K,V shaped (B?,heads,M,dh)."""
        squeeze = H_P.dim() == 2
        if squeeze:
            H_P = H_P.unsqueeze(0)
        B, M, d = H_P.shape
        K = self.W_k(H_P).view(B, M, self.heads, self.dh).transpose(1, 2)
        V = self.W_v(H_P).view(B, M, self.heads, self.dh).transpose(1, 2)
        if squeeze:
            K, V = K[0], V[0]
        return K, V

    def forward(self, H_L: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                pocket_mask: torch.Tensor | None = None,
                ligand_mask: torch.Tensor | None = None) -> torch.Tensor:
        """H_L (B,A,d); K,V (B,h,M,dh) or (h,M,dh) broadcast; masks bool."""
        B, A, d = H_L.shape
        h, dh = self.heads, self.dh
        Q = self.W_q(H_L).view(B, A, h, dh).transpose(1, 2)  # (B,h,A,dh)
        if K.dim() == 3:
            K = K.unsqueeze(0).expand(B, -1, -1, -1)
            V = V.unsqueeze(0).expand(B, -1, -1, -1)
        attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dh)  # (B,h,A,M)
        if pocket_mask is not None:
            if pocket_mask.dim() == 1:
                pocket_mask = pocket_mask.unsqueeze(0).expand(B, -1)
            attn = attn.masked_fill(~pocket_mask[:, None, None, :],
                                    float("-inf"))
        attn = F.softmax(attn, dim=-1)
        C = torch.matmul(attn, V)  # (B,h,A,dh)
        C = C.transpose(1, 2).reshape(B, A, d)
        out = self.ln1(H_L + self.drop(self.W_o(C)))
        out = self.ln2(out + self.drop(self.ffn(out)))
        if ligand_mask is not None:
            out = out * ligand_mask.unsqueeze(-1).to(out.dtype)
        return out


class InteractionStack(nn.Module):
    def __init__(self, dim: int = 128, layers: int = 2, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttnLayer(dim, heads, dropout) for _ in range(layers)]
        )

    def kv_all(self, H_P_padded: torch.Tensor):
        return [lyr.kv(H_P_padded) for lyr in self.layers]

    def forward(self, H_L, kv_list, pocket_mask=None, ligand_mask=None):
        for lyr, (K, V) in zip(self.layers, kv_list):
            H_L = lyr(H_L, K, V, pocket_mask=pocket_mask,
                      ligand_mask=ligand_mask)
        return H_L
