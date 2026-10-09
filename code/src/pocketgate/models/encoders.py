"""Pocket and ligand encoders (thin wrappers over shared GraphEncoder)."""
from __future__ import annotations

from pocketgate.data.features import (ATOM_DIM, BOND_DIM, POCKET_EDGE_DIM,
                                      POCKET_NODE_DIM)

from .mpnn import GraphEncoder


class PocketEncoder(GraphEncoder):
    def __init__(self, dim=128, layers=3, dropout=0.1):
        super().__init__(POCKET_NODE_DIM, POCKET_EDGE_DIM, dim, layers,
                         dropout)


class LigandEncoder(GraphEncoder):
    def __init__(self, dim=128, layers=3, dropout=0.1):
        super().__init__(ATOM_DIM, BOND_DIM, dim, layers, dropout)
