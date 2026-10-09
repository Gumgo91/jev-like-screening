"""Screening driver with pocket-cache timing instrumentation (plan 11).

Separately measures: pocket encode time (once/target vs per-batch),
ligand encode+interaction time, featurization time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd
import torch

from pocketgate.data.graphbatch import collate_ligands, collate_pockets
from pocketgate.engine import move


@dataclass
class ScreenTiming:
    pocket_encode_s: float = 0.0
    pocket_encode_calls: int = 0
    ligand_score_s: float = 0.0
    ligand_score_calls: int = 0
    n_ligands: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class Screener:
    """cache_mode: 'per_target' (encode once per target, the cached path)
    or 'per_batch' (recompute pocket ctx every batch, the uncached path)."""

    def __init__(self, model, pockets: dict, ligs: dict, device: str,
                 cache_mode: str = "per_target", pocket_override: dict | None = None):
        assert cache_mode in ("per_target", "per_batch")
        self.m = model
        self.pockets = pockets
        self.ligs = ligs
        self.dev = device
        self.cache_mode = cache_mode
        self.pocket_override = pocket_override or {}
        self._ctx_cache: dict[str, dict] = {}
        self.timing = ScreenTiming()

    def _encode(self, target_id: str):
        t0 = time.time()
        if self.dev == "cuda":
            torch.cuda.synchronize()
        pb = collate_pockets([self.pockets[target_id]])
        ctx = self.m.encode_pocket(move(pb, self.dev))
        if self.dev == "cuda":
            torch.cuda.synchronize()
        self.timing.pocket_encode_s += time.time() - t0
        self.timing.pocket_encode_calls += 1
        return ctx

    def _ctx(self, target_id: str):
        if self.cache_mode == "per_batch":
            return self._encode(self.pocket_override.get(target_id, target_id))
        key = self.pocket_override.get(target_id, target_id)
        if key in self._ctx_cache:
            self.timing.cache_hits += 1
            return self._ctx_cache[key]
        self.timing.cache_misses += 1
        ctx = self._encode(key)
        self._ctx_cache[key] = ctx
        return ctx

    def score(self, df: pd.DataFrame, batch_size: int = 256,
              target_idx_of: dict | None = None) -> pd.DataFrame:
        """df: target_id, ligand_id."""
        self.m.eval()
        out_rows = []
        with torch.no_grad():
            for t, sub in df.groupby("target_id", sort=False):
                ids = sub["ligand_id"].tolist()
                for i in range(0, len(ids), batch_size):
                    ctx = self._ctx(t)
                    lb = collate_ligands(
                        [self.ligs[x] for x in ids[i:i + batch_size]])
                    if self.dev == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.time()
                    tidx = (torch.full((len(lb["n_atoms"]),),
                                       target_idx_of[t], dtype=torch.long,
                                       device=self.dev)
                            if target_idx_of is not None else None)
                    out = self.m(ctx, move(lb, self.dev), target_idx=tidx)
                    p = torch.sigmoid(out["logit"]).cpu().numpy()
                    if self.dev == "cuda":
                        torch.cuda.synchronize()
                    self.timing.ligand_score_s += time.time() - t0
                    self.timing.ligand_score_calls += 1
                    self.timing.n_ligands += len(ids[i:i + batch_size])
                    out_rows.append(pd.DataFrame({
                        "target_id": t,
                        "ligand_id": ids[i:i + batch_size],
                        "logit": out["logit"].cpu().numpy(),
                        "score_pred": out["score"].cpu().numpy(),
                        "p_hit": p,
                    }))
        return pd.concat(out_rows, ignore_index=True)
