"""Pocket cache (plan 6.5).

PocketCache stores per-target, ligand-independent tensors:
  H_P (padded), pocket_mask, per-layer K/V, pocket_pool.

Cache key = sha256(receptor_hash | box_hash | preparation_version |
featurizer_version | checkpoint_hash | precision). Any change ->
different key -> invalidate. Entries live in an in-memory dict plus an
optional directory of .pt files.
"""
from __future__ import annotations

from pathlib import Path

import torch


def pocket_cache_key(receptor_hash: str, box_hash: str,
                     preparation_version: str, featurizer_version: str,
                     checkpoint_hash: str, precision: str = "fp32") -> str:
    import hashlib
    s = "|".join([receptor_hash, box_hash, preparation_version,
                  featurizer_version, checkpoint_hash, precision])
    return hashlib.sha256(s.encode()).hexdigest()


class PocketCacheStore:
    def __init__(self, checkpoint_hash: str, cache_dir: Path | None = None,
                 precision: str = "fp32"):
        self.checkpoint_hash = checkpoint_hash
        self.precision = precision
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._mem: dict[str, dict] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"hit": 0, "miss": 0}

    def _key(self, receptor_hash: str, box_hash: str, preparation_version: str,
             featurizer_version: str) -> str:
        return pocket_cache_key(
            receptor_hash, box_hash, preparation_version,
            featurizer_version, self.checkpoint_hash, self.precision)

    def get(self, receptor_hash, box_hash, preparation_version,
            featurizer_version) -> dict | None:
        key = self._key(receptor_hash, box_hash, preparation_version,
                        featurizer_version)
        if key in self._mem:
            self.stats["hit"] += 1
            return self._mem[key]
        if self.cache_dir:
            p = self.cache_dir / f"{key}.pt"
            if p.exists():
                ctx = torch.load(p, weights_only=True)
                self._mem[key] = ctx
                self.stats["hit"] += 1
                return ctx
        self.stats["miss"] += 1
        return None

    def put(self, ctx: dict, receptor_hash, box_hash, preparation_version,
            featurizer_version) -> str:
        key = self._key(receptor_hash, box_hash, preparation_version,
                        featurizer_version)
        self._mem[key] = ctx
        if self.cache_dir:
            torch.save(ctx, self.cache_dir / f"{key}.pt")
        return key
