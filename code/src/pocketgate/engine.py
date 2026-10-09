"""Training and inference engine (plan 7).

- Loss: BCEWithLogits(a, y) + aux_weight * Huber(s_hat, s_std).
  Score scaler is global over P_train x L_train only (plan 7.1).
- Sampling: target-balanced - each batch draws one train target and
  batch_size ligands from its train pool, so the pocket is encoded once
  per batch with shared gradient (plan 6.5, allowed: no detach).
- Early stopping on development macro Recall_1%@10% (P_train x L_dev +
  P_dev x L_dev, or a caller-provided frame).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pocketgate.data.graphbatch import collate_ligands, collate_pockets


@dataclass
class TrainConfig:
    dim: int = 128
    seed: int = 11
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_updates: int = 10000
    max_epochs: int = 10
    eval_every_updates: int = 2500
    patience_rounds: int = 3
    aux_weight: float = 0.2
    device: str = "cuda"
    deterministic: bool = True
    ckpt_path: str = "best.pt"


@dataclass
class TrainLog:
    updates: int = 0
    epochs: float = 0.0
    train_examples: int = 0
    wall_seconds: float = 0.0
    evals: list = field(default_factory=list)
    best_metric: float = -1.0
    best_update: int = 0
    peak_vram_mb: float = 0.0
    stop_reason: str = ""


def score_scaler(train_df: pd.DataFrame) -> dict:
    s = train_df["score"].dropna()
    return {"mean": float(s.mean()), "std": float(s.std())}


def hit_labels(train_df: pd.DataFrame, q: float = 0.01) -> pd.Series:
    """tau_P per target from TRAIN ligands only; y = 1[score <= tau]."""
    tau = train_df.dropna(subset=["score"]).groupby("target_id")["score"] \
        .quantile(q)
    return train_df["target_id"].map(tau).rename("tau"), tau


def move(d: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in d.items()}


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig,
                 pocket_graphs: dict, ligand_graphs: dict):
        self.m = model
        self.cfg = cfg
        self.pockets = pocket_graphs  # target_id -> graph dict
        self.ligs = ligand_graphs     # ligand_id -> graph dict
        self.dev = cfg.device
        self.m.to(self.dev)

    def pocket_ctx(self, target_id: str):
        pb = collate_pockets([self.pockets[target_id]])
        return self.m.encode_pocket(move(pb, self.dev))

    def predict_pairs(self, df: pd.DataFrame, batch_size: int = 256,
                      use_cache=None) -> pd.DataFrame:
        """df: target_id, ligand_id. Returns df + logit, p_hit, score."""
        self.m.eval()
        rows = []
        with torch.no_grad():
            for t, sub in df.groupby("target_id", sort=False):
                if use_cache is not None:
                    ctx = use_cache.get_ctx(t, lambda: self.pocket_ctx(t))
                else:
                    ctx = self.pocket_ctx(t)
                ids = sub["ligand_id"].tolist()
                for i in range(0, len(ids), batch_size):
                    lb = collate_ligands(
                        [self.ligs[x] for x in ids[i:i + batch_size]])
                    out = self.m(ctx, move(lb, self.dev))
                    rows.append(pd.DataFrame({
                        "target_id": t,
                        "ligand_id": ids[i:i + batch_size],
                        "logit": out["logit"].cpu().numpy(),
                        "score_pred": out["score"].cpu().numpy(),
                        "p_hit": torch.sigmoid(out["logit"]).cpu().numpy(),
                    }))
        return pd.concat(rows, ignore_index=True)

    def train(self, pairs: pd.DataFrame, dev_eval_fn=None,
              target_idx_of: dict | None = None,
              scaler: dict | None = None):
        """pairs: target_id, ligand_id, y, score."""
        cfg = self.cfg
        opt = torch.optim.AdamW(self.m.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        bce = nn.BCEWithLogitsLoss()
        hub = nn.HuberLoss()
        log = TrainLog()
        smean, sstd = scaler["mean"], scaler["std"]

        by_target = {t: g.reset_index(drop=True)
                     for t, g in pairs.groupby("target_id")}
        tgt_list = sorted(by_target)
        rng = np.random.default_rng(cfg.seed)
        updates = 0
        t0 = time.time()
        evals_since_best = 0
        n_pairs = len(pairs)
        epoch_of = lambda u: u * cfg.batch_size / n_pairs

        torch.manual_seed(cfg.seed)
        self.m.train()
        while updates < cfg.max_updates and epoch_of(updates) < cfg.max_epochs:
            t = tgt_list[rng.integers(len(tgt_list))]
            pool = by_target[t]
            idx = rng.integers(0, len(pool), size=min(cfg.batch_size,
                                                     len(pool)))
            batch = pool.iloc[idx]
            lb = move(collate_ligands(
                [self.ligs[x] for x in batch["ligand_id"]]), self.dev)
            ctx = self.pocket_ctx(t)
            tidx = (torch.full((len(batch),), target_idx_of[t],
                               dtype=torch.long, device=self.dev)
                    if target_idx_of is not None else None)
            out = self.m(ctx, lb, target_idx=tidx)
            y = torch.from_numpy(batch["y"].to_numpy(np.float32)).to(self.dev)
            loss = bce(out["logit"], y)
            smask = batch["score"].notna().to_numpy()
            if cfg.aux_weight > 0 and smask.any():
                s_true = torch.from_numpy(
                    (batch["score"].to_numpy(np.float32) - smean) / sstd
                ).to(self.dev)
                loss = loss + cfg.aux_weight * hub(
                    out["score"][torch.from_numpy(smask).to(self.dev)],
                    s_true[torch.from_numpy(smask).to(self.dev)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            updates += 1
            log.train_examples += len(batch)

            if dev_eval_fn and updates % cfg.eval_every_updates == 0:
                metric = dev_eval_fn()
                log.evals.append({"update": updates, "metric": metric})
                if metric > log.best_metric:
                    log.best_metric, log.best_update = metric, updates
                    evals_since_best = 0
                    torch.save(self.m.state_dict(), cfg.ckpt_path)
                else:
                    evals_since_best += 1
                self.m.train()
                if evals_since_best >= cfg.patience_rounds:
                    log.stop_reason = "early_stop_patience"
                    break
        else:
            log.stop_reason = "budget"

        if not log.stop_reason:
            log.stop_reason = "budget"
        log.updates = updates
        log.epochs = epoch_of(updates)
        log.wall_seconds = time.time() - t0
        if self.dev == "cuda":
            log.peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
        return log
