"""Trainer of the dual-encoder networks (real pockets and constant pocket).

Per update: one train target is drawn uniformly, batch_size of its ligands are
drawn with replacement, the pocket is encoded once, and the loss is
BCEWithLogits(logit, y) + aux_weight * Huber(score_hat, standardized score).

Everything is logged per update (bce, huber, grad_norm) and at each
evaluation (cold and warm macro Recall_1%@10%). The best checkpoint is saved
at the best cold evaluation; training stops after patience_evals evaluations
without improvement or at max_updates.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pocketgate.data.graphbatch import collate_ligands, collate_pockets
from pocketgate.engine import move


@dataclass
class TrainConfigV1:
    dim: int = 128
    seed: int = 11
    batch_size: int = 256          # base batch (effective, no accum)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_updates: int = 60000
    eval_every_updates: int = 6094  # ~1 data-epoch at 256 bs / 1.56M pairs
    patience_evals: int = 3
    aux_weight: float = 0.2
    device: str = "cuda"
    ckpt_path: str = "best.pt"


class TrainerV1:
    """Trainer of the dual-encoder baselines."""

    def __init__(self, model, cfg: TrainConfigV1, pocket_graphs: dict,
                 ligand_graphs: dict):
        self.m = model
        self.cfg = cfg
        self.pockets = pocket_graphs
        self.ligs = ligand_graphs
        self.dev = cfg.device
        self.m.to(self.dev)

    def train(self, pairs: pd.DataFrame, scaler: dict, dev_eval_fn):
        """pairs: target_id, ligand_id, y, score (train roles only).
        dev_eval_fn() -> dict with 'cold_recall_1@10' and 'warm_recall_1@10'."""
        cfg = self.cfg
        opt = torch.optim.AdamW(self.m.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        bce = nn.BCEWithLogitsLoss()
        hub = nn.HuberLoss()
        smean, sstd = scaler["mean"], scaler["std"]

        by_target = {t: g.reset_index(drop=True)
                     for t, g in pairs.groupby("target_id")}
        tgt_list = sorted(by_target)
        rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        n_pairs = len(pairs)
        epoch_of = lambda u: u * cfg.batch_size / n_pairs

        hist = []
        evals = []
        updates = 0
        evals_since_best = 0
        best_metric = -1.0
        best_update = 0
        t0 = time.time()
        stop_reason = "budget"
        self.m.train()
        while updates < cfg.max_updates:
            t = tgt_list[rng.integers(len(tgt_list))]
            pool = by_target[t]
            idx = rng.integers(0, len(pool), size=cfg.batch_size)
            batch = pool.iloc[idx]
            lb = move(collate_ligands(
                [self.ligs[x] for x in batch["ligand_id"]]), self.dev)
            ctx = self.m.encode_pocket(
                move(collate_pockets([self.pockets[t]]), self.dev))
            out = self.m(ctx, lb)
            y = torch.from_numpy(batch["y"].to_numpy(np.float32)) \
                .to(self.dev)
            l_bce = bce(out["logit"], y)
            smask = torch.from_numpy(batch["score"].notna().to_numpy()) \
                .to(self.dev)
            s_true = torch.from_numpy(
                (batch["score"].fillna(0).to_numpy(np.float32) - smean)
                / sstd).to(self.dev)
            l_hub = hub(out["score"][smask], s_true[smask])
            loss = l_bce + cfg.aux_weight * l_hub

            opt.zero_grad()
            loss.backward()
            gn = torch.sqrt(sum(p.grad.detach().pow(2).sum()
                                for p in self.m.parameters()
                                if p.grad is not None))
            opt.step()
            updates += 1
            hist.append({"update": updates, "epoch": epoch_of(updates),
                         "bce": float(l_bce.detach().cpu()),
                         "huber": float(l_hub.detach().cpu()),
                         "grad_norm": float(gn.detach().cpu())})

            if updates % cfg.eval_every_updates == 0:
                met = dev_eval_fn()   # dict with 'cold' and 'warm' macro
                met["update"] = updates
                evals.append(met)
                if met["cold_recall_1@10"] > best_metric:
                    best_metric = met["cold_recall_1@10"]
                    best_update = updates
                    evals_since_best = 0
                    torch.save(self.m.state_dict(), cfg.ckpt_path)
                else:
                    evals_since_best += 1
                self.m.train()
                print(f"  upd {updates} cold={met['cold_recall_1@10']:.3f} "
                      f"warm={met['warm_recall_1@10']:.3f}", flush=True)
                if evals_since_best >= cfg.patience_evals:
                    stop_reason = "early_stop_patience"
                    break

        return {
            "updates": updates, "epochs": epoch_of(updates),
            "train_examples": updates * cfg.batch_size,
            "best_metric": best_metric, "best_update": best_update,
            "stop_reason": stop_reason,
            "wall_seconds": time.time() - t0,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 1e6
            if self.dev == "cuda" else 0.0,
            "loss_curve": hist, "eval_curve": evals,
        }
