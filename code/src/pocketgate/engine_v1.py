"""V1 trainer: B5 backbone + optional pairwise ranking supervision.

Adds to the v0 recipe only a ranking batch per update:
- mode 'none':     L_base only (C0)
- mode 'within':   within-target pairs (A,i,j), |d_A| >= delta (C1)
- mode 'reversal': quadruplets (A,B,i,j) with d_A*d_B<0 and margin
                   >= delta (C2, and C3 with a constant pocket input)

Per update the ranking batch contributes 4*Q forward pairs and
2*Q loss terms for 'reversal' (Q quadruplets); 'within' uses 2*Q pairs
to match both counts exactly.

Everything is logged per update (bce, huber, l_rev, grad_norm) and at
each eval (cold/warm macro recall_1@10). Best checkpoint + optimizer
state are saved at the best cold eval; early stop = patience evals.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pocketgate.data.graphbatch import collate_ligands, collate_pockets
from pocketgate.engine import move


@dataclass
class TrainConfigV1:
    dim: int = 128
    seed: int = 11
    batch_size: int = 256          # base batch (effective, no accum)
    quads_per_update: int = 64     # reversal quads; within uses 2x pairs
    lam: float = 0.1               # weight of ranking loss
    delta: float = 1.0
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_updates: int = 60000
    eval_every_updates: int = 6094  # ~1 data-epoch at 256 bs / 1.56M pairs
    patience_evals: int = 3
    aux_weight: float = 0.2
    device: str = "cuda"
    ckpt_path: str = "best.pt"
    optim_path: str = "best.optim.pt"


class TrainerV1:
    """B5-only trainer with optional pairwise ranking supervision."""

    def __init__(self, model, cfg: TrainConfigV1, pocket_graphs: dict,
                 ligand_graphs: dict):
        self.m = model
        self.cfg = cfg
        self.pockets = pocket_graphs
        self.ligs = ligand_graphs
        self.dev = cfg.device
        self.m.to(self.dev)

    # ---- encoders reused for pair-indexed logits ---------------------
    def _lig_vecs(self, ligand_ids: list[str]):
        lb = move(collate_ligands([self.ligs[x] for x in ligand_ids]),
                  self.dev)
        H_L = self.m.ligand_encoder(lb["x"], lb["edge_index"],
                                    lb["edge_attr"])
        from pocketgate.data.graphbatch import pad_flat
        from pocketgate.models.mpnn import masked_mean
        H_pad, mask = pad_flat(H_L, lb["batch"], lb["n_atoms"],
                               lb["n_graphs"])
        return self.m.W_l(masked_mean(H_pad, mask))

    def _poc_vecs(self, target_ids: list[str]):
        pb = move(collate_pockets([self.pockets[t] for t in target_ids]),
                  self.dev)
        ctx = self.m.encode_pocket(pb)
        return ctx["pocket_vec"]            # (B,d) after W_p

    def _logits(self, lv, pv, t_idx, l_idx):
        scale = self.m.logit_scale.clamp(min=1e-3)
        return (lv[l_idx] * pv[t_idx]).sum(-1) / scale

    def train(self, pairs: pd.DataFrame, scaler: dict,
              rank_batch: pd.DataFrame | None, mode: str,
              dev_eval_fn):
        """pairs: target_id, ligand_id, y, score (train roles only).
        rank_batch: mode 'within' -> cols target_a, ligand_i, ligand_j,
        d_a; mode 'reversal' -> target_a, target_b, ligand_i, ligand_j,
        d_a, d_b."""
        cfg = self.cfg
        assert mode in ("none", "within", "reversal")
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

        rb = rank_batch.reset_index(drop=True) if rank_batch is not None \
            else None
        hist = []
        evals = []
        updates = 0
        evals_since_best = 0
        best_metric = -1.0
        best_update = 0
        t0 = time.time()
        stop_reason = "budget"
        self.m.train()
        n_rank_fw = 0
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

            l_rev = torch.zeros((), device=self.dev)
            if mode == "reversal" and rb is not None:
                Q = cfg.quads_per_update
                qi = rng.integers(0, len(rb), size=Q)
                qd = rb.iloc[qi]
                tids = pd.concat([qd["target_a"], qd["target_b"]]) \
                    .tolist()
                lids = pd.concat([qd["ligand_i"], qd["ligand_j"]]) \
                    .tolist()
                tu = {t: k for k, t in enumerate(sorted(set(tids)))}
                lu = {l: k for k, l in enumerate(sorted(set(lids)))}
                pv = self._poc_vecs(list(tu.keys()))
                lv = self._lig_vecs(list(lu.keys()))
                ia = torch.tensor([tu[t] for t in qd["target_a"]],
                                  device=self.dev)
                ib = torch.tensor([tu[t] for t in qd["target_b"]],
                                  device=self.dev)
                ii = torch.tensor([lu[l] for l in qd["ligand_i"]],
                                  device=self.dev)
                ij = torch.tensor([lu[l] for l in qd["ligand_j"]],
                                  device=self.dev)
                fAi = self._logits(lv, pv, ia, ii)
                fAj = self._logits(lv, pv, ia, ij)
                fBi = self._logits(lv, pv, ib, ii)
                fBj = self._logits(lv, pv, ib, ij)
                qA, qB = fAi - fAj, fBi - fBj
                # score direction: lower docking score = better ligand.
                # d_A = S_i - S_j > 0 means j is better, so the model must
                # output f_i < f_j, i.e. sign(q_A) = -sign(d_A).
                yA = -torch.tensor(np.sign(qd["d_a"].to_numpy()),
                                   dtype=torch.float32, device=self.dev)
                yB = -torch.tensor(np.sign(qd["d_b"].to_numpy()),
                                   dtype=torch.float32, device=self.dev)
                l_rev = 0.5 * (F.softplus(-yA * qA)
                               + F.softplus(-yB * qB)).mean()
                n_rank_fw += 4 * Q
            elif mode == "within" and rb is not None:
                P = 2 * cfg.quads_per_update
                qi = rng.integers(0, len(rb), size=P)
                qd = rb.iloc[qi]
                tids = qd["target_a"].tolist()
                lids = pd.concat([qd["ligand_i"], qd["ligand_j"]]) \
                    .tolist()
                tu = {t: k for k, t in enumerate(sorted(set(tids)))}
                lu = {l: k for k, l in enumerate(sorted(set(lids)))}
                pv = self._poc_vecs(list(tu.keys()))
                lv = self._lig_vecs(list(lu.keys()))
                ia = torch.tensor([tu[t] for t in qd["target_a"]],
                                  device=self.dev)
                ii = torch.tensor([lu[l] for l in qd["ligand_i"]],
                                  device=self.dev)
                ij = torch.tensor([lu[l] for l in qd["ligand_j"]],
                                  device=self.dev)
                q = self._logits(lv, pv, ia, ii) \
                    - self._logits(lv, pv, ia, ij)
                # same score-direction convention as 'reversal'
                yA = -torch.tensor(np.sign(qd["d_a"].to_numpy()),
                                   dtype=torch.float32, device=self.dev)
                l_rev = F.softplus(-yA * q).mean()
                n_rank_fw += 2 * P

            loss = loss + cfg.lam * l_rev
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
                         "l_rev": float(l_rev.detach().cpu()),
                         "grad_norm": float(gn.detach().cpu())})

            if updates % cfg.eval_every_updates == 0:
                met = dev_eval_fn()   # dict with 'cold' and 'warm' macro
                met["update"] = updates
                evals.append(met)
                # preserve EVERY eval checkpoint + optimizer state
                tag = f"_upd{updates}"
                torch.save(self.m.state_dict(),
                           cfg.ckpt_path.replace(".pt", f"{tag}.pt"))
                torch.save(opt.state_dict(),
                           cfg.optim_path.replace(".pt", f"{tag}.pt"))
                if met["cold_recall_1@10"] > best_metric:
                    best_metric = met["cold_recall_1@10"]
                    best_update = updates
                    evals_since_best = 0
                    torch.save(self.m.state_dict(), cfg.ckpt_path)
                    torch.save(opt.state_dict(), cfg.optim_path)
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
            "rank_forward_pairs": n_rank_fw,
            "best_metric": best_metric, "best_update": best_update,
            "stop_reason": stop_reason,
            "wall_seconds": time.time() - t0,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 1e6
            if self.dev == "cuda" else 0.0,
            "loss_curve": hist, "eval_curve": evals,
        }
