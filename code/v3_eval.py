"""Evaluation of the joint networks trained by scripts/v3_train_joint.py (addendum 8).

dev   : top-1% recall metrics on the 10 development targets, per checkpoint (the checkpoint rule used these targets).
test  : after the checkpoints are frozen and hashed (runs/v3/v3_registry.json), the test frame is scored first and the sealed
        test labels are read ONCE (guarded interface, caller v3_test_eval) for all joint networks together.
Writes runs/v3/metrics_<COND>_s<seed>_cold.csv and runs/v3/test_metrics_<COND>_s<seed>_cold.csv (COND = XA for the cross-attention
network, PC for the pooled-concatenation network), the same per-target tables as runs/v2, so that wt_budget_summary.py can read them.
Usage: python dockmut/v3_eval.py dev | test
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketgate.common import PROCESSED_DIR  # noqa: E402
from pocketgate.data import labels as L  # noqa: E402
from pocketgate.evaluation.metrics import macro_eval  # noqa: E402
from pocketgate.inference.screen import Screener  # noqa: E402
from pocketgate.models.pocketgate import PocketGate, PooledConcatNet  # noqa: E402

OUT = ROOT / "runs/v3"
COND = {"xattn": ("XA", PocketGate), "pooled": ("PC", PooledConcatNet)}


def runs():
    reg = {}
    for f in sorted(OUT.glob("registry_*.json")):
        reg[f.stem.replace("registry_", "")] = json.loads(f.read_text())
    return reg


def predict(kind, ckpt, ligs, pockets, frame, dev):
    net = COND[kind][1](dim=128)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net = net.to(dev).eval()
    scr = Screener(net, pockets, ligs, dev, cache_mode="per_target")
    return scr.score(frame, batch_size=512)


def main():
    mode = sys.argv[1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads((ROOT / "configs/g2_config.json").read_text())
    tg = pd.read_parquet(ROOT / "data/splits/targets.parquet")
    pockets = torch.load(PROCESSED_DIR / "v1_pocket_graphs.pt", weights_only=False)
    reg = runs()
    if mode == "dev":
        ligs = torch.load(PROCESSED_DIR / "g2_ligand_graphs.pt", weights_only=False)
        dev_t = tg[tg.split_role == "dev"]["target_id"].tolist()
        dev_ligs = [l for l in cfg["dev_ligands"] if l in ligs]
        frame = pd.DataFrame([(t, l) for t in dev_t for l in dev_ligs], columns=["target_id", "ligand_id"])
        lab = pd.concat([L.load_labels("train", "dev", caller="v3_eval_dev"), L.load_labels("dev", "dev", caller="v3_eval_dev")])
        lab = lab[lab.ligand_id.isin(dev_ligs)].merge(frame, on=["target_id", "ligand_id"])
        for tag, r in reg.items():
            cond = COND[r["model"]][0]
            pred = predict(r["model"], OUT / f"ckpt_v3_{tag}.pt", ligs, pockets, frame, dev)
            me = macro_eval(pred.merge(lab, on=["target_id", "ligand_id"]).dropna(subset=["score", "p_hit"]))
            me.to_csv(OUT / f"metrics_{cond}_s{r['seed']}_cold.csv", index=False)
            print(tag, "dev recall@10%", round(float(me["recall_1@10"].mean()), 4), flush=True)
    else:
        for tag, r in reg.items():                                   # the checkpoints must match the registry
            sha = hashlib.sha256((OUT / f"ckpt_v3_{tag}.pt").read_bytes()).hexdigest()
            assert sha == r["ckpt_sha256"], tag
        ligs = torch.load(PROCESSED_DIR / "v2_test_ligand_graphs.pt", weights_only=False)
        pockets.update(torch.load(PROCESSED_DIR / "v2_pocket_graphs.pt", weights_only=False))
        test_t = tg[tg.split_role == "test"]["target_id"].tolist()
        frame = pd.DataFrame([(t, l) for t in test_t for l in ligs], columns=["target_id", "ligand_id"])
        preds = {tag: predict(r["model"], OUT / f"ckpt_v3_{tag}.pt", ligs, pockets, frame, dev) for tag, r in reg.items()}
        for tag in preds:
            preds[tag].to_parquet(OUT / f"test_predictions_{tag}.parquet", index=False)
        lab = L.load_labels("test", "test", caller="v3_test_eval", allow_test=True)          # the only read of the test labels in this step
        (OUT / "test_exposure_ledger.json").write_text(json.dumps({"caller": "v3_test_eval", "cells": ["test_test"], "models": list(reg),
                                                                   "ts": pd.Timestamp.now("UTC").isoformat()}, indent=1))
        for tag, r in reg.items():
            cond = COND[r["model"]][0]
            me = macro_eval(preds[tag].merge(lab, on=["target_id", "ligand_id"]).dropna(subset=["score", "p_hit"]))
            me.to_csv(OUT / f"test_metrics_{cond}_s{r['seed']}_cold.csv", index=False)
            print(tag, "test recall@10%", round(float(me["recall_1@10"].mean()), 4), flush=True)


if __name__ == "__main__":
    main()
