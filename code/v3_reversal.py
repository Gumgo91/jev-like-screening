"""Registered outcome 2 of addendum 8: reversal accuracy of the joint networks on the frozen quadruplets.

The test value is computed from the saved test predictions (runs/v3/test_predictions_<tag>.parquet) and the frozen quadruplet file of the
preceding analysis (runs/v2/test_quads_cold.parquet); no label vault is opened, so the count of reads of the sealed test labels does not change.
The development value scores the development frame with the frozen checkpoints and reads runs/v1/eval_quads_dev.parquet only.
Writes dockmut/eval/joint_reversal.json.
Usage: python dockmut/v3_reversal.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dockmut"))
from pocketgate.common import PROCESSED_DIR  # noqa: E402
from v3_eval import COND, OUT, predict, runs  # noqa: E402


def joint_reversal(pred: pd.DataFrame, quads: pd.DataFrame) -> dict:
    key = pred.set_index(["target_id", "ligand_id"])["p_hit"]
    q = quads.copy()
    for tgt, lig, col in [("target_a", "ligand_i", "fAi"), ("target_a", "ligand_j", "fAj"),
                          ("target_b", "ligand_i", "fBi"), ("target_b", "ligand_j", "fBj")]:
        q[col] = key.reindex(pd.MultiIndex.from_frame(q[[tgt, lig]])).to_numpy()
    q = q.dropna(subset=["fAi", "fAj", "fBi", "fBj"])
    qa, qb = q.fAi - q.fAj, q.fBi - q.fBj
    ok_a, ok_b = np.sign(qa) == -np.sign(q.d_a), np.sign(qb) == -np.sign(q.d_b)
    return {"joint": float((ok_a & ok_b).mean()), "n_quads": int(len(q)), "n_total": int(len(quads))}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads((ROOT / "configs/g2_config.json").read_text())
    tg = pd.read_parquet(ROOT / "data/splits/targets.parquet")
    reg = runs()
    res = {"test": {}, "dev": {}}
    q_test = pd.read_parquet(ROOT / "runs/v2/test_quads_cold.parquet")
    for tag in reg:
        pred = pd.read_parquet(OUT / f"test_predictions_{tag}.parquet")
        res["test"][tag] = joint_reversal(pred, q_test)
        print("test", tag, res["test"][tag], flush=True)
    ligs = torch.load(PROCESSED_DIR / "g2_ligand_graphs.pt", weights_only=False)
    pockets = torch.load(PROCESSED_DIR / "v1_pocket_graphs.pt", weights_only=False)
    dev_t = tg[tg.split_role == "dev"]["target_id"].tolist()
    dev_ligs = [l for l in cfg["dev_ligands"] if l in ligs]
    frame = pd.DataFrame([(t, l) for t in dev_t for l in dev_ligs], columns=["target_id", "ligand_id"])
    q_dev = pd.read_parquet(ROOT / "runs/v1/eval_quads_dev.parquet")
    for tag, r in reg.items():
        pred = predict(r["model"], OUT / f"ckpt_v3_{tag}.pt", ligs, pockets, frame, dev)
        res["dev"][tag] = joint_reversal(pred, q_dev)
        print("dev", tag, res["dev"][tag], flush=True)
    for split in ("dev", "test"):
        for kind, cond in (("xattn", "XA"), ("pooled", "PC")):
            v = [res[split][t]["joint"] for t in reg if t.startswith(kind)]
            res[split][f"mean_{cond}"] = float(np.mean(v))
    (ROOT / "dockmut/eval/joint_reversal.json").write_text(json.dumps(res, indent=1))
    print("mean dev", res["dev"]["mean_XA"], res["dev"]["mean_PC"], "test", res["test"]["mean_XA"], res["test"]["mean_PC"])


if __name__ == "__main__":
    main()
