"""Known-pocket evaluation of the joint-network comparison: dual encoder, constant-pocket dual encoder, pooled network, cross-attention.

The 39 training targets x the 10,000 development ligands (labels of the cell train x dev). Networks: dual encoder (C0, seeds 11/22/33),
dual encoder with one constant pocket (C0c), pooled-concatenation network (PC) and cross-attention network (XA). Reads the
checkpoints of runs/v1, runs/v2 and runs/v3, writes runs/v5/metrics_<COND>_s<seed>_known.csv (macro_eval per target, skipped when it
exists) and eval/v5_known_pockets_summary.json (macro means, bootstrap intervals, paired differences). Only the label cell
train x dev is read.

Usage (repository root): python scripts/v5_known_pockets_eval.py [--device cuda|cpu] [--n-ligands N] [--conds C0,C0c,PC,XA]
                                                                  [--summary-only] [--summary PATH]
  --summary-only  rebuild eval/v5_known_pockets_summary.json from the 12 tables in runs/v5 without checkpoints, labels, graphs or a GPU
                  (imports neither torch nor rdkit).
Scoring needs data/processed/g2_ligand_graphs.pt and data/processed/v1_pocket_graphs.pt. The default device is cuda when available.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROCESSED_DIR = ROOT / "data" / "processed"
OUT = ROOT / "runs" / "v5"
SUMMARY = ROOT / "eval" / "v5_known_pockets_summary.json"
CKPT = {
    "C0": [("runs/v1/ckpt_v1_C0.pt", 11), ("runs/v2/ckpt_v1_C0_s22.pt", 22), ("runs/v2/ckpt_v1_C0_s33.pt", 33)],
    "C0c": [(f"runs/v2/ckpt_v1_C0c_s{s}.pt", s) for s in (11, 22, 33)],
    "PC": [(f"runs/v3/ckpt_v3_pooled_s{s}.pt", s) for s in (11, 22, 33)],
    "XA": [(f"runs/v3/ckpt_v3_xattn_s{s}.pt", s) for s in (11, 22, 33)],
}
COLS = ["recall_1@1", "recall_1@5", "recall_1@10", "recall_1@20"]
SYNTHETIC_POCKET_HASH = "1cb5081f2d82f7695252d3c809d636acd3cb15020aeb249d67d42bb56b0f4c81"


def boot(v, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    b = v[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def score_networks(a, device, conds):
    """Score every checkpoint of the requested conditions and write runs/v5/metrics_<cond>_s<seed>_known.csv."""
    import torch

    from pocketgate.data import labels as L
    from pocketgate.data.synthetic import make_synthetic_pocket
    from pocketgate.evaluation.metrics import macro_eval
    from pocketgate.inference.screen import Screener
    from pocketgate.models.pocketgate import DualEncoderNet, PocketGate, PooledConcatNet

    arch_of = {"C0": DualEncoderNet, "C0c": DualEncoderNet, "PC": PooledConcatNet, "XA": PocketGate}
    cfg = json.loads((ROOT / "configs/g2_config.json").read_text())
    tg = pd.read_parquet(ROOT / "data/splits/targets.parquet")
    train_t = tg[tg.split_role == "train"]["target_id"].tolist()
    ligs = torch.load(PROCESSED_DIR / "g2_ligand_graphs.pt", weights_only=False)
    pockets = torch.load(PROCESSED_DIR / "v1_pocket_graphs.pt", weights_only=False)
    synth = make_synthetic_pocket({t: pockets[t] for t in train_t})
    assert synth["hash"] == SYNTHETIC_POCKET_HASH, synth["hash"]
    pockets_const = {t: synth for t in pockets}
    dev_ligs = [l for l in cfg["dev_ligands"] if l in ligs]
    if a.n_ligands:
        dev_ligs = dev_ligs[:a.n_ligands]
    frame = pd.DataFrame([(t, l) for t in train_t for l in dev_ligs], columns=["target_id", "ligand_id"])
    lab = L.load_labels("train", "dev", caller="v5_known_pockets_eval")        # training targets x development ligands
    lab = lab[lab.ligand_id.isin(set(dev_ligs)) & lab.target_id.isin(train_t)]
    for cond in conds:
        for path, seed in CKPT[cond]:
            f = OUT / f"metrics_{cond}_s{seed}_known{'_smoke' if a.n_ligands else ''}.csv"
            if f.exists():
                continue
            net = arch_of[cond](dim=128)
            net.load_state_dict(torch.load(ROOT / path, map_location="cpu"))
            net = net.to(device).eval()
            scr = Screener(net, pockets_const if cond == "C0c" else pockets, ligs, device, cache_mode="per_target")
            pred = scr.score(frame, batch_size=512)
            me = macro_eval(pred.merge(lab, on=["target_id", "ligand_id"]).dropna(subset=["score", "p_hit"]))
            me.to_csv(f, index=False)
            print(cond, seed, "known-pocket recall@10% =", round(float(me["recall_1@10"].mean()), 4), flush=True)


def summarize(conds, summary_path):
    pt = {}
    for cond in conds:
        fs = [pd.read_csv(OUT / f"metrics_{cond}_s{s}_known.csv") for _, s in CKPT[cond]]
        pt[cond] = pd.concat(fs).groupby("target_id")[COLS].mean()
    summ = {"n_targets": int(len(next(iter(pt.values())))), "targets": sorted(next(iter(pt.values())).index), "mean": {}, "ci": {}, "paired": {}, "ratio_to_XA": {}}
    for c, d in pt.items():
        summ["mean"][c] = {m: float(d[m].mean()) for m in COLS}
        summ["ci"][c] = {m: boot(d[m].to_numpy()) for m in COLS}
        summ["mean"][c]["per_target"] = {m: {str(k): float(v) for k, v in d[m].items()} for m in COLS}
    pairs = [("XA", "C0"), ("XA", "PC"), ("PC", "C0"), ("C0", "C0c"), ("PC", "C0c"), ("XA", "C0c")]
    for x, y in pairs:
        if x in pt and y in pt:
            diff = (pt[x] - pt[y])
            summ["paired"][f"{x}_minus_{y}"] = {m: {"mean": float(diff[m].mean()), "ci": boot(diff[m].to_numpy())} for m in COLS}
    if "XA" in pt:
        for c in pt:
            if c != "XA":
                rng = np.random.default_rng(1)
                v, w = pt[c]["recall_1@10"].to_numpy(), pt["XA"]["recall_1@10"].to_numpy()
                idx = rng.integers(0, len(v), size=(5000, len(v)))
                r = v[idx].mean(1) / w[idx].mean(1)
                summ["ratio_to_XA"][c] = {"mean": float(v.mean() / w.mean()), "ci": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summ, indent=1))
    for c in pt:
        print(c, {m: round(summ["mean"][c][m], 3) for m in COLS}, summ["ci"][c]["recall_1@10"])
    for k, v in summ["paired"].items():
        print(k, round(v["recall_1@10"]["mean"], 3), [round(x, 3) for x in v["recall_1@10"]["ci"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available, else cpu)")
    ap.add_argument("--n-ligands", type=int, default=0, help="use only the first N development ligands (smoke test, no summary)")
    ap.add_argument("--conds", default="C0,C0c,PC,XA")
    ap.add_argument("--summary-only", action="store_true", help="rebuild the summary from runs/v5/metrics_*_known.csv without scoring")
    ap.add_argument("--summary", type=Path, default=SUMMARY, help="summary JSON to write (default eval/v5_known_pockets_summary.json)")
    a = ap.parse_args()
    summary = a.summary if a.summary.is_absolute() else ROOT / a.summary
    OUT.mkdir(parents=True, exist_ok=True)
    conds = a.conds.split(",")
    if not a.summary_only:
        device = a.device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        score_networks(a, device, conds)
    if a.n_ligands:
        return
    summarize(conds, summary)


if __name__ == "__main__":
    main()
