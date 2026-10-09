"""Addendum 11, known-pocket evaluation of the cross-fitted networks (checkpoints of the cross-validation runs of addendum 9).

A network of fold k was trained on the targets R_k with its checkpoint chosen on seen targets (six validation targets of R_k against 10,000 calibration
ligands). Here every network is evaluated on the pockets of R_k (its known pockets) against the 10,000 development ligands (open labels, never used for
training or for its checkpoint choice). Per pocket the recall is the mean over the folds in which the pocket is a training target (four of five), and the summary is
the macro mean over the pockets with 95% bootstrap intervals. Writes runs/v5/cv_known_<tag>.csv and dockmut/eval/v5_known_cv_summary.json.
Usage: python dockmut/v5_known_cv_eval.py [--device cuda] [--summary-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pocketgate.common import PROCESSED_DIR  # noqa: E402
from pocketgate.data import labels as L  # noqa: E402
from pocketgate.evaluation.metrics import macro_eval  # noqa: E402
from pocketgate.inference.screen import Screener  # noqa: E402
from pocketgate.models.pocketgate import DualEncoderNet, PocketGate, PooledConcatNet  # noqa: E402
from v1b_train import make_synthetic_pocket  # noqa: E402

RUNS = ROOT / "runs/v4"
OUT = ROOT / "runs/v5"
ARCH = {"dual": DualEncoderNet, "pooled": PooledConcatNet, "xattn": PocketGate}
COLS = ["recall_1@1", "recall_1@5", "recall_1@10", "recall_1@20"]


def boot(v, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    b = v[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--n-ligands", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    folds = {d["fold"]: d for d in json.loads((ROOT / "data/splits/v4_cv_folds.json").read_text())["folds"]}
    regs = sorted(RUNS.glob("registry_*.json"))
    if not a.summary_only:
        cfg = json.loads((ROOT / "configs/g2_config.json").read_text())
        ligs = torch.load(PROCESSED_DIR / "g2_ligand_graphs.pt", weights_only=False)
        pockets = torch.load(PROCESSED_DIR / "v1_pocket_graphs.pt", weights_only=False)
        dev_ligs = [l for l in cfg["dev_ligands"] if l in ligs]
        if a.n_ligands:
            dev_ligs = dev_ligs[:a.n_ligands]
        lab = pd.concat([L.load_labels("train", "dev", caller="v5_known_cv_eval"), L.load_labels("dev", "dev", caller="v5_known_cv_eval")])
        lab = lab[lab.ligand_id.isin(set(dev_ligs))]
        for rf in regs:
            r = json.loads(rf.read_text())
            tag = rf.stem.replace("registry_", "")
            f = OUT / f"cv_known_{tag}{'_smoke' if a.n_ligands else ''}.csv"
            if f.exists():
                continue
            fold = folds[r["fold"]]
            cond = r["cond"]
            arch, const = (cond[:-6], True) if cond.endswith("_const") else (cond, False)
            used = pockets
            if const:
                synth = make_synthetic_pocket({t: pockets[t] for t in fold["train"]})
                used = {t: synth for t in pockets}
            net = ARCH[arch](dim=128)
            net.load_state_dict(torch.load(RUNS / f"ckpt_v4_{tag}.pt", map_location="cpu"))
            net = net.to(a.device).eval()
            frame = pd.DataFrame([(t, l) for t in fold["train"] for l in dev_ligs], columns=["target_id", "ligand_id"])
            scr = Screener(net, used, ligs, a.device, cache_mode="per_target")
            pred = scr.score(frame, batch_size=512)
            me = macro_eval(pred.merge(lab, on=["target_id", "ligand_id"]).dropna(subset=["score", "p_hit"]))
            me["tag"], me["cond"], me["recipe"], me["fold"] = tag, cond, r["recipe"], r["fold"]
            me.to_csv(f, index=False)
            print(tag, "known-pocket recall@10% =", round(float(me["recall_1@10"].mean()), 4), flush=True)
    if a.n_ligands:
        return
    df = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("cv_known_*.csv"))])
    df["c"] = df.cond + "_" + df.recipe
    pt = {c: g.groupby("target_id")[COLS].mean() for c, g in df.groupby("c")}
    summ = {"n_targets": {c: int(len(d)) for c, d in pt.items()}, "mean": {}, "ci": {}, "paired": {}, "per_target": {}}
    for c, d in pt.items():
        summ["mean"][c] = {m: float(d[m].mean()) for m in COLS}
        summ["ci"][c] = {m: boot(d[m].to_numpy()) for m in COLS}
        summ["per_target"][c] = {m: {str(k): float(v) for k, v in d[m].items()} for m in COLS}
    pairs = [("xattn_R0", "dual_R0"), ("xattn_R0", "pooled_R0"), ("pooled_R0", "dual_R0"), ("dual_R0", "dual_const_R0"), ("pooled_R0", "pooled_const_R0"),
             ("xattn_R0", "xattn_const_R0"), ("xattn_R1", "dual_R1"), ("xattn_R1", "xattn_R0"), ("dual_R1", "dual_R0")]
    for x, y in pairs:
        if x in pt and y in pt:
            diff = pt[x].reindex(pt[y].index) - pt[y]
            summ["paired"][f"{x}_minus_{y}"] = {m: {"mean": float(diff[m].mean()), "ci": boot(diff[m].dropna().to_numpy())} for m in COLS}
    (ROOT / "dockmut/eval/v5_known_cv_summary.json").write_text(json.dumps(summ, indent=1))
    for c in pt:
        print(c, {m: round(summ["mean"][c][m], 3) for m in COLS}, [round(x, 3) for x in summ["ci"][c]["recall_1@10"]])
    for k, v in summ["paired"].items():
        print(k, round(v["recall_1@10"]["mean"], 3), [round(x, 3) for x in v["recall_1@10"]["ci"]])


if __name__ == "__main__":
    main()
