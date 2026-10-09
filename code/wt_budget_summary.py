"""Screening at several selection budgets: top-1% recall at 1, 5, 10 and 20% budgets, enrichment factor at 1% and the budget that
recovers 95% of the top-1% hits. Reads existing per-target result files only (no label access).

Conditions: the hit-head networks, the regression-trained geometry networks, the baseline surrogates read out with their
hit-probability head (preceding analysis) and the pooled ligand-only fingerprint model. Per target values are means over seeds, and the
summary is the macro mean over targets with a 95% bootstrap interval over targets.
Usage: python wt_budget_summary.py out.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "dockmut/eval"
V2 = ROOT / "runs/v2"
V3 = ROOT / "runs/v3"
COLS = ["recall_1@1", "recall_1@5", "recall_1@10", "recall_1@20", "ef1@1", "recall95_budget"]
rng = np.random.default_rng(0)


def boot(v: np.ndarray, n=5000):
    idx = rng.integers(0, len(v), size=(n, len(v)))
    b = v[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def per_target(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("target_id")[COLS].mean()


res = {}
for split, pre in (("dev", "metrics"), ("test", "test_metrics")):
    conds = {}
    for c, tag in (("C0", "Plain objective, hit head"), ("C0c", "Constant pocket, hit head")):
        fs = sorted(glob.glob(str(V2 / f"{pre}_{c}_s*_cold.csv")))
        conds[c] = pd.concat([pd.read_csv(f) for f in fs])
    conds["CB"] = pd.read_csv(V2 / f"{pre}_CB_s11_cold.csv")
    for c in ("XA", "PC"):                                   # joint networks of addendum 8 (cross-attention, pooled concatenation)
        fs = sorted(glob.glob(str(V3 / f"{pre}_{c}_s*_cold.csv")))
        if fs:
            conds[c] = pd.concat([pd.read_csv(f) for f in fs])
    h = pd.read_csv(EV / "hit" / f"wt_{split}_hit.per_target.csv")
    h["cond"] = h.model.str.replace(r"_s\d+$", "", regex=True)
    for c in ("rsgh_wt", "rsgh_int"):
        conds[c] = h[h.cond == c]
    r = pd.read_csv(EV / f"wt_{split}_all.per_target.csv")
    r["cond"] = r.model.str.replace(r"_s\d+$", "", regex=True)
    for c in ("rsg_wt", "rsg_int"):
        conds[c] = r[r.cond == c]
    out = {}
    for c, df in conds.items():
        pt = per_target(df)
        out[c] = {"n_targets": int(len(pt)), **{m: {"mean": float(pt[m].mean()), "ci": boot(pt[m].to_numpy()), "per_target": {str(k): float(v) for k, v in pt[m].items()}} for m in COLS}}
    pair = {}
    pt0 = per_target(conds["C0"])["recall_1@10"]
    for c in ("XA", "PC"):
        if c in conds:
            d = (pt0 - per_target(conds[c])["recall_1@10"]).dropna().to_numpy()
            pair[f"C0_minus_{c}"] = {"mean": float(d.mean()), "ci": boot(d), "n_targets": int(len(d))}
    out["paired_recall_1@10"] = pair
    res[split] = out
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
for split, out in res.items():
    for c, v in out.items():
        if c == "paired_recall_1@10":
            print(split, "paired", {k: round(x["mean"], 3) for k, x in v.items()})
            continue
        print(f"{split:4s} {c:9s} " + " ".join(f"{m.split('_')[-1] if m.startswith('recall_1') else m}={v[m]['mean']:.3f}" for m in COLS))
