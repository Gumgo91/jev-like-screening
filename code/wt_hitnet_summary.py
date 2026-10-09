"""Screening of the hit-head networks (addendum 7): recall at a 10% budget with the hit logit on the development and test split.

Development metrics were computed on the pod, test metrics locally (sealed labels stay on the workstation).
Reports per condition the macro recall over targets with a bootstrap interval, the reversal accuracy, and paired differences
over targets against the hit-trained baselines of the preceding analysis (result files), the ligand-only fingerprint model and
the regression-trained geometry networks of the registered runs.
Usage: python wt_hitnet_summary.py out.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "runs/v2"
EV = ROOT / "dockmut/eval"
out = Path(sys.argv[1])
rng = np.random.default_rng(0)
BASE = {"C0": ["C0_s11", "C0_s22", "C0_s33"], "C0c": ["C0c_s11", "C0c_s22", "C0c_s33"], "C1": ["C1_s11", "C1_s22", "C1_s33"], "CB": ["CB_s11"]}
PAT = {"dev": "metrics_{run}_cold.csv", "test": "test_metrics_{run}_cold.csv"}


def boot(v: np.ndarray, n=5000):
    idx = rng.integers(0, len(v), size=(n, len(v)))
    b = v[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


res = {}
for split in ("dev", "test"):
    per = {}
    for cond, runs in BASE.items():
        M = pd.concat([pd.read_csv(V2 / PAT[split].format(run=r)).set_index("target_id")["recall_1@10"].rename(r) for r in runs], axis=1)
        per[cond] = M.mean(axis=1)
    reg = pd.read_csv(EV / f"wt_{split}_all.per_target.csv")
    reg["cond"] = reg.model.str.replace(r"_s\d+$", "", regex=True)
    for c in ("rsg_wt", "rsg_int"):
        per[c] = reg[reg.cond == c].groupby("target_id")["recall_1@10"].mean()
    hit = pd.read_csv(EV / "hit" / f"wt_{split}_hit.per_target.csv")
    hit["cond"] = hit.model.str.replace(r"_s\d+$", "", regex=True)
    js = json.loads((EV / "hit" / f"wt_{split}_hit.json").read_text())
    entry = {"conditions": {}, "pairs": {}}
    for c in ("rsgh_wt", "rsgh_int"):
        sub = hit[hit.cond == c]
        per[c] = sub.groupby("target_id")["recall_1@10"].mean()
        v = per[c].to_numpy()
        tags = sorted(t for t in js if t.startswith(c + "_s"))
        j = np.array([js[t]["joint"] for t in tags])
        seeds = [float(sub[sub.model == t]["recall_1@10"].mean()) for t in tags]
        entry["conditions"][c] = {"recall_1@10": float(v.mean()), "ci": boot(v), "seeds": seeds, "n_targets": int(len(v)),
                                  "reversal": {"mean": float(j.mean()), "min": float(j.min()), "max": float(j.max()), "n": int(len(j))}}
    for a, b in (("rsgh_wt", "C0"), ("rsgh_int", "C0"), ("rsgh_wt", "C0c"), ("rsgh_int", "C0c"), ("rsgh_wt", "CB"), ("rsgh_int", "CB"), ("rsgh_wt", "C1"), ("rsgh_int", "C1"),
                 ("rsgh_wt", "rsg_wt"), ("rsgh_int", "rsg_int"), ("rsgh_int", "rsgh_wt")):
        common = per[a].index.intersection(per[b].index)
        d = (per[a].loc[common] - per[b].loc[common]).to_numpy()
        entry["pairs"][f"{a} - {b}"] = {"diff": float(d.mean()), "ci": boot(d), "n_targets": int(len(d))}
    res[split] = entry
out.write_text(json.dumps(res, indent=1))
for split, e in res.items():
    for c, v in e["conditions"].items():
        print(f"{split:4s} {c:8s} recall@10% {v['recall_1@10']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] seeds {[round(x, 3) for x in v['seeds']]} reversal {v['reversal']['mean']:.3f} ({v['reversal']['min']:.3f}-{v['reversal']['max']:.3f})")
    for k, p in e["pairs"].items():
        print(f"{split:4s} {k:22s} diff {p['diff']:+.3f} [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] (n={p['n_targets']})")
