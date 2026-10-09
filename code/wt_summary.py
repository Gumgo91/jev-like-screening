"""Summarise wild-type screening of all models: condition means over seeds with target-bootstrap intervals.

Usage: python wt_summary.py wt_dev_all.json wt_dev_all.per_target.csv out.json
Reports top-1% recall at a 10% budget (macro over targets, bootstrap over targets), paired differences of conditions
against the baseline surrogates and against the geometry network trained on wild-type scores, and the reversal accuracy
(mean, SD and range over seeds).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

res = json.loads(Path(sys.argv[1]).read_text())
pt = pd.read_csv(sys.argv[2])
out = Path(sys.argv[3])


def cond(tag: str) -> str:
    m = re.match(r"^(.*)_s(\d+)$", tag)
    return m.group(1) if m else tag


pt["cond"] = pt.model.map(cond)
# per condition and target: mean over seeds of the top-1% recall at a 10% budget
R = pt.groupby(["cond", "target_id"]).agg(r=("recall_1@10", "mean"), sp=("spearman", "mean")).reset_index()
P = R.pivot(index="target_id", columns="cond", values="r")
rng = np.random.default_rng(0)
n = len(P)
idx = rng.integers(0, n, size=(5000, n))


def boot_mean(v: np.ndarray):
    b = v[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


summary = {"n_targets": int(n), "conditions": {}, "pairs": {}}
for c in P.columns:
    v = P[c].to_numpy()
    summary["conditions"][c] = {"recall_1@10": float(v.mean()), "ci": boot_mean(v)}
base = P[["C0", "C1", "C2"]].mean(axis=1).to_numpy()
summary["baseline_mean"] = {"recall_1@10": float(base.mean()), "ci": boot_mean(base)}
for c in P.columns:
    for ref, rv in (("baseline", base), ("rsg_wt", P["rsg_wt"].to_numpy())):
        if c in ("C0", "C1", "C2", ref):
            continue
        d = P[c].to_numpy() - rv
        summary["pairs"][f"{c} - {ref}"] = {"diff": float(d.mean()), "ci": boot_mean(d)}
for a_, b_ in (("rsg_wt", "rs_wt"), ("rsg_wt", "b5_wt"), ("rsg_int", "rs_int"), ("rsg_int", "b5_int"), ("rsg_int", "rsg_int_shm2"), ("rsg_int", "rsg_int_sha2"),
               ("rsg_int", "rsg_int_shl"), ("rsg_wt", "C0c"), ("rs_wt", "b5_wt"), ("rsg_int", "rsg_wt"), ("rs_int", "rs_wt"), ("rse_int", "rse_wt"), ("b5_int", "b5_wt"), ("rsg_int", "C0c")):
    if a_ in P.columns and b_ in P.columns:
        d = P[a_].to_numpy() - P[b_].to_numpy()
        summary["pairs"][f"{a_} - {b_}"] = {"diff": float(d.mean()), "ci": boot_mean(d)}
groups: dict[str, list] = {}
for t, v in res.items():
    groups.setdefault(cond(t), []).append(v)
summary["reversal"] = {c: {"mean": float(np.mean([v["joint"] for v in vs])), "sd": float(np.std([v["joint"] for v in vs])),
                           "min": float(np.min([v["joint"] for v in vs])), "max": float(np.max([v["joint"] for v in vs])),
                           "n": len(vs)} for c, vs in groups.items()}
summary["spearman"] = {c: {"mean": float(np.mean([v["spearman"] for v in vs])), "n": len(vs)} for c, vs in groups.items()}
out.write_text(json.dumps(summary, indent=1))
for c in ["C0", "C1", "C2", "C0c", "b5_wt", "b5_int", "rs_wt", "rs_int", "rse_wt", "rse_int", "rsg_wt", "rsg_int", "rsg_int_shl", "rsg_int_shm", "rsg_int_sha", "rsg_int_shm2", "rsg_int_sha2", "rsg_const"]:
    if c in summary["conditions"]:
        s = summary["conditions"][c]
        rv = summary["reversal"][c]
        print(f"{c:14s} R1@10 {s['recall_1@10']:.3f} [{s['ci'][0]:.3f},{s['ci'][1]:.3f}]  reversal {rv['mean']:.3f} (sd {rv['sd']:.3f}, {rv['min']:.2f}-{rv['max']:.2f}, n={rv['n']})  spearman {summary['spearman'][c]['mean']:.3f}")
for k in ("rsg_wt - baseline", "rsg_int - baseline", "rsg_int - rsg_wt", "rs_int - baseline", "rsg_int_shm2 - rsg_wt"):
    if k in summary["pairs"]:
        v = summary["pairs"][k]
        print(f"{k:28s} diff {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
