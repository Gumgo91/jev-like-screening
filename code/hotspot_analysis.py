"""Hot-spot ranking: agreement between the predicted and the measured mean absolute change of single contact truncations.

For every held-out target the contact residues are ranked by the mean absolute change over the shared ligands, and the
Spearman correlation with the ranking by the prediction is averaged over targets. Distance to the box center and the
number of removed atoms serve as heuristics. For the scan target the same comparison is made with the library-scale scan.
Usage: python hotspot_analysis.py summary_all.parquet scan.csv target out.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
from dm_core import delta_table, read_results  # noqa: E402

df = pd.read_parquet(sys.argv[1])
df = df[df.kind == "single_contact"]
scan = pd.read_csv(sys.argv[2])
target = sys.argv[3]
out = Path(sys.argv[4])
df["cond"] = df.model.map(lambda t: re.sub(r"_s\d+$", "", t))
meta = {}
for t in df.target.unique():
    m = json.loads((ROOT / f"dockmut/plan/receptors/{t}/manifest.json").read_text())
    for r in m["receptors"]:
        if r["id"] != "wt":
            ds = [x["d_sc"] for x in r["residues"] if x["d_sc"] is not None]
            meta[(t, r["id"])] = (float(r["n_removed"]), min(ds) if ds else 30.0)
res = {}
for c, g in df.groupby("cond"):
    e = g.groupby(["target", "receptor", "ligand"], as_index=False).agg(pred=("pred", "mean"), dS=("dS", "first"))
    rs = []
    for t, gt in e.groupby("target"):
        a = gt.assign(ap=gt.pred.abs(), ad=gt.dS.abs()).groupby("receptor", as_index=False).agg(ap=("ap", "mean"), ad=("ad", "mean"))
        if len(a) >= 5:
            rs.append(spearmanr(a.ap, a.ad)[0])
    res[c] = float(np.nanmean(rs))
ref = df[df.cond == "rsg_int"].groupby(["target", "receptor"], as_index=False).agg(ad=("dS", lambda x: float(np.mean(np.abs(x)))))
rem, dist = [], []
for t, gt in ref.groupby("target"):
    n = np.array([meta[(t, r)][0] for r in gt.receptor])
    d = np.array([meta[(t, r)][1] for r in gt.receptor])
    rem.append(spearmanr(n, gt.ad)[0])
    dist.append(spearmanr(-d, gt.ad)[0])
res["distance"] = float(np.nanmean(dist))
res["removed_atoms"] = float(np.nanmean(rem))
# scan target
man = json.loads((ROOT / f"dockmut/plan/receptors/{target}/manifest.json").read_text())
tab, _, _ = delta_table(read_results(ROOT / "dockmut/results", target), man)
rows = []
for rid, (ids, dS, kind) in tab.items():
    if kind in ("single_contact", "single_shell"):
        rec = [r for r in man["receptors"] if r["id"] == rid][0]
        rows.append({"residue": rec["residues"][0]["label"], "kind": kind, "measured": float(np.mean(dS)), "meas_abs": float(np.mean(np.abs(dS)))})
D = scan.merge(pd.DataFrame(rows), on="residue")
c = D[D.kind == "single_contact"]
res["scan_target"] = target
res["scan_n_all"] = int(len(D))
res["scan_n_contact"] = int(len(c))
res["scan_spearman_abs_all"] = float(spearmanr(D.mean_abs_dS, D.meas_abs)[0])
res["scan_spearman_abs_contact"] = float(spearmanr(c.mean_abs_dS, c.meas_abs)[0])
res["scan_r_signed_all"] = float(np.corrcoef(D.mean_dS, D.measured)[0, 1])
res["scan_r_signed_contact"] = float(np.corrcoef(c.mean_dS, c.measured)[0, 1])
out.write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(k, v if isinstance(v, (str, int)) else round(v, 3))
