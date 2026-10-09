"""Recompute the edit-level metrics of the withheld-kinase runs and of the kinase descriptor models with receptors grouped
by (target, receptor). The pooled r is recomputed as a check and has to equal the value stored by the training run.
Usage: python kin_recompute.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
sys.path.insert(0, str(ROOT / "src"))
from dm_core import delta_table, lig_graphs_for, load_pockets, metric_block, predict_delta, read_results  # noqa: E402
from dm_eval import load_model  # noqa: E402

KIN = ["ABL1", "EGFR", "SRC", "JAK2"]
EV = ROOT / "dockmut/eval"
plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
pockets = load_pockets(ROOT / "dockmut/plan/pocket_graphs.pt", plan)
tabs = {t: delta_table(read_results(ROOT / "dockmut/results", t), json.loads((ROOT / f"dockmut/plan/receptors/{t}/manifest.json").read_text()))[0] for t in KIN}
need = sorted({i for t in tabs for _, (ids, _, _) in tabs[t].items() for i in ids})
smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
ligs = lig_graphs_for(need, ROOT / "data/processed/g2_ligand_graphs.pt", smiles)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def contact(g):
    return g[g.kind.isin(["single_contact"]) | g.kind.str.startswith("multi")]


per = {}
for ck in sorted((ROOT / "dockmut").rglob("*_kin_s*.pt")):
    tag = ck.stem
    cond = re.sub(r"_kin_s\d+$", "", tag)
    model, _ = load_model("dm:" + str(ck))
    g = contact(pd.concat([predict_delta(model, pockets, ligs, tabs[t], t, dev) for t in KIN]))
    mb = metric_block(g)
    old = json.loads(next((ROOT / "dockmut").rglob(f"{tag}_diag.json")).read_text())["val_families"]
    assert abs(mb["pooled_r"] - old["pooled_r"]) < 2e-3, (tag, mb["pooled_r"], old["pooled_r"])
    per.setdefault(cond, {})[tag] = mb
    print(f"{tag:26s} pooled {mb['pooled_r']:.3f} (stored {old['pooled_r']:.3f}) edit-mean {old['mutant_r']:.3f} -> {mb['mutant_r']:.3f}  within {old['within_r']:.3f} -> {mb['within_r']:.3f}", flush=True)

K = json.loads((EV / "kin_summary.json").read_text())
for cond, d in per.items():
    assert len(d) == K[cond]["n"], (cond, len(d), K[cond]["n"])
    K[cond]["mutant_r_mean"] = float(np.mean([v["mutant_r"] for v in d.values()]))
    K[cond]["within_r_mean"] = float(np.mean([v["within_r"] for v in d.values()]))
    K[cond]["mutant_r_seeds"] = [v["mutant_r"] for v in d.values()]
(EV / "kin_summary.json").write_text(json.dumps(K, indent=1))

D = json.loads((EV / "descriptor_baselines_kin.json").read_text())
P = pd.read_parquet(EV / "descriptor_baselines_kin.parquet")
for name in D:
    g = contact(P[P.model == name])
    mb = metric_block(g)
    assert abs(mb["pooled_r"] - D[name]["pooled_r"]) < 2e-3, (name, mb["pooled_r"], D[name]["pooled_r"])
    print(f"{name:24s} edit-mean {D[name]['mutant_r']:.3f} -> {mb['mutant_r']:.3f}  within {D[name]['within_r']:.3f} -> {mb['within_r']:.3f}")
    D[name]["mutant_r"], D[name]["within_r"] = mb["mutant_r"], mb["within_r"]
(EV / "descriptor_baselines_kin.json").write_text(json.dumps(D, indent=1))
print("kin summaries updated")
