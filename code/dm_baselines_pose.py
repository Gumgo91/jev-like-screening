"""Pose-informed descriptor reference (exploratory, post hoc).

Adds to the descriptor features of dm_baselines.py the contacts between the wild-type docked pose of a ligand and the
removed side-chain atoms: the number of ligand heavy atoms within 4.5 A of the removed atoms and the smallest distance.
A surrogate that scores 2D ligands against a pocket state has no access to these quantities, because they require one
docking of the ligand into the wild-type receptor. The model shows how much of the measured change is pose-dependent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
from dm_baselines import FEATS, add_interactions, build  # noqa: E402
from dm_core import metric_block  # noqa: E402
from geometry import GEO_NAMES  # noqa: E402

out = Path(sys.argv[1])
plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
plan_dir = ROOT / "dockmut/plan"
train_t = [t for t, v in plan["targets"].items() if v["role"] == "train"]
test_t = [t for t, v in plan["targets"].items() if v["role"] in ("dev", "test")]
smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
pockets = torch.load(ROOT / "dockmut/plan/pocket_graphs.pt", weights_only=False)
cache: dict = {}
ct = pd.read_parquet(ROOT / "dockmut/analysis/contacts/contact_changes.parquet")[["target", "receptor", "inchikey", "n_contact", "min_dist"]]
ct = ct.rename(columns={"inchikey": "ligand"})
tr = add_interactions(build(ROOT / "dockmut/results", plan, plan_dir, train_t, smiles, cache, pockets)).merge(ct, on=["target", "receptor", "ligand"], how="left")
te = add_interactions(build(ROOT / "dockmut/results", plan, plan_dir, test_t, smiles, cache, pockets)).merge(ct, on=["target", "receptor", "ligand"], how="left")
print("rows", len(tr), len(te), "missing contacts", int(tr.n_contact.isna().sum()), int(te.n_contact.isna().sum()))
for d in (tr, te):
    d["n_contact"] = d.n_contact.fillna(0)
    d["min_dist"] = d.min_dist.fillna(30.0)
    d["contact_x_removed"] = d.n_contact * d.n_removed
    d["contact_x_ha"] = d.n_contact * d.ha
    d["ha_x_dvol6"] = d.ha * d["dg_vol_r6"]
    d["ha_x_g_vol6"] = d.ha * d["g_vol_r6"]
geo = [c for c in tr.columns if c.startswith(("g_", "dg_"))]
base = FEATS + ["ha_x_removed", "ha_x_inv_d", "removed_x_inv_d"]
pose = ["n_contact", "min_dist", "contact_x_removed", "contact_x_ha"]
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
sets = {"gbm_pose": base + pose, "gbm_pose_geo": base + geo + ["ha_x_dvol6", "ha_x_g_vol6"] + pose}
frames, res = [], {}
for name, feats in sets.items():
    m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=0).fit(tr[feats], tr.dS)
    g = te[["target", "receptor", "kind", "ligand", "dS"]].copy()
    g["pred"] = m.predict(te[feats])
    g["model"] = name
    frames.append(g)
    c = g[g.kind.isin(["single_contact"]) | g.kind.str.startswith("multi")]
    res[name] = metric_block(c)
    print(f"{name:14s} pooled r={res[name]['pooled_r']:.3f} mutant r={res[name]['mutant_r']:.3f} within r={res[name]['within_r']:.3f} "
          f"gain={res[name]['gain']:.2f} predSD={res[name]['pred_sd']:.2f}")
pd.concat(frames).to_parquet(out.with_suffix(".parquet"), index=False)
out.write_text(json.dumps(res, indent=1))
