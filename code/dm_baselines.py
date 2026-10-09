"""Descriptor baselines for the interventional benchmark.

Features use only ligand descriptors and a description of the edit (removed atoms, residue classes,
distance of the edited residues to the box center, pocket size). No pocket graph is seen. The baselines
show how much of the measured change a model can explain without reading the pocket, and they are
trained on the same training targets as the neural surrogates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
from dm_core import delta_table, metric_block, read_results  # noqa: E402

RDLogger.DisableLog("rdApp.*")
HYDROPHOBIC = {"VAL", "LEU", "ILE", "MET", "PHE", "TRP"}
AROMATIC = {"PHE", "TRP", "TYR", "HIS", "HID", "HIE", "HIP"}
POLAR = {"SER", "THR", "ASN", "GLN", "CYS", "TYR", "HIS", "HID", "HIE", "HIP"}
CHARGED = {"ASP", "GLU", "LYS", "ARG"}


def lig_desc(smiles: str) -> list[float]:
    m = Chem.MolFromSmiles(smiles)
    return [m.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumRotatableBonds(m),
            Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m), Crippen.MolLogP(m), rdMolDescriptors.CalcTPSA(m),
            rdMolDescriptors.CalcFractionCSP3(m)]


def edit_desc(rec: dict) -> list[float]:
    rs = rec["residues"]
    d = [x["d_sc"] for x in rs if x["d_sc"] is not None]
    cnt = lambda S: sum(x["n_sc_heavy"] for x in rs if x["res"] in S)
    return [rec["n_removed"], len(rs), min(d) if d else 30.0, float(np.mean(d)) if d else 30.0,
            cnt(HYDROPHOBIC), cnt(AROMATIC), cnt(POLAR), cnt(CHARGED),
            float(rec["kind"].startswith("multi")), float(rec["kind"] in ("single_shell", "single_far"))]


def build(results: Path, plan: dict, plan_dir: Path, targets: list[str], smiles: pd.Series, ligcache: dict, pockets=None):
    rows = []
    for t in targets:
        man = json.loads((plan_dir / "receptors" / t / "manifest.json").read_text())
        tab, _, _ = delta_table(read_results(results, t), man)
        recs = {r["id"]: r for r in man["receptors"]}
        npocket = float(np.mean([x["n_sc_heavy"] for r in man["receptors"] for x in r["residues"]] or [0]))
        for rid, (ids, dS, kind) in tab.items():
            e = edit_desc({**recs[rid], "kind": kind})
            if pockets is not None:
                g0 = pockets[t]["wt"]["geo"].numpy()
                g1 = pockets[t][rid]["geo"].numpy()
                e = e + list(g0) + list(g1 - g0)
            for i, d in zip(ids, dS):
                if i not in ligcache:
                    ligcache[i] = lig_desc(smiles[i])
                rows.append([t, rid, kind, i, float(d), *ligcache[i], *e])
    cols = ["target", "receptor", "kind", "ligand", "dS", "ha", "arom", "rot", "hbd", "hba", "logp", "tpsa", "fsp3",
            "n_removed", "n_res", "d_min", "d_mean", "n_hphob", "n_arom", "n_polar", "n_charged", "is_multi", "is_null"]
    if pockets is not None:
        from geometry import GEO_NAMES
        cols = cols + ["g_" + n for n in GEO_NAMES] + ["dg_" + n for n in GEO_NAMES]
    return pd.DataFrame(rows, columns=cols)


FEATS = ["ha", "arom", "rot", "hbd", "hba", "logp", "tpsa", "fsp3", "n_removed", "n_res", "d_min", "d_mean",
         "n_hphob", "n_arom", "n_polar", "n_charged", "is_multi", "is_null"]


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ha_x_removed"] = d.ha * d.n_removed
    d["ha_x_inv_d"] = d.ha / (1.0 + d.d_min)
    d["removed_x_inv_d"] = d.n_removed / (1.0 + d.d_min)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "dockmut/results")
    p.add_argument("--plan", type=Path, default=ROOT / "dockmut/plan")
    p.add_argument("--train-targets", default="")
    p.add_argument("--test-targets", default="")
    p.add_argument("--pockets", type=Path, default=ROOT / "dockmut/plan/pocket_graphs.pt")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    plan = json.loads((a.plan / "plan.json").read_text())
    train_t = [t for t in a.train_targets.split(",") if t] or [t for t, v in plan["targets"].items() if v["role"] == "train"]
    test_t = [t for t in a.test_targets.split(",") if t] or [t for t, v in plan["targets"].items() if v["role"] in ("dev", "test")]
    smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
    cache: dict = {}
    import torch
    pockets = torch.load(a.pockets, weights_only=False)
    tr = add_interactions(build(a.results, plan, a.plan, train_t, smiles, cache, pockets))
    te = add_interactions(build(a.results, plan, a.plan, test_t, smiles, cache, pockets))
    feats = FEATS + ["ha_x_removed", "ha_x_inv_d", "removed_x_inv_d"]
    geo = [c for c in tr.columns if c.startswith(("g_", "dg_"))]
    tr["ha_x_dvol6"] = tr.ha * tr["dg_vol_r6"]
    te["ha_x_dvol6"] = te.ha * te["dg_vol_r6"]
    tr["ha_x_g_vol6"] = tr.ha * tr["g_vol_r6"]
    te["ha_x_g_vol6"] = te.ha * te["g_vol_r6"]
    feats_geo = feats + geo + ["ha_x_dvol6", "ha_x_g_vol6"]
    out, frames = {}, []
    ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit((tr[feats] - tr[feats].mean()) / tr[feats].std(), tr.dS)
    gbm = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.06, max_iter=300, l2_regularization=1.0,
                                        random_state=0).fit(tr[feats], tr.dS)
    size_only = RidgeCV(alphas=[1.0]).fit(tr[["ha", "n_removed", "ha_x_removed"]], tr.dS)
    mu, sd = tr[feats_geo].mean(), tr[feats_geo].std().replace(0, 1)
    ridge_g = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit((tr[feats_geo] - mu) / sd, tr.dS)
    gbm_g = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.06, max_iter=300, l2_regularization=1.0,
                                          random_state=0).fit(tr[feats_geo], tr.dS)
    preds = {"ridge_descriptors": ridge.predict((te[feats] - tr[feats].mean()) / tr[feats].std()),
             "gbm_descriptors": gbm.predict(te[feats]),
             "ridge_descriptors_geo": ridge_g.predict((te[feats_geo] - mu) / sd),
             "gbm_descriptors_geo": gbm_g.predict(te[feats_geo]),
             "size_only": size_only.predict(te[["ha", "n_removed", "ha_x_removed"]])}
    for name, pr in preds.items():
        g = te[["target", "receptor", "kind", "ligand", "dS"]].copy()
        g["pred"] = pr
        g["model"] = name
        frames.append(g)
        c = g[g.kind.isin(["single_contact"]) | g.kind.str.startswith("multi")]
        out[name] = metric_block(c)
        print(f"{name:18s} pooled r={out[name]['pooled_r']:.3f} mutant r={out[name]['mutant_r']:.3f} "
              f"within r={out[name]['within_r']:.3f} gain={out[name]['gain']:.2f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_parquet(a.out.with_suffix(".parquet"), index=False)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
