"""Does pocket geometry explain how truncations act on ligands of different size?

For each target the ligand main effect of contact truncations (mean change per ligand over edits) is regressed on
the heavy-atom count; the slope measures how much larger ligands gain. The slope and the mean multi-residue
change are related to wild-type pocket geometry. Leave-one-target-out and leave-one-family-out ridge regression
give out-of-sample R2 for the slope from two or three descriptors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
from dm_core import delta_table, read_results  # noqa: E402
from geometry import GEO_NAMES  # noqa: E402


def target_table(results: Path, plan_dir: Path, plan: dict, pockets: dict, smiles: pd.Series) -> pd.DataFrame:
    tg = pd.read_parquet(ROOT / "data/splits/targets.parquet").set_index("target_id")
    rows, ha = [], {}
    for f in sorted(results.glob("*.csv")):
        t = f.stem
        man = json.loads((plan_dir / "receptors" / t / "manifest.json").read_text())
        tab, sigma, _ = delta_table(read_results(results, t), man)
        cm = [r for r, v in tab.items() if v[2] == "single_contact" or v[2].startswith("multi")]
        if len(cm) < 6:
            continue
        common = sorted(set.intersection(*[set(tab[r][0]) for r in cm]))
        if len(common) < 30:
            continue
        X = np.array([[dict(zip(tab[r][0], tab[r][1]))[l] for l in common] for r in cm])
        H = []
        for l in common:
            if l not in ha:
                ha[l] = Chem.MolFromSmiles(smiles[l]).GetNumHeavyAtoms()
            H.append(ha[l])
        H = np.array(H, float)
        lam = X.mean(0)
        slope = float(np.polyfit(H, lam, 1)[0])
        multi = [i for i, r in enumerate(cm) if tab[r][2].startswith("multi")]
        g = pockets[t]["wt"]["geo"].numpy()
        rows.append({"target": t, "family": tg.loc[t, "family"], "role": plan["targets"][t]["role"], "slope_ha": slope,
                     "mean_dS_contact": float(X.mean()), "mean_dS_multi": float(X[multi].mean()) if multi else np.nan,
                     **dict(zip(GEO_NAMES, g.tolist()))})
    return pd.DataFrame(rows)


def loo(df: pd.DataFrame, feats: list[str], y: str, group: str | None = None) -> float:
    from sklearn.linear_model import Ridge
    pred = np.zeros(len(df))
    keys = df[group] if group else df.index.to_series()
    for k in keys.unique():
        te = (keys == k).to_numpy()
        tr = ~te
        mu, sd = df.loc[tr, feats].mean(), df.loc[tr, feats].std().replace(0, 1)
        m = Ridge(alpha=1.0).fit((df.loc[tr, feats] - mu) / sd, df.loc[tr, y])
        pred[te] = m.predict((df.loc[te, feats] - mu) / sd)
    ss = ((df[y] - pred) ** 2).sum()
    return float(1 - ss / ((df[y] - df[y].mean()) ** 2).sum()), pred


if __name__ == "__main__":
    res = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
    pockets = torch.load(ROOT / "dockmut/plan/pocket_graphs.pt", weights_only=False)
    smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
    df = target_table(res, ROOT / "dockmut/plan", plan, pockets, smiles)
    df.to_csv(out / "target_geometry.csv", index=False)
    print(len(df), "targets")
    for c in GEO_NAMES:
        print(f"{c:12s} slope: pearson {pearsonr(df[c], df.slope_ha)[0]:+.2f} spearman {spearmanr(df[c], df.slope_ha)[0]:+.2f}"
              f" | multi mean: pearson {pearsonr(df[c], df.mean_dS_multi)[0]:+.2f}")
    feats = ["vol_r6", "polar_frac", "buried_frac"]
    r2t, _ = loo(df, feats, "slope_ha")
    r2f, _ = loo(df, feats, "slope_ha", group="family")
    print("LOO-target R2 of slope from", feats, round(r2t, 3), "| leave-family-out R2", round(r2f, 3))
    json.dump({"n": len(df), "loo_target_r2": r2t, "loo_family_r2": r2f, "features": feats}, open(out / "geometry_loo.json", "w"))
