"""Score-blind ligand selection for interventional docking.

Ligands are drawn by ligand-split role and stratified by heavy-atom count, using
SMILES only. No docking score is read here, so the choice cannot depend on any
label. The same ligand list is docked against every receptor variant of a target,
which gives exactly matched wild-type/mutant pairs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["train", "dev", "probcal", "policycal", "test"])
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--min-ha", type=int, default=15)
    p.add_argument("--max-ha", type=int, default=40)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exclude", type=Path, default=None, help="csv with inchikey column to exclude")
    a = p.parse_args()

    split = pd.read_parquet("data/splits/ligand_splits.parquet")
    split = split[(split.split_role == a.role) & (split.feature_status == "OK")]
    raw = pd.read_csv("data/raw/dockstring-dataset.tsv", sep="\t", usecols=["inchikey", "smiles"])
    df = split.merge(raw, left_on="ligand_id", right_on="inchikey", how="inner")
    if a.exclude is not None:
        ex = set(pd.read_csv(a.exclude)["inchikey"])
        df = df[~df.inchikey.isin(ex)]
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(df))[: max(a.n * 6, 2000)]
    df = df.iloc[idx].copy()
    ha = []
    for s in df.smiles:
        m = Chem.MolFromSmiles(s)
        ha.append(m.GetNumHeavyAtoms() if m is not None else -1)
    df["heavy_atoms"] = ha
    df = df[(df.heavy_atoms >= a.min_ha) & (df.heavy_atoms <= a.max_ha)]
    df["stratum"] = pd.qcut(df.heavy_atoms.rank(method="first"), 4, labels=False)
    per = a.n // 4
    parts = [g.head(per) for _, g in df.groupby("stratum")]
    out = pd.concat(parts)
    if len(out) < a.n:
        rest = df[~df.inchikey.isin(out.inchikey)].head(a.n - len(out))
        out = pd.concat([out, rest])
    out = out[["inchikey", "smiles", "heavy_atoms", "ligand_group_id"]].reset_index(drop=True)
    out["role"] = a.role
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(a.out, len(out), "heavy atoms", out.heavy_atoms.min(), out.heavy_atoms.median(), out.heavy_atoms.max())


if __name__ == "__main__":
    main()
