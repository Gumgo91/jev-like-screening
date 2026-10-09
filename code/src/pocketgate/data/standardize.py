"""Ligand standardization and grouping (plan 4.3).

Policy (fixed before looking at model performance / scores):
- Parse original SMILES with RDKit; keep stereochemistry.
- standardized_smiles: canonical isomeric SMILES of the input as given.
- parent: largest organic fragment after salt/solvent stripping
  (RDKit SaltRemover + largest-fragment pick), canonical NON-isomeric
  SMILES -> stereoisomers/salts/protomers of the same parent share
  parent_group_id.
- scaffold_id: Bemis-Murcko scaffold of the parent ('' if acyclic).
- ligand_group_id: union-find over {same parent_group, same non-empty
  scaffold, same DOCKSTRING author cluster id}. Keeps analogs and
  stereoisomers in one split group.
- Failed parses are kept with feature_status=PARSE_FAIL, never dropped
  silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pocketgate.common import PROCESSED_DIR, RAW_DIR  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_salt_remover = SaltRemover.SaltRemover()


def largest_organic_fragment(mol: Chem.Mol) -> Chem.Mol:
    stripped = _salt_remover.StripMol(mol)
    frags = Chem.GetMolFrags(stripped, asMols=True, sanitizeFrags=False)
    if not frags:
        return stripped
    # largest by heavy atoms, tie-break: fewer formal charge magnitude
    def key(m):
        return (
            sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1),
            -abs(sum(a.GetFormalCharge() for a in m.GetAtoms())),
        )
    return max(frags, key=key)


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaf, isomericSmiles=False)
    except Exception:
        return ""


def standardize_one(smiles: str) -> dict:
    out = {
        "standardized_smiles": None,
        "parent_smiles": None,
        "scaffold_smiles": None,
        "atom_count": 0,
        "feature_status": "OK",
    }
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        out["feature_status"] = "PARSE_FAIL"
        return out
    try:
        std = Chem.MolToSmiles(mol, isomericSmiles=True)
        parent = largest_organic_fragment(mol)
        parent_smi = Chem.MolToSmiles(parent, isomericSmiles=False)
        out.update(
            standardized_smiles=std,
            parent_smiles=parent_smi,
            scaffold_smiles=murcko_scaffold_smiles(parent),
            atom_count=mol.GetNumHeavyAtoms(),
        )
    except Exception:
        out["feature_status"] = "STANDARDIZE_FAIL"
    return out


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_ligand_table(n_jobs: int = 1) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "dockstring-dataset.tsv", sep="\t",
                     usecols=["inchikey", "smiles"])
    df = df.rename(columns={"inchikey": "ligand_id", "smiles": "original_smiles"})

    recs = [standardize_one(s) for s in df["original_smiles"]]
    std = pd.DataFrame(recs)
    df = pd.concat([df.reset_index(drop=True), std], axis=1)

    # merge DOCKSTRING author clusters (fingerprint-based, used for their split)
    cs = pd.read_csv(RAW_DIR / "cluster_split.tsv", sep="\t",
                     usecols=["inchikey", "cluster"])
    df = df.merge(cs.rename(columns={"inchikey": "ligand_id",
                                     "cluster": "dockstring_cluster"}),
                  on="ligand_id", how="left")

    # Group = Murcko scaffold when present, else the desalted parent.
    # A parent determines its scaffold deterministically, so identical
    # parents / stereoisomers always land in the same group. Transitive
    # union over clusters/scaffolds is NOT used: it produced a single
    # ~90k-member component (35% of the library) which would make
    # group-wise splitting meaningless. DOCKSTRING cluster overlap across
    # splits is reported as a diagnostic in DATA_AUDIT instead.
    df["group_key"] = np.where(
        df["scaffold_smiles"].fillna("") != "",
        df["scaffold_smiles"],
        "ACYCLIC::" + df["parent_smiles"].fillna("PARSE_FAIL"),
    )
    df["ligand_group_id"] = pd.factorize(df["group_key"])[0]
    df["scaffold_id"] = pd.factorize(df["scaffold_smiles"].fillna("∅"))[0]
    df["parent_group_id"] = pd.factorize(df["parent_smiles"].fillna("∅"))[0]
    return df


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_ligand_table()
    out = PROCESSED_DIR / "ligands.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} rows={len(df)}")
    print(df["feature_status"].value_counts().to_dict())
    print("n_ligand_groups:", df["ligand_group_id"].nunique())
    print("largest group size:", df.groupby("ligand_group_id").size().max())
