"""Ligand preparation that follows the DOCKSTRING recipe step by step.

canonical SMILES -> Uncharger -> single fragment -> Open Babel protonation at pH 7.4
-> RDKit AddHs + ETKDG embedding (seed 974528263) -> MMFF94 (UFF fallback) ->
MOL file -> Open Babel PDBQT with Gasteiger charges.

Runs in parallel over CPU cores and writes one PDBQT per ligand plus an index, so every
receptor variant is docked with byte-identical ligand inputs. A failed ligand keeps its
failure state; no score is ever substituted.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
SEED = 974528263
OBABEL = os.environ.get("OBABEL", "obabel")


def protonate_smiles(smiles: str, ph: float = 7.4) -> str:
    r = subprocess.run([OBABEL, f"-:{smiles}", "-ismi", "-ocan", f"-p{ph}"],
                       capture_output=True, text=True, timeout=60)
    toks = r.stdout.strip().split()
    if not toks:
        raise ValueError("protonation returned nothing")
    return toks[0]


def prepare_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(smiles))
    if mol is None:
        raise ValueError("unparsable")
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("multiple fragments")
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError("radical")
    mol = Chem.MolFromSmiles(protonate_smiles(Chem.MolToSmiles(mol)))
    if mol is None:
        raise ValueError("protonated smiles unparsable")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=SEED, maxAttempts=10 * mol.GetNumAtoms()) != 0:
        raise ValueError("embedding failed")
    try:
        if AllChem.MMFFOptimizeMolecule(mol, maxIters=1000) == -1:
            raise ValueError("no MMFF params")
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def prepare_one(args):
    k, inchikey, smiles, outdir = args
    target = Path(outdir) / f"{k}.pdbqt"
    if target.exists() and target.stat().st_size > 0:
        return k, inchikey, "cached"
    try:
        mol = prepare_mol(smiles)
        with tempfile.TemporaryDirectory() as td:
            molfile = Path(td) / "l.mol"
            Chem.MolToMolFile(mol, str(molfile))
            subprocess.run([OBABEL, "-imol", str(molfile), "-opdbqt", "-O", str(target),
                            "--partialcharge", "gasteiger"],
                           capture_output=True, text=True, timeout=120)
        if not target.exists() or target.stat().st_size == 0:
            return k, inchikey, "pdbqt_failed"
        return k, inchikey, "ok"
    except Exception as e:
        return k, inchikey, f"error:{e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ligands", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--procs", type=int, default=16)
    a = p.parse_args()
    df = pd.read_csv(a.ligands)
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, r.inchikey, r.smiles, str(a.out)) for k, r in enumerate(df.itertuples())]
    with mp.Pool(a.procs) as pool:
        res = pool.map(prepare_one, jobs, chunksize=4)
    idx = pd.DataFrame(res, columns=["k", "inchikey", "status"])
    idx.to_csv(a.out / "index.csv", index=False)
    print(json.dumps(idx.status.value_counts().to_dict()))


if __name__ == "__main__":
    main()
