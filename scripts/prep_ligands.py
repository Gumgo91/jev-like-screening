"""Ligand preparation that follows the DOCKSTRING recipe step by step.

canonical SMILES -> Uncharger -> single fragment -> Open Babel protonation at pH 7.4
-> RDKit AddHs + ETKDG embedding (seed 974528263) -> MMFF94 (UFF fallback) ->
MOL file -> Open Babel PDBQT with Gasteiger charges.

Runs in parallel over CPU cores and writes one PDBQT per ligand (<k>.pdbqt, k = row of the ligand list) plus index.csv (k,
inchikey, status), so every receptor is docked with byte-identical ligand inputs. A failed ligand keeps its failure state in the
index. The count of each status is printed as JSON on stdout.
Needs RDKit and the Open Babel executable (OBABEL environment variable, default obabel on PATH).

Usage (repository root): python scripts/prep_ligands.py --ligands bench/ligands.csv --out bench/lig --procs 64
The ligand list has the columns inchikey and smiles (scripts/make_bench_ligands.py). Relative paths are taken relative to the
repository root. The exact commands of the Uni-Dock benchmark are in the docstring of scripts/dock_rate_bench.py.
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

ROOT = Path(__file__).resolve().parents[1]
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
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ligands", type=Path, default=Path("bench/ligands.csv"), help="CSV with the columns inchikey and smiles")
    p.add_argument("--out", type=Path, default=Path("bench/lig"), help="output folder (<k>.pdbqt and index.csv)")
    p.add_argument("--procs", type=int, default=16)
    a = p.parse_args()
    ligands = a.ligands if a.ligands.is_absolute() else ROOT / a.ligands
    out = a.out if a.out.is_absolute() else ROOT / a.out
    df = pd.read_csv(ligands)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, r.inchikey, r.smiles, str(out)) for k, r in enumerate(df.itertuples())]
    with mp.Pool(a.procs) as pool:
        res = pool.map(prepare_one, jobs, chunksize=4)
    idx = pd.DataFrame(res, columns=["k", "inchikey", "status"])
    idx.to_csv(out / "index.csv", index=False)
    print(json.dumps(idx.status.value_counts().to_dict()))


if __name__ == "__main__":
    main()
