"""Ligand lists of the Uni-Dock rate benchmark: seeded random samples of the DOCKSTRING collection.

The collection is taken in the file order of data/raw/dockstring-dataset.tsv (260,155 molecules). The first sample draws 1,024
molecules with pandas DataFrame.sample(1024, random_state=0). The second draws 3,072 further molecules from the remaining ones with
DataFrame.sample(3072, random_state=1). Two files are written with the columns inchikey and smiles:
  bench/ligands.csv    the 1,024 molecules of the first sample, in sampling order
  bench/ligands4k.csv  the 1,024 molecules of the first sample followed by the 3,072 of the second (4,096 molecules)
smiles is the standardized SMILES of the molecule, the RDKit canonical isomeric SMILES of the DOCKSTRING SMILES (the same string as
the column standardized_smiles of data/processed/ligands.parquet, which scripts/make_bench_ligands.py does not need). Because the
sampling is positional, only the sampled molecules are standardized.
The row k of a file is ligand k of scripts/prep_ligands.py, so the column k of eval/docking_scores*.csv indexes these rows.

Usage (repository root): python scripts/make_bench_ligands.py [--tsv data/raw/dockstring-dataset.tsv] [--out-dir bench]
Relative paths are taken relative to the repository root.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
N_FIRST, SEED_FIRST = 1024, 0
N_SECOND, SEED_SECOND = 3072, 1
N_LIBRARY = 260155

RDLogger.DisableLog("rdApp.*")


def standardize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def sample_lists(tsv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (first sample, first sample + second sample) as frames with the columns inchikey and smiles."""
    d = pd.read_csv(tsv, sep="\t", usecols=["inchikey", "smiles"])
    assert d.inchikey.is_unique, "inchikey is not unique in the DOCKSTRING table"
    if len(d) != N_LIBRARY:
        print(f"[warning] {len(d)} rows in {tsv.name}, the benchmark lists were drawn from {N_LIBRARY}", file=sys.stderr)
    first = d.sample(N_FIRST, random_state=SEED_FIRST)
    second = d.drop(first.index).sample(N_SECOND, random_state=SEED_SECOND)
    both = pd.concat([first, second])
    both = pd.DataFrame({"inchikey": both.inchikey.to_numpy(), "smiles": [standardize(s) for s in both.smiles]})
    return both.iloc[:N_FIRST].reset_index(drop=True), both.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tsv", type=Path, default=Path("data/raw/dockstring-dataset.tsv"))
    ap.add_argument("--out-dir", type=Path, default=Path("bench"))
    a = ap.parse_args()
    tsv = a.tsv if a.tsv.is_absolute() else ROOT / a.tsv
    out = a.out_dir if a.out_dir.is_absolute() else ROOT / a.out_dir
    if not tsv.exists():
        sys.exit(f"{tsv} not found; download the DOCKSTRING dataset (figshare article 16511577) first")
    small, big = sample_lists(tsv)
    out.mkdir(parents=True, exist_ok=True)
    small.to_csv(out / "ligands.csv", index=False)
    big.to_csv(out / "ligands4k.csv", index=False)
    print(f"wrote {len(small)} molecules to {out / 'ligands.csv'} and {len(big)} to {out / 'ligands4k.csv'}")


if __name__ == "__main__":
    main()
