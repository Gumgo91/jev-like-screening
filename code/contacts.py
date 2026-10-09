"""Pose-based contact analysis of receptor truncations.

For each (truncated receptor, ligand) the script counts heavy atoms of the wild-type docked pose that
lie within a cutoff of the removed side-chain atoms. The resulting contact matrix is a pose-informed
reference that a two-dimensional surrogate cannot see. It explains which ligands are affected by which edit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
from make_mutants import BACKBONE, KEEP_ALA, parse_line  # noqa: E402


def removed_coords(wt_pdbqt: Path, residues: list[dict]) -> np.ndarray:
    want = {(r["res"], r["resid"]) for r in residues}
    xyz = []
    for line in wt_pdbqt.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            name, res, chain, resid, c, adt = parse_line(line)
            if (res, resid) in want and name not in KEEP_ALA:
                xyz.append(c)
    return np.array(xyz)


def pose_heavy_atoms(path: Path) -> np.ndarray:
    xyz = []
    for line in path.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM", "HETATM")):
            adt = line[77:79].strip()
            if not adt.startswith("H"):
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(xyz)


def contact_table(target: str, plan_dir: Path, pose_dir: Path, lig_index: pd.DataFrame, cutoff=4.5):
    man = json.loads((plan_dir / "receptors" / target / "manifest.json").read_text())
    wt = plan_dir / "wt_raw" / f"{target}_target.pdbqt"
    rows = []
    poses = {}
    for k, inchikey in zip(lig_index.k, lig_index.inchikey):
        f = pose_dir / target / f"{k}.pdbqt"
        if f.exists():
            poses[inchikey] = pose_heavy_atoms(f)
    for rec in man["receptors"]:
        if rec["id"] == "wt":
            continue
        rc = removed_coords(wt, rec["residues"])
        if len(rc) == 0:
            continue
        for ik, P in poses.items():
            if len(P) == 0:
                continue
            d = np.sqrt(((P[:, None, :] - rc[None, :, :]) ** 2).sum(-1))
            rows.append((target, rec["id"], ik, int((d.min(1) < cutoff).sum()), float(d.min())))
    return pd.DataFrame(rows, columns=["target", "receptor", "inchikey", "n_contact", "min_dist"])
