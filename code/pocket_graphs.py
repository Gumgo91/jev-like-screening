"""Pocket graphs for wild-type and mutated DOCKSTRING receptors.

Uses the exact featurizer of the original study (pocketgate.data.features) so that
a wild-type receptor written by make_mutants.py reproduces the stored V1 graph
bit for bit. Mutants differ only through the residues whose side chains were removed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketgate.data.audit import parse_conf, parse_pdbqt_atoms  # noqa: E402
from pocketgate.data.features import featurize_pocket_atoms  # noqa: E402
from geometry import pocket_geometry  # noqa: E402
import numpy as np  # noqa: E402


def graph_from_pdbqt(pdbqt: Path, conf: Path, target_id: str, mutated=()) -> dict:
    """Pocket graph of a receptor file. Residues listed in `mutated` as (resname, resid) pairs
    were truncated to alanine; they are relabelled ALA so the graph shows a genuine alanine."""
    atoms = parse_pdbqt_atoms(pdbqt)
    if mutated:
        want = {(r, int("".join(ch for ch in str(i) if ch.isdigit()))) for r, i in mutated}
        atoms = [(a[0], "ALA" if (a[1], a[2]) in want else a[1], *a[2:]) for a in atoms]
    box = parse_conf(conf)
    g = featurize_pocket_atoms(atoms, box, target_id=target_id)
    out = {k: (torch.as_tensor(v) if hasattr(v, "dtype") else v) for k, v in g.items()}
    out["geo"] = torch.as_tensor(pocket_geometry(pdbqt, np.array([box["center_x"], box["center_y"], box["center_z"]])))
    return out


def build_target_graphs(target: str, receptor_dir: Path, conf_dir: Path) -> dict:
    manifest = json.loads((receptor_dir / target / "manifest.json").read_text())
    out = {}
    for rec in manifest["receptors"]:
        mutated = [(x["res"], x["resid"]) for x in rec["residues"]]
        g = graph_from_pdbqt(receptor_dir / target / f"{rec['id']}.pdbqt",
                             conf_dir / f"{target}_conf.txt", target, mutated)
        ds = [x["d_sc"] for x in rec["residues"] if x["d_sc"] is not None]
        g["edit_meta"] = torch.tensor([float(rec["n_removed"]), float(len(rec["residues"])), float(min(ds)) if ds else 0.0])
        out[rec["id"]] = g
    return out


if __name__ == "__main__":
    # consistency check against the stored V1 wild-type graphs
    ref = torch.load(ROOT / "data/processed/v1_pocket_graphs.pt", weights_only=False)
    rec_dir = ROOT / "dockmut/receptors_test"
    conf_dir = ROOT / "data/raw/targets"
    for t in sorted(p.name for p in rec_dir.iterdir() if p.is_dir()):
        if t not in ref:
            print(t, "not in V1 graphs (test target)")
            continue
        g = graph_from_pdbqt(rec_dir / t / "wt.pdbqt", conf_dir / f"{t}_conf.txt", t)
        same = all(torch.equal(g[k], ref[t][k]) for k in ("x", "edge_index", "edge_attr"))
        print(t, "WT graph identical to stored V1 graph:", same, "| residues", int(g["n_residues"]))
        graphs = build_target_graphs(t, rec_dir, conf_dir)
        dx = {rid: float((gg["x"] - g["x"]).abs().sum()) if gg["x"].shape == g["x"].shape else None
              for rid, gg in graphs.items()}
        nz = {k: round(v, 2) for k, v in list(dx.items())[:6] if v is not None}
        print("   feature change vs WT (first receptors):", nz)
