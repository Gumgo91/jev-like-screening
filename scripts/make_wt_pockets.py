"""Build the pocket graphs of the full-scale timing from the public DOCKSTRING receptors.

One residue graph per DOCKSTRING target (pocketgate.data.features.featurize_pocket on <targets-dir>/<T>_target.pdbqt and
<T>_conf.txt), for all targets except DRD2 (its search box exceeds the Uni-Dock grid limit), in sorted order. For the DOCKSTRING
release this gives the 57 pockets of the timing. The graphs are saved as {target: {x, edge_index, edge_attr}} of torch tensors to
data/processed/pocket_graphs_wt.pt, the default --pockets file of scripts/v4_fair_timing.py.

Usage (repository root): python scripts/make_wt_pockets.py [--targets-dir data/raw/targets] [--out data/processed/pocket_graphs_wt.pt]
Relative paths are taken relative to the repository root.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketgate.data.features import featurize_pocket  # noqa: E402

EXCLUDE = ("DRD2",)


def target_names(targets_dir: Path, exclude=EXCLUDE) -> list[str]:
    names = sorted(p.name[: -len("_target.pdbqt")] for p in targets_dir.glob("*_target.pdbqt"))
    return [t for t in names if t not in exclude]


def build(targets_dir: Path, exclude=EXCLUDE) -> dict:
    out = {}
    for t in target_names(targets_dir, exclude):
        g = featurize_pocket(t, targets_dir=targets_dir)
        out[t] = {"x": torch.from_numpy(g["x"]), "edge_index": torch.from_numpy(g["edge_index"]),
                  "edge_attr": torch.from_numpy(g["edge_attr"])}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets-dir", type=Path, default=Path("data/raw/targets"),
                    help="folder with <T>_target.pdbqt and <T>_conf.txt (DOCKSTRING resources/targets)")
    ap.add_argument("--out", type=Path, default=Path("data/processed/pocket_graphs_wt.pt"))
    ap.add_argument("--exclude", default=",".join(EXCLUDE), help="comma list of targets to leave out")
    a = ap.parse_args()
    tdir = a.targets_dir if a.targets_dir.is_absolute() else ROOT / a.targets_dir
    out = a.out if a.out.is_absolute() else ROOT / a.out
    exclude = tuple(t for t in a.exclude.split(",") if t)
    if not tdir.is_dir() or not list(tdir.glob("*_target.pdbqt")):
        sys.exit(f"no *_target.pdbqt files in {tdir}; download the DOCKSTRING targets first")
    graphs = build(tdir, exclude)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, out)
    sizes = [g["x"].shape[0] for g in graphs.values()]
    print(f"wrote {len(graphs)} pocket graphs ({min(sizes)} to {max(sizes)} residues) to {out}")


if __name__ == "__main__":
    main()
