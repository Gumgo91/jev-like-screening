"""Global geometric descriptors of a receptor pocket computed from coordinates.

Probe points on a 1 A grid inside a 10 A sphere around the box center are free when no receptor heavy
atom lies within 2.6 A. A free point counts as enclosed when at least 10 of 14 rays of length 10 A hit the
receptor. Enclosed free volume at several radii, buriedness, pocket polarity, aromatic content and net charge
summarise the pocket in ten numbers. Truncating a side chain opens volume, which the descriptors register.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

GEO_NAMES = ["vol_r4", "vol_r6", "vol_r8", "vol_r10", "buried_frac", "n_atoms_r10", "polar_frac", "arom_frac",
             "charge_r10", "width"]
_DIRS = None


def _directions(n=14):
    global _DIRS
    if _DIRS is None:
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        theta = np.pi * (1 + 5 ** 0.5) * i
        _DIRS = np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    return _DIRS


def read_atoms(pdbqt: Path):
    xyz, adt, chg = [], [], []
    for line in pdbqt.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            t = line[77:79].strip() if len(line) >= 79 else line.split()[-1]
            if t.startswith("H"):
                continue
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            adt.append(t)
            try:
                chg.append(float(line[70:76]))
            except ValueError:
                chg.append(0.0)
    return np.array(xyz), np.array(adt), np.array(chg)


def pocket_geometry(pdbqt: Path, center: np.ndarray) -> np.ndarray:
    xyz, adt, chg = read_atoms(pdbqt)
    tree = cKDTree(xyz)
    g = np.arange(-10, 10.01, 1.0)
    grid = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
    grid = grid[(grid ** 2).sum(1) <= 100.0] + center
    d_near, _ = tree.query(grid)
    free = grid[d_near > 2.6]
    dirs = _directions()
    if len(free):
        steps = np.arange(1.5, 10.01, 1.5)
        pts = free[:, None, None, :] + steps[None, None, :, None] * dirs[None, :, None, :]
        dd, _ = tree.query(pts.reshape(-1, 3))
        hit = (dd.reshape(len(free), len(dirs), len(steps)) < 1.8).any(2).sum(1)
        enclosed = hit >= 10
    else:
        enclosed = np.zeros(0, bool)
    rc = np.linalg.norm(free - center, axis=1) if len(free) else np.zeros(0)
    vols = [float(((rc <= r) & enclosed).sum()) for r in (4, 6, 8, 10)]
    near = np.linalg.norm(xyz - center, axis=1) <= 10.0
    at, ch = adt[near], chg[near]
    n = max(int(near.sum()), 1)
    polar = float(np.isin(at, ["N", "NA", "OA", "SA"]).sum()) / n
    arom = float((at == "A").sum()) / n
    width = float(d_near[d_near > 2.6][enclosed.nonzero()[0].clip(0, max(len(free) - 1, 0))].mean()) if 0 else (
        float(np.mean(tree.query(free[enclosed])[0])) if enclosed.any() else 0.0)
    buried = float(enclosed.mean()) if len(enclosed) else 0.0
    return np.array([*vols, buried, n / 100.0, polar, arom, float(ch.sum()) / 10.0, width], dtype=np.float32)
