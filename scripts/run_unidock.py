"""Uni-Dock call used by scripts/dock_rate_bench.py.

read_box parses the search box of a DOCKSTRING target (<T>_conf.txt). dock_receptor docks one prepared ligand set (<ligdir>/<k>.pdbqt)
into one receptor with a single Uni-Dock call (Vina scoring function, one output pose, the given search mode) and returns the best-pose
Vina score of every ligand that produced one, the wall time of the call, the return code and the end of stderr.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

RESULT = re.compile(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)")


def read_box(path: Path) -> dict:
    v = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, val = line.split("=", 1)
            v[k.strip()] = float(val)
    return v


def dock_receptor(receptor: Path, box: dict, ligdir: Path, ks: list[int], out: Path,
                  seed: int, mode: str, exhaustiveness: int | None, extra: list[str]):
    """Dock the ligands <ligdir>/<k>.pdbqt, k in ks, into `receptor`. The output folder `out` is emptied first and kept afterwards.
    -> (scores {k: best-pose score}, wall seconds of the unidock call, return code, last 300 characters of stderr)."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    idx = out / "index.txt"
    idx.write_text("\n".join(str((ligdir / f"{k}.pdbqt").resolve()) for k in ks) + "\n")
    cmd = ["unidock", "--receptor", str(receptor), "--ligand_index", str(idx),
           "--center_x", str(box["center_x"]), "--center_y", str(box["center_y"]),
           "--center_z", str(box["center_z"]), "--size_x", str(box["size_x"]),
           "--size_y", str(box["size_y"]), "--size_z", str(box["size_z"]),
           "--scoring", "vina", "--num_modes", "1", "--seed", str(seed),
           "--dir", str(out), "--verbosity", "0"]
    if exhaustiveness:
        cmd += ["--exhaustiveness", str(exhaustiveness)]
    else:
        cmd += ["--search_mode", mode]
    cmd += extra
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    scores = {}
    for k in ks:
        f = out / f"{k}_out.pdbqt"
        if f.exists():
            m = RESULT.search(f.read_text())
            if m:
                scores[k] = float(m.group(1))
    return scores, wall, r.returncode, r.stderr[-300:]
