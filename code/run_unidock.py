"""Batch docking of receptor variants with Uni-Dock.

For one target and a list of receptor ids the script docks the same prepared ligand
set into every receptor, parses the best-pose Vina score from each output PDBQT and
appends rows to a CSV. Timing is recorded per receptor so that throughput can be
audited. Failed or missing ligands keep an empty score.
"""
from __future__ import annotations

import argparse
import csv
import json
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--receptor-dir", type=Path, required=True)
    p.add_argument("--conf-dir", type=Path, required=True)
    p.add_argument("--ligdir", type=Path, required=True)
    p.add_argument("--receptors", default="all")
    p.add_argument("--seed", type=int, default=974528263)
    p.add_argument("--mode", default="balance", choices=["fast", "balance", "detail"])
    p.add_argument("--exhaustiveness", type=int, default=None)
    p.add_argument("--tag", default="run")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keep", action="store_true", help="keep raw outputs")
    a = p.parse_args()

    manifest = json.loads((a.receptor_dir / a.target / "manifest.json").read_text())
    ids = [r["id"] for r in manifest["receptors"]] if a.receptors == "all" else a.receptors.split(",")
    box = read_box(a.conf_dir / f"{a.target}_conf.txt")
    idx = list(csv.DictReader(open(a.ligdir / "index.csv")))
    ks = [int(r["k"]) for r in idx if r["status"] in ("ok", "cached")]
    names = {int(r["k"]): r["inchikey"] for r in idx}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    with open(a.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["target", "receptor", "tag", "mode", "seed", "k", "inchikey", "score"])
        for rid in ids:
            tmp = a.out.parent / f"_tmp_{a.target}_{rid}_{a.tag}"
            scores, wall, rc, err = dock_receptor(
                a.receptor_dir / a.target / f"{rid}.pdbqt", box, a.ligdir, ks, tmp,
                a.seed, a.mode, a.exhaustiveness, [])
            for k in ks:
                w.writerow([a.target, rid, a.tag, a.mode if not a.exhaustiveness else f"e{a.exhaustiveness}",
                            a.seed, k, names[k], scores.get(k, "")])
            fh.flush()
            print(json.dumps({"target": a.target, "receptor": rid, "tag": a.tag, "n_ok": len(scores),
                              "n": len(ks), "sec": round(wall, 1), "rc": rc, "err": err if rc else ""}),
                  flush=True)
            if not a.keep:
                shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
