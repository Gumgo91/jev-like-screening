"""Docking rate of Uni-Dock on one GPU, with a record of every call.

For each target the wild-type receptor is docked against the first n prepared ligands in every configuration (search mode and batch
size). Every call is written as one row (target, mode, n ligands, seconds, ligands docked, return code), so that the rate in the
manuscript can be recomputed from the file. The first call of a run is a warm-up and is labeled. For the first repetition the best-pose
scores are kept, to compare the faster modes with the detail mode.
Usage: python dock_rate_bench.py --ligdir DIR --targets ABL1,MAOB,PTGS2 --out timing.csv --scores scores.csv
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_unidock import dock_receptor, read_box  # noqa: E402

CONFIGS = [("detail", 160, 3), ("detail", 256, 3), ("detail", 1000, 2), ("balance", 256, 3), ("balance", 1000, 2), ("fast", 256, 3), ("fast", 1000, 2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ligdir", type=Path, required=True)
    p.add_argument("--targets", default="ABL1,MAOB,PTGS2")
    p.add_argument("--receptor-dir", type=Path, default=HERE.parent / "data/raw/targets")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--min-n", type=int, default=0, help="run only the configurations with at least this many ligands per call")
    p.add_argument("--configs", default=None, help="mode:n:repetitions,... in place of the default configurations")
    a = p.parse_args()
    idx = [r for r in csv.DictReader(open(a.ligdir / "index.csv")) if r["status"] in ("ok", "cached")]
    ks = [int(r["k"]) for r in idx]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()[0]
    print("gpu", gpu, "ligands prepared", len(ks), flush=True)
    rows, srows = [], []
    for t in a.targets.split(","):
        rec = a.receptor_dir / f"{t}_target.pdbqt"
        box = read_box(a.receptor_dir / f"{t}_conf.txt")
        warm = True
        configs = [(c.split(":")[0], int(c.split(":")[1]), int(c.split(":")[2])) for c in a.configs.split(",")] if a.configs else CONFIGS
        for mode, n, reps in configs:
            if n > len(ks) or n < a.min_n:
                continue
            for rep in range(reps):
                tmp = a.out.parent / f"_tmp_{t}_{mode}_{n}"
                scores, wall, rc, err = dock_receptor(rec, box, a.ligdir, ks[:n], tmp, 974528263, mode, None, [])
                rows.append({"target": t, "mode": mode, "n": n, "rep": rep, "sec": round(wall, 3), "n_ok": len(scores), "rc": rc, "warmup": int(warm), "gpu": gpu})
                if rep == 0:
                    srows += [{"target": t, "mode": mode, "n": n, "k": k, "score": v} for k, v in scores.items()]
                print(rows[-1], flush=True)
                warm = False
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(a.scores, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "mode", "n", "k", "score"])
        w.writeheader()
        w.writerows(srows)
    print("done", len(rows), "calls", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
