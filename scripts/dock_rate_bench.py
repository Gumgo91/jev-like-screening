"""Docking rate of Uni-Dock on one GPU, with a record of every call.

For each target the receptor is docked against the first n prepared ligands in every configuration (search mode and batch size, the
number n of ligands per call is the batch). Every call is written as one row of the timing table (target, mode, n ligands, repetition,
seconds, ligands docked, return code, warm-up flag, GPU name and driver), so that the rate in the manuscript can be recomputed from the
file. The first call of each target is a warm-up and is labeled. For the first repetition of each configuration the best-pose scores
are written to the score table (column k is the row of the ligand list), to compare the faster modes with the detail mode. The
outputs of a call are written to <workdir>/<target>_<mode>_<n> and removed after the call (--keep keeps them).

The benchmark behind eval/docking_timing*.csv, eval/docking_scores*.csv and eval/prep4k.log ran on a RunPod pod with one RTX 4090
(NVIDIA driver 570.169, 96 visible CPU threads) in the micromamba environment of scripts/pod_setup.sh (Uni-Dock 1.2.0, Open Babel 3.2.1,
RDKit 2026.03.6) with the repository as working directory and /workspace/mamba/envs/ud/bin first on PATH. The commands, in order:

  python scripts/make_bench_ligands.py          # bench/ligands.csv (1,024 molecules) and bench/ligands4k.csv (4,096 molecules)

  python scripts/prep_ligands.py --ligands bench/ligands.csv --out bench/lig --procs 64
  python scripts/dock_rate_bench.py --ligdir bench/lig --targets ABL1,MAOB,PTGS2 \\
      --configs detail:160:3,detail:256:3,balance:256:3,fast:256:3 --out eval/docking_timing.csv --scores eval/docking_scores.csv
  python scripts/dock_rate_bench.py --ligdir bench/lig --targets ABL1,MAOB,PTGS2 --min-n 1000 \\
      --out eval/docking_timing_big.csv --scores eval/docking_scores_big.csv

  python scripts/prep_ligands.py --ligands bench/ligands4k.csv --out bench/lig4k --procs 32 > eval/prep4k.log
  python scripts/dock_rate_bench.py --ligdir bench/lig4k --targets ABL1,MAOB,PTGS2 \\
      --configs fast:4000:2,balance:4000:1,detail:4000:1 --out eval/docking_timing_4k.csv --scores eval/docking_scores_4k.csv

The first docking command covers the batches of 160 and 256 ligands (1,023 of the 1,024 ligands could be prepared), the second the
batches of 1,000 ligands (the default configuration list restricted to n >= 1000) and the third the batches of 4,000 ligands (4,093 of
the 4,096 ligands could be prepared). The tables are the record of that pod; a rerun on another GPU gives other seconds.
Relative paths are taken relative to the repository root. --receptor-dir holds <T>_target.pdbqt and <T>_conf.txt.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_unidock import dock_receptor, read_box  # noqa: E402

CONFIGS = [("detail", 160, 3), ("detail", 256, 3), ("detail", 1000, 2), ("balance", 256, 3), ("balance", 1000, 2), ("fast", 256, 3), ("fast", 1000, 2)]
SEED = 974528263


def resolve(p: Path) -> Path:
    return p if p.is_absolute() else ROOT / p


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ligdir", type=Path, required=True, help="folder written by scripts/prep_ligands.py (<k>.pdbqt and index.csv)")
    p.add_argument("--targets", default="ABL1,MAOB,PTGS2")
    p.add_argument("--receptor-dir", type=Path, default=Path("data/raw/targets"))
    p.add_argument("--out", type=Path, default=Path("bench/docking_timing.csv"), help="timing table (one row per call)")
    p.add_argument("--scores", type=Path, default=Path("bench/docking_scores.csv"), help="score table (first repetition of each configuration)")
    p.add_argument("--workdir", type=Path, default=Path("bench/work"), help="folder for the Uni-Dock outputs of the running call")
    p.add_argument("--keep", action="store_true", help="keep the Uni-Dock outputs of every call")
    p.add_argument("--min-n", type=int, default=0, help="run only the configurations with at least this many ligands per call")
    p.add_argument("--configs", default=None, help="mode:n:repetitions,... in place of the default configurations")
    a = p.parse_args()
    ligdir, receptor_dir, out, scores_out, workdir = (resolve(x) for x in (a.ligdir, a.receptor_dir, a.out, a.scores, a.workdir))
    idx = [r for r in csv.DictReader(open(ligdir / "index.csv")) if r["status"] in ("ok", "cached")]
    ks = [int(r["k"]) for r in idx]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()[0]
    print("gpu", gpu, "ligands prepared", len(ks), flush=True)
    configs = [(c.split(":")[0], int(c.split(":")[1]), int(c.split(":")[2])) for c in a.configs.split(",")] if a.configs else CONFIGS
    rows, srows = [], []
    for t in a.targets.split(","):
        rec = receptor_dir / f"{t}_target.pdbqt"
        box = read_box(receptor_dir / f"{t}_conf.txt")
        warm = True
        for mode, n, reps in configs:
            if n > len(ks) or n < a.min_n:
                continue
            for rep in range(reps):
                tmp = workdir / f"{t}_{mode}_{n}"
                scores, wall, rc, err = dock_receptor(rec, box, ligdir, ks[:n], tmp, SEED, mode, None, [])
                rows.append({"target": t, "mode": mode, "n": n, "rep": rep, "sec": round(wall, 3), "n_ok": len(scores), "rc": rc, "warmup": int(warm), "gpu": gpu})
                if rep == 0:
                    srows += [{"target": t, "mode": mode, "n": n, "k": k, "score": v} for k, v in scores.items()]
                if not a.keep:
                    shutil.rmtree(tmp, ignore_errors=True)
                print(rows[-1], flush=True)
                warm = False
    if not rows:
        sys.exit("no configuration was run (n larger than the number of prepared ligands, or --min-n too large)")
    for path in (out, scores_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(scores_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "mode", "n", "k", "score"])
        w.writeheader()
        w.writerows(srows)
    if not a.keep:
        try:
            workdir.rmdir()
        except OSError:
            pass
    print("done", len(rows), "calls", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
