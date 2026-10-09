"""Cross-check of measured score changes with the original AutoDock Vina 1.1.2 (CPU).

For a few targets the script picks the two single and two multi-residue truncations with the largest measured
effect in the Uni-Dock data, re-docks 32 shared ligands into those receptors and the wild type with Vina 1.1.2
(exhaustiveness 8, the DOCKSTRING seed) and writes the scores. Comparing the two change tables shows whether the
measured effects depend on the engine. Runs on a pod where the receptors, prepared ligands and results exist.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def delta_table(results_dir, target, man):
    """Minimal copy of the change table (pandas only): rid -> (ids, dS, kind)."""
    df = pd.read_csv(Path(results_dir) / f"{target}.csv").dropna(subset=["score"])
    df = df.drop_duplicates(["receptor", "tag", "inchikey"], keep="last")
    wt = df[df.receptor == "wt"].pivot_table(index="inchikey", columns="seed", values="score")
    mean = wt.mean(axis=1)
    sd = wt.std(axis=1, ddof=1)
    stable = sd[(sd < 1.0) & (mean.reindex(sd.index) < 0)].index
    kinds = {r["id"]: r["kind"] for r in man["receptors"]}
    out = {}
    for rid, g in df[(df.receptor != "wt") & (df.tag == "mut_s1")].groupby("receptor"):
        s = g.set_index("inchikey").score
        ids = s.index.intersection(stable)
        if len(ids) >= 20:
            out[rid] = (np.array(ids), (s[ids] - mean[ids]).clip(-4, 4).to_numpy(), kinds[rid])
    return out

VINA = "/workspace/vina/vina_linux"


def job(args):
    target, rid, k, rec, conf, lig, out = args
    f = Path(out) / f"{target}_{rid}_{k}.pdbqt"
    subprocess.run([VINA, "--receptor", rec, "--ligand", lig, "--config", conf, "--seed", "974528263", "--cpu", "1",
                    "--out", str(f)], capture_output=True, text=True, timeout=1800)
    score = ""
    if f.exists():
        for line in f.read_text().splitlines():
            if line.startswith("REMARK VINA RESULT"):
                score = line.split()[3]
                break
    return target, rid, k, score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("/workspace/dm"))
    p.add_argument("--targets", required=True)
    p.add_argument("--n-lig", type=int, default=32)
    p.add_argument("--procs", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("/workspace/dm/vina_check"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((a.base / "plan.json").read_text())
    tasks = []
    chosen = {}
    for t in a.targets.split(","):
        man = json.loads((a.base / "receptors" / t / "manifest.json").read_text())
        tab = delta_table(a.base / "results", t, man)
        singles = sorted([r for r, v in tab.items() if v[2] == "single_contact"], key=lambda r: -tab[r][1].std())[:2]
        multis = sorted([r for r, v in tab.items() if v[2].startswith("multi")], key=lambda r: -tab[r][1].std())[:2]
        chosen[t] = ["wt"] + singles + multis
        lig_dir = a.base / "lig" / Path(plan["targets"][t]["ligand_file"]).stem
        idx = list(csv.DictReader(open(lig_dir / "index.csv")))[: a.n_lig]
        for rid in chosen[t]:
            for r in idx:
                tasks.append((t, rid, int(r["k"]), str(a.base / "receptors" / t / f"{rid}.pdbqt"),
                              str(a.base / "conf" / f"{t}_conf.txt"), str(lig_dir / f"{r['k']}.pdbqt"), str(a.out)))
    print(len(tasks), "dockings", flush=True)
    with mp.Pool(a.procs) as pool:
        res = pool.map(job, tasks, chunksize=1)
    with open(a.out / "vina_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "receptor", "k", "score_vina"])
        w.writerows(res)
    (a.out / "chosen.json").write_text(json.dumps(chosen, indent=1))
    print("VINA_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
