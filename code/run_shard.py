"""Run one DockMut shard on a GPU pod (resumable).

1. regenerate the truncated receptors from the wild-type PDBQTs with the same seeds as the
   plan and verify every manifest hash against the plan
2. prepare ligand PDBQTs for the shard (DOCKSTRING recipe)
3. dock wild-type replicates and every mutant with Uni-Dock, appending rows to one CSV per
   target; already finished (target, receptor, tag) units are skipped after an interruption
4. keep the seed-1 wild-type poses (compact) for contact analysis
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_mutants import build_target  # noqa: E402
from run_unidock import dock_receptor, read_box  # noqa: E402

SEED = 20261001


def sha(path: Path) -> str:
    """Canonical JSON hash, independent of newline style and key order."""
    return hashlib.sha256(json.dumps(json.loads(path.read_text()), sort_keys=True).encode()).hexdigest()


def _sha_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def done_units(csv_path: Path) -> dict:
    done = {}
    if csv_path.exists():
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                done[(row["receptor"], row["tag"])] = done.get((row["receptor"], row["tag"]), 0) + 1
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("/workspace/dm"))
    p.add_argument("--shard", required=True)
    a = p.parse_args()
    plan = json.loads((a.base / "plan.json").read_text())
    shards = json.loads((a.base / "shards.json").read_text())
    targets = shards[a.shard]
    (a.base / "results").mkdir(exist_ok=True)
    (a.base / "poses").mkdir(exist_ok=True)
    log = open(a.base / f"shard_{a.shard}.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    t_all = time.time()
    for t in targets:
        info = plan["targets"][t]
        rec_dir = a.base / "receptors"
        build_target(t, a.base / "wt_raw", rec_dir, SEED + sum(map(ord, t)), 12, 3, 1, 8)
        if sha(rec_dir / t / "manifest.json") != info["manifest_sha256"]:
            say(f"ABORT {t}: regenerated manifest differs from plan")
            sys.exit(3)
        lig_csv = a.base / "ligands" / info["ligand_file"]
        lig_dir = a.base / "lig" / Path(info["ligand_file"]).stem
        if not (lig_dir / "index.csv").exists():
            subprocess.run([sys.executable, str(HERE / "prep_ligands.py"), "--ligands", str(lig_csv),
                            "--out", str(lig_dir), "--procs", "16"], check=True)
        idx = list(csv.DictReader(open(lig_dir / "index.csv")))
        ks = [int(r["k"]) for r in idx if r["status"] in ("ok", "cached")]
        names = {int(r["k"]): r["inchikey"] for r in idx}
        box = read_box(a.base / "conf" / f"{t}_conf.txt")
        out_csv = a.base / "results" / f"{t}.csv"
        done = done_units(out_csv)
        manifest = json.loads((rec_dir / t / "manifest.json").read_text())
        jobs = [("wt", f"wt_s{s}", s) for s in info["wt_seeds"]] + \
               [(r["id"], "mut_s1", 1) for r in manifest["receptors"] if r["id"] != "wt"]
        new = not out_csv.exists()
        with open(out_csv, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["target", "receptor", "tag", "mode", "seed", "k", "inchikey", "score"])
            for rid, tag, seed in jobs:
                if done.get((rid, tag), 0) >= len(ks):
                    continue
                tmp = a.base / "tmp" / f"{t}_{rid}_{tag}"
                scores, wall, rc, err = dock_receptor(rec_dir / t / f"{rid}.pdbqt", box, lig_dir, ks, tmp,
                                                       seed, "detail", None, [])
                for k in ks:
                    w.writerow([t, rid, tag, "detail", seed, k, names[k], scores.get(k, "")])
                fh.flush()
                if rid == "wt" and seed == 1:
                    keep = a.base / "poses" / t
                    keep.mkdir(parents=True, exist_ok=True)
                    for k in ks:
                        f = tmp / f"{k}_out.pdbqt"
                        if f.exists():
                            shutil.copy(f, keep / f"{k}.pdbqt")
                shutil.rmtree(tmp, ignore_errors=True)
                say(json.dumps({"target": t, "receptor": rid, "tag": tag, "n_ok": len(scores), "n": len(ks),
                                "sec": round(wall, 1), "rc": rc, "elapsed_min": round((time.time() - t_all) / 60, 1)}))
        say(f"TARGET_DONE {t}")
    say("SHARD_DONE " + a.shard)


if __name__ == "__main__":
    main()
