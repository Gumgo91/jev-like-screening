"""Pooled pose-contact analysis over all targets.

For every target the seed-1 wild-type poses are matched with the measured changes. Contacts are the number of ligand
heavy atoms within 4.5 A of the removed side-chain atoms. Outputs: changes by contact bin, correlations by edit class,
and a per-target table.
"""
from __future__ import annotations

import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
from contacts import contact_table  # noqa: E402
from dm_core import delta_table, read_results  # noqa: E402


def main(results: Path, poses: Path, out: Path):
    plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for t, info in plan["targets"].items():
        f = poses / f"{t}.tgz"
        if not f.exists() or not (results / f"{t}.csv").exists():
            continue
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(f) as tar:
                tar.extractall(td)
            lig = pd.read_csv(ROOT / "dockmut/plan/ligands" / info["ligand_file"])
            lig["k"] = range(len(lig))
            ct = contact_table(t, ROOT / "dockmut/plan", Path(td), lig)
        man = json.loads((ROOT / "dockmut/plan/receptors" / t / "manifest.json").read_text())
        tab, _, _ = delta_table(read_results(results, t), man)
        rows = [(rid, kind, i, d) for rid, (ids, dS, kind) in tab.items() for i, d in zip(ids, dS)]
        D = pd.DataFrame(rows, columns=["receptor", "kind", "inchikey", "dS"]).merge(ct, on=["receptor", "inchikey"])
        D["role"] = info["role"]
        frames.append(D)
        print(t, len(D), flush=True)
    D = pd.concat(frames)
    D.to_parquet(out / "contact_changes.parquet", index=False)
    cont = D[D.kind.isin(["single_contact"]) | D.kind.str.startswith("multi")]
    null = D[D.kind.isin(["single_shell", "single_far"])]
    bins = pd.cut(D.n_contact, [-1, 0, 2, 5, 100], labels=["0", "1-2", "3-5", "6+"])
    by = D.assign(bin=bins, absdS=D.dS.abs(), cls=np.where(D.kind.isin(["single_shell", "single_far"]), "control", "contact")) \
        .groupby(["cls", "bin"], observed=True).agg(n=("dS", "size"), mean_abs=("absdS", "mean"), mean=("dS", "mean")).round(3)
    print(by)
    res = {"r_contacts_absdS_contact": float(np.corrcoef(cont.n_contact, cont.dS.abs())[0, 1]),
           "r_contacts_absdS_control": float(np.corrcoef(null.n_contact, null.dS.abs())[0, 1]),
           "by_bin": {f"{a}|{b}": v for (a, b), v in by.to_dict("index").items()}}
    (out / "contacts_summary.json").write_text(json.dumps(res, indent=1, default=str))
    print({k: v for k, v in res.items() if k != "by_bin"})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
