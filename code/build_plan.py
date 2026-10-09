"""Build the DockMut-1 docking plan and the per-shard upload bundles.

Design (fixed before any main-scale docking):
  targets     : 57 DOCKSTRING targets (DRD2 excluded, its 30x30x32 A box exceeds the
                Uni-Dock grid limit)
  receptors   : wild type, 12 single contact truncations, 3 shell and 1 far single
                truncation (null controls), 8 multi-residue contact truncations
  ligands     : train targets  -> 256 ligands per target from the 40k G2 train pool
                held-out targets (dev + test roles) -> one shared set of 160 dev ligands
  replicates  : wild type docked with 2 seeds (train targets) or 3 seeds (held-out)
  engine      : Uni-Dock 1.2.0, search_mode detail, one pose, single GPU
Ligand choice uses SMILES only (heavy atoms 15-40, stratified in quartiles); no score is
read. Shards balance the docking count across pods.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
from make_mutants import build_target  # noqa: E402

OUT = ROOT / "dockmut/plan"
N_TRAIN_LIG, N_HELD_LIG = 256, 160
SEED = 20261001


def pick_ligands(pool_ids: list[str], n: int, seed: int, smiles_of: dict, min_ha=15, max_ha=40):
    rng = np.random.default_rng(seed)
    ids = [pool_ids[i] for i in rng.permutation(len(pool_ids))[: n * 8]]
    rows = []
    for i in ids:
        m = Chem.MolFromSmiles(smiles_of[i])
        if m is None:
            continue
        ha = m.GetNumHeavyAtoms()
        if min_ha <= ha <= max_ha:
            rows.append((i, smiles_of[i], ha))
    df = pd.DataFrame(rows, columns=["inchikey", "smiles", "heavy_atoms"])
    df["q"] = pd.qcut(df.heavy_atoms.rank(method="first"), 4, labels=False)
    per = n // 4
    out = pd.concat([g.head(per) for _, g in df.groupby("q")])
    return out.drop(columns="q").reset_index(drop=True)


def main():
    cfg = json.loads((ROOT / "configs/g2_config.json").read_text())
    tg = pd.read_parquet(ROOT / "data/splits/targets.parquet")
    std = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id")
    raw = pd.read_csv(ROOT / "data/raw/dockstring-dataset.tsv", sep="\t", usecols=["inchikey", "smiles"]).set_index("inchikey")
    smiles_of = {i: raw.loc[i, "smiles"] for i in set(cfg["train_ligands"]) | set(cfg["dev_ligands"])}
    tg = tg[tg.target_id != "DRD2"]
    (OUT / "receptors").mkdir(parents=True, exist_ok=True)
    (OUT / "ligands").mkdir(parents=True, exist_ok=True)
    held = tg[tg.split_role.isin(["dev", "test"])].target_id.tolist()
    train = tg[tg.split_role == "train"].target_id.tolist()
    role = dict(zip(tg.target_id, tg.split_role))

    held_lig = pick_ligands(cfg["dev_ligands"], N_HELD_LIG, SEED + 1, smiles_of)
    held_lig.to_csv(OUT / "ligands/heldout_shared.csv", index=False)
    plan = {"targets": {}, "design": {"n_train_lig": N_TRAIN_LIG, "n_held_lig": len(held_lig),
                                      "seed": SEED, "engine": "unidock-1.2.0 detail"}}
    for t in tg.target_id:
        man = build_target(t, ROOT / "data/raw/targets", OUT / "receptors", SEED + sum(map(ord, t)),
                           n_contact=12, n_shell=3, n_far=1, n_multi=8)
        n_rec = len(man["receptors"])
        if t in held:
            lig_file = "heldout_shared.csv"
            n_lig, wt_seeds = len(held_lig), [1, 2, 3]
        else:
            lig = pick_ligands(cfg["train_ligands"], N_TRAIN_LIG, SEED + 100 + sum(map(ord, t)), smiles_of)
            lig.to_csv(OUT / f"ligands/{t}.csv", index=False)
            lig_file, n_lig, wt_seeds = f"{t}.csv", len(lig), [1, 2]
        plan["targets"][t] = {"role": role[t], "n_receptors": n_rec, "ligand_file": lig_file,
                              "n_ligands": n_lig, "wt_seeds": wt_seeds,
                              "n_dockings": (n_rec - 1 + len(wt_seeds)) * n_lig}
    total = sum(v["n_dockings"] for v in plan["targets"].values())
    plan["total_dockings"] = total
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    print("targets", len(plan["targets"]), "total dockings", total)
    print("held-out", sum(v["n_dockings"] for t, v in plan["targets"].items() if t in held),
          "train", sum(v["n_dockings"] for t, v in plan["targets"].items() if t in train))


if __name__ == "__main__":
    main()
