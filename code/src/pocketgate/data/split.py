"""Split construction (plan section 5).

Two independent axes:
- Target axis: protein_cluster_id from sequence links (>=30% identity,
  >=80% shorter coverage) UNION curated ligand-binding family
  (FAMILY_MAP). Roles assigned at cluster level, ~70/15/15 by target
  count: clusters with >2 members go to train; remaining clusters are
  round-robin assigned to test/dev (test first) until each reaches
  >=ceil(0.15 * n_targets) targets and test has >=6 clusters; leftovers
  go to train. Deterministic, score-blind.
- Ligand axis: ligand_group_id -> sha256 hash bucketed into
  train/dev/probcal/policycal/test at 60/10/10/10/10.

Outputs (data/splits/):
- targets.parquet: target_id, family, protein_cluster_id, split_role,
  sequence_len, receptor_hash, box_hash, pocket_residue_count
- ligand_splits.parquet: ligand_id, ligand_group_id, split_role
- split_manifest.json: params, counts, hashes
- data/label_vault/labels_<trole>_<lrole>.parquet: per-cell label files.
  Test cells (any P_test label, or P_train x L_test) are written only
  under label_vault/ with a sha256 manifest; nothing else reads them.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pocketgate.common import (  # noqa: E402
    LABEL_VAULT_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SPLITS_DIR,
    TARGETS_DIR,
    sha256_file,
)
from pocketgate.data.audit import parse_conf, pocket_residue_count  # noqa: E402
from pocketgate.data.standardize import UnionFind  # noqa: E402
from pocketgate.data.targetseq import FAMILY_MAP  # noqa: E402

LIGAND_RATIOS = {
    "train": 0.60,
    "dev": 0.10,
    "probcal": 0.10,
    "policycal": 0.10,
    "test": 0.10,
}
SPLIT_SALT = "pocketgate_split_v1"


def assign_target_roles(clusters: dict[int, list[str]], n_targets: int) -> dict[str, str]:
    """cluster_id -> list of targets. Returns target_id -> role."""
    cl_list = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), sorted(kv[1])))
    big = [(cid, ts) for cid, ts in cl_list if len(ts) > 2]
    small = [(cid, ts) for cid, ts in cl_list if len(ts) <= 2]
    # deterministic order for small clusters: size asc, then first member name
    small.sort(key=lambda kv: (len(kv[1]), sorted(kv[1])[0]))

    roles: dict[str, str] = {}
    for _, ts in big:
        for t in ts:
            roles[t] = "train"

    need = int(np.ceil(0.15 * n_targets))
    test_targets: list[str] = []
    dev_targets: list[str] = []
    test_clusters: set[int] = set()
    i = 0
    # fill test first until >=need targets AND >=6 clusters
    while i < len(small) and (len(test_targets) < need or len(test_clusters) < 6):
        cid, ts = small[i]
        i += 1
        test_targets += ts
        test_clusters.add(cid)
    while i < len(small) and len(dev_targets) < need:
        _, ts = small[i]
        i += 1
        dev_targets += ts
    for t in test_targets:
        roles[t] = "test"
    for t in dev_targets:
        roles[t] = "dev"
    for _, ts in small[i:]:
        for t in ts:
            roles[t] = "train"
    return roles


def build_target_table() -> tuple[pd.DataFrame, list[tuple]]:
    from pocketgate.data.targetseq import main as seq_main

    targets, seqs, seq_links = seq_main()
    uf = UnionFind(len(targets))
    idx = {t: i for i, t in enumerate(targets)}
    for a, b, *_ in seq_links:
        uf.union(idx[a], idx[b])
    # family merge layer
    fam_groups: dict[str, list[str]] = {}
    for t in targets:
        fam_groups.setdefault(FAMILY_MAP.get(t, t), []).append(t)
    for members in fam_groups.values():
        for t in members[1:]:
            uf.union(idx[members[0]], idx[t])

    roots = [uf.find(i) for i in range(len(targets))]
    codes, uniques = pd.factorize(pd.Series(roots))
    cluster_of = dict(zip(targets, codes))
    clusters: dict[int, list[str]] = {}
    for t, c in cluster_of.items():
        clusters.setdefault(int(c), []).append(t)

    roles = assign_target_roles(clusters, len(targets))

    rows = []
    for t in targets:
        conf = parse_conf(TARGETS_DIR / f"{t}_conf.txt")
        n_pocket, n_res, n_atoms = pocket_residue_count(
            TARGETS_DIR / f"{t}_target.pdbqt", conf
        )
        rows.append(
            dict(
                target_id=t,
                family=FAMILY_MAP.get(t, t),
                protein_cluster_id=int(cluster_of[t]),
                split_role=roles[t],
                sequence_len=len(seqs[t]),
                sequence=seqs[t],
                receptor_hash=sha256_file(TARGETS_DIR / f"{t}_target.pdbqt")[:16],
                box_hash=sha256_file(TARGETS_DIR / f"{t}_conf.txt")[:16],
                pocket_residues=n_pocket,
                total_residues=n_res,
            )
        )
    return pd.DataFrame(rows), seq_links


_ROLE_ORDER = ["train", "dev", "probcal", "policycal", "test"]


def assign_ligand_roles(ligands: pd.DataFrame) -> pd.Series:
    """Deterministic scaffold-split style assignment.

    Groups sorted by (size desc, group_key asc); each goes to the role
    that stays furthest below its target share after absorbing it
    (minimizing (cur+size)/target), ties preferring earlier roles in
    _ROLE_ORDER. Common/large scaffold groups land in train; test gets
    smaller, rarer scaffold groups -> scaffold-disjoint-ish eval.
    """
    gsize = ligands.groupby("group_key").size().rename("n").reset_index()
    gsize = gsize.sort_values(["n", "group_key"], ascending=[False, True],
                              kind="mergesort")
    target = {r: LIGAND_RATIOS[r] * len(ligands) for r in _ROLE_ORDER}
    cur = {r: 0 for r in _ROLE_ORDER}
    role_of: dict[str, str] = {}
    for key, n in gsize[["group_key", "n"]].itertuples(index=False):
        best, best_cost = "train", float("inf")
        for r in _ROLE_ORDER:
            cost = (cur[r] + n) / target[r]
            if cost < best_cost - 1e-12:
                best, best_cost = r, cost
        role_of[key] = best
        cur[best] += n
    return ligands["group_key"].map(role_of)


def main() -> dict:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_VAULT_DIR.mkdir(parents=True, exist_ok=True)

    targets_df, seq_links = build_target_table()
    ligands = pd.read_parquet(PROCESSED_DIR / "ligands.parquet")
    ligands["split_role"] = assign_ligand_roles(ligands)

    targets_df.to_parquet(SPLITS_DIR / "targets.parquet", index=False)
    ligands[["ligand_id", "ligand_group_id", "split_role",
             "feature_status"]].to_parquet(
        SPLITS_DIR / "ligand_splits.parquet", index=False
    )

    # ---- label files -------------------------------------------------
    tsv = RAW_DIR / "dockstring-dataset.tsv"
    scores = pd.read_csv(tsv, sep="\t")
    scores = scores.rename(columns={"inchikey": "ligand_id", "smiles": "smiles"})
    role_of_lig = dict(zip(ligands["ligand_id"], ligands["split_role"]))
    role_of_tgt = dict(zip(targets_df["target_id"], targets_df["split_role"]))

    long = scores.melt(
        id_vars=["ligand_id"], value_vars=list(role_of_tgt),
        var_name="target_id", value_name="score",
    )
    long["target_role"] = long["target_id"].map(role_of_tgt)
    long["ligand_role"] = long["ligand_id"].map(role_of_lig)

    manifest = {"target_roles": targets_df["split_role"].value_counts().to_dict(),
                "ligand_roles": ligands["split_role"].value_counts().to_dict(),
                "label_files": {}}
    for (tr, lr), sub in long.groupby(["target_role", "ligand_role"]):
        name = f"labels_{tr}_{lr}.parquet"
        sub[["target_id", "ligand_id", "score"]].to_parquet(
            LABEL_VAULT_DIR / name, index=False
        )
        manifest["label_files"][f"{tr}_{lr}"] = {
            "file": name,
            "sha256": sha256_file(LABEL_VAULT_DIR / name),
            "n_cells": int(len(sub)),
            "n_valid": int(sub["score"].notna().sum()),
        }

    manifest["n_seq_links"] = len(seq_links)
    manifest["seq_links"] = [
        [a, b, i, c] for a, b, i, c in seq_links
    ]
    manifest["n_target_clusters"] = int(targets_df["protein_cluster_id"].nunique())
    manifest["test_clusters"] = sorted(
        targets_df.loc[targets_df["split_role"] == "test", "protein_cluster_id"]
        .unique().tolist()
    )
    manifest["ligand_ratios"] = LIGAND_RATIOS
    manifest["split_salt"] = SPLIT_SALT
    manifest["files"] = {
        "targets.parquet": sha256_file(SPLITS_DIR / "targets.parquet"),
        "ligand_splits.parquet": sha256_file(SPLITS_DIR / "ligand_splits.parquet"),
    }
    (SPLITS_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    m = main()
    print(json.dumps({k: v for k, v in m.items() if k != "label_files" and k != "seq_links"}, indent=2))
    for k, v in m["label_files"].items():
        print(k, v["n_cells"], v["n_valid"])
