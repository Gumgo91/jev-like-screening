"""Selection of the training and development ligands (and of the 8 + 4
family-diverse targets stored next to them), deterministic and independent
of the docking scores. Writes configs/g2_config.json.

Targets: round-robin over families sorted by size desc, targets sorted
by name inside a family -> maximally family-diverse picks.
Ligands: fixed subset by sha256(ligand_id|salt) order - identical sets
for every model.

Needs data/processed/ligands.parquet (src/pocketgate/data/standardize.py) and
data/splits/{targets,ligand_splits}.parquet (src/pocketgate/data/split.py).

Usage: python scripts/g2_prepare.py [--out configs/g2_config.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketgate.common import PROCESSED_DIR, SPLITS_DIR  # noqa: E402

N_TRAIN_T = 8
N_DEV_T = 4
N_TRAIN_L = 40000
N_DEV_L = 10000
LIGAND_SALT = "g2_ligand_pick_v1"


def pick_targets(df: pd.DataFrame, role: str, n: int) -> list[str]:
    sub = df[df["split_role"] == role]
    fam_size = sub.groupby("family")["target_id"].count()
    fams = fam_size.sort_values(ascending=False).index.tolist()
    picks: list[str] = []
    i = 0
    while len(picks) < n:
        fam = fams[i % len(fams)]
        cands = sorted(
            sub[(sub["family"] == fam) & ~sub["target_id"].isin(picks)]
            ["target_id"])
        if cands:
            picks.append(cands[0])
        i += 1
        if i > 1000:
            break
    return picks


def pick_ligands(lig: pd.DataFrame, role: str, n: int) -> list[str]:
    sub = lig[(lig["split_role"] == role) & (lig["feature_status"] == "OK")]
    h = sub["ligand_id"].map(
        lambda s: hashlib.sha256(f"{s}|{LIGAND_SALT}".encode()).hexdigest())
    order = h.sort_values().index
    return sub.loc[order[:n], "ligand_id"].tolist()


def main():
    ap = argparse.ArgumentParser(
        description="Write the training/development ligand selection "
                    "(configs/g2_config.json).")
    ap.add_argument("--out", default=None,
                    help="output file, relative paths are relative to the "
                         "repository root (default configs/g2_config.json)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path("configs") / "g2_config.json"
    out = out if out.is_absolute() else ROOT / out

    targets = pd.read_parquet(SPLITS_DIR / "targets.parquet")
    ligands = pd.read_parquet(PROCESSED_DIR / "ligands.parquet").merge(
        pd.read_parquet(SPLITS_DIR / "ligand_splits.parquet")
        .drop(columns=["feature_status"]),
        on=["ligand_id", "ligand_group_id"])
    cfg = {
        "train_targets": pick_targets(targets, "train", N_TRAIN_T),
        "dev_targets": pick_targets(targets, "dev", N_DEV_T),
        "train_ligands": pick_ligands(ligands, "train", N_TRAIN_L),
        "dev_ligands": pick_ligands(ligands, "dev", N_DEV_L),
        "seed": 11,
        "ligand_salt": LIGAND_SALT,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v)
                      for k, v in cfg.items()}, indent=2))
    print("train targets:", cfg["train_targets"])
    print("dev targets:", cfg["dev_targets"])
    print("wrote", out)


if __name__ == "__main__":
    main()
