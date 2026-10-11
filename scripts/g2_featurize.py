"""Ligand graphs of the 50,000 ligands of configs/g2_config.json (40,000
train + 10,000 dev), cached to data/processed/g2_ligand_graphs.pt.

Needs configs/g2_config.json (scripts/g2_prepare.py) and
data/processed/ligands.parquet (src/pocketgate/data/standardize.py).
Writes manifests/g2_featurize_meta.json (graph count, failed ligands,
featurization time).

Usage: python scripts/g2_featurize.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketgate.common import MANIFESTS_DIR, PROCESSED_DIR  # noqa: E402
from pocketgate.data.features import featurize_ligand  # noqa: E402


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    cfg = json.loads((ROOT / "configs" / "g2_config.json").read_text())
    lig = pd.read_parquet(PROCESSED_DIR / "ligands.parquet")
    smi_of = dict(zip(lig["ligand_id"], lig["standardized_smiles"]))

    ids = sorted(set(cfg["train_ligands"]) | set(cfg["dev_ligands"]))
    t0 = time.time()
    ligs, failed = {}, []
    for i, lid in enumerate(ids):
        g = featurize_ligand(smi_of[lid])
        if g is None:
            failed.append(lid)
        else:
            ligs[lid] = {k: torch.from_numpy(np.asarray(v))
                         for k, v in g.items()}
        if (i + 1) % 5000 == 0:
            print(f"  ligands {i+1}/{len(ids)} "
                  f"({(time.time()-t0)/(i+1)*1000:.0f} ms/mol)", flush=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ligs, PROCESSED_DIR / "g2_ligand_graphs.pt")
    ligand_feat_seconds = time.time() - t0

    meta = {
        "n_ligand_graphs": len(ligs), "featurize_failed": failed,
        "ligand_feat_seconds": ligand_feat_seconds,
    }
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFESTS_DIR / "g2_featurize_meta.json").write_text(
        json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2)[:800])


if __name__ == "__main__":
    main()
