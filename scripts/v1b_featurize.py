"""Featurize the 39 train + 10 dev pockets into data/processed/v1_pocket_graphs.pt.

Needs data/splits/targets.parquet and the receptor files in data/raw/targets.

Usage: python scripts/v1b_featurize.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketgate.common import PROCESSED_DIR  # noqa: E402
from pocketgate.data.features import featurize_pocket  # noqa: E402


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    tg = pd.read_parquet(ROOT / "data" / "splits" / "targets.parquet")
    need = tg[tg.split_role.isin(["train", "dev"])]["target_id"].tolist()
    out = {}
    for t in need:
        g = featurize_pocket(t)
        out[t] = {k: (torch.as_tensor(v) if hasattr(v, "dtype") else v)
                  for k, v in g.items()}
        print(t, g["n_residues"], "residues",
              "TRUNC" if g["truncated"] else "", flush=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out, PROCESSED_DIR / "v1_pocket_graphs.pt")
    print("saved", len(out), "pockets")


if __name__ == "__main__":
    main()
