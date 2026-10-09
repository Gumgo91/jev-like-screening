"""Concatenate the held-out predictions of all runs, baselines and descriptor models into one parquet.

Usage: python combine_heldout.py out.parquet runs_dir [parquet_with_model_column ...]
Files named <tag>_heldout.parquet in runs_dir get model = tag; other parquet files must already carry a model column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
frames = []
for f in sorted(Path(sys.argv[2]).glob("*_heldout.parquet")):
    d = pd.read_parquet(f)
    d["model"] = f.stem.replace("_heldout", "")
    frames.append(d)
for p in sys.argv[3:]:
    frames.append(pd.read_parquet(p))
df = pd.concat(frames, ignore_index=True)
df = df[["model", "target", "receptor", "kind", "ligand", "dS", "pred"]]
df.to_parquet(out, index=False)
print(out, len(df), df.model.nunique(), "models")
