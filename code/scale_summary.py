"""Scaling study summary: pooled r on the 19 held-out targets versus number of training targets and number of edits per
target, with the registered final runs (all 38 targets, all edits) as the endpoint. Two seeds per setting (11, 22).

Usage: python scale_summary.py out.json runs_dir [runs_dir ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
from dm_core import metric_block  # noqa: E402


def load(paths):
    out = {}
    for p in paths:
        for f in sorted(Path(p).rglob("*_heldout.parquet")):
            out[f.stem.replace("_heldout", "")] = f
    return out


def contact(d):
    return d[d.kind.isin(["single_contact"]) | d.kind.str.startswith("multi")]


def main():
    out = Path(sys.argv[1])
    files = load(sys.argv[2:])
    runs = {}
    for tag, f in files.items():
        m = re.match(r"^rsg_int(_nt\d+|_mk\d+)?_s(\d+)$", tag)
        if not m:
            continue
        variant = (m.group(1) or "_full").lstrip("_")
        seed = int(m.group(2))
        if variant == "full" and seed not in (11, 22):
            continue
        d = contact(pd.read_parquet(f))
        mb = metric_block(d)
        diag = json.loads(f.with_name(tag + "_diag.json").read_text()) if f.with_name(tag + "_diag.json").exists() else {}
        runs.setdefault(variant, {})[seed] = {"pooled_r": mb["pooled_r"], "mutant_r": mb["mutant_r"], "within_r": mb["within_r"], "pred_sd": mb["pred_sd"],
                                               "new_edits_r": diag.get("val_mutants", {}).get("pooled_r"), "frame": d}
    res = {}
    for v, seeds in runs.items():
        frames = [s["frame"] for s in seeds.values()]
        ens = pd.concat(frames).groupby(["target", "receptor", "ligand", "kind"], as_index=False).agg(pred=("pred", "mean"), dS=("dS", "first"))
        mb = metric_block(ens)
        res[v] = {"n_seeds": len(seeds), "pooled_r_seeds": {str(k): s["pooled_r"] for k, s in seeds.items()},
                  "pooled_r_mean": float(np.mean([s["pooled_r"] for s in seeds.values()])),
                  "ensemble_r": mb["pooled_r"], "mutant_r_mean": float(np.mean([s["mutant_r"] for s in seeds.values()])),
                  "within_r_mean": float(np.mean([s["within_r"] for s in seeds.values()])),
                  "new_edits_r_mean": float(np.nanmean([s["new_edits_r"] if s["new_edits_r"] is not None else np.nan for s in seeds.values()]))}
        print(f"{v:6s} seeds={len(seeds)} r seeds={[round(s['pooled_r'], 3) for s in seeds.values()]} mean={res[v]['pooled_r_mean']:+.3f} ens={mb['pooled_r']:+.3f} "
              f"edit-mean r={res[v]['mutant_r_mean']:+.3f} within={res[v]['within_r_mean']:+.3f} new-edits r={res[v]['new_edits_r_mean']:+.3f}")
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
