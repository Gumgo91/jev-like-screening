"""Pooled r of seed ensembles and paired differences between conditions, bootstrapped over targets.

Sufficient statistics per target (n, sum x, sum y, sum x^2, sum y^2, sum xy) make the pooled correlation of a
resampled set of targets a sum over the drawn targets, so 10,000 resamples take seconds.
Usage: python paired_bootstrap.py out.json in1.parquet [in2.parquet ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = [("rsg_int", "C0"), ("rsg_int", "rsg_wt"), ("rsg_int", "rsg_int_shm2"), ("rsg_int", "rsg_int_sha2"),
         ("rsg_int", "rsg_int_shm"), ("rsg_int", "rsg_int_sha"), ("rsg_int", "rsg_int_shl"), ("rsg_int", "gbm_descriptors_geo"),
         ("rsg_int", "rs_int"), ("rsg_int", "rse_int"), ("rs_int", "rs_wt"), ("rse_int", "rse_wt"), ("rsg_int", "b5_int"),
         ("b5_int", "b5_wt"), ("gbm_descriptors_geo", "gbm_descriptors"), ("gbm_pose_geo", "gbm_descriptors_geo"),
         ("rsg_wt", "C0"), ("rse_int", "gbm_descriptors_geo"), ("gbm_descriptors_geo", "rsg_int"), ("rsg_int", "rsg_int_shl"),
         ("rsgh_int", "rsgh_wt"), ("rsgh_int", "rsg_int"), ("rsgh_wt", "rsg_wt"), ("rsgh_int", "C0"), ("rsgh_int", "gbm_descriptors_geo"), ("rsgh_wt", "C0")]


def cond_of(tag: str) -> str:
    m = re.match(r"^(.*)_s(\d+)$", tag)
    return m.group(1) if m else tag


def suff(d: pd.DataFrame) -> pd.DataFrame:
    g = d.assign(xx=d.pred ** 2, yy=d.dS ** 2, xy=d.pred * d.dS).groupby("target").agg(
        n=("pred", "size"), sx=("pred", "sum"), sy=("dS", "sum"), sxx=("xx", "sum"), syy=("yy", "sum"), sxy=("xy", "sum"))
    return g


def r_from(s: np.ndarray) -> np.ndarray:
    n, sx, sy, sxx, syy, sxy = s.T if s.ndim == 2 else s
    num = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sxx - sx ** 2, 0) * np.maximum(n * syy - sy ** 2, 0))
    return np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1), 0.0)


def main():
    out = Path(sys.argv[1])
    frames = [pd.read_parquet(p) for p in sys.argv[2:]]
    df = pd.concat(frames)
    df = df[df.kind.isin(["single_contact"]) | df.kind.str.startswith("multi")]
    df["cond"] = df.model.map(cond_of)
    S = {}
    for c, g in df.groupby("cond"):
        if c.endswith("_const"):
            continue
        ens = g.groupby(["target", "receptor", "ligand", "kind"], as_index=False).agg(pred=("pred", "mean"), dS=("dS", "first"))
        S[c] = suff(ens)
    targets = sorted(set.intersection(*[set(v.index) for v in S.values()]))
    A = {c: v.loc[targets].to_numpy() for c, v in S.items()}
    rng = np.random.default_rng(1)
    B = 10000
    idx = rng.integers(0, len(targets), size=(B, len(targets)))
    res = {"n_targets": len(targets), "conditions": {}, "pairs": {}}
    R = {}
    for c, a in A.items():
        point = float(r_from(a.sum(0)))
        boots = np.array([float(r_from(a[i].sum(0))) for i in idx])
        R[c] = boots
        res["conditions"][c] = {"r": point, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    for a_, b_ in PAIRS:
        if a_ in R and b_ in R:
            d = R[a_] - R[b_]
            res["pairs"][f"{a_} - {b_}"] = {"diff": res["conditions"][a_]["r"] - res["conditions"][b_]["r"],
                                            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                            "p_le0": float((d <= 0).mean())}
    out.write_text(json.dumps(res, indent=1))
    for c, v in res["conditions"].items():
        print(f"{c:24s} r={v['r']:+.3f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]")
    for k, v in res["pairs"].items():
        print(f"{k:48s} diff={v['diff']:+.3f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}] P(diff<=0)={v['p_le0']:.3f}")


if __name__ == "__main__":
    main()
