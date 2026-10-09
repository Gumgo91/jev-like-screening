"""Decompose held-out faithfulness into three levels (exploratory, written after the registered outcomes were known).

Pooled r over all ligand-receptor pairs mixes (1) differences between targets, (2) differences between edits of one
target and (3) ligand-specific responses to one edit. For each condition the script reports the mean over the 19
held-out targets of
  within-target r   : Pearson r over all contact pairs of one target,
  edit-mean r       : Pearson r of the receptor means of the contact edits of one target,
  ligand r          : mean within-receptor Pearson r (receptor means removed),
and the across-target correlation of the target-level SD of the change. Intervals are bootstrap percentiles over targets.
Paired differences between conditions use the same resampled targets.

Usage: python analysis_levels.py summary_final.parquet out.json [reference_condition]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_EDITS = 4


def corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def per_target(g: pd.DataFrame) -> dict:
    pooled = corr(g.pred.to_numpy(), g.dS.to_numpy())
    m = g.groupby("receptor").agg(p=("pred", "mean"), d=("dS", "mean"))
    edit = corr(m.p.to_numpy(), m.d.to_numpy()) if len(m) >= MIN_EDITS else np.nan
    gp = g.assign(pc=g.pred - g.groupby("receptor").pred.transform("mean"), dc=g.dS - g.groupby("receptor").dS.transform("mean"))
    lig = np.nanmean([corr(x.pc.to_numpy(), x.dc.to_numpy()) for _, x in gp.groupby("receptor")])
    return {"within_target_r": pooled, "edit_mean_r": edit, "ligand_r": float(lig),
            "sd_pred": float(g.pred.std()), "sd_true": float(g.dS.std()), "mean_abs_pred": float(g.pred.abs().mean()),
            "mean_abs_true": float(g.dS.abs().mean())}


def condition_table(df: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    """Per-target metrics of a condition. Seeds are averaged per target; the ensemble uses the mean prediction."""
    rows = []
    for t, gt in df[df.model.isin(tags)].groupby("target"):
        seeds = []
        for tag in tags:
            seeds.append(per_target(gt[gt.model == tag]))
        avg = {k: float(np.nanmean([s[k] for s in seeds])) for k in seeds[0]}
        ens = gt.groupby(["receptor", "ligand"], as_index=False).agg(pred=("pred", "mean"), dS=("dS", "first"))
        e = per_target(ens)
        rows.append({"target": t, **{k: avg[k] for k in avg}, **{"ens_" + k: v for k, v in e.items()}})
    return pd.DataFrame(rows).set_index("target")


def boot(vals: pd.Series, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = vals.dropna().to_numpy()
    b = [np.mean(rng.choice(v, len(v))) for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def boot_diff(a: pd.Series, b: pd.Series, n=2000, seed=0):
    d = (a - b).dropna()
    rng = np.random.default_rng(seed)
    v = d.to_numpy()
    bs = [np.mean(rng.choice(v, len(v))) for _ in range(n)]
    return float(v.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    ref = sys.argv[3] if len(sys.argv) > 3 else "rsg_int"
    df = pd.read_parquet(src)
    df = df[df.kind.isin(["single_contact"]) | df.kind.str.startswith("multi")]
    conds: dict[str, list[str]] = {}
    for tag in sorted(df.model.unique()):
        m = re.match(r"^(.*)_s(\d+)$", tag)
        conds.setdefault(m.group(1) if m else tag, []).append(tag)
    tabs = {c: condition_table(df, tags) for c, tags in conds.items() if not c.endswith("_const")}
    res = {}
    for c, T in tabs.items():
        r = {"n_targets": int(len(T))}
        for k in ("within_target_r", "edit_mean_r", "ligand_r", "ens_within_target_r", "ens_edit_mean_r", "ens_ligand_r"):
            r[k] = float(T[k].mean())
            r[k + "_ci"] = boot(T[k])
        r["sd_pred"] = float(T.sd_pred.mean())
        r["sd_true"] = float(T.sd_true.mean())
        r["target_level_sd_r"] = corr(T.sd_pred.to_numpy(), T.sd_true.to_numpy())
        r["target_level_abs_r"] = corr(T.mean_abs_pred.to_numpy(), T.mean_abs_true.to_numpy())
        if c != ref and ref in tabs:
            for k in ("within_target_r", "edit_mean_r", "ligand_r"):
                d, ci = boot_diff(T[k], tabs[ref][k])
                r["vs_" + ref + "_" + k] = {"diff": d, "ci": ci}
        res[c] = r
    out.write_text(json.dumps(res, indent=1))
    hdr = f"{'condition':22s} {'within-target r':>16s} {'edit-mean r':>14s} {'ligand r':>10s} {'sd pred':>8s} {'tgt-level SD r':>15s} {'tgt-level |d| r':>16s}"
    print(hdr)
    for c, r in res.items():
        print(f"{c:22s} {r['within_target_r']:+.3f} [{r['within_target_r_ci'][0]:+.2f},{r['within_target_r_ci'][1]:+.2f}] "
              f"{r['edit_mean_r']:+.3f} [{r['edit_mean_r_ci'][0]:+.2f},{r['edit_mean_r_ci'][1]:+.2f}] {r['ligand_r']:+.3f} "
              f"{r['sd_pred']:8.2f} {r['target_level_sd_r']:+15.2f} {r['target_level_abs_r']:+16.2f}")


if __name__ == "__main__":
    main()
