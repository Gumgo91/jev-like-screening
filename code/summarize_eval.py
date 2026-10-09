"""Summarise held-out benchmark predictions by condition.

Reads every parquet with columns model, target, receptor, kind, ligand, dS, pred. The model tag ends with _s<seed> for seeded
conditions. For each condition the script reports, per seed and for the seed ensemble, the pooled, receptor-level and
within-receptor correlations, the regression gain, the predicted SD and the detection AUC, with a bootstrap over targets.
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
from dm_eval import detect_auc  # noqa: E402


def is_contact(df):
    return df.kind.isin(["single_contact"]) | df.kind.str.startswith("multi")


def boot_r(g: pd.DataFrame, n=4000, seed=0):
    """95% percentile interval of the pooled r, resampling targets (sufficient statistics per target)."""
    from paired_bootstrap import r_from, suff
    a = suff(g).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    v = np.array([float(r_from(a[i].sum(0))) for i in idx])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def condition_of(tag: str) -> tuple[str, int | None]:
    m = re.match(r"^(.*)_s(\d+)$", tag)
    return (m.group(1), int(m.group(2))) if m else (tag, None)


def main(paths: list[Path], out: Path):
    frames = []
    for p in paths:
        for f in ([p] if p.is_file() else sorted(p.rglob("*.parquet"))):
            d = pd.read_parquet(f)
            if "model" not in d.columns:
                tag = f.stem.replace("_heldout", "")
                d["model"] = tag
            frames.append(d)
    df = pd.concat(frames)
    df = df[is_contact(df) | df.kind.isin(["single_shell", "single_far"])]
    conds = {}
    for tag in df.model.unique():
        c, s = condition_of(tag)
        conds.setdefault(c, []).append(tag)
    res = {}
    for c, tags in sorted(conds.items()):
        sub = df[df.model.isin(tags)]
        per_seed = {}
        for t in tags:
            g = sub[(sub.model == t) & is_contact(sub)]
            per_seed[t] = metric_block(g)
            per_seed[t]["detect_auc"] = detect_auc(sub[sub.model == t])
        ens = sub.groupby(["target", "receptor", "kind", "ligand"], as_index=False).agg(pred=("pred", "mean"), dS=("dS", "first"))
        ge = ens[is_contact(ens)]
        mb = metric_block(ge)
        mb["detect_auc"] = detect_auc(ens)
        mb["ci95_pooled_r"] = boot_r(ge)
        rs = [v["pooled_r"] for v in per_seed.values()]
        res[c] = {"n_seeds": len(tags), "pooled_r_seeds": rs, "pooled_r_mean": float(np.mean(rs)), "pooled_r_sd": float(np.std(rs)),
                  "ensemble": mb, "per_seed": per_seed}
        print(f"{c:22s} seeds={len(tags)} pooled r mean={np.mean(rs):+.3f} ens={mb['pooled_r']:+.3f} "
              f"CI[{mb['ci95_pooled_r'][0]:+.2f},{mb['ci95_pooled_r'][1]:+.2f}] mutant r={mb['mutant_r']:+.3f} within={mb['within_r']:+.3f} "
              f"gain={mb['gain']:+.2f} predSD={mb['pred_sd']:.2f} AUC={mb['detect_auc']:.2f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    df.to_parquet(out.with_suffix(".parquet"), index=False)


if __name__ == "__main__":
    main([Path(x) for x in sys.argv[2:]], Path(sys.argv[1]))
