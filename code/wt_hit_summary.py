"""Wild-type screening of the baseline surrogates with their own hit-probability head.

The preceding analysis ranked ligands with the hit logit of the dual encoders. Its per-target metrics (development and
sealed test split, cold scope) and reversal accuracies are summarised here in the same form as wt_summary.py, so that the
screening recall of the baselines can be compared with the regression-trained networks of this study on the same frames.
No labels are read: all inputs are result files of the preceding analysis. Reproduction check: the development recall of
dm_baseline_logit.py (hit logit evaluated with this study's pipeline) matches these files.

Usage: python wt_hit_summary.py out.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "runs/v2"
EV = ROOT / "dockmut/eval"
out = Path(sys.argv[1])
COND = {"C0": ["C0_s11", "C0_s22", "C0_s33"], "C1": ["C1_s11", "C1_s22", "C1_s33"], "C2": ["C2_s11", "C2_s22", "C2_s33"],
        "C0c": ["C0c_s11", "C0c_s22", "C0c_s33"], "CB": ["CB_s11"]}
SPLIT = {"dev": ("metrics_{run}_cold.csv", "v2_reversal_eval.csv", "wt_dev_all.per_target.csv"),
         "test": ("test_metrics_{run}_cold.csv", "v2_test_reversal_eval.csv", "wt_test_all.per_target.csv")}
rng = np.random.default_rng(0)


def boot(v: np.ndarray, n=5000):
    idx = rng.integers(0, len(v), size=(n, len(v)))
    b = v[idx].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


res = {}
for split, (pat, rev_file, mine) in SPLIT.items():
    rev = pd.read_csv(V2 / rev_file)
    rev = rev[rev.scope == "cold"]
    per = {}
    entry = {"conditions": {}, "pairs": {}}
    for cond, runs in COND.items():
        frames = [pd.read_csv(V2 / pat.format(run=r)).set_index("target_id")["recall_1@10"].rename(r) for r in runs]
        M = pd.concat(frames, axis=1)
        per[cond] = M.mean(axis=1)
        v = per[cond].to_numpy()
        j = rev[rev.run.isin(runs)].joint_acc.to_numpy()
        entry["conditions"][cond] = {"recall_1@10": float(v.mean()), "ci": boot(v), "seeds": [float(M[r].mean()) for r in runs],
                                     "reversal": {"mean": float(j.mean()), "min": float(j.min()), "max": float(j.max()), "n": int(len(j))},
                                     "n_targets": int(len(v))}
    mine_df = pd.read_csv(EV / mine)
    mine_df["cond"] = mine_df.model.str.replace(r"_s\d+$", "", regex=True)
    mine_t = mine_df.groupby(["cond", "target_id"])["recall_1@10"].mean().unstack(0)
    for a in ("rsg_wt", "rsg_int", "rs_wt", "b5_wt", "rse_wt", "rse_int", "rs_int"):
        if a not in mine_t:
            continue
        for b in ("C0", "C1", "CB"):
            common = mine_t.index.intersection(per[b].index)
            d = (mine_t.loc[common, a] - per[b].loc[common]).to_numpy()
            entry["pairs"][f"{a} - {b}"] = {"diff": float(d.mean()), "ci": boot(d), "n_targets": int(len(d))}
    res[split] = entry
out.write_text(json.dumps(res, indent=1))
for split, e in res.items():
    for c, v in e["conditions"].items():
        print(f"{split:4s} {c:4s} recall@10% {v['recall_1@10']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}] reversal {v['reversal']['mean']:.3f} ({v['reversal']['min']:.3f}-{v['reversal']['max']:.3f})")
    for k in ("rsg_wt - C0", "rsg_int - C0", "rsg_wt - CB", "b5_wt - C0"):
        if k in e["pairs"]:
            p = e["pairs"][k]
            print(f"{split:4s} {k:14s} diff {p['diff']:+.3f} [{p['ci'][0]:+.3f},{p['ci'][1]:+.3f}] (n={p['n_targets']})")
