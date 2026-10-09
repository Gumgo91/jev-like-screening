"""Figure 6: wild-type screening and target-specific ranking of every condition on the development and test targets.

Usage: python fig6_wt.py <wt_dev_all.json> <wt_test_all.json> <wt_dev_descriptor.json> <wt_hit_summary.json> <wt_hitnet_summary.json>
Bars give the mean over seeds. Dots are single seeds (recall) and the line gives the range over seeds (reversal) for the
baselines read out with their hit-probability head, which come from the result files of the preceding analysis.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from conds import GCOL, GROUP  # noqa: E402
from figstyle import DOUBLE, PAL, apply, panel  # noqa: E402

apply()
dev = json.loads(Path(sys.argv[1]).read_text())
test = json.loads(Path(sys.argv[2]).read_text())
desc = json.loads(Path(sys.argv[3]).read_text())
hit = json.loads(Path(sys.argv[4]).read_text())
hn = json.loads(Path(sys.argv[5]).read_text())
GROUP.update({"C0c": "reference", "rsg_const": "control"})
# (key, label, readout); "H:" keys come from the hit-probability head of the preceding analysis
rows = [("H:C0", "Baseline, hit head", "hit"), ("H:C1", "Baseline, within-target rank, hit head", "hit"),
        ("H:C2", "Baseline, cross-target rank, hit head", "hit"), ("H:C0c", "Baseline, constant pocket, hit head", "hit"),
        ("H:CB", "Ligand-only fingerprint model", "hit"),
        ("C0", "Baseline, regression head", "reg"), ("C1", "Baseline, within-target rank, regr. head", "reg"),
        ("C2", "Baseline, cross-target rank, regr. head", "reg"), ("C0c", "Baseline, constant pocket, regr. head", "reg"),
        ("b5_wt", "Dual encoder, WT", "reg"), ("b5_int", "Dual encoder, interv.", "reg"), ("rs_wt", "ResidueSum, WT", "reg"), ("rs_int", "ResidueSum, interv.", "reg"),
        ("rse_wt", "ResidueSum-Edit, WT", "reg"), ("rse_int", "ResidueSum-Edit, interv.", "reg"), ("rsg_wt", "ResidueSum-Geo, WT", "reg"),
        ("rsg_int", "ResidueSum-Geo, interv.", "reg"),
        ("N:rsgh_wt", "Geo + hit head, WT (hit logit)", "net"), ("N:rsgh_int", "Geo + hit head, interv. (hit logit)", "net"), ("rsg_int_shl", "Permuted over ligands", "reg"), ("rsg_int_shm2", "Deranged over edits", "reg"),
        ("rsg_int_sha2", "All deranged", "reg")]


def by_cond(res):
    g = {}
    for t, v in res.items():
        m = re.match(r"^(.*)_s(\d+)$", t)
        g.setdefault(m.group(1) if m else t, []).append(v)
    return g


G = {"dev": by_cond(dev), "test": by_cond(test)}
fig = plt.figure(figsize=(DOUBLE - 0.25, 4.7))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.12)
specs = [("dev", "joint", "Reversal, development"), ("test", "joint", "Reversal, test"),
         ("dev", "recall_1@10", "Recall, development"), ("test", "recall_1@10", "Recall, test")]
n_rows = len(rows)
for j, (split, key, title) in enumerate(specs):
    ax = fig.add_subplot(gs[j])
    for i, (c, label, kind) in enumerate(rows):
        if kind == "hit":
            e = hit[split]["conditions"][c[2:]]
            if key == "joint":
                v = e["reversal"]
                ax.barh(i, v["mean"], height=0.64, color=GCOL["reference"], edgecolor="black", lw=0.4, zorder=2, hatch="////")
                ax.plot([v["min"], v["max"]], [i, i], color="black", lw=0.8, zorder=3)
            else:
                ax.barh(i, e["recall_1@10"], height=0.64, color=GCOL["reference"], edgecolor="black", lw=0.4, zorder=2, hatch="////")
                ax.scatter(e["seeds"], [i] * len(e["seeds"]), s=4, color="black", zorder=3, lw=0)
            continue
        if kind == "net":
            e = hn[split]["conditions"][c[2:]]
            colr = GCOL["wt"] if c.endswith("_wt") else GCOL["int"]
            if key == "joint":
                v = e["reversal"]
                ax.barh(i, v["mean"], height=0.64, color=colr, edgecolor="black", lw=0.4, zorder=2, hatch="////")
                ax.plot([v["min"], v["max"]], [i, i], color="black", lw=0.8, zorder=3)
            else:
                ax.barh(i, e["recall_1@10"], height=0.64, color=colr, edgecolor="black", lw=0.4, zorder=2, hatch="////")
                ax.scatter(e["seeds"], [i] * len(e["seeds"]), s=4, color="black", zorder=3, lw=0)
            continue
        vals = np.array([v[key] for v in G[split].get(c, [])])
        if not len(vals):
            continue
        grp = GROUP.get(c, "control")
        ax.barh(i, vals.mean(), height=0.64, color=GCOL[grp], edgecolor="black", lw=0.4, zorder=2)
        ax.scatter(vals, [i] * len(vals), s=4, color="black", zorder=3, lw=0)
    if split == "dev":
        for name, y in (("wt_gbm_geometry", n_rows), ("wt_gbm_ligand_only", n_rows + 1)):
            ax.barh(y, desc[name][key], height=0.64, color=GCOL["descriptor"], edgecolor="black", lw=0.4, zorder=2)
    if key == "joint":
        ax.axvline(0.25, color=PAL["red"], lw=0.7, ls="--", zorder=1)
        ax.set_xlim(0, 0.65)
        ax.set_xlabel("both orderings reproduced", fontsize=6.8)
    else:
        ax.set_xlim(0.5, 0.9)
        ax.set_xlabel("top-1% recall at 10% budget", fontsize=6.8)
    ax.set_ylim(n_rows + 1.6, -0.7)
    names = [r[1] for r in rows] + ["Boosting + geometry (dev.)", "Boosting, ligand only (dev.)"]
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names if j == 0 else [], fontsize=6.1)
    ax.set_title(title, fontsize=6.9, pad=3)
    panel(ax, "abcd"[j], dx=-0.08 if j else -1.15, dy=1.07)
plt.savefig(Path(__file__).parent / "figures/fig6_wt.pdf")
plt.savefig(Path(__file__).parent / "figures/fig6_wt.png", dpi=200)
print("saved")
