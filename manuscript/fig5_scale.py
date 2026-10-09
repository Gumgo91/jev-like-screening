"""Figure 5: scaling of the interventional response and transfer within a protein family.

(a) pooled r on the 19 held-out targets against the number of training targets, (b) against the number of edits per
training target, (c) r on four kinases withheld from training for the geometry network and its controls.
Usage: python fig5_scale.py <scale_summary.json> <kin_summary.json> <descriptor_baselines_kin.json> <summary_all.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import DOUBLE, PAL, apply, panel  # noqa: E402

apply()
Z = json.loads(Path(sys.argv[1]).read_text())
K = json.loads(Path(sys.argv[2]).read_text())
D = json.loads(Path(sys.argv[3]).read_text())
S = json.loads(Path(sys.argv[4]).read_text())
fig = plt.figure(figsize=(DOUBLE, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.45], wspace=0.42)

for j, (prefix, xs, xlabel, full_x) in enumerate([("nt", [4, 8, 16, 24], "training targets", 38), ("mk", [2, 4, 8, 12], "edits per training target", 20)]):
    ax = fig.add_subplot(gs[j])
    X, M = [], []
    for x in xs + [None]:
        key = f"{prefix}{x}" if x else "full"
        px = x if x else full_x
        seeds = list(Z[key]["pooled_r_seeds"].values())
        ax.scatter([px] * len(seeds), seeds, s=9, color=PAL["blue"], alpha=0.65, lw=0, zorder=3)
        X.append(px)
        M.append(np.mean(seeds))
    ax.plot(X, M, color=PAL["blue"], lw=1.2, zorder=2)
    ref = S["rsg_wt"]["pooled_r_mean"]
    ax.axhline(ref, color=PAL["teal"], lw=0.8, ls="--")
    ax.text(X[0], ref + 0.012, "no edited receptors", fontsize=6.0, color=PAL["teal"], va="bottom")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks(X)
    ax.set_xticklabels([str(x) if x != 20 else "all" for x in X], fontsize=6.6)
    ax.minorticks_off()
    ax.set_xlabel(xlabel, fontsize=6.9)
    ax.set_ylabel("pooled $r$ on held-out targets" if j == 0 else "", fontsize=6.9)
    ax.set_ylim(-0.17, 0.38)
    panel(ax, "ab"[j], dx=-0.2)

ax = fig.add_subplot(gs[2])
rows = [("rsg_int", "Measured\nchanges", PAL["blue"]), ("rsg_int_shm2", "Edits\nderanged", PAL["orange"]), ("rsg_int_sha2", "All\nderanged", PAL["orange"]),
        ("rsg_wt", "No edited\nreceptors", PAL["teal"])]
for i, (k, lab, col) in enumerate(rows):
    ax.bar(i, K[k]["pooled_r_mean"], width=0.62, color=col, edgecolor="black", lw=0.4)
    ax.scatter([i] * len(K[k]["pooled_r_seeds"]), K[k]["pooled_r_seeds"], s=6, color="black", zorder=3, lw=0)
desc = [("gbm_descriptors_geo", "Boosting\n+ geometry"), ("gbm_descriptors", "Boosting"), ("size_only", "Size\nmodel")]
for j2, (k, lab) in enumerate(desc):
    ax.bar(len(rows) + j2, D[k]["pooled_r"], width=0.62, color="#d9d9d9", edgecolor="black", lw=0.4)
ax.axhline(S["rsg_int"]["pooled_r_mean"], color=PAL["blue"], lw=0.8, ls="--")
ax.text(len(rows) + 2.45, S["rsg_int"]["pooled_r_mean"] + 0.012, "held-out families", fontsize=6.0, color=PAL["blue"], ha="right", va="bottom")
ax.set_xticks(range(len(rows) + len(desc)))
ax.set_xticklabels([r[1].replace(chr(10), " ") for r in rows] + [d[1].replace(chr(10), " ") for d in desc], fontsize=6.2, rotation=40, ha="right")
ax.set_ylabel("pooled $r$ on four kinases", fontsize=6.9)
ax.set_ylim(0, 0.45)
panel(ax, "c", dx=-0.2)
plt.savefig(Path(__file__).parent / "figures/fig5_scale.pdf")
plt.savefig(Path(__file__).parent / "figures/fig5_scale.png", dpi=200)
print("saved")
