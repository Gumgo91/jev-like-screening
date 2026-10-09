"""Figure 3: predicted against measured changes for four models, and the pooled correlation of every condition.

Usage: python fig3_main.py <summary_all.parquet> <summary_all.json> <paired_all.json> <baselines_v2.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from conds import GCOL, GNAME, GROUP, LABEL, ORDER  # noqa: E402
from figstyle import DOUBLE, PAL, apply, panel  # noqa: E402

apply()
df = pd.read_parquet(sys.argv[1])
S = json.loads(Path(sys.argv[2]).read_text())
P = json.loads(Path(sys.argv[3]).read_text())
ceiling = json.loads(Path(sys.argv[4]).read_text())["_noise"]["r_max"]
df = df[df.kind.isin(["single_contact"]) | df.kind.str.startswith("multi")]


def ens(prefix: str) -> pd.DataFrame:
    g = df[df.model.str.match(rf"^{prefix}_s\d+$")] if prefix not in ("gbm_descriptors_geo",) else df[df.model == prefix]
    return g.groupby(["target", "receptor", "ligand", "kind"], as_index=False).agg(pred=("pred", "mean"), dS=("dS", "first"))


fig = plt.figure(figsize=(DOUBLE, 5.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.62, wspace=0.42)
panels = [("C0", "Baseline surrogate", PAL["grey"]), ("rsg_wt", "Geometry, WT only", PAL["teal"]),
          ("rsg_int", "Geometry, interventional", PAL["blue"]), ("gbm_descriptors_geo", "Boosting + geometry", PAL["grey"])]
for i, (key, title, col) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    g = ens(key)
    sub = g.sample(min(len(g), 25000), random_state=0)
    ax.hexbin(sub.pred, sub.dS, gridsize=38, extent=(-2.5, 2.5, -2.5, 2.5), bins="log", cmap="Blues" if col == PAL["blue"] else ("GnBu" if col == PAL["teal"] else "Greys"),
              mincnt=1, linewidths=0)
    ax.plot([-2.5, 2.5], [-2.5, 2.5], color=PAL["red"], lw=0.7, ls="--")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    r = P["conditions"][key]["r"]
    ax.text(0.04, 0.96, f"$r$ = {r:.2f}\nSD {g.pred.std():.2f}", transform=ax.transAxes, va="top", fontsize=6.6)
    ax.set_title(title, fontsize=7.2, pad=3)
    ax.set_xlabel("predicted change", fontsize=6.8)
    if i == 0:
        ax.set_ylabel("measured change (kcal mol$^{-1}$)", fontsize=6.8)
    panel(ax, "abcd"[i], dx=-0.27 if i == 0 else -0.16, dy=1.17)

ax = fig.add_subplot(gs[1, :])
x, xs, labels, prev = 0.0, [], [], None
for key, label, grp in ORDER:
    if key not in P["conditions"]:
        continue
    if prev is not None and grp != prev:
        x += 0.7
    c = P["conditions"][key]
    ax.bar(x, c["r"], width=0.72, color=GCOL[grp], edgecolor="black", lw=0.4, zorder=2)
    ax.plot([x, x], c["ci"], color="black", lw=0.9, zorder=3)
    seeds = S.get(key, {}).get("pooled_r_seeds", [])
    if len(seeds) > 1:
        ax.scatter([x + np.random.default_rng(1).uniform(-0.18, 0.18) for _ in seeds], seeds, s=5, color="black", zorder=4, lw=0)
    xs.append(x)
    labels.append(label)
    prev = grp
    x += 1
ax.axhline(0, color="black", lw=0.6)
ax.axhline(ceiling, color=PAL["red"], lw=0.8, ls="--")
ax.text(x - 0.3, ceiling - 0.03, "noise ceiling", ha="right", va="top", fontsize=6.4, color=PAL["red"])
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=6.1)
ax.set_ylabel("pooled $r$ on held-out targets\n(seed ensemble, 95% CI over targets)", fontsize=6.9)
ax.set_ylim(-0.12, 1.0)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
handles = [Patch(facecolor=GCOL[g], edgecolor="black", lw=0.4, label=GNAME[g]) for g in GNAME]
handles.append(Line2D([], [], marker="o", ls="", color="black", markersize=2.5, label="single seeds"))
ax.legend(handles=handles, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=6.2, handlelength=1.0, columnspacing=0.9)
panel(ax, "e", dx=-0.05)
plt.savefig(Path(__file__).parent / "figures/fig3_main.pdf")
plt.savefig(Path(__file__).parent / "figures/fig3_main.png", dpi=200)
print("saved")
