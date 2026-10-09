"""Figure 2: how the docking engine responds to receptor truncations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "dockmut"))
from dm_core import delta_table, read_results  # noqa: E402
from decomp import decompose  # noqa: E402
from figstyle import DOUBLE, PAL, apply, panel  # noqa: E402

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dockmut/results"
ANA = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "dockmut/analysis"
EX = sys.argv[3] if len(sys.argv) > 3 else "ACHE"

apply()
T = pd.read_csv(ANA / "per_target.csv")
M = pd.read_csv(ANA / "per_mutant.csv")
plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
roles = {t: v["role"] for t, v in plan["targets"].items()}
D = decompose(RES, ROOT / "dockmut/plan", roles)
fig = plt.figure(figsize=(DOUBLE, 4.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.25], hspace=0.85, wspace=0.52)

# a: replicate noise per target
ax = fig.add_subplot(gs[0, 0])
Ts = T.sort_values("sigma_seed").reset_index(drop=True)
cmap = {"train": PAL["blue"], "dev": PAL["orange"], "test": PAL["green"]}
ax.scatter(Ts.index, Ts.sigma_seed, s=10, c=[cmap[r] for r in Ts.role], lw=0)
ax.axhline(Ts.sigma_seed.median(), color=PAL["grey"], lw=0.7, ls="--")
ax.text(len(Ts) * 0.02, Ts.sigma_seed.median() + 0.02, f"median {Ts.sigma_seed.median():.2f}", fontsize=6.3, color=PAL["grey"])
ax.set_xlabel("targets, sorted")
ax.set_ylabel("replicate noise $\\sigma$ (kcal mol$^{-1}$)")
ax.set_ylim(0, min(1.0, Ts.sigma_seed.max() * 1.1))
for r, c in cmap.items():
    ax.scatter([], [], s=10, c=c, label=r)
ax.legend(frameon=False, loc="lower right", handlelength=0.8, borderpad=0.1, labelspacing=0.2, bbox_to_anchor=(1.0, 0.02))
panel(ax, "a")

# b: SD of measured change by class against noise floor
ax = fig.add_subplot(gs[0, 1])
order = ["far control", "shell control", "single contact", "multi contact"]
cols = [PAL["light"], PAL["light"], PAL["blue"], PAL["purple"]]
data = [M[M["class"] == c].sd_dS.dropna().to_numpy() for c in order]
bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=0.9),
                whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7), boxprops=dict(lw=0.7))
for b, c in zip(bp["boxes"], cols):
    b.set_facecolor(c)
ax.axhline(M.sigma_delta.median(), color=PAL["red"], lw=0.8, ls="--")
ax.text(0.03, 0.97, "dashed line: noise floor", transform=ax.transAxes, fontsize=6.0, color=PAL["red"], va="top", ha="left")
ax.set_xticks(range(1, 5))
ax.set_xticklabels(["far", "shell", "single", "multi"], fontsize=6.6)
ax.set_xlabel("far and shell: controls; single and multi: contact", fontsize=5.8)
ax.set_ylabel("SD of change over ligands\n(kcal mol$^{-1}$)")
panel(ax, "b")

# c: variance decomposition
ax = fig.add_subplot(gs[0, 2])
comp = ["mutant_mean", "ligand_main", "interaction", "noise"]
names = ["mutant mean", "ligand main effect", "mutant $\\times$ ligand", "replicate noise"]
ccol = [PAL["blue"], PAL["orange"], PAL["purple"], PAL["light"]]
grp = {"training targets": D[D.role == "train"], "held-out targets": D[D.role != "train"]}
left = np.zeros(len(grp))
for c, n, col in zip(comp, names, ccol):
    v = np.array([g[c].mean() for g in grp.values()])
    ax.barh(range(len(grp)), v, left=left, color=col, edgecolor="white", lw=0.6, label=n, height=0.55)
    for i, (l, w) in enumerate(zip(left, v)):
        if w > 0.06:
            ax.text(l + w / 2, i, f"{w*100:.0f}", ha="center", va="center", fontsize=6.3, color="white" if col != PAL["light"] else "black")
    left += v
ax.set_yticks(range(len(grp)))
ax.set_yticklabels(list(grp), fontsize=6.6)
ax.set_xlim(0, 1)
ax.set_xlabel("share of variance of measured change")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, handlelength=0.9, columnspacing=1.0)
panel(ax, "c")

# d: change matrix of one held-out target
ax = fig.add_subplot(gs[1, :])
man = json.loads((ROOT / "dockmut/plan/receptors" / EX / "manifest.json").read_text())
tab, sigma, _ = delta_table(read_results(RES, EX), man)
cm = [r for r, v in tab.items() if v[2] == "single_contact" or v[2].startswith("multi")]
common = sorted(set.intersection(*[set(tab[r][0]) for r in cm]))
lig = pd.read_csv(ROOT / "dockmut/plan/ligands/heldout_shared.csv").set_index("inchikey")
ha = lig.heavy_atoms.reindex(common)
common = [c for c, _ in sorted(zip(common, ha), key=lambda x: x[1])]
X = np.array([[dict(zip(tab[r][0], tab[r][1]))[l] for l in common] for r in cm])
dmin = []
for r in cm:
    rec = [x for x in man["receptors"] if x["id"] == r][0]
    dmin.append(min([z["d_sc"] for z in rec["residues"] if z["d_sc"] is not None]))
idx = np.argsort([(0 if tab[r][2] == "single_contact" else 1, d) for r, d in zip(cm, dmin)], axis=0)[:, 0]
kinds = [tab[cm[i]][2] for i in range(len(cm))]
order = sorted(range(len(cm)), key=lambda i: (0 if kinds[i] == "single_contact" else 1, dmin[i]))
X = X[order]
im = ax.imshow(X, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5, interpolation="nearest")
ax.set_yticks([])
ax.set_ylabel(f"{len(cm)} truncations\n(single, then multi)", fontsize=6.8)
ax.set_xlabel(f"{X.shape[1]} ligands of {EX}, sorted by heavy-atom count")
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
cb.set_label("measured change (kcal mol$^{-1}$)", fontsize=6.6)
cb.outline.set_linewidth(0.5)
n_single = sum(1 for k in kinds if k == "single_contact")
ax.axhline(n_single - 0.5, color="black", lw=0.6)
panel(ax, "d", dx=-0.045)
plt.savefig(Path(__file__).parent / "figures/fig2_response.pdf")
plt.savefig(Path(__file__).parent / "figures/fig2_response.png", dpi=200)
print("saved; decomposition:", D[comp].mean().round(3).to_dict())
