"""Figure 4: where the response of a surrogate comes from.

(a-c) three levels of faithfulness on the 19 held-out targets: within-target r over all contact pairs, r of the edit means
within a target, and the ligand-specific r within an edit. (d) new edits of known pockets against new targets.
Usage: python fig4_levels.py <levels_all.json> <runs_dir_with_diag_json> <paired_all.json>
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from conds import GCOL, GROUP, LABEL  # noqa: E402
from figstyle import DOUBLE, PAL, apply, panel  # noqa: E402

apply()
L = json.loads(Path(sys.argv[1]).read_text())
runs = Path(sys.argv[2])
P = json.loads(Path(sys.argv[3]).read_text())
rows = ["C0", "rsg_wt", "rsg_int", "rsg_int_shl", "rsg_int_shm2", "rsg_int_sha2", "gbm_descriptors_geo", "gbm_pose_geo"]
short = {"C0": "Baseline", "rsg_wt": "Geometry, WT only", "rsg_int": "Geometry, interventional", "rsg_int_shl": "Permuted over ligands",
         "rsg_int_shm2": "Deranged over edits", "rsg_int_sha2": "All deranged", "gbm_descriptors_geo": "Boosting + geometry",
         "gbm_pose_geo": "Boosting + geometry + pose"}
fig = plt.figure(figsize=(DOUBLE, 2.9))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.5], wspace=0.16)
titles = [("within_target_r", "Within target"), ("edit_mean_r", "Edit means"), ("ligand_r", "Ligand-specific")]
for j, (key, title) in enumerate(titles):
    ax = fig.add_subplot(gs[j])
    for i, r in enumerate(rows):
        v = L[r][key]
        ci = L[r][key + "_ci"]
        ax.barh(i, v, height=0.62, color=GCOL[GROUP[r]], edgecolor="black", lw=0.4)
        ax.plot(ci, [i, i], color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([short[r] for r in rows] if j == 0 else [], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=7.4, pad=3)
    ax.set_xlabel("mean $r$ over targets", fontsize=6.8)
    ax.set_xlim(-0.12, 0.72 if key != "ligand_r" else 0.3)
    panel(ax, "abc"[j], dx=-0.12 if j else -0.75, dy=1.12)

ax = fig.add_subplot(gs[3])
cond = ["rsg_wt", "rsg_int", "rsg_int_shl", "rsg_int_shm2", "rsg_int_sha2"]
known = {}
for c in cond:
    vals = []
    for f in glob.glob(str(runs / f"{c}_s*_diag.json")):
        name = Path(f).name
        if name.startswith(c + "_s") and name[len(c) + 2:len(c) + 4].isdigit():
            vals.append(json.loads(Path(f).read_text())["val_mutants"]["pooled_r"])
    known[c] = vals
w = 0.38
for i, c in enumerate(cond):
    k = np.array(known[c])
    ax.bar(i - w / 2, k.mean(), width=w, color=GCOL[GROUP[c]], edgecolor="black", lw=0.4)
    ax.scatter([i - w / 2] * len(k), k, s=4, color="black", zorder=3, lw=0)
    n = P["conditions"][c]
    ax.bar(i + w / 2, n["r"], width=w, color="white", edgecolor=GCOL[GROUP[c]], lw=1.0, hatch="////")
    ax.plot([i + w / 2] * 2, n["ci"], color="black", lw=0.8)
ax.set_xticks(range(len(cond)))
ax.set_xticklabels(["WT\nonly", "Interv.", "Perm.\nligands", "Der.\nedits", "Der.\nall"], fontsize=6.4)
ax.set_ylabel("pooled $r$", fontsize=6.9)
ax.axhline(0, color="black", lw=0.6)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(facecolor="0.75", edgecolor="black", lw=0.4, label="new edits, known pockets"),
                   Patch(facecolor="white", edgecolor="0.3", lw=1.0, hatch="////", label="new targets")],
          frameon=False, loc="upper right", fontsize=6.0, handlelength=1.3, bbox_to_anchor=(1.02, 1.04))
ax.set_ylim(-0.05, 0.8)
panel(ax, "d", dx=-0.18, dy=1.12)
plt.savefig(Path(__file__).parent / "figures/fig4_levels.pdf")
plt.savefig(Path(__file__).parent / "figures/fig4_levels.png", dpi=200)
print("saved", {c: (round(float(np.mean(v)), 3), len(v)) for c, v in known.items()})
