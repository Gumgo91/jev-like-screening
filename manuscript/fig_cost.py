"""Cost and recall of Jev-like and joint networks on known pockets (Figure 2).

(a) End-to-end time from SMILES to scores for the 260,155 ligands of DOCKSTRING against P pockets, measured at the full scale
    (dockmut/v4_fair_timing.json, addendum 9), and the time of Uni-Dock (median rate of the fastest search mode at 4,000 ligands per call).
(b) Top-1% recall at a 10% budget on the known pockets (dockmut/eval/v5_known_cv_summary.json) against the end-to-end time for 1,000 pockets.
Usage: python fig_cost.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figstyle import DOUBLE, PAL, apply, panel

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "dockmut/eval"
apply()
f = EV / "v4_fair_timing.json"
try:
    _ok = "1000" in json.loads(f.read_text())["blocks"].get("xattn_fp16", {})
except Exception:
    _ok = False
if not _ok:
    f = EV / "v4_fair_timing_L40S.json"
T = json.loads(f.read_text())
GPU = T["environment"]["gpu"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
CM = json.loads((EV / "cost_model.json").read_text())
L0 = T["library_size"]
rate = CM["unidock_rate"]["fast"]
PS = [1, 5, 20, 57, 1000]
COL = {"dual": PAL["orange"], "xattn_fp16": PAL["purple"], "xattn_fp32": PAL["purple"], "dock": PAL["grey"]}
LAB = {"dual": "dual encoder", "xattn_fp16": "cross-attention, float16", "xattn_fp32": "cross-attention, float32"}
LS = {"dual": "-", "xattn_fp16": "-", "xattn_fp32": "--"}


def series(net, key):
    return np.array([T["blocks"][net][str(P)][key] for P in PS])


fig = plt.figure(figsize=(DOUBLE, 2.7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.35)
Pg = np.logspace(0, np.log10(1000), 30)
ax = fig.add_subplot(gs[0])
for net in ("xattn_fp32", "xattn_fp16", "dual"):
    ax.plot(PS, series(net, "end_to_end_s"), color=COL[net], ls=LS[net], lw=1.3, marker="o", ms=2.8, label=LAB[net])
ax.plot(Pg, L0 * Pg / rate, color=COL["dock"], lw=1.3, label="Uni-Dock, fast mode")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("pockets (260,155 ligands)", fontsize=7)
ax.set_ylabel("seconds from SMILES to scores", fontsize=7)
ax.set_ylim(10, 3e7)
ax.axvline(57, color=PAL["light"], lw=0.8, zorder=0)
ax.legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.8, labelspacing=0.15)
panel(ax, "a", dx=-0.17)

ax = fig.add_subplot(gs[1])
K = EV / "v5_known_cv_summary.json"
if K.exists():
    s = json.loads(K.read_text())
    t1000 = {n: T["blocks"][n]["1000"]["end_to_end_s"] for n in ("dual", "xattn_fp16", "xattn_fp32")}
    pts = [("dual encoder", "dual_R0", t1000["dual"], COL["dual"], 0.95, "o"),
           ("constant pocket", "dual_const_R0", t1000["dual"], PAL["grey"], 1.05, "o"),
           ("cross-attention, float16", "xattn_R0", t1000["xattn_fp16"], COL["xattn_fp16"], 1.0, "o")]
    for lab, cond, tx, col, fx, mk in pts:
        m = s["mean"][cond]["recall_1@10"]
        lo, hi = s["ci"][cond]["recall_1@10"]
        ax.errorbar([tx * fx], [m], yerr=[[m - lo], [hi - m]], fmt=mk, ms=3.8, color=col, lw=0.9, capsize=1.5, label=lab)
    mx = s["mean"]["xattn_R0"]["recall_1@10"]
    lo, hi = s["ci"]["xattn_R0"]["recall_1@10"]
    ax.errorbar([t1000["xattn_fp32"]], [mx], yerr=[[mx - lo], [hi - mx]], fmt="s", ms=3.4, color=COL["xattn_fp32"], mfc="white", lw=0.9, capsize=1.5, label="cross-attention, float32")
    ax.legend(frameon=False, fontsize=5.8, loc="lower right", handlelength=1.0, labelspacing=0.2, bbox_to_anchor=(1.02, 0.0))
    ax.set_xscale("log")
    ax.set_xlabel("seconds from SMILES to scores for 1,000 pockets", fontsize=7)
    ax.set_ylabel("top-1% recall at 10% budget", fontsize=7)
    ax.set_title(str(s["n_targets"]["dual_R0"]) + " known pockets, new ligands", fontsize=6.8, pad=2)
    xs = list(t1000.values())
    ax.set_xlim(min(xs) * 0.55, max(xs) * 2.2)
panel(ax, "b")
plt.savefig(Path(__file__).parent / "figures/fig_cost.pdf")
plt.savefig(Path(__file__).parent / "figures/fig_cost.png", dpi=200)
print("saved fig_cost, timing from", f.name, GPU)
