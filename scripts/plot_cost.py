"""Cost and recall of Jev-like and joint networks on known pockets (Figure 2).

(a) End-to-end time from SMILES to scores for the 260,155 ligands of DOCKSTRING against P pockets, measured at the full scale
    (eval/v4_fair_timing.json), and the time of Uni-Dock (median rate of the fast mode at 4,000 ligands per call, from eval/docking_timing*.csv).
(b) Top-1% recall at a 10% budget on the known pockets (eval/v5_known_cv_summary.json) against the end-to-end time for 1,000 pockets.
Writes figures/fig_cost.png.
Usage: python scripts/plot_cost.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
FIGURES = ROOT / "figures"

DOUBLE = 7.0    # inches, ACS double column
PAL = {"orange": "#d9730d", "purple": "#6f4a9e", "grey": "#6b6b6b", "light": "#d9d9d9"}


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
    })


def panel(ax, letter, dx=-0.16, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def load_timing(path):
    T = json.loads(Path(path).read_text(encoding="utf-8"))
    if "1000" not in T.get("blocks", {}).get("xattn_fp16", {}):
        raise ValueError(f"{Path(path).name} has no xattn_fp16 block for 1000 pockets")
    return T


def unidock_fast_rate():
    """Median fast-mode rate (ligands per second) at the number of ligands per call with the highest rate, from the timed calls."""
    files = [EVAL / n for n in ("docking_timing.csv", "docking_timing_big.csv", "docking_timing_4k.csv")]
    dt = pd.concat([pd.read_csv(f) for f in files])
    dt = dt[dt.warmup == 0].copy()
    dt["rate"] = dt.n_ok / dt.sec
    return float(dt[dt["mode"] == "fast"].groupby("n").rate.median().max())


style()
T = load_timing(EVAL / "v4_fair_timing.json")
GPU = T["environment"]["gpu"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
L0 = T["library_size"]
rate = unidock_fast_rate()
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
s = json.loads((EVAL / "v5_known_cv_summary.json").read_text())
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
FIGURES.mkdir(exist_ok=True)
plt.savefig(FIGURES / "fig_cost.png", dpi=200)
print("saved fig_cost, timing from eval/v4_fair_timing.json", GPU)
