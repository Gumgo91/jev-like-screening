"""Figure 7: virtual alanine scan of a held-out target with a cached pocket state.

(a) Pocket map: residue C-alpha positions projected on the plane of largest spread, filled by the predicted mean absolute
    change of the truncation over the library, ringed by the measured mean absolute change for residues that were docked.
(b) Predicted against measured mean absolute change per docked residue.
(c) Ligand-edit pairs per second of GPU docking and of the surrogate scan (end to end, including ligand featurisation).
Usage: python fig7_scan.py <target> <scan.csv> <timing.json> <results_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "dockmut"))
from dm_core import delta_table, read_results  # noqa: E402
from figstyle import DOUBLE, PAL, apply, panel  # noqa: E402
from make_mutants import read_box, read_receptor  # noqa: E402

target, scan_csv, timing_json, res_dir = sys.argv[1:5]
EVD = ROOT / "dockmut/eval"
_dt = pd.concat([pd.read_csv(f) for f in (EVD / "docking_timing.csv", EVD / "docking_timing_big.csv", EVD / "docking_timing_4k.csv") if f.exists()])
_dt = _dt[_dt.warmup == 0].copy()
_dt["rate"] = _dt.n_ok / _dt.sec
DOCK_RATE = float(_dt[_dt["mode"] == "detail"].groupby("n").rate.median().max())
FAST_RATE = float(_dt[_dt["mode"] == "fast"].groupby("n").rate.median().max())
apply()
scan = pd.read_csv(scan_csv)
timing = json.loads(Path(timing_json).read_text())
man = json.loads((ROOT / "dockmut/plan/receptors" / target / "manifest.json").read_text())
tab, _, _ = delta_table(read_results(Path(res_dir), target), man)
meas = []
for rid, (ids, dS, kind) in tab.items():
    if kind in ("single_contact", "single_shell"):
        rec = [r for r in man["receptors"] if r["id"] == rid][0]
        meas.append({"residue": rec["residues"][0]["label"], "meas_abs": float(np.mean(np.abs(dS))), "kind": kind})
scan = scan.merge(pd.DataFrame(meas), on="residue", how="left")
lines, atoms = read_receptor(ROOT / "dockmut/plan/wt_raw" / f"{target}_target.pdbqt")
ca = {f"{a['res']}{a['resid']}": a["xyz"] for a in atoms if a["name"] == "CA"}
scan = scan[scan.residue.isin(ca)].copy()
X = np.array([ca[r] for r in scan.residue])
box = read_box(ROOT / "data/raw/targets" / f"{target}_conf.txt")
c = np.array([box["center_x"], box["center_y"], box["center_z"]])
u, s, vt = np.linalg.svd(X - c, full_matrices=False)
P = (X - c) @ vt[:2].T

fig = plt.figure(figsize=(DOUBLE, 2.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.0], wspace=0.5)
ax = fig.add_subplot(gs[0])
vmax = float(np.nanpercentile(scan.mean_abs_dS, 98))
sc = ax.scatter(P[:, 0], P[:, 1], c=scan.mean_abs_dS, cmap="Blues", vmin=0, vmax=vmax, s=34, edgecolor="none", zorder=2)
m = scan.meas_abs.notna().to_numpy()
ax.scatter(P[m, 0], P[m, 1], s=70, facecolors="none", edgecolors=plt.cm.Oranges(np.clip(scan.meas_abs[m] / max(scan.meas_abs.max(), 1e-6), 0.25, 1)), linewidths=1.7, zorder=3)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.8)
cb.set_label("predicted mean |change|", fontsize=6.4)
cb.outline.set_linewidth(0.5)
ax.set_title(f"{target}: fill predicted, ring measured", fontsize=6.8, pad=2)
panel(ax, "a", dx=-0.06, dy=1.16)

ax = fig.add_subplot(gs[1])
d = scan[scan.meas_abs.notna()]
colors = [PAL["blue"] if k == "single_contact" else PAL["grey"] for k in d.kind]
ax.scatter(d.mean_abs_dS, d.meas_abs, s=16, c=colors, lw=0)
cont = d[d.kind == "single_contact"]
rho = spearmanr(cont.mean_abs_dS, cont.meas_abs)[0]
ax.text(0.05, 0.95, f"Spearman {rho:.2f}\n({len(cont)} contact residues)", transform=ax.transAxes, va="top", fontsize=6.6)
ax.scatter([], [], s=16, color=PAL["blue"], label="contact")
ax.scatter([], [], s=16, color=PAL["grey"], label="shell control")
ax.legend(frameon=False, loc="lower right", fontsize=6.2, handletextpad=0.2)
ax.set_xlabel("predicted mean |change|", fontsize=7)
ax.set_ylabel("measured mean |change|\n(kcal mol$^{-1}$)", fontsize=7)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
panel(ax, "b")

ax = fig.add_subplot(gs[2])
thr = json.loads((EVD / "throughput" / "thr_rsg_run2.json").read_text())
lib, sc, pk = thr["library"], thr["scan"], thr["pockets"]
scan_total = lib["sec_featurise_cpu"] + lib["sec_embed_gpu_all_models"] + sc["sec_build_edited_graphs_cpu"] + sc["sec_state_encoding_all_models"] + sc["sec_queries_all_models"]
pk_total = lib["sec_featurise_cpu"] + lib["sec_embed_gpu_all_models"] + pk["sec_state_encoding"] + pk["sec_scoring_all_models"]
vals = [DOCK_RATE, FAST_RATE, sc["pairs"] / scan_total, pk["pairs"] / pk_total]
ax.bar([0, 1, 2, 3], vals, color=[PAL["grey"], PAL["grey"], PAL["blue"], PAL["teal"]], width=0.66)
ax.set_yscale("log")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["Uni-Dock\ndetail", "Uni-Dock\nfast", "scan of\n47 edits", "library x\n57 pockets"], fontsize=5.8)
ax.set_ylabel("ligand-pocket pairs per second\n(one RTX 4090)", fontsize=6.8)
for x, v in zip([0, 1, 2, 3], vals):
    ax.text(x, v * 1.25, (f"{v:,.1f}" if v < 100 else f"{v:,.0f}"), ha="center", fontsize=6.6)
ax.set_ylim(1, max(vals) * 60)
ax.text(0.5, 0.98, f"260,155 ligands: scan {scan_total:.0f} s, 57 pockets {pk_total:.0f} s\n"
        f"docking the scan pairs: {sc['pairs'] / DOCK_RATE / 86400:.0f} GPU-days (detail),\n"
        f"{sc['pairs'] / FAST_RATE / 86400:.0f} GPU-days (fast)",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.8)
panel(ax, "c")
plt.savefig(Path(__file__).parent / "figures/fig7_scan.pdf")
plt.savefig(Path(__file__).parent / "figures/fig7_scan.png", dpi=200)
print("saved; spearman contact =", round(float(rho), 3))
