"""Supporting Information figures.

S1 per-target correlation of selected models on the 19 held-out targets,
S2 the size law and the pose-contact dependence of the measured changes,
S3 training curves of the registered conditions.
Usage: python figS_all.py
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from figstyle import DOUBLE, PAL, SINGLE, apply, panel  # noqa: E402

apply()
OUT = Path(__file__).parent / "figures"
EV = ROOT / "dockmut/eval"

# ---------------------------------------------------------------- S1 per-target correlation
df = pd.read_parquet(EV / "summary_all.parquet")
df = df[df.kind.isin(["single_contact"]) | df.kind.str.startswith("multi")]
df["cond"] = df.model.map(lambda t: re.sub(r"_s\d+$", "", t))
tg = pd.read_parquet(ROOT / "data/splits/targets.parquet")[["target_id", "family", "split_role"]].rename(columns={"target_id": "target"})


def per_target_r(cond: str) -> pd.Series:
    g = df[df.cond == cond]
    e = g.groupby(["target", "receptor", "ligand"], as_index=False).agg(pred=("pred", "mean"), dS=("dS", "first"))
    return e.groupby("target").apply(lambda x: np.corrcoef(x.pred, x.dS)[0, 1] if x.pred.std() > 1e-9 else np.nan)


R = pd.DataFrame({c: per_target_r(c) for c in ["C0", "rsg_wt", "rsg_int", "rsg_int_shm2", "gbm_descriptors_geo"]})
R = R.join(tg.set_index("target"))
R = R.sort_values("rsg_int")
fig, ax = plt.subplots(figsize=(SINGLE * 1.25, 4.6))
styles = {"C0": ("Baseline", PAL["grey"], "s"), "rsg_wt": ("Geometry, WT only", PAL["teal"], "^"), "rsg_int": ("Geometry, interventional", PAL["blue"], "o"),
          "rsg_int_shm2": ("Deranged over edits", PAL["orange"], "v"), "gbm_descriptors_geo": ("Boosting + geometry", "black", "D")}
for c, (lab, col, mk) in styles.items():
    ax.scatter(R[c], range(len(R)), s=16, color=col, marker=mk, label=lab, zorder=3, lw=0)
ax.axvline(0, color="black", lw=0.6)
ax.set_yticks(range(len(R)))
ax.set_yticklabels([f"{t} ({r})" for t, r in zip(R.index, R.split_role)], fontsize=6.5)
ax.set_xlabel("correlation over the contact pairs of one target")
ax.legend(frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, handletextpad=0.2, columnspacing=0.8)
ax.grid(axis="y", lw=0.3, color="0.85", zorder=0)
plt.savefig(OUT / "figS1_pertarget.pdf")
plt.savefig(OUT / "figS1_pertarget.png", dpi=200)
plt.close()

# ---------------------------------------------------------------- S2 size law and pose contacts
G = pd.read_csv(ROOT / "dockmut/analysis/geometry/target_geometry.csv")
C = pd.read_parquet(ROOT / "dockmut/analysis/contacts/contact_changes.parquet")
fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.4), gridspec_kw={"wspace": 0.5})
ax = axs[0]
cols = {"train": PAL["blue"], "dev": PAL["orange"], "test": PAL["green"]}
ax.scatter(G.buried_frac, G.slope_ha, c=[cols[r] for r in G.role], s=12, lw=0)
r = np.corrcoef(G.buried_frac, G.slope_ha)[0, 1]
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("buried fraction of free space")
ax.set_ylabel("slope of change with heavy atoms\n(kcal mol$^{-1}$ per atom)", fontsize=6.8)
ax.text(0.05, 0.05, f"$r$ = {r:.2f}", transform=ax.transAxes, fontsize=6.8)
from matplotlib.lines import Line2D  # noqa: E402
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, markersize=3.5, label=k) for k, c in cols.items()], frameon=False, fontsize=6.2,
          loc="upper right", handletextpad=0.1, bbox_to_anchor=(1.0, 0.6))
panel(ax, "a", dx=-0.28)
ax = axs[1]
ax.scatter(G.n_atoms_r10, G.slope_ha, c=[cols[r] for r in G.role], s=12, lw=0)
r2 = np.corrcoef(G.n_atoms_r10, G.slope_ha)[0, 1]
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("receptor atoms within 10 Å (hundreds)")
ax.text(0.05, 0.05, f"$r$ = {r2:.2f}", transform=ax.transAxes, fontsize=6.8)
panel(ax, "b", dx=-0.2)
ax = axs[2]
C["abs"] = C.dS.abs()
C["bin"] = pd.cut(C.n_contact, [-1, 0, 2, 5, 100], labels=["0", "1-2", "3-5", "6+"])
C["cls"] = np.where(C.kind.isin(["single_shell", "single_far"]), "controls", np.where(C.kind == "single_contact", "single contact", "multi contact"))
S = C.groupby(["cls", "bin"], observed=True)["abs"].mean().unstack(0)
x = np.arange(len(S))
w = 0.26
for i, (cl, col) in enumerate([("controls", PAL["light"]), ("single contact", PAL["blue"]), ("multi contact", PAL["purple"])]):
    ax.bar(x + (i - 1) * w, S[cl], width=w, color=col, edgecolor="black", lw=0.4, label=cl)
ax.set_xticks(x)
ax.set_xticklabels(S.index)
ax.set_xlabel("ligand atoms within 4.5 Å")
ax.set_ylabel("mean |change| (kcal mol$^{-1}$)")
ax.legend(frameon=False, fontsize=6.2, loc="upper left")
panel(ax, "c", dx=-0.2)
plt.savefig(OUT / "figS2_law.pdf")
plt.savefig(OUT / "figS2_law.png", dpi=200)
plt.close()

# ---------------------------------------------------------------- S3 training curves
fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), gridspec_kw={"wspace": 0.3})
base = ROOT / "dockmut/runs_collected/final"
curves = {"rsg_wt": ("Geometry, WT only", PAL["teal"]), "rsg_int": ("Geometry, interventional", PAL["blue"]),
          "rsg_int_shm2": ("Deranged over edits", PAL["orange"]), "rsg_int_sha2": ("All deranged", PAL["red"])}
for c, (lab, col) in curves.items():
    for k, f in enumerate(sorted(glob.glob(str(base / f"{c}_s11_loss.csv")) + glob.glob(str(base / f"{c}_s22_loss.csv")) + glob.glob(str(base / f"{c}_s33_loss.csv")))):
        d = pd.read_csv(f)
        axs[0].plot(d["update"], d.loss_wt, color=col, lw=0.8, alpha=0.8, label=lab if k == 0 else None)
        if c != "rsg_wt":
            axs[1].plot(d["update"], d.loss_delta, color=col, lw=0.8, alpha=0.8, label=lab if k == 0 else None)
axs[0].set_yscale("log")
axs[0].set_xlabel("update")
axs[0].set_ylabel("Huber loss, wild-type scores")
axs[0].legend(frameon=False, fontsize=6.3)
panel(axs[0], "a", dx=-0.16)
axs[1].set_xlabel("update")
axs[1].set_ylabel("Huber loss, measured changes")
axs[1].legend(frameon=False, fontsize=6.3)
panel(axs[1], "b", dx=-0.16)
plt.savefig(OUT / "figS3_curves.pdf")
plt.savefig(OUT / "figS3_curves.png", dpi=200)
print("saved S1 to S3")
