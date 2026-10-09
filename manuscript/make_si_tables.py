"""Supporting Information tables generated from analysis outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "dockmut/analysis"
OUT = Path(__file__).parent


def num(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{d}f}".replace("-", "$-$")


def table_targets():
    T = pd.read_csv(ANA / "per_target.csv")
    D = pd.read_csv(ANA / "decomposition.csv")[["target", "mutant_mean", "ligand_main", "interaction", "noise"]]
    P = pd.read_csv(ANA / "wt_parity.csv")[["target", "pearson", "mae"]]
    tg = pd.read_parquet(ROOT / "data/splits/targets.parquet")[["target_id", "family"]].rename(columns={"target_id": "target"})
    df = T.merge(tg, on="target").merge(D, on="target", how="left").merge(P, on="target", how="left")
    df = df.sort_values(["role", "family", "target"])
    L = [r"\begin{longtable}{llp{2.4cm}rrrrrrrr}",
         r"\caption{Per-target summary of DockMut-1. $\sigma$ is the replicate noise of the wild-type score, $n_\mathrm{stable}$ the ligands kept after the stability filter, the four shares are fractions of the variance of measured changes in percent, and $r_\mathrm{WT}$ and MAE compare the wild-type scores of the GPU engine with the DOCKSTRING table (training and development targets only).}",
         r"\label{tab:si_targets}\\", r"\toprule",
         r"Target & Role & Family & Edits & $n_\mathrm{stable}$ & $\sigma$ & Mean & Ligand & Inter. & Noise & $r_\mathrm{WT}$ \\", r"\midrule", r"\endfirsthead",
         r"Target & Role & Family & Edits & $n_\mathrm{stable}$ & $\sigma$ & Mean & Ligand & Inter. & Noise & $r_\mathrm{WT}$ \\", r"\midrule", r"\endhead"]
    for r in df.itertuples():
        fam = str(r.family).replace("_", " ")
        L.append(f"{r.target} & {r.role} & {fam} & {int(r.n_mutants)} & {int(r.n_stable)} & {num(r.sigma_seed)} & "
                 f"{num(100 * r.mutant_mean, 0)} & {num(100 * r.ligand_main, 0)} & {num(100 * r.interaction, 0)} & {num(100 * r.noise, 0)} & {num(r.pearson)} " + r"\\")
    L += [r"\bottomrule", r"\end{longtable}"]
    (OUT / "si_table_targets.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("si_table_targets written", len(df))


if __name__ == "__main__":
    table_targets()


VAL_NAMES = {
    "rsg_wt_V0": ("A", "ResidueSum-Geo", "no intervention"), "rs_int_V1": ("A", "ResidueSum", "$\lambda$ = 3"),
    "rs_int_V2": ("A", "ResidueSum", "$\lambda$ = 10"), "rsg_int_V3": ("A", "ResidueSum-Geo", "$\lambda$ = 3"),
    "rsg_int_V4": ("A", "ResidueSum-Geo", "$\lambda$ = 10"), "rsg_int_V5": ("A", "ResidueSum-Geo", "$\lambda$ = 30"),
    "rsg_int_V6": ("A", "ResidueSum-Geo", "$\lambda$ = 10, rate $10^{-3}$"), "b5_int_V7": ("A", "Dual encoder", "$\lambda$ = 10"),
    "rs_int_V8": ("A", "ResidueSum", "$\lambda$ = 10, family balance"), "rsg_int_V9": ("A", "ResidueSum-Geo", "$\lambda$ = 10, family balance"),
    "rs_wt_W_A": ("A", "ResidueSum", "no intervention"), "rsg_int_G1_A": ("A", "ResidueSum-Geo", "$\lambda$ = 1"),
    "rse_int_E3_A": ("A", "ResidueSum-Edit", "$\lambda$ = 3"), "rse_int_E10_A": ("A", "ResidueSum-Edit", "$\lambda$ = 10"),
    "rse_wt_EW_A": ("A", "ResidueSum-Edit", "no intervention"),
    "rs_int_R10_B": ("B", "ResidueSum", "$\lambda$ = 10"), "rs_int_R3_B": ("B", "ResidueSum", "$\lambda$ = 3"),
    "rs_wt_W_B": ("B", "ResidueSum", "no intervention"), "rsg_wt_GW_B": ("B", "ResidueSum-Geo", "no intervention"),
    "rsg_int_G1_B": ("B", "ResidueSum-Geo", "$\lambda$ = 1"), "rsg_int_G3_B": ("B", "ResidueSum-Geo", "$\lambda$ = 3"),
    "rsg_int_G10_B": ("B", "ResidueSum-Geo", "$\lambda$ = 10"), "rse_int_E3_B": ("B", "ResidueSum-Edit", "$\lambda$ = 3"),
    "rse_int_E10_B": ("B", "ResidueSum-Edit", "$\lambda$ = 10"), "rse_wt_EW_B": ("B", "ResidueSum-Edit", "no intervention"),
}


def table_validation():
    import re
    rows = []
    for f in ("validation_round1_table.txt", "validation_round2_table.txt"):
        for line in (ROOT / "dockmut/eval" / f).read_text().splitlines():
            m = re.match(r"^(\S+)_s11\s+pod\d\s+\|\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)", line)
            if m and m.group(1) in VAL_NAMES:
                fold, arch, cfg = VAL_NAMES[m.group(1)]
                rows.append((fold, arch, cfg, float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(6)), float(m.group(7))))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    L = [r"\begin{longtable}{llp{3.3cm}rrrrr}",
         r"\caption{Validation of candidate configurations. Fold A withholds class A GPCRs, cytochromes P450 and S1 serine proteases, fold B withholds the nuclear receptors. Pearson $r$ over contact edits of the withheld families, the correlation of edit means, the within-edit correlation, the SD of the predicted change in \kcal{}, and $r$ for edits that were withheld from training targets (new edits of known pockets).}",
         r"\label{tab:si_val}\\", r"\toprule", r"Fold & Model & Setting & $r$ & Edit mean $r$ & Within $r$ & Pred. SD & New edits $r$ \\", r"\midrule", r"\endfirsthead",
         r"Fold & Model & Setting & $r$ & Edit mean $r$ & Within $r$ & Pred. SD & New edits $r$ \\", r"\midrule", r"\endhead"]
    for fold, arch, cfg, r, mr, wr, sd, nr in rows:
        L.append(f"{fold} & {arch} & {cfg} & {num(r)} & {num(mr)} & {num(wr)} & {sd:.2f} & {num(nr)} " + r"\\")
    L += [r"\bottomrule", r"\end{longtable}"]
    (OUT / "si_table_validation.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("si_table_validation written", len(rows))


if __name__ == "__main__":
    table_validation()
