"""Supporting Information tables: si_table_speed.tex (rates of the Uni-Dock benchmark, eval/docking_timing*.csv) and
si_table_joint.tex (runs of the existing pooled and cross-attention networks, runs/v3/registry_*.json).
Usage: python manuscript/make_si_tables2.py
"""
from __future__ import annotations

import json

import paths


def table_speed_si():
    """Per-receptor rates of the docking benchmark."""
    import pandas as pd
    for f in paths.DOCKING_TIMING_FILES:
        if not f.exists():
            raise FileNotFoundError(f"docking timing file not found: {f}")
    dt = pd.concat([pd.read_csv(f) for f in paths.DOCKING_TIMING_FILES])
    dt["rate"] = dt.n_ok / dt.sec
    lines = [r"\begin{table}[h]",
             r"\caption{Rates of the Uni-Dock benchmark. Ligands docked per second by Uni-Dock 1.2.0 on one RTX 4090 for each receptor, search mode and number of ligands per call "
             r"(median over the timed calls, whose number is given in the last column; the first call of each job and receptor is a warm-up and is not counted). "
             r"The rates of the main text are the medians over the three receptors at 4{,}000 ligands per call.}",
             r"\label{tab:si_speed}", r"\small", r"\begin{tabular}{llrrr}", r"\toprule",
             r"Receptor & Mode & Per call & Rate & Calls " + r"\\", r"\midrule"]
    for (t, mode, n), g in dt[dt.warmup == 0].groupby(["target", "mode", "n"], sort=False):
        lines.append(f"{t} & {mode} & {n:,} & {g.rate.median():.1f} & {len(g)} ".replace(",", "{,}", 1) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (paths.MS / "si_table_speed.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("si_table_speed written")


def table_joint_si():
    """Per-run record of the existing pooled and cross-attention networks."""
    reg = {f.stem.replace("registry_", ""): json.loads(f.read_text()) for f in sorted(paths.RUNS_V3.glob("registry_*.json"))}
    if not reg:
        raise FileNotFoundError(f"no registry_*.json in {paths.RUNS_V3}")
    lines = [r"\begin{table}[h]",
             r"\caption{Runs of the existing pooled and cross-attention networks. Updates and best update count training updates of 256 ligands; "
             r"the checkpoint is the evaluation point with the highest macro recall on the ten development targets (top-1\% recall at the 10\% budget, Dev. recall).}",
             r"\label{tab:si_joint_runs}", r"\small", r"\begin{tabular}{llrrrr}", r"\toprule", r"Network & Seed & Updates & Best update & Stop & Dev. recall " + r"\\", r"\midrule"]
    for tag, r in reg.items():
        stop = "patience" if "patience" in r["stop_reason"] else "budget"
        nm = "Cross-attention" if r["model"] == "xattn" else "Pooled concatenation"
        lines.append(f"{nm} & {r['seed']} & {r['updates']:,} & {r['best_update']:,} & {stop} & {r['best_cold_recall_1@10']:.3f} ".replace(",", "{,}") + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (paths.MS / "si_table_joint.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("si_table_joint written")


if __name__ == "__main__":
    table_speed_si()
    table_joint_si()
