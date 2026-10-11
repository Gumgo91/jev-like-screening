"""Tables of the manuscript: table_cost.tex (Table 2, full-scale timing), si_table_timing.tex and si_table_timing_b.tex (all stages of every block on the two GPUs).
The recall tables (Table 1 and the SI tables) are written by make_tables_known.py, which this script calls.

Reads eval/v4_fair_timing.json (RTX 4090 pod), eval/v4_fair_timing_L40S.json and eval/docking_timing*.csv (Uni-Dock rate of the fast mode).
Usage: python manuscript/make_tables_v5.py
"""
from __future__ import annotations

import paths

T = paths.load_timing(paths.TIMING_PRIMARY)
GPU = paths.gpu_name(T)
UNIDOCK_FAST_RATE = paths.unidock_rates()["fast"]       # ligands per second, 4,000 ligands per call
L0 = T["library_size"]
B = T["blocks"]


def fs(x):
    return f"{x:,.0f}".replace(",", "{,}") if x >= 100 else (f"{x:.1f}" if x >= 10 else (f"{x:.2f}" if x >= 0.1 else f"{x:.4f}"))


def table_cost():
    """Table of the main text: end-to-end times of the dual encoder, the cross-attention network and Uni-Dock for 57 and 1,000 pockets."""
    rows = [("Dual encoder", "dual"), ("Cross-attention, float16", "xattn_fp16"), ("Cross-attention, float32", "xattn_fp32")]
    lines = [r"\begin{table}[t]",
             r"\caption{Time to screen the " + f"{L0:,}".replace(",", "{,}") + r" ligands of DOCKSTRING from SMILES to scores at the full scale on one " + GPU + r" (seconds; best of two runs for the GPU stages, "
             r"featurization of the SMILES on " + str(T["environment"]["featurization_processes"]) + r" CPU processes included). 1{,}000 pockets are the 57 pockets of DOCKSTRING used cyclically. "
             r"Uni-Dock: library size times the number of pockets divided by the median docking rate of the fast mode at 4{,}000 ligands per call, without ligand preparation. "
             r"The model time and every stage are in the Supporting Information.}",
             r"\label{tab:cost}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrr}", r"\toprule",
             r"Network & 57 pockets & 1{,}000 pockets \\", r"\midrule"]
    for nm, k in rows:
        lines.append(f"{nm} & {fs(B[k]['57']['end_to_end_s'])} & {fs(B[k]['1000']['end_to_end_s'])} " + r"\\")
    r_fast = UNIDOCK_FAST_RATE
    lines.append(f"Uni-Dock, fast mode & {fs(L0 * 57 / r_fast)} & {fs(L0 * 1000 / r_fast)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (paths.MS / "table_cost.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_timing_si(Tx=None, outname="si_table_timing.tex", label="tab:si_timing"):
    Tx = Tx or T
    B = Tx["blocks"]
    GPU = paths.gpu_name(Tx)
    nets = [("Dual encoder", "dual"), ("Pooled concatenation", "pooled"), ("Cross-attention, float16", "xattn_fp16"), ("Cross-attention, float32", "xattn_fp32")]
    lines = [r"\begin{table}[h]", r"\caption{Stages of the full-scale timing on one " + GPU + r" (seconds, best of two runs for the GPU stages). Model time is the sum of the embedding, the pocket states, the scoring and the selection of the best 10\% of the ligands per pocket. "
             r"End to end is the model time plus the featurization of the SMILES on " + str(Tx["environment"]["featurization_processes"]) + r" CPU processes (" + fs(Tx["featurization"]["seconds"]) +
             r" s) and the collation of the batches (" + fs(next(iter(B["dual"].values()))["stages_s"]["collate_cpu"]) + r" s) and of the pocket graphs, which every network needs. Pockets beyond 57 use the 57 pockets cyclically.}",
             r"\label{" + label + "}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule", r"Network & Pockets & Embedding & Pocket states & Scoring & Selection & Model time & End to end \\", r"\midrule"]
    for nm, k in nets:
        for P in ("1", "5", "20", "57", "1000"):
            b = B[k][P]
            st = b["stages_s"]
            lines.append(f"{nm} & {P} & {fs(st['embed_gpu'])} & {fs(st['pocket_states_gpu'])} & {fs(st['score_gpu'])} & {fs(st['topk_gpu'])} & {fs(b['model_time_s'])} & {fs(b['end_to_end_s'])} " + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (paths.MS / outname).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    table_cost()
    import make_tables_known
    make_tables_known.main()
    table_timing_si()
    table_timing_si(paths.load_timing(paths.TIMING_SECOND), "si_table_timing_b.tex", "tab:si_timing_b")
    print("tables written, timing from", paths.TIMING_PRIMARY.name, GPU, "and", paths.TIMING_SECOND.name)
