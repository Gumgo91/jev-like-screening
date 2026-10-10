"""Tables of the known-pocket manuscript: table_cost.tex (full-scale timing), table_known.tex (recall on known pockets), si_table_known_targets.tex,
si_table_timing.tex (all stages of every block).

Reads dockmut/eval/v4_fair_timing.json (4090 pod; falls back to the L40S file), cost_model.json (Uni-Dock rates) and v5_known_pockets_summary.json.
Usage: python make_tables_v5.py
"""
from __future__ import annotations

import json
from pathlib import Path

here = Path(__file__).parent
EV = here.parent / "dockmut/eval"
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
    r_fast = CM["unidock_rate"]["fast"]
    lines.append(f"Uni-Dock, fast mode & {fs(L0 * 57 / r_fast)} & {fs(L0 * 1000 / r_fast)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (here / "table_cost.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_cost_full():
    rows = [("Dual encoder", "dual"), ("Pooled concatenation", "pooled"),
            ("Cross-attention, float16", "xattn_fp16"), ("Cross-attention, float32", "xattn_fp32")]
    lines = [r"\begin{table}[h]",
             r"\caption{Cost of screening the " + f"{L0:,}".replace(",", "{,}") + r" ligands of DOCKSTRING at the full scale on one " + GPU + r" (seconds, best of two runs for the GPU stages). "
             r"Model time: ligand embedding, pocket states, scoring and selection of the best 10\% per pocket. End to end: model time plus the featurization of the SMILES on " + str(T["environment"]["featurization_processes"]) +
             r" CPU processes (" + fs(T["featurization"]["seconds"]) + r" s) and the collation of the batches. 1{,}000 pockets are the 57 pockets of DOCKSTRING used cyclically. "
             r"Mpairs/s: million pairs per second of the scoring and selection stages for 57 pockets. Uni-Dock: library size times the number of pockets divided by the median docking rate at 4{,}000 ligands per call on one RTX 4090, without ligand preparation.}",
             r"\label{tab:si_cost_full}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{57 pockets} & \multicolumn{2}{c}{1{,}000 pockets} & Scoring rate \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Network & Model & End to end & Model & End to end & Mpairs/s \\", r"\midrule"]
    for nm, k in rows:
        a, b = B[k]["57"], B[k]["1000"]
        lines.append(f"{nm} & {fs(a['model_time_s'])} & {fs(a['end_to_end_s'])} & {fs(b['model_time_s'])} & {fs(b['end_to_end_s'])} & {fs(a['scoring_pairs_per_s'] / 1e6)} " + r"\\")
    for nm, key in (("Uni-Dock, fast mode", "fast"), ("Uni-Dock, detail mode", "detail")):
        r = CM["unidock_rate"][key]
        lines.append(f"{nm} & {fs(L0 * 57 / r)} & {fs(L0 * 57 / r)} & {fs(L0 * 1000 / r)} & {fs(L0 * 1000 / r)} & $" + f"{r / 1e6 * 1e5:.1f}" + r"\times10^{-5}$ " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (here / "si_table_cost_full.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_known():
    K = EV / "v5_known_pockets_summary.json"
    if not K.exists():
        return
    s = json.loads(K.read_text())
    n = s["n_targets"]
    rows = [("Dual encoder", "C0", "dual"), ("Pooled concatenation", "PC", "pooled"), ("Cross-attention", "XA", "xattn_fp16"), ("Constant pocket (dual encoder)", "C0c", "dual")]
    cols = [("recall_1@1", "1\\%"), ("recall_1@5", "5\\%"), ("recall_1@10", "10\\%"), ("recall_1@20", "20\\%")]
    lines = [r"\begin{table}[t]",
             r"\caption{Top-1\% recall of the networks on the " + str(n) + r" known pockets for 10{,}000 ligands that were not used in training (macro mean over the pockets of the mean over three seeds; 95\% interval from bootstrapping the pockets), "
             r"within a budget of the best 1, 5, 10 and 20\% of the ranking. The constant-pocket network is the dual encoder trained with one dummy pocket for every target. "
             r"Model time: seconds for the library of " + f"{L0:,}".replace(",", "{,}") + r" ligands against 57 pockets on one " + GPU + r" (cross-attention in ragged float16).}",
             r"\label{tab:known}", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccccr}", r"\toprule",
             r"Network & " + " & ".join(f"Recall, {c[1]} budget" for c in cols) + r" & Model time (s) \\", r"\midrule"]
    for nm, c, net in rows:
        cells = []
        for m, _ in cols:
            v, ci = s["mean"][c][m], s["ci"][c][m]
            cells.append(f"{v:.2f} ({ci[0]:.2f} to {ci[1]:.2f})")
        lines.append(f"{nm} & " + " & ".join(cells) + f" & {fs(B[net]['57']['model_time_s'])} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (here / "table_known.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # per-target table for the SI
    pt = {c: s["mean"][c]["per_target"]["recall_1@10"] for c in ("C0", "PC", "XA", "C0c")}
    lines = [r"\begin{longtable}{lrrrr}", r"\caption{Top-1\% recall at the 10\% budget of every known pocket (mean over three seeds; 10{,}000 ligands that were not used in training).}\label{tab:si_known_targets}\\",
             r"\toprule", r"Pocket & Dual encoder & Pooled concatenation & Cross-attention & Constant pocket \\", r"\midrule", r"\endfirsthead", r"\toprule",
             r"Pocket & Dual encoder & Pooled concatenation & Cross-attention & Constant pocket \\", r"\midrule", r"\endhead"]
    for t in sorted(pt["C0"]):
        lines.append(f"{t} & {pt['C0'][t]:.2f} & {pt['PC'][t]:.2f} & {pt['XA'][t]:.2f} & {pt['C0c'][t]:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    (here / "si_table_known_targets.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_timing_si(Tx=None, outname="si_table_timing.tex", label="tab:si_timing"):
    Tx = Tx or T
    B = Tx["blocks"]
    GPU = Tx["environment"]["gpu"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
    nets = [("Dual encoder", "dual"), ("Pooled concatenation", "pooled"), ("Cross-attention, float16", "xattn_fp16"), ("Cross-attention, float32", "xattn_fp32")]
    lines = [r"\begin{table}[h]", r"\caption{Stages of the full-scale timing on one " + GPU + r" (seconds). Featurization and collation of the ligands on the CPU are " + fs(Tx["featurization"]["seconds"]) +
             r" s and " + fs(next(iter(B["dual"].values()))["stages_s"]["collate_cpu"]) + r" s for every network.}", r"\label{" + label + "}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Network & Pockets & Embedding & Pocket states & Scoring & Selection & Model time \\", r"\midrule"]
    for nm, k in nets:
        for P in ("1", "5", "20", "57", "1000"):
            b = B[k][P]
            st = b["stages_s"]
            lines.append(f"{nm} & {P} & {fs(st['embed_gpu'])} & {fs(st['pocket_states_gpu'])} & {fs(st['score_gpu'])} & {fs(st['topk_gpu'])} & {fs(b['model_time_s'])} " + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (here / outname).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    table_cost()
    table_cost_full()
    import make_tables_known
    make_tables_known.main()
    table_timing_si()
    table_timing_si(json.loads((EV / "v4_fair_timing_L40S.json").read_text()), "si_table_timing_b.tex", "tab:si_timing_b")
    print("tables written, timing from", f.name, GPU)
