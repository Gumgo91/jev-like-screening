"""Known-pocket tables: table_known.tex (main text; experiments E3 and E4) and si_table_known_targets.tex (per pocket, SI).

E3: existing checkpoints chosen on new targets, 39 training pockets, three seeds (dockmut/eval/v5_known_pockets_summary.json).
E4: cross-fitted networks with checkpoints chosen on seen pockets, 49 pockets (dockmut/eval/v5_known_cv_summary.json), when present.
Usage: python make_tables_known.py
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
B = T["blocks"]
GPU = T["environment"]["gpu"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
COLS = ["recall_1@1", "recall_1@5", "recall_1@10", "recall_1@20"]


def fs(x):
    return f"{x:,.0f}".replace(",", "{,}") if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")


def row(name, s, key, net):
    v = [s["mean"][key][m] for m in COLS]
    ci = s["ci"][key]["recall_1@10"]
    return f"\\quad {name} & {v[0]:.2f} & {v[1]:.2f} & {v[2]:.2f} ({ci[0]:.2f} to {ci[1]:.2f}) & {v[3]:.2f} & {fs(B[net]['57']['model_time_s'])} " + r"\\"


def main():
    s3 = json.loads((EV / "v5_known_pockets_summary.json").read_text())
    f4 = EV / "v5_known_cv_summary.json"
    s4 = json.loads(f4.read_text()) if f4.exists() else None
    cap = (r"\caption{Top-1\% recall of the networks on known pockets for ligands that were not used in training, within a budget of the best 1, 5, 10 and 20\% of the ranking "
           r"(macro mean over the pockets; 95\% interval from bootstrapping the pockets). ")
    if s4:
        cap += (r"Checkpoint chosen on known pockets: networks of five cross-fitted folds that stop on seen targets, evaluated on the pockets of their training targets, " +
                str(s4["n_targets"]["dual_R0"]) + r" pockets, mean over the folds that train on a pocket. ")
    cap += (r"Checkpoint chosen on new targets: the existing networks of the preceding analysis, the " + str(s3["n_targets"]) + r" training pockets, mean over three seeds. ")
    cap += (r"The constant-pocket network is trained with one dummy pocket for every target. Model time: seconds for the " + f"{T['library_size']:,}".replace(",", "{,}") +
            r" ligands of DOCKSTRING against 57 pockets on one " + GPU + r" (cross-attention in float16).}")
    lines = [r"\begin{table}[t]", cap, r"\label{tab:known}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccrr}", r"\toprule",
             r"Network & 1\% & 5\% & 10\% (95\% interval) & 20\% & Model time (s) \\", r"\midrule"]
    if s4:
        lines.append(r"\multicolumn{6}{l}{\emph{Checkpoint chosen on known pockets}} \\")
        for nm, k, net in (("Dual encoder", "dual_R0", "dual"), ("Pooled concatenation", "pooled_R0", "pooled"), ("Cross-attention", "xattn_R0", "xattn_fp16"),
                           ("Dual encoder, longer training", "dual_R1", "dual"), ("Cross-attention, longer training", "xattn_R1", "xattn_fp16"), ("Constant pocket", "dual_const_R0", "dual")):
            if k in s4["mean"]:
                lines.append(row(nm, s4, k, net))
        lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\emph{Checkpoint chosen on new targets}} \\")
    for nm, k, net in (("Dual encoder", "C0", "dual"), ("Pooled concatenation", "PC", "pooled"), ("Cross-attention", "XA", "xattn_fp16"), ("Constant pocket", "C0c", "dual")):
        lines.append(row(nm, s3, k, net))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (here / "table_known.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pt = {c: s3["mean"][c]["per_target"]["recall_1@10"] for c in ("C0", "PC", "XA", "C0c")}
    out = [r"\begin{longtable}{lrrrr}", r"\caption{Top-1\% recall at the 10\% budget of every known pocket (checkpoint chosen on new targets; mean over three seeds; 10{,}000 ligands that were not used in training).}\label{tab:si_known_targets}\\",
           r"\toprule", r"Pocket & Dual encoder & Pooled concatenation & Cross-attention & Constant pocket \\", r"\midrule", r"\endfirsthead", r"\toprule",
           r"Pocket & Dual encoder & Pooled concatenation & Cross-attention & Constant pocket \\", r"\midrule", r"\endhead"]
    for t in sorted(pt["C0"]):
        out.append(f"{t} & {pt['C0'][t]:.2f} & {pt['PC'][t]:.2f} & {pt['XA'][t]:.2f} & {pt['C0c'][t]:.2f} " + r"\\")
    out += [r"\bottomrule", r"\end{longtable}"]
    (here / "si_table_known_targets.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    if s4:
        c4 = ("dual_R0", "pooled_R0", "xattn_R0", "dual_const_R0")
        pt4 = {c: s4["per_target"][c]["recall_1@10"] for c in c4 if c in s4["per_target"]}
        out = [r"\begin{longtable}{lrrrr}", r"\caption{Top-1\% recall at the 10\% budget of every known pocket for the cross-fitted networks (checkpoint chosen on known pockets; mean over the folds that train on the pocket).}\label{tab:si_known_cv_targets}\\",
               r"\toprule", r"Pocket & Dual encoder & Pooled concatenation & Cross-attention & Constant pocket \\", r"\midrule", r"\endfirsthead", r"\toprule",
               r"Pocket & Dual encoder & Pooled concatenation & Cross-attention & Constant pocket \\", r"\midrule", r"\endhead"]
        for t in sorted(pt4["dual_R0"]):
            out.append(f"{t} & " + " & ".join(f"{pt4[c].get(t, float('nan')):.2f}" for c in c4) + r" \\")
        out += [r"\bottomrule", r"\end{longtable}"]
        (here / "si_table_known_cv_targets.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("known tables written", "(E3 and E4)" if s4 else "(E3 only)")


if __name__ == "__main__":
    main()
