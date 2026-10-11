"""Known-pocket tables: table_known.tex (main text, Table 1), si_table_recall_all.tex (SI, both checkpoint choices side by side) and the per-pocket SI table of the
49 known pockets (family, role, held-out fold; si_table_known_cv_targets.tex).

Existing networks, checkpoint chosen on new targets: 39 training pockets, three seeds (eval/v5_known_pockets_summary.json).
Cross-fitted networks, checkpoint chosen on known pockets: 49 pockets (eval/v5_known_cv_summary.json).
Model times come from eval/v4_fair_timing.json (RTX 4090 pod).
Usage: python manuscript/make_tables_known.py
"""
from __future__ import annotations

import json

import paths

T = paths.load_timing(paths.TIMING_PRIMARY)
B = T["blocks"]
GPU = paths.gpu_name(T)
COLS = ["recall_1@1", "recall_1@5", "recall_1@10", "recall_1@20"]


def fs(x):
    return f"{x:,.0f}".replace(",", "{,}") if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")


def row(name, s, key, net):
    v = [s["mean"][key][m] for m in COLS]
    ci = s["ci"][key]["recall_1@10"]
    return f"\\quad {name} & {v[0]:.2f} & {v[1]:.2f} & {v[2]:.2f} ({ci[0]:.2f} to {ci[1]:.2f}) & {v[3]:.2f} & {fs(B[net]['57']['model_time_s'])} " + r"\\"


def main_table(s_cv):
    """Table of the main text: the cross-fitted networks (checkpoint chosen on known pockets), the dual encoder against the cross-attention network."""
    rows = (("Dual encoder", "dual_R0", "dual"), ("Cross-attention", "xattn_R0", "xattn_fp16"), ("Constant pocket", "dual_const_R0", "dual"),
            ("Pooled concatenation (Jev-like, learned readout)", "pooled_R0", "pooled"))
    cap = (r"\caption{Top-1\% recall of the networks on the " + str(s_cv["n_targets"]["dual_R0"]) + r" known pockets for ligands that were not used in training, within a budget of the best 1, 5, 10 and 20\% of the ranking "
           r"(macro mean over the pockets; 95\% interval from bootstrapping the pockets). Each network is the mean over the cross-fitted folds that train on a pocket, with the checkpoint chosen on known pockets. "
           r"The constant-pocket network is the dual encoder trained with one dummy pocket for every target. The pooled network replaces the product of the dual encoder by a learned readout. "
           r"Time: seconds from SMILES to scores for the " + f"{T['library_size']:,}".replace(",", "{,}") + r" ligands of DOCKSTRING against 1{,}000 pockets on one " + GPU + r" (cross-attention in float16). "
           r"Percentages and differences in the text are computed from unrounded values. The recall of every network with the checkpoint chosen on new targets and with longer training is in the Supporting Information.}")
    lines = [r"\begin{table}[t]", cap, r"\label{tab:known}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccrr}", r"\toprule",
             r"Network & 1\% & 5\% & 10\% (95\% interval) & 20\% & Time (s) \\", r"\midrule"]
    for nm, k, net in rows:
        v = [s_cv["mean"][k][m] for m in COLS]
        ci = s_cv["ci"][k]["recall_1@10"]
        lines.append(f"{nm} & {v[0]:.2f} & {v[1]:.2f} & {v[2]:.2f} ({ci[0]:.2f} to {ci[1]:.2f}) & {v[3]:.2f} & {fs(B[net]['1000']['end_to_end_s'])} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (paths.MS / "table_known.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    s_new = json.loads(paths.KNOWN_POCKETS_SUMMARY.read_text())      # existing networks, checkpoint chosen on new targets
    s_cv = json.loads(paths.KNOWN_CV_SUMMARY.read_text())            # cross-fitted networks, checkpoint chosen on known pockets
    cap = (r"\caption{Top-1\% recall of the networks on known pockets for ligands that were not used in training, within a budget of the best 1, 5, 10 and 20\% of the ranking "
           r"(macro mean over the pockets; 95\% interval from bootstrapping the pockets). ")
    cap += (r"Checkpoint chosen on known pockets: networks of five cross-fitted folds that stop on seen targets, evaluated on the pockets of their training targets, " +
            str(s_cv["n_targets"]["dual_R0"]) + r" pockets, mean over the folds that train on a pocket. ")
    cap += (r"Checkpoint chosen on new targets: the existing networks, the " + str(s_new["n_targets"]) + r" training pockets, mean over three seeds. ")
    cap += (r"The constant-pocket network is trained with one dummy pocket for every target. Model time: seconds for the " + f"{T['library_size']:,}".replace(",", "{,}") +
            r" ligands of DOCKSTRING against 57 pockets on one " + GPU + r" (cross-attention in float16).}")
    lines = [r"\begin{table}[h]", cap, r"\label{tab:si_recall_all}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccrr}", r"\toprule",
             r"Network & 1\% & 5\% & 10\% (95\% interval) & 20\% & Model time (s) \\", r"\midrule"]
    lines.append(r"\multicolumn{6}{l}{\emph{Checkpoint chosen on known pockets}} \\")
    for nm, k, net in (("Dual encoder", "dual_R0", "dual"), ("Pooled concatenation", "pooled_R0", "pooled"), ("Cross-attention", "xattn_R0", "xattn_fp16"),
                       ("Dual encoder, longer training", "dual_R1", "dual"), ("Cross-attention, longer training", "xattn_R1", "xattn_fp16"), ("Constant pocket", "dual_const_R0", "dual")):
        if k in s_cv["mean"]:
            lines.append(row(nm, s_cv, k, net))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\emph{Checkpoint chosen on new targets}} \\")
    for nm, k, net in (("Dual encoder", "C0", "dual"), ("Pooled concatenation", "PC", "pooled"), ("Cross-attention", "XA", "xattn_fp16"), ("Constant pocket", "C0c", "dual")):
        lines.append(row(nm, s_new, k, net))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (paths.MS / "si_table_recall_all.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    main_table(s_cv)

    c_cv = ("dual_R0", "pooled_R0", "xattn_R0", "dual_const_R0")
    pt = {c: s_cv["per_target"][c]["recall_1@10"] for c in c_cv if c in s_cv["per_target"]}
    import pandas as pd
    tg = pd.read_parquet(paths.TARGETS_PARQUET).set_index("target_id")
    fa = json.loads(paths.FOLDS_JSON.read_text(encoding="utf-8"))["fold_assignment"]
    held = {t: int(k) for k, v in fa.items() for t in v["heldout"]}
    role = {"train": "training", "dev": "development"}
    head = r"Pocket & Family & Role & Fold & Dual & Pooled & Cross-att. & Constant \\"
    out = [r"\begin{longtable}{llllrrrr}", r"\caption{Top-1\% recall at the 10\% budget of every known pocket for the cross-fitted networks (checkpoint chosen on known pockets; mean over the folds that train on the pocket). "
           r"Dual: dual encoder; Pooled: pooled concatenation; Cross-att.: cross-attention; Constant: dual encoder with one constant pocket. "
           r"Role is the role of the target in the split of the ligands and targets. Fold is the index (0 to 4) of the cross-fitted fold that holds the target out.}\label{tab:si_known_cv_targets}\\",
           r"\toprule", head, r"\midrule", r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead"]
    for t in sorted(pt["dual_R0"]):
        fam = tg.loc[t, "family"].replace("_", " ")
        out.append(f"{t} & {fam} & {role[tg.loc[t, 'split_role']]} & {held[t]} & " + " & ".join(f"{pt[c].get(t, float('nan')):.2f}" for c in c_cv) + r" \\")
    out += [r"\bottomrule", r"\end{longtable}"]
    (paths.MS / "si_table_known_cv_targets.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("known tables written (checkpoint chosen on new targets and on known pockets)")


if __name__ == "__main__":
    main()
