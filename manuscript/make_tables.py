"""LaTeX tables generated from analysis outputs (no number is typed by hand).

table1 : composition of DockMut-1 (per_target.csv and per_mutant.csv of analyze_dockmut.py)
table2 : held-out benchmark by condition (summary_all.json and paired_all.json)
table3 : wild-type screening and reversal accuracy on the development and test targets
table_levels (SI): three levels of faithfulness for every condition
table_scale (SI): scaling study and within-family transfer
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from conds import LABEL  # noqa: E402

ANA = ROOT / "dockmut/analysis"
EV = ROOT / "dockmut/eval"
OUT = Path(__file__).parent

GROUPS = [
    ("Baseline surrogates, regression head", ["C0", "C1", "C2"]),
    ("Baseline surrogates, hit-probability head", ["C0L", "C1L", "C2L"]),
    ("Descriptor models", ["size_only", "ridge_descriptors", "gbm_descriptors", "ridge_descriptors_geo", "gbm_descriptors_geo"]),
    ("Descriptor model with wild-type pose (post hoc)", ["gbm_pose", "gbm_pose_geo"]),
    ("Wild-type training only", ["b5_wt", "rs_wt", "rse_wt", "rsg_wt"]),
    ("Interventional training", ["b5_int", "rs_int", "rse_int", "rsg_int"]),
    ("Network with a hit head (addendum 7, post hoc)", ["rsgh_wt", "rsgh_int"]),
    ("Label controls for the geometry network", ["rsg_int_shl", "rsg_int_shm", "rsg_int_sha", "rsg_int_shm2", "rsg_int_sha2"]),
]
NAMES = {
    "C0": "Plain objective", "C1": "Within-target ranking", "C2": "Cross-target ranking",
    "C0L": "Plain objective", "C1L": "Within-target ranking", "C2L": "Cross-target ranking",
    "size_only": "Ligand size and removed atoms", "ridge_descriptors": "Ridge, ligand and edit descriptors",
    "gbm_descriptors": "Boosting, ligand and edit descriptors", "ridge_descriptors_geo": "Ridge with pocket geometry",
    "gbm_descriptors_geo": "Boosting with pocket geometry", "gbm_pose": "Boosting with pose contacts",
    "gbm_pose_geo": "Boosting with pose contacts and geometry",
    "b5_wt": "Dual encoder", "rs_wt": "ResidueSum", "rse_wt": "ResidueSum-Edit", "rsg_wt": "ResidueSum-Geo",
    "b5_int": "Dual encoder", "rs_int": "ResidueSum", "rse_int": "ResidueSum-Edit", "rsg_int": "ResidueSum-Geo",
    "rsgh_wt": "ResidueSum-Geo with hit head, wild type only", "rsgh_int": "ResidueSum-Geo with hit head, measured changes",
    "rsg_int_shl": "Labels permuted over ligands", "rsg_int_shm": "Labels permuted over edits (v1)",
    "rsg_int_sha": "All labels permuted (v1)", "rsg_int_shm2": "Labels deranged over edits (v2)",
    "rsg_int_sha2": "All labels deranged (v2)", "C0c": "Plain objective, constant pocket", "rsg_const": "Geometry network, constant pocket",
}


def num(x, d=2):
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "$-$")


def fmt_int(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def table1():
    T = pd.read_csv(ANA / "per_target.csv")
    M = pd.read_csv(ANA / "per_mutant.csv")
    rows = []
    for name, sel in (("Training", T.role == "train"), ("Development", T.role == "dev"), ("Test", T.role == "test")):
        t = T[sel]
        m = M[M.target.isin(t.target)]
        cls = m["class"].value_counts()
        lig = int(t.n_ligands.iloc[0]) if t.n_ligands.nunique() == 1 else f"{int(t.n_ligands.min())} to {int(t.n_ligands.max())}"
        rows.append((name, len(t), int(cls.get("single contact", 0)), int(cls.get("multi contact", 0)),
                     int(cls.get("shell control", 0) + cls.get("far control", 0)), lig, int(t.n_dockings_ok.sum())))
    tot = ("All", len(T), int((M["class"] == "single contact").sum()), int((M["class"] == "multi contact").sum()),
           int(M["class"].isin(["shell control", "far control"]).sum()), "", int(T.n_dockings_ok.sum()))
    lines = [r"\begin{table}[t]",
             r"\caption{Composition of DockMut-1. Edited receptors are those with at least 20 valid ligands. Dockings are the scores obtained, wild-type replicates included.}",
             r"\label{tab:data}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Targets & n & Single contact & Multi contact & Controls & Ligands & Dockings \\", r"\midrule"]
    for r in rows + [tot]:
        lines.append(" & ".join([str(r[0]), str(r[1]), fmt_int(r[2]), fmt_int(r[3]), fmt_int(r[4]), str(r[5]), fmt_int(r[6])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table1.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("table1 written")


def table2():
    S = json.loads((EV / "summary_all.json").read_text())
    P = json.loads((EV / "paired_all.json").read_text())["conditions"]
    cap = (r"\caption{Prediction of measured score changes on the 19 held-out targets. $r$ is the Pearson correlation over all ligand-edit pairs of "
           r"contact truncations for the seed ensemble, with the 95\% interval from bootstrapping targets. The next column gives mean and SD of $r$ over seeds (number of seeds in parentheses). "
           r"Edit mean $r$ is the correlation of the receptor-mean changes and within $r$ the mean correlation within a receptor, both pooled over the receptors of all 19 targets (Figure~\ref{fig:levels} and Table~\ref{tab:si_levels} give the means of the per-target correlations), and AUC the separation of contact truncations from shell and far controls. "
           r"SD is the standard deviation of the predicted change in \kcal{}, against 0.71 for the measured change. v1 marks the first implementation of a permutation control and v2 its correction.}")
    lines = [r"\begin{table}[t]", cap, r"\label{tab:main}", r"\centering", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Model & $r$ (ensemble) & 95\% interval & $r$ over seeds & Edit mean $r$ & Within $r$ & SD & AUC \\", r"\midrule"]
    first = True
    for gname, keys in GROUPS:
        keys = [k for k in keys if k in S]
        if not keys:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(r"\multicolumn{8}{l}{\emph{" + gname + r"}} \\")
        for k in keys:
            e = S[k]["ensemble"]
            c = P[k]
            seeds = f"{num(S[k]['pooled_r_mean'])} $\\pm$ {S[k]['pooled_r_sd']:.2f} ({S[k]['n_seeds']})" if S[k]["n_seeds"] > 1 else "single fit"
            lines.append(f"\\quad {NAMES[k]} & {num(c['r'])} & {num(c['ci'][0])} to {num(c['ci'][1])} & {seeds} & {num(e['mutant_r'])} & "
                         f"{num(e['within_r'])} & {e['pred_sd']:.2f} & {e['detect_auc']:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table2.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("table2 written")


def table3():
    W = {}
    for sp in ("dev", "test"):
        res = json.loads((EV / f"wt_{sp}_all.json").read_text())
        sm = json.loads((EV / f"wt_{sp}_summary.json").read_text())
        W[sp] = (res, sm)
    keys = ["C0", "C1", "C2", "C0c", "b5_wt", "rs_wt", "rse_wt", "rsg_wt", "b5_int", "rs_int", "rse_int", "rsg_int",
            "rsg_int_shl", "rsg_int_shm2", "rsg_int_sha2"]
    cap = (r"\caption{Wild-type screening and cross-target ordering. Top-1\% recall within the best 10\% of the ranking (macro average over targets; 95\% interval from bootstrapping targets) "
           r"and reversal accuracy, the fraction of quadruplets of two ligands and two targets in which both opposite orderings are reproduced (mean over seeds with range). "
           r"The baseline surrogates have two readouts. The hit-probability head ranked ligands in the preceding analysis (values taken from its result files), the regression head gives the predicted score that is used for the edits in this study. "
           r"The development split holds 10 targets and 10{,}025 quadruplets, the test split 9 targets and 3{,}536 quadruplets. Recall of the hit-head networks is the mean of single networks, and the test labels were read once for all models except these networks and once more for them (a third read served the six networks of the comparison with joint networks, Table~\ref{tab:si_joint_runs}).}")
    lines = [r"\begin{table}[t]", cap, r"\label{tab:wt}", r"\centering", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{Development targets} & \multicolumn{3}{c}{Test targets} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Model & Recall & 95\% interval & Reversal & Recall & 95\% interval & Reversal \\", r"\midrule"]
    H = json.loads((EV / "wt_hit_summary.json").read_text())
    HN = json.loads((EV / "wt_hitnet_summary.json").read_text())
    NAMES.update({"N:rsgh_wt": "ResidueSum-Geo with hit head, wild type only", "N:rsgh_int": "ResidueSum-Geo with hit head, measured changes"})
    NAMES.update({"H:C0": "Plain objective", "H:C1": "Within-target ranking", "H:C2": "Cross-target ranking", "H:C0c": "Plain objective, constant pocket",
                  "H:CB": "Ligand-only fingerprint model (CatBoost)"})
    groups = [("Baseline surrogates, hit-probability head (readout of the preceding analysis)", ["H:C0", "H:C1", "H:C2", "H:C0c", "H:CB"]),
              ("Baseline surrogates, regression head (readout used for edits)", ["C0", "C1", "C2", "C0c"]),
              ("Wild-type training only", ["b5_wt", "rs_wt", "rse_wt", "rsg_wt"]),
              ("Interventional training", ["b5_int", "rs_int", "rse_int", "rsg_int"]),
              ("Network with a hit head, hit-logit readout (addendum 7)", ["N:rsgh_wt", "N:rsgh_int"]),
              ("Label controls", ["rsg_int_shl", "rsg_int_shm2", "rsg_int_sha2"])]
    first = True
    for gname, ks in groups:
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(r"\multicolumn{7}{l}{\emph{" + gname + r"}} \\")
        for k in ks:
            cells = []
            for sp in ("dev", "test"):
                if k.startswith("H:"):
                    c = H[sp]["conditions"][k[2:]]
                    rv = c["reversal"]
                elif k.startswith("N:"):
                    c = HN[sp]["conditions"][k[2:]]
                    rv = c["reversal"]
                else:
                    res, sm = W[sp]
                    c = sm["conditions"][k]
                    rv = sm["reversal"][k]
                cells += [num(c["recall_1@10"]), f"{num(c['ci'][0])} to {num(c['ci'][1])}", f"{num(rv['mean'])} ({num(rv['min'])} to {num(rv['max'])})"]
            lines.append(f"\\quad {NAMES[k]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table3.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("table3 written")


def table_levels():
    L = json.loads((EV / "levels_all.json").read_text())
    lines = [r"\begin{longtable}{p{4.6cm}rrrr}",
             r"\caption{Three levels of faithfulness on the 19 held-out targets (mean over targets with 95\% interval from bootstrapping targets). Within-target $r$ is computed over all contact pairs of one target, "
             r"edit-mean $r$ over the receptor means of its contact edits, and ligand-specific $r$ within a receptor after removing the receptor mean. SD is the standard deviation of the predicted change in \kcal{}.}",
             r"\label{tab:si_levels}\\", r"\toprule", r"Model & Within target & Edit means & Ligand-specific & SD \\", r"\midrule", r"\endfirsthead",
             r"Model & Within target & Edit means & Ligand-specific & SD \\", r"\midrule", r"\endhead"]
    for gname, keys in GROUPS:
        keys = [k for k in keys if k in L]
        if not keys:
            continue
        lines.append(r"\multicolumn{5}{l}{\emph{" + gname + r"}} \\")
        for k in keys:
            v = L[k]
            f = lambda key: f"{num(v[key])} [{num(v[key + '_ci'][0])}, {num(v[key + '_ci'][1])}]"
            lines.append(f"\\quad {NAMES[k]} & {f('within_target_r')} & {f('edit_mean_r')} & {f('ligand_r')} & {v['sd_pred']:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    (OUT / "si_table_levels.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("si_table_levels written")


def table_scale():
    Z = json.loads((EV / "scale_summary.json").read_text())
    K = json.loads((EV / "kin_summary.json").read_text())
    D = json.loads((EV / "descriptor_baselines_kin.json").read_text())
    lines = [r"\begin{table}[h]",
             r"\caption{Scaling of the geometry network trained on measured changes and within-family transfer. Pooled $r$ on the 19 held-out targets for two seeds (mean), and the same for the four withheld kinases (seeds in parentheses).}",
             r"\label{tab:si_scale}", r"\begin{tabular}{lrrr}", r"\toprule", r"Setting & Pooled $r$ & Ensemble $r$ & Edit mean $r$ \\", r"\midrule"]
    names = {"nt4": "4 training targets", "nt8": "8 training targets", "nt16": "16 training targets", "nt24": "24 training targets",
             "mk2": "2 edits per target", "mk4": "4 edits per target", "mk8": "8 edits per target", "mk12": "12 edits per target", "full": "38 targets, all edits"}
    for k in ("nt4", "nt8", "nt16", "nt24", "mk2", "mk4", "mk8", "mk12", "full"):
        z = Z[k]
        lines.append(f"{names[k]} & {num(z['pooled_r_mean'])} & {num(z['ensemble_r'])} & {num(z['mutant_r_mean'])} " + r"\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\emph{Four withheld kinases}} \\")
    kn = {"rsg_int": "Geometry network, measured changes", "rsg_wt": "Geometry network, wild type only", "rsg_int_shm2": "Labels deranged over edits",
          "rsg_int_sha2": "All labels deranged"}
    for k, nm in kn.items():
        z = K[k]
        lines.append(f"{nm} ({z['n']}) & {num(z['pooled_r_mean'])} & -- & {num(z['mutant_r_mean'])} " + r"\\")
    for k, nm in (("gbm_descriptors_geo", "Boosting with pocket geometry"), ("gbm_descriptors", "Boosting without pocket geometry"), ("size_only", "Ligand size and removed atoms")):
        lines.append(f"{nm} & {num(D[k]['pooled_r'])} & -- & {num(D[k]['mutant_r'])} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "si_table_scale.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("si_table_scale written")


def boot_ci(v, n=5000):
    import numpy as np
    rng = np.random.default_rng(0)
    v = np.asarray(v, dtype=float)
    b = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def table_speed():
    """Docking rate of Uni-Dock by search mode and batch size, and the cost of the surrogate ensembles for 260,155 ligands x 57 pockets."""
    dt = pd.concat([pd.read_csv(f) for f in (EV / "docking_timing.csv", EV / "docking_timing_big.csv", EV / "docking_timing_4k.csv") if f.exists()])
    dt = dt[dt.warmup == 0].copy()
    dt["rate"] = dt.n_ok / dt.sec
    thr = {fam: json.loads((EV / "throughput" / f"thr_{fam}_run2.json").read_text()) for fam in ("c0", "rsg", "hit")}
    pairs = thr["rsg"]["pockets"]["pairs"]
    lib = thr["rsg"]["library"]["featurised"]
    lines = [r"\begin{table*}[t]",
             r"\caption{Cost of scoring " + f"{lib:,}".replace(",", "{,}") + r" ligands against 57 pockets (" + f"{pairs / 1e6:.1f}" + r" million pairs) on one RTX 4090. "
             r"Uni-Dock 1.2.0 was timed over " + str(len(dt)) + r" calls (median of the ligands docked per second with the 10th to 90th percentile in brackets; three receptors; the first call of each job and receptor was discarded, see Methods), and the time of docking is the number of pairs divided by the highest median rate. "
             r"The networks were timed as ensembles of three from the ligand SMILES to the scores: featurization of the ligands on 32 CPU processes, one ligand embedding per ensemble member and 57 pocket states (second of two identical runs).}",
             r"\label{tab:speed}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lcccr}", r"\toprule",
             r"Docking engine & " + r"\multicolumn{3}{c}{Ligands per second} & Time for the pairs " + r"\\", r"\cmidrule(lr){2-4}", r" & 160 to 256 per call & 1{,}000 per call & 4{,}000 per call & " + r"\\", r"\midrule"]
    for mode in ("detail", "balance", "fast"):
        cells, med = [], {}
        for n, grp in ((256, (160, 256)), (1000, (1000,)), (4000, (4000,))):
            g = dt[(dt["mode"] == mode) & dt.n.isin(grp)].rate
            if len(g) == 0:
                cells.append("not timed")
                continue
            med[n] = float(g.median())
            cells.append(f"{g.median():.1f} ({g.quantile(0.1):.1f} to {g.quantile(0.9):.1f})")
        d = pairs / max(med.values()) / 86400
        lines.append(f"Uni-Dock, {mode} mode & {cells[0]} & {cells[1]} & {cells[2]} & {d:.1f} GPU-days " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\medskip", "",
              r"\begin{tabular}{lccccr}", r"\toprule",
              r"Ensemble of three networks & Rate, $10^6$ pairs/s & Featurization (s) & Embedding (s) & Scoring (s) & Total (s) " + r"\\",
              r"\midrule"]
    for fam, nm in (("c0", "Baseline dual encoders"), ("rsg", "Geometry networks"), ("hit", "Hit-head networks")):
        j = thr[fam]
        pk, lb = j["pockets"], j["library"]
        emb = lb["sec_embed_gpu_all_models"] + pk["sec_state_encoding"]
        tot = lb["sec_featurise_cpu"] + emb + pk["sec_scoring_all_models"]
        lines.append(f"{nm} & {pk['per_model_pairs_per_sec'] / 1e6:.1f} & {lb['sec_featurise_cpu']:.0f} & {emb:.1f} & {pk['sec_scoring_all_models']:.1f} & {tot:.0f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "table_speed.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("table_speed written")


def table_budget():
    """Top-1% recall at several selection budgets on the test and development targets (wt_budget_summary.json)."""
    B = json.loads((EV / "wt_budget_summary.json").read_text())
    names = [("Hit-head networks (hit logit, no docked sample)", [("rsgh_wt", "Wild type only"), ("rsgh_int", "Measured changes")]),
             ("Baseline dual encoders (hit-probability head)", [("C0", "Plain objective"), ("C0c", "Constant dummy pocket")]),
             ("Pooled fingerprint model without protein input", [("CB", "Ligand-only boosting")]),
             ("Geometry networks (regression head)", [("rsg_wt", "Wild type only"), ("rsg_int", "Measured changes")])]
    lines = [r"\begin{table*}[t]",
             r"\caption{Top-1\% recall at four selection budgets on the nine test targets (macro mean over targets; 95\% interval over targets for the 10\% budget), enrichment factor of the best 1\% of the ranking (maximum 100) and the fraction of the library that recovers 95\% of the top-1\% molecules. "
             r"The last column gives the recall at the 10\% budget on the ten development targets. Random selection recovers a fraction of the molecules that equals the budget. All values are means over seeds per target. Wild type only: trained on DOCKSTRING scores alone; measured changes: trained with the receptor edits of DockMut-1 in addition.}",
             r"\label{tab:budget}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Model & 1\% & 5\% & 10\% & 20\% & EF 1\% & 95\% recall & Dev. 10\% \\", r"\midrule"]
    for gname, items in names:
        lines.append(r"\multicolumn{8}{l}{\emph{" + gname + r"}} \\")
        for key, nm in items:
            t, d = B["test"][key], B["dev"][key]
            ten = t["recall_1@10"]
            lines.append(f"\\quad {nm} & {t['recall_1@1']['mean']:.2f} & {t['recall_1@5']['mean']:.2f} & {ten['mean']:.2f} ({ten['ci'][0]:.2f} to {ten['ci'][1]:.2f}) & {t['recall_1@20']['mean']:.2f} & "
                         f"{t['ef1@1']['mean']:.0f} & {100 * t['recall95_budget']['mean']:.0f}\\% & {d['recall_1@10']['mean']:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "table_budget.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("table_budget written")


def table_ml():
    """Zero-shot networks against an ML-guided docking workflow on the development targets (ml_guided_dev.json)."""
    import numpy as np
    M = json.loads((EV / "ml_guided_dev.json").read_text())
    lines = [r"\begin{table}[t]",
             r"\caption{Zero-shot networks against a target-specific ML-guided docking workflow on the ten development targets (library of " + f"{M['n_library']:,}".replace(",", "{,}") + r" ligands per target; macro mean top-1\% recall with a 95\% interval over targets). "
             r"The ML-guided workflow docks a random sample, fits a boosting model on Morgan fingerprints to the sample scores, and docks the top-ranked remaining ligands until the total docked fraction, the sample included, equals the budget (five repeats; one round, boosting settings not tuned). The networks need no docked sample. Their recall differs from the development column of Table~\ref{tab:budget} because this library holds all development ligands that have scores for the ten targets, whereas the development frame of that table holds 10{,}000 ligands.}",
             r"\label{tab:ml}", r"\centering", r"\small", r"\begin{tabular}{lrrr}", r"\toprule", r"Workflow & Docked sample & 10\% budget & 20\% budget \\", r"\midrule"]
    for key, nm in (("rsgh_wt", "Hit-head network, wild type only"), ("rsgh_int", "Hit-head network, measured changes")):
        cells = []
        for b in ("0.10", "0.20"):
            v = M["zero_shot"][key][b]
            lo, hi = boot_ci(v)
            cells.append(f"{np.mean(v):.2f} ({lo:.2f} to {hi:.2f})")
        lines.append(f"{nm} & none & {cells[0]} & {cells[1]} " + r"\\")
    lines.append(r"\midrule")
    for f in ("0.01", "0.02", "0.05"):
        e = M["ml_guided"][f]
        cells = []
        for b in ("0.10", "0.20"):
            lo, hi = boot_ci(e[b])
            cells.append(f"{np.mean(e[b]):.2f} ({lo:.2f} to {hi:.2f})")
        lines.append(f"ML-guided docking & {e['sample']:,}".replace(",", "{,}") + f" ({float(f):.0%}".replace("%", r"\%") + f") & {cells[0]} & {cells[1]} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_ml.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("table_ml written")


def table_joint():
    """Jev-like and joint networks: parameters, scoring rate, model time for 57 and 1,000 pockets, recall (cost_model.json, wt_budget_summary.json)."""
    CM = json.loads((EV / "cost_model.json").read_text())
    BU = json.loads((EV / "wt_budget_summary.json").read_text())
    reg = {f.stem.replace("registry_", ""): json.loads(f.read_text()) for f in sorted((ROOT / "runs/v3").glob("registry_*.json"))}
    par = {"XA": [v["params"] for v in reg.values() if v["model"] == "xattn"], "PC": [v["params"] for v in reg.values() if v["model"] == "pooled"]}
    groups = [("Jev-like (shared ligand embedding)", [("Dual encoder", 848002, "dual", "C0"), ("Residue-sum, hit head", 917638, "rsgh", "rsgh_wt"),
                                                       ("Pooled concatenation", par["PC"][0], "pooled", "PC")]),
              ("Joint (interaction layers for every pair)", [("Cross-attention, cached states", par["XA"][0], "xattn_cached", "XA"), ("Cross-attention, whole network", par["XA"][0], "xattn_naive", "XA")])]
    sec = CM["seconds_by_pockets"]

    def f(x):
        return f"{x:,.0f}".replace(",", "{,}") if x >= 10 else f"{x:.1f}"
    lines = [r"\begin{table*}[t]",
             r"\caption{Jev-like and joint networks on one " + CM["gpu"].replace("NVIDIA GeForce ", "") + r". The dual encoder, the pooled network and the cross-attention network were trained with the same recipe on the same 39 targets (three seeds each); the residue-sum network was trained on 38 targets for 20{,}000 updates without a checkpoint rule. Rate: million ligand-pocket pairs scored per second after the ligand embeddings exist. "
             r"57 and 1{,}000 pockets: model time in seconds (embedding, pocket states and scoring, without ligand featurization) for 260{,}155 ligands from the cost model of the Supporting Information. "
             r"Recall: top-1\% recall at the 10\% budget (mean over seeds per target; 95\% interval over the nine test targets, and the mean over the ten development targets). "
             r"The residue-sum network is trained on wild-type scores. Uni-Dock needs " + f"{sec['57']['unidock_fast']:,.0f}".replace(",", "{,}") + r" s (fast mode, RTX 4090) for 57 pockets.}",
             r"\label{tab:joint}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Network & Params. & Rate & 57 pockets & 1{,}000 pockets & Test recall & Dev. recall " + r"\\", r"\midrule"]
    for gname, items in groups:
        lines.append(r"\multicolumn{7}{l}{\emph{" + gname + r"}} \\")
        for nm, npar, ck, cond in items:
            rate = CM["scoring_pairs_per_s"][ck] / 1e6
            rate_s = f"{rate:,.0f}" if rate >= 10 else (f"{rate:.1f}" if rate >= 1 else f"{rate:.2f}")
            r = BU["test"][cond]["recall_1@10"]
            rt = f"{r['mean']:.2f} ({r['ci'][0]:.2f} to {r['ci'][1]:.2f})"
            rd = f"{BU['dev'][cond]['recall_1@10']['mean']:.2f}"
            nps = f"{npar:,}".replace(",", "{,}")
            lines.append(f"\\quad {nm} & {nps} & {rate_s} & {f(sec['57'][ck])} & {f(sec['1000'][ck])} & {rt} & {rd} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "table_joint.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("table_joint written")


if __name__ == "__main__":
    table1()
    table2()
    table3()
    table_levels()
    table_scale()
    if (EV / "cost_model.json").exists() and (EV / "wt_budget_summary.json").exists():
        table_joint()
    for f, fn in (("docking_timing.csv", table_speed), ("wt_budget_summary.json", table_budget), ("ml_guided_dev.json", table_ml)):
        if (EV / f).exists() and (f != "docking_timing.csv" or (EV / "throughput" / "thr_c0_run2.json").exists()):
            fn()
