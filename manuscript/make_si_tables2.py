"""More Supporting Information tables: paired differences (S5), label access log (S6) and hyperparameters (S7)."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "dockmut/eval"
OUT = Path(__file__).parent

NAMES = {
    "C0": "baseline, plain objective", "rsg_int": "geometry network, measured labels", "rsg_wt": "geometry network, wild type only",
    "rsg_int_shm2": "labels deranged over edits (v2)", "rsg_int_sha2": "all labels deranged (v2)", "rsg_int_shm": "labels permuted over edits (v1)",
    "rsg_int_sha": "all labels permuted (v1)", "rsg_int_shl": "labels permuted over ligands", "gbm_descriptors_geo": "boosting with geometry",
    "gbm_descriptors": "boosting without geometry", "rs_int": "ResidueSum, measured labels", "rs_wt": "ResidueSum, wild type only",
    "rse_int": "ResidueSum-Edit, measured labels", "rse_wt": "ResidueSum-Edit, wild type only", "b5_int": "dual encoder, measured labels",
    "b5_wt": "dual encoder, wild type only", "gbm_pose_geo": "boosting with pose contacts and geometry",
    "rsgh_int": "geometry network with hit head, measured labels", "rsgh_wt": "geometry network with hit head, wild type only",
}


def num(x, d=2):
    return f"{x:.{d}f}".replace("-", "$-$")


def table_pairs():
    P = json.loads((EV / "paired_all.json").read_text())["pairs"]
    rows = [r"\begin{longtable}{p{8.6cm}rrr}",
            r"\caption{Paired differences of the pooled correlation on the 19 held-out targets (seed ensembles). The interval is the 95\% percentile interval of 10{,}000 bootstrap resamples of targets, and $P$ is the fraction of resamples with a difference of zero or less.}",
            r"\label{tab:si_pairs}\\", r"\toprule", r"Comparison (first minus second) & Difference & 95\% interval & $P$ \\", r"\midrule", r"\endfirsthead",
            r"Comparison (first minus second) & Difference & 95\% interval & $P$ \\", r"\midrule", r"\endhead"]
    for k, v in P.items():
        a, b = [x.strip() for x in k.split(" - ")]
        rows.append(f"{NAMES.get(a, a)} minus {NAMES.get(b, b)} & {num(v['diff'])} & {num(v['ci'][0])} to {num(v['ci'][1])} & {v['p_le0']:.3f} " + r"\\")
    rows += [r"\bottomrule", r"\end{longtable}"]
    (OUT / "si_table_pairs.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("si_table_pairs written", len(P))


def table_access():
    rows = [json.loads(line) for line in (ROOT / "runs/label_access.log").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "test" in r["cell"]]
    lines = [r"\begin{table}[h]", r"\caption{Reads of the sealed test labels through the guarded label interface (local time, UTC+9). The first three rows are single-row reads by unit tests of the label guard before any model existed, the next two belong to the preceding analysis, the sixth to the 69 models of the first round, the seventh to the six hit-head networks (addendum 7) and the last to the six joint networks (addendum 8).}",
             r"\label{tab:si_access}", r"\small", r"\begin{tabular}{llll}", r"\toprule", r"Time & Caller & Cell & Rows \\", r"\midrule"]
    for r in rows:
        t = datetime.datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{t} & \\texttt{{{r['caller'].replace('_', chr(92) + '_')}}} & \\texttt{{{r['cell'].replace('_', chr(92) + '_')}}} & {r['n']:,} ".replace(",", "{,}") + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "si_table_access.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("si_table_access written", len(rows))


def table_speed_si():
    """Per-receptor rates of the docking benchmark and the two timing runs of every network family."""
    import pandas as pd
    fs = [EV / n for n in ("docking_timing.csv", "docking_timing_big.csv", "docking_timing_4k.csv") if (EV / n).exists()]
    dt = pd.concat([pd.read_csv(f) for f in fs])
    dt["rate"] = dt.n_ok / dt.sec
    lines = [r"\begin{table}[h]",
             r"\caption{Speed records. Upper block: ligands docked per second by Uni-Dock 1.2.0 on one RTX 4090 for each receptor, search mode and number of ligands per call (median over the timed calls, whose number is given in the last column; the first call of each job and receptor is a warm-up and is not counted). "
             r"Lower block: the two timing runs of each network family (ensembles of three networks, 260{,}155 ligands against 57 pockets, seconds).}",
             r"\label{tab:si_speed}", r"\small", r"\begin{tabular}{llrrr}", r"\toprule",
             r"Receptor & Mode & Per call & Rate & Calls " + r"\\", r"\midrule"]
    for (t, mode, n), g in dt[dt.warmup == 0].groupby(["target", "mode", "n"], sort=False):
        lines.append(f"{t} & {mode} & {n:,} & {g.rate.median():.1f} & {len(g)} ".replace(",", "{,}", 1) + r"\\")
    lines += [r"\midrule", r"Network family & Run & Featurization & Embedding and states & Scoring " + r"\\", r"\midrule"]
    for fam, nm in (("c0", "Baseline dual encoders"), ("rsg", "Geometry networks"), ("hit", "Hit-head networks")):
        for run in (1, 2):
            j = json.loads((EV / "throughput" / f"thr_{fam}_run{run}.json").read_text())
            pk, lb = j["pockets"], j["library"]
            lines.append(f"{nm} & {run} & {lb['sec_featurise_cpu']:.1f} & {lb['sec_embed_gpu_all_models'] + pk['sec_state_encoding']:.2f} & {pk['sec_scoring_all_models']:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "si_table_speed.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("si_table_speed written")


def table_joint_si():
    """Per-run record of the joint networks (addendum 8) and the per-stage times of the cost model."""
    import pandas as pd
    ROOTD = ROOT
    reg = {f.stem.replace("registry_", ""): json.loads(f.read_text()) for f in sorted((ROOTD / "runs/v3").glob("registry_*.json"))}
    cond = {"xattn": "XA", "pooled": "PC"}
    lines = [r"\begin{table}[h]",
             r"\caption{Joint networks of addendum 8. Updates and best update count training updates of 256 ligands; the checkpoint is the evaluation point with the highest macro recall on the ten development targets (top-1\% recall at the 10\% budget). "
             r"Test recall: the nine test targets, read once for all six networks after the checkpoints were frozen. The registry file of the first three runs was overwritten by concurrent runs and was rebuilt from the training logs and the checkpoint hashes.}",
             r"\label{tab:si_joint_runs}", r"\small", r"\begin{tabular}{llrrrrr}", r"\toprule", r"Network & Seed & Updates & Best update & Stop & Dev. recall & Test recall " + r"\\", r"\midrule"]
    for tag, r in reg.items():
        c = cond[r["model"]]
        tm = pd.read_csv(ROOTD / f"runs/v3/test_metrics_{c}_s{r['seed']}_cold.csv")["recall_1@10"].mean()
        stop = "patience" if "patience" in r["stop_reason"] else "budget"
        nm = "Cross-attention" if r["model"] == "xattn" else "Pooled concatenation"
        lines.append(f"{nm} & {r['seed']} & {r['updates']:,} & {r['best_update']:,} & {stop} & {r['best_cold_recall_1@10']:.3f} & {tm:.3f} ".replace(",", "{,}") + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    T = json.loads((EV / "joint_timing.json").read_text())
    nl, npk = T["n_lig"], T["n_pockets"]
    lines += [r"\begin{table}[h]",
              r"\caption{Per-stage times of the cost model on one " + T["gpu"].replace("NVIDIA GeForce ", "") + f" ({nl:,} ligands against {npk} pockets, float32 unless noted, shorter of two runs, seconds)".replace(",", "{,}") + r". "
              r"Embedding: the pocket-independent ligand representation (for the cross-attention network the atom states). State: the pocket states of all pockets. Scoring: all ligand-pocket pairs.}",
              r"\label{tab:si_joint_timing}", r"\small", r"\begin{tabular}{lrrr}", r"\toprule", r"Network and path & Embedding & State & Scoring " + r"\\", r"\midrule"]
    rows = [("Dual encoder", T["dual"], "score_s"), ("Pooled concatenation", T["pooled"], "score_s"), ("Residue-sum, hit head", T["rsgh"], "score_s"),
            ("Cross-attention, cached ligand states", T["xattn"], "score_cached_s"), ("Cross-attention, whole network per pair", T["xattn"], "score_naive_s")]
    if "xattn_amp" in T:
        rows += [("Cross-attention, cached, float16", T["xattn_amp"], "score_cached_s"), ("Cross-attention, whole network, float16", T["xattn_amp"], "score_naive_s")]
    for nm, m, k in rows:
        lines.append(f"{nm} & {m['embed_s']:.3f} & {m['state_s']:.3f} & {m[k]:.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "si_table_joint.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("si_table_joint written")


if __name__ == "__main__":
    table_pairs()
    table_access()
    table_speed_si()
    if (EV / "joint_timing.json").exists():
        table_joint_si()
