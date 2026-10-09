"""Macros of the known-pocket manuscript: full-scale timing (addendum 9, experiment E2) and the known-pocket evaluation (addendum 10, E3).

Reads dockmut/eval/v4_fair_timing.json (A, the RTX 4090 pod), v4_fair_timing_L40S.json (B) and v5_known_pockets_summary.json.
Writes numbers_v5.tex (LaTeX macro names contain letters only).
"""
from __future__ import annotations

import json
from pathlib import Path

here = Path(__file__).parent
EV = here.parent / "dockmut/eval"
defs: dict[str, str] = {}


def put(name: str, value, fmt: str = "{:.2f}"):
    s = value if isinstance(value, str) else fmt.format(value)
    if s in ("-0.00", "-0.0", "-0"):
        s = s[1:]
    defs[name] = s.replace("-", "$-$")


def thousands(x) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def fsec(x: float) -> str:
    return f"{x:.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else (f"{x:.2f}" if x >= 0.1 else f"{x:.4f}"))


def fratio(x: float) -> str:
    return thousands(x) if x >= 100 else (f"{x:.0f}" if x >= 10 else f"{x:.1f}")


PW = {"1": "One", "5": "Five", "20": "Twenty", "57": "FiftySeven", "1000": "Thousand"}
NETS = {"dual": "Dual", "pooled": "Pooled", "xattn_fp32": "XaSingle", "xattn_fp16": "XaHalf"}
def _complete(fname):
    f_ = EV / fname
    if not f_.exists():
        return False
    try:
        return "1000" in json.loads(f_.read_text())["blocks"].get("xattn_fp16", {})
    except Exception:
        return False


PRIMARY = "v4_fair_timing.json" if _complete("v4_fair_timing.json") else "v4_fair_timing_L40S.json"      # A: the RTX 4090 pod when available, else the L40S
for g, fname in (("A", PRIMARY), ("B", "v4_fair_timing_L40S.json")):
    f = EV / fname
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    env = d["environment"]
    put(f"tm{g}Gpu", env["gpu"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", ""))
    put(f"tm{g}Procs", str(env["featurization_processes"]))
    put(f"tm{g}Cpus", str(env["cpu_count"]))
    put(f"tm{g}Feat", fsec(d["featurization"]["seconds"]))
    put(f"tm{g}Lig", thousands(d["library_size"]))
    blk = d["blocks"]
    for P, pw in PW.items():
        for net, nm in NETS.items():
            b = blk.get(net, {}).get(P)
            if not b:
                continue
            put(f"tm{g}{nm}{pw}Model", fsec(b["model_time_s"]))
            put(f"tm{g}{nm}{pw}Full", fsec(b["end_to_end_s"]))
            put(f"tm{g}{nm}{pw}Score", fsec(b["stages_s"]["score_gpu"]))
            r_ = b["scoring_pairs_per_s"] / 1e6
            put(f"tm{g}{nm}{pw}Rate", thousands(r_) if r_ >= 100 else f"{r_:.1f}")
        base = blk["dual"].get(P)
        for net, nm in NETS.items():
            b = blk.get(net, {}).get(P)
            if net == "dual" or not b or not base:
                continue
            put(f"tm{g}Ratio{nm}{pw}Model", fratio(b["model_time_s"] / base["model_time_s"]))
            put(f"tm{g}Ratio{nm}{pw}Full", fratio(b["end_to_end_s"] / base["end_to_end_s"]))
    b57 = blk["dual"]["57"]
    put(f"tm{g}Pairs", f"{b57['n_pairs'] / 1e6:.1f}")
    pad = d["ratios"].get("padded_vs_ragged", {})
    for k, v in pad.items():
        put(f"tm{g}PaddedSingle", v.get("ragged_fp32_over_padded_scoring_pairs_per_s", 0), "{:.1f}")
        put(f"tm{g}PaddedHalf", v.get("ragged_fp16_over_padded_scoring_pairs_per_s", 0), "{:.0f}")

put("paramsDual", "848{,}002")
for g, fname in (("A", PRIMARY), ("B", "v4_fair_timing_L40S.json")):
    f = EV / fname
    if not f.exists():
        continue
    blk = json.loads(f.read_text())["blocks"]
    for P, pw in PW.items():
        for net, nm in NETS.items():
            b = blk.get(net, {}).get(P)
            if b:
                put(f"tm{g}{nm}{pw}Select", fsec(b["stages_s"]["score_gpu"] + b["stages_s"]["topk_gpu"]))
    for net, nm in NETS.items():                                  # marginal model time of one more pocket (1,000 against 57 pockets)
        x = (blk[net]["1000"]["model_time_s"] - blk[net]["57"]["model_time_s"]) / 943
        put(f"tm{g}{nm}PerPocket", fsec(x))
        put(f"tm{g}{nm}PerPocketMs", f"{1e3 * x:.1f}")
folds_f = here.parent / "data/splits/v4_cv_folds.json"
if folds_f.exists():
    fl = json.loads(folds_f.read_text())["folds"]
    put("kfFoldTrainMin", str(min(len(x["train"]) for x in fl)))
    put("kfFoldTrainMax", str(max(len(x["train"]) for x in fl)))
import glob as _glob

def _best(pattern):
    v = [json.loads(Path(p_).read_text())["best_update"] for p_ in _glob.glob(str(here.parent / pattern))]
    return (min(v), max(v)) if v else None

b3 = _best("runs/v3/registry_xattn_s*.json")
b4 = _best("runs/v4/registry_f*_xattn_R0_s11.json")
if b3:
    put("kpXaBestLo", thousands(b3[0]))
    put("kpXaBestHi", thousands(b3[1]))
if b4:
    put("kfXaBestLo", thousands(b4[0]))
    put("kfXaBestHi", thousands(b4[1]))
share = 40000 / 260155
put("trainShareLib", 100 * share, "{:.1f}")
put("shareLabelShortlist", 100 * (share + 0.10), "{:.0f}")
put("factorLabelShortlist", 1 / (share + 0.10), "{:.1f}")
K = EV / "v5_known_pockets_summary.json"
if K.exists():
    s = json.loads(K.read_text())
    CN = {"C0": "Dual", "PC": "Pooled", "XA": "Xattn", "C0c": "Const"}
    BN = {"recall_1@1": "One", "recall_1@5": "Five", "recall_1@10": "Ten", "recall_1@20": "Twenty"}
    put("kpNTargets", str(s["n_targets"]))
    tens_ = [s["mean"][c]["recall_1@10"] for c in ("C0", "PC", "XA") if c in s["mean"]]
    put("kpSpreadTen", max(tens_) - min(tens_))
    for c, cn in CN.items():
        if c not in s["mean"]:
            continue
        for m, bn in BN.items():
            put(f"kp{cn}{bn}", s["mean"][c][m])
            put(f"kp{cn}{bn}Lo", s["ci"][c][m][0])
            put(f"kp{cn}{bn}Hi", s["ci"][c][m][1])
    for k, v in s["paired"].items():
        x, y = k.split("_minus_")
        for m, bn in BN.items():
            put(f"kpDiff{CN[x]}{CN[y]}{bn}", v[m]["mean"])
            put(f"kpDiff{CN[x]}{CN[y]}{bn}Lo", v[m]["ci"][0])
            put(f"kpDiff{CN[x]}{CN[y]}{bn}Hi", v[m]["ci"][1])
    for c, v in s.get("ratio_to_XA", {}).items():
        put(f"kpRatio{CN[c]}Xattn", 100 * v["mean"], "{:.0f}")
        put(f"kpRatio{CN[c]}XattnLo", 100 * v["ci"][0], "{:.0f}")
        put(f"kpRatio{CN[c]}XattnHi", 100 * v["ci"][1], "{:.0f}")

K4 = EV / "v5_known_cv_summary.json"
if K4.exists():
    import numpy as np

    s4 = json.loads(K4.read_text())
    FN = {"dual_R0": "Dual", "pooled_R0": "Pooled", "xattn_R0": "Xattn", "dual_const_R0": "Const", "pooled_const_R0": "PooledConst", "xattn_const_R0": "XattnConst",
          "dual_R1": "DualLong", "xattn_R1": "XattnLong"}
    BN = {"recall_1@1": "One", "recall_1@5": "Five", "recall_1@10": "Ten", "recall_1@20": "Twenty"}
    put("kfNTargets", str(s4["n_targets"]["dual_R0"]))
    for c, cn in FN.items():
        if c not in s4["mean"]:
            continue
        for m, bn in BN.items():
            put(f"kf{cn}{bn}", s4["mean"][c][m])
            put(f"kf{cn}{bn}Lo", s4["ci"][c][m][0])
            put(f"kf{cn}{bn}Hi", s4["ci"][c][m][1])
    for k, v in s4["paired"].items():
        x, y = k.split("_minus_")
        for m, bn in BN.items():
            put(f"kfDiff{FN[x]}{FN[y]}{bn}", v[m]["mean"])
            put(f"kfDiff{FN[x]}{FN[y]}{bn}Lo", v[m]["ci"][0])
            put(f"kfDiff{FN[x]}{FN[y]}{bn}Hi", v[m]["ci"][1])
    rng = np.random.default_rng(1)
    for c, ref in (("dual_R0", "xattn_R0"), ("pooled_R0", "xattn_R0"), ("dual_R1", "xattn_R1"), ("dual_const_R0", "xattn_R0")):
        pt = s4["per_target"]
        keys = sorted(set(pt[c]["recall_1@10"]) & set(pt[ref]["recall_1@10"]))
        v = np.array([pt[c]["recall_1@10"][k] for k in keys])
        w = np.array([pt[ref]["recall_1@10"][k] for k in keys])
        idx = rng.integers(0, len(v), size=(5000, len(v)))
        r = v[idx].mean(1) / w[idx].mean(1)
        put(f"kfRatio{FN[c]}", 100 * v.mean() / w.mean(), "{:.0f}")
        put(f"kfRatio{FN[c]}Lo", 100 * float(np.percentile(r, 2.5)), "{:.0f}")
        put(f"kfRatio{FN[c]}Hi", 100 * float(np.percentile(r, 97.5)), "{:.0f}")

KH = EV / "v4_cv_summary.json"
if KH.exists():                                                  # held-out target families of the cross-validation (addendum 9), used in the Supporting Information
    sh = json.loads(KH.read_text())
    HN = {"dual_R0": "Dual", "pooled_R0": "Pooled", "xattn_R0": "Xattn", "dual_const_R0": "Const"}
    for c, cn in HN.items():
        e = sh["conditions"][c]["overall"]["recall_1@10"]
        put(f"ho{cn}Ten", e["mean"])
        put(f"ho{cn}TenLo", e["lo"])
        put(f"ho{cn}TenHi", e["hi"])
    put("hoNTargets", str(sh["conditions"]["dual_R0"]["overall"]["recall_1@10"]["n"]))
    for p_ in sh["paired"]:
        if p_["kind"] == "pocket_gain":
            e = p_["overall"]["recall_1@10"]
            nm = {"dual_R0": "Dual", "pooled_R0": "Pooled", "xattn_R0": "Xattn"}[p_["a"]]
            put(f"hoGain{nm}Ten", e["mean"])
            put(f"hoGain{nm}TenLo", e["lo"])
            put(f"hoGain{nm}TenHi", e["hi"])

out = here / "numbers_v5.tex"
out.write_text("\n".join("\\newcommand{\\%s}{%s}" % (k, defs[k]) for k in sorted(defs)) + "\n", encoding="utf-8")
print(len(defs), "macros written to", out.name)
