"""Append model-result macros to numbers.tex (run after make_numbers.py).

Every number quoted in the model results is defined here from the evaluation outputs. Missing inputs are skipped, and
the fallback definitions of numbers_placeholder.tex print a visible placeholder for them.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from conds import MACRO  # noqa: E402

EV = ROOT / "dockmut/eval"
OUT = Path(__file__).parent / "numbers.tex"
defs: dict[str, str] = {}
MARK = "% model results below"


def put(name: str, value, fmt: str = "{:.2f}"):
    s = value if isinstance(value, str) else fmt.format(value)
    if s in ("-0.00", "-0.0", "-0"):
        s = s[1:]
    defs[name] = s.replace("-", "$-$")


def load(name: str):
    p = EV / name
    return json.loads(p.read_text()) if p.exists() else None


def thousands(n) -> str:
    return f"{int(round(n)):,}".replace(",", "{,}")


S = load("summary_all.json")
P = load("paired_all.json")
L = load("levels_all.json")
B = load("baselines_v2.json")
if S and P:
    for key, suf in MACRO.items():
        if key in P["conditions"]:
            c = P["conditions"][key]
            put("rEns" + suf, c["r"])
            put("rLo" + suf, c["ci"][0])
            put("rHi" + suf, c["ci"][1])
        if key in S:
            s = S[key]
            e = s["ensemble"]
            put("rSeed" + suf, s["pooled_r_mean"])
            put("rSeedSd" + suf, s["pooled_r_sd"])
            put("nSeed" + suf, str(s["n_seeds"]))
            put("editR" + suf, e["mutant_r"])
            put("withinR" + suf, e["within_r"])
            put("sdPred" + suf, e["pred_sd"])
            put("auc" + suf, e["detect_auc"])
            put("gain" + suf, e["gain"])
            if s["pooled_r_seeds"]:
                put("rSeedMin" + suf, min(s["pooled_r_seeds"]))
                put("rSeedMax" + suf, max(s["pooled_r_seeds"]))
    for pair, v in P["pairs"].items():
        a, b = [x.strip() for x in pair.split(" - ")]
        if a in MACRO and b in MACRO:
            nm = MACRO[a] + "Minus" + MACRO[b]
            put("diff" + nm, v["diff"])
            put("diffLo" + nm, v["ci"][0])
            put("diffHi" + nm, v["ci"][1])
    put("nHeldTargetsCI", str(P["n_targets"]))
if S and B:
    g = B["C0_s11"]["contact_all"]
    put("sdTrue", g["true_sd"])
    put("ceilingR", B["_noise"]["r_max"], "{:.2f}")
    sds = [S[k]["ensemble"]["pred_sd"] for k in ("C0", "C1", "C2")]
    put("sdPredRefMin", min(sds))
    put("sdPredRefMax", max(sds))
    put("ratioSdRefMin", g["true_sd"] / max(sds), "{:.0f}")
    put("ratioSdRefMax", g["true_sd"] / min(sds), "{:.0f}")
    ck = [v["pred_sd"] for k in ("C0", "C1", "C2") for v in S[k]["per_seed"].values()]      # the nine single checkpoints
    put("sdCkptMin", min(ck))
    put("sdCkptMax", max(ck))
    put("ratioCkptMin", g["true_sd"] / max(ck), "{:.0f}")
    put("ratioCkptMax", g["true_sd"] / min(ck), "{:.0f}")
    allr = [r for k in ("C0", "C1", "C2") for r in S[k]["pooled_r_seeds"]]
    put("rRefSeedMin", min(allr))
    put("rRefSeedMax", max(allr))
    aucs = [S[k]["ensemble"]["detect_auc"] for k in ("C0", "C1", "C2")]
    put("aucRefMin", min(aucs))
    put("aucRefMax", max(aucs))
if L:
    for key, suf in MACRO.items():
        if key in L:
            v = L[key]
            put("lvTarget" + suf, v["within_target_r"])
            put("lvEdit" + suf, v["edit_mean_r"])
            put("lvLig" + suf, v["ligand_r"])
            put("lvTargetLo" + suf, v["within_target_r_ci"][0])
            put("lvTargetHi" + suf, v["within_target_r_ci"][1])
            put("lvEditLo" + suf, v["edit_mean_r_ci"][0])
            put("lvEditHi" + suf, v["edit_mean_r_ci"][1])
            put("lvLigLo" + suf, v["ligand_r_ci"][0])
            put("lvLigHi" + suf, v["ligand_r_ci"][1])
            for k2, tag in (("within_target_r", "Target"), ("edit_mean_r", "Edit"), ("ligand_r", "Lig")):
                d = v.get("vs_rsg_int_" + k2)
                if d:
                    put(f"lvd{tag}{suf}", d["diff"])
                    put(f"lvd{tag}Lo{suf}", d["ci"][0])
                    put(f"lvd{tag}Hi{suf}", d["ci"][1])
# new edits of known pockets (15% of the edits of every training target, withheld from training)
known = {}
for f in glob.glob(str(ROOT / "dockmut/runs_collected/final/*_diag.json")):
    stem = Path(f).name[:-len("_diag.json")]
    c, s_ = stem.rsplit("_s", 1)
    known.setdefault(c, []).append(json.loads(Path(f).read_text())["val_mutants"])
for c, vs in known.items():
    if c in MACRO:
        suf = MACRO[c]
        r = [v["pooled_r"] for v in vs if v["pooled_r"] == v["pooled_r"]]
        if r:
            put("knownR" + suf, float(np.mean(r)))
            put("knownRSd" + suf, float(np.std(r)), "{:.3f}")
            put("knownEdit" + suf, float(np.nanmean([v["mutant_r"] for v in vs])))
            put("knownWithin" + suf, float(np.nanmean([v["within_r"] for v in vs])))
# scaling study
WORDS = {"2": "Two", "4": "Four", "8": "Eight", "12": "Twelve", "16": "Sixteen", "24": "Twentyfour"}
Z = load("scale_summary.json")
if Z:
    for v, d in Z.items():
        m = re.match(r"^(nt|mk)(\d+)$", v)
        name = (m.group(1).capitalize() + WORDS[m.group(2)]) if m else v.capitalize()
        put("scaleR" + name, d["pooled_r_mean"])
        put("scaleEns" + name, d["ensemble_r"])
        put("scaleEdit" + name, d["mutant_r_mean"])
        put("scaleNew" + name, d["new_edits_r_mean"])
# hot-spot ranking
H = load("hotspot.json")
if H:
    for key, suf in MACRO.items():
        if key in H and H[key] == H[key]:
            put("hot" + suf, H[key])
    put("hotDist", H["distance"])
    put("hotRemoved", H["removed_atoms"])
    put("scanSpearmanAbs", H["scan_spearman_abs_contact"])
    put("scanSpearmanAbsAll", H["scan_spearman_abs_all"])
    put("scanRSigned", H["scan_r_signed_contact"])
    put("scanRSignedAll", H["scan_r_signed_all"])
    put("scanNContact", str(H["scan_n_contact"]))
    put("scanNAll", str(H["scan_n_all"]))
# within-family transfer (four withheld kinases)
K = load("kin_summary.json")
if K:
    for c, d in K.items():
        if c in MACRO:
            put("kinR" + MACRO[c], d["pooled_r_mean"])
            put("kinRSd" + MACRO[c], d["pooled_r_sd"])
            put("kinN" + MACRO[c], str(d["n"]))
            put("kinEdit" + MACRO[c], d["mutant_r_mean"])
            put("kinWithin" + MACRO[c], d["within_r_mean"])
    if "rsg_int" in K and "rsg_int_shm2" in K and "rsg_int_sha2" in K:
        put("kinGapRsgShmTwo", K["rsg_int"]["pooled_r_mean"] - K["rsg_int_shm2"]["pooled_r_mean"])
        put("kinGapRsgShaTwo", K["rsg_int"]["pooled_r_mean"] - K["rsg_int_sha2"]["pooled_r_mean"])
D = load("descriptor_baselines_kin.json")
if D:
    for c, d in D.items():
        if c in MACRO:
            put("kinR" + MACRO[c], d["pooled_r"])
# wild-type screening
for split, tag in (("dev", "Dev"), ("test", "Test")):
    W = load(f"wt_{split}_all.json")
    if not W:
        continue
    groups: dict[str, list] = {}
    for t, v in W.items():
        m = re.match(r"^(.*)_s(\d+)$", t)
        groups.setdefault(m.group(1) if m else t, []).append(v)
    for c, vs in groups.items():
        if c in MACRO or c == "C0c":
            suf = MACRO.get(c, "CZeroC")
            put(f"wtR{tag}{suf}", np.mean([v["recall_1@10"] for v in vs]), "{:.2f}")
            put(f"wtRmin{tag}{suf}", np.min([v["recall_1@10"] for v in vs]), "{:.2f}")
            put(f"wtRmax{tag}{suf}", np.max([v["recall_1@10"] for v in vs]), "{:.2f}")
            put(f"wtSp{tag}{suf}", np.mean([v["spearman"] for v in vs]), "{:.2f}")
            put(f"wtJ{tag}{suf}", np.mean([v["joint"] for v in vs]), "{:.2f}")
            put(f"wtJmin{tag}{suf}", np.min([v["joint"] for v in vs]), "{:.2f}")
            put(f"wtJmax{tag}{suf}", np.max([v["joint"] for v in vs]), "{:.2f}")
            put(f"wtJsd{tag}{suf}", np.std([v["joint"] for v in vs]), "{:.2f}")
            put(f"wtN{tag}{suf}", str(len(vs)))
    put(f"nQuads{tag}", thousands(next(iter(W.values()))["n_quads"]))
    WS = load(f"wt_{split}_summary.json")
    if WS:
        put(f"nTarget{tag}", str(WS["n_targets"]))
        put(f"wtRBase{tag}", WS["baseline_mean"]["recall_1@10"], "{:.2f}")
        for c, v in WS["conditions"].items():
            suf = MACRO.get(c, "CZeroC" if c == "C0c" else None)
            if suf:
                put(f"wtRLo{tag}{suf}", v["ci"][0], "{:.2f}")
                put(f"wtRHi{tag}{suf}", v["ci"][1], "{:.2f}")
        for pair, v in WS["pairs"].items():
            a_, b_ = [x.strip() for x in pair.split(" - ")]
            sa = MACRO.get(a_, "CZeroC" if a_ == "C0c" else None)
            sb = "Base" if b_ == "baseline" else ("Minus" + MACRO.get(b_, "CZeroC" if b_ == "C0c" else "X"))
            if sa:
                put(f"wtd{tag}{sa}{sb}", v["diff"], "{:.2f}")
                put(f"wtdLo{tag}{sa}{sb}", v["ci"][0], "{:.2f}")
                put(f"wtdHi{tag}{sa}{sb}", v["ci"][1], "{:.2f}")
        for c, v in WS["reversal"].items():
            suf = MACRO.get(c, "CZeroC" if c == "C0c" else None)
            if suf:
                put(f"wtJSdAll{tag}{suf}", v["sd"], "{:.2f}")
    DD = load("wt_dev_descriptor.json") if split == "dev" else None
    if DD:
        put("wtRDevBoostGeo", DD["wt_gbm_geometry"]["recall_1@10"], "{:.2f}")
        put("wtJDevBoostGeo", DD["wt_gbm_geometry"]["joint"], "{:.2f}")
        put("wtRDevBoostLig", DD["wt_gbm_ligand_only"]["recall_1@10"], "{:.2f}")
        put("wtJDevBoostLig", DD["wt_gbm_ligand_only"]["joint"], "{:.2f}")
    ref = [v for t, v in W.items() if re.match(r"^C[012]_s\d+$", t)]
    if ref:
        put(f"wtR{tag}RefMin", min(v["recall_1@10"] for v in ref), "{:.2f}")
        put(f"wtR{tag}RefMax", max(v["recall_1@10"] for v in ref), "{:.2f}")
        put(f"wtJ{tag}RefMin", min(v["joint"] for v in ref), "{:.2f}")
        put(f"wtJ{tag}RefMax", max(v["joint"] for v in ref), "{:.2f}")
        put(f"wtSp{tag}RefMin", min(v["spearman"] for v in ref), "{:.2f}")
        put(f"wtSp{tag}RefMax", max(v["spearman"] for v in ref), "{:.2f}")
# baselines read out with their own hit-probability head (result files of the preceding analysis)
HS = load("wt_hit_summary.json")
HSUF = {"C0": "CZero", "C1": "COne", "C2": "CTwo", "C0c": "CZeroC", "CB": "LigOnly"}
if HS:
    for split, tag in (("dev", "Dev"), ("test", "Test")):
        e = HS[split]
        for c, v in e["conditions"].items():
            suf = HSUF[c]
            put(f"hitR{tag}{suf}", v["recall_1@10"], "{:.2f}")
            put(f"hitRLo{tag}{suf}", v["ci"][0], "{:.2f}")
            put(f"hitRHi{tag}{suf}", v["ci"][1], "{:.2f}")
            put(f"hitJ{tag}{suf}", v["reversal"]["mean"], "{:.2f}")
            put(f"hitJmin{tag}{suf}", v["reversal"]["min"], "{:.2f}")
            put(f"hitJmax{tag}{suf}", v["reversal"]["max"], "{:.2f}")
        for pair, v in e["pairs"].items():
            a, b = [x.strip() for x in pair.split(" - ")]
            if a in MACRO and b in HSUF:
                nm = MACRO[a] + "Minus" + HSUF[b]
                put(f"hitd{tag}{nm}", v["diff"], "{:.2f}")
                put(f"hitdLo{tag}{nm}", v["ci"][0], "{:.2f}")
                put(f"hitdHi{tag}{nm}", v["ci"][1], "{:.2f}")
# hit-head networks (addendum 7): screening with the hit logit
HN = load("wt_hitnet_summary.json")
HNSUF = {"rsgh_wt": "RsghWt", "rsgh_int": "RsghInt"}
BSUF = {"C0": "CZero", "C0c": "CZeroC", "C1": "COne", "CB": "LigOnly", "rsg_wt": "RsgWt", "rsg_int": "RsgInt", "rsgh_wt": "RsghWt"}
if HN:
    for split, tag in (("dev", "Dev"), ("test", "Test")):
        e = HN[split]
        for c, v in e["conditions"].items():
            suf = HNSUF[c]
            put(f"hnR{tag}{suf}", v["recall_1@10"], "{:.2f}")
            put(f"hnRLo{tag}{suf}", v["ci"][0], "{:.2f}")
            put(f"hnRHi{tag}{suf}", v["ci"][1], "{:.2f}")
            put(f"hnJ{tag}{suf}", v["reversal"]["mean"], "{:.2f}")
            put(f"hnJmin{tag}{suf}", v["reversal"]["min"], "{:.2f}")
            put(f"hnJmax{tag}{suf}", v["reversal"]["max"], "{:.2f}")
        for pair, v in e["pairs"].items():
            a, b = [x.strip() for x in pair.split(" - ")]
            nm = HNSUF[a] + "Minus" + BSUF[b]
            put(f"hnd{tag}{nm}", v["diff"], "{:.2f}")
            put(f"hndLo{tag}{nm}", v["ci"][0], "{:.2f}")
            put(f"hndHi{tag}{nm}", v["ci"][1], "{:.2f}")
SH = load("hit/wt_dev_score.json")
if SH:
    for c, suf in (("rsgh_wt", "RsghWt"), ("rsgh_int", "RsghInt")):
        v = [x["joint"] for t, x in SH.items() if t.startswith(c + "_s")]
        put(f"hnScoreJDev{suf}", float(np.mean(v)), "{:.2f}")
        v = [x["recall_1@10"] for t, x in SH.items() if t.startswith(c + "_s")]
        put(f"hnScoreRDev{suf}", float(np.mean(v)), "{:.2f}")
# screening budget of the hit-head networks and of the baselines on the nine test targets (existing result files, no label access)
import pandas as pd  # noqa: E402

PTH = EV / "hit" / "wt_test_hit.per_target.csv"
if PTH.exists():
    h = pd.read_csv(PTH)
    h["cond"] = h.model.str.replace(r"_s\d+$", "", regex=True)
    for c, suf in (("rsgh_wt", "RsghWt"), ("rsgh_int", "RsghInt")):
        g = h[h.cond == c]
        put(f"hnBudOneTest{suf}", g["recall_1@1"].mean(), "{:.2f}")
        put(f"hnBudTwentyTest{suf}", g["recall_1@20"].mean(), "{:.2f}")
        put(f"hnBudNinetyFiveTest{suf}", 100 * g["recall95_budget"].mean(), "{:.0f}")
        put(f"hnSpearmanTest{suf}", g["spearman"].mean(), "{:.2f}")
        put(f"hnMissTest{suf}", 1 - g["recall_1@10"].mean(), "{:.2f}")
for c, suf in (("C0", "CZero"), ("C0c", "CZeroC")):
    fs = sorted(glob.glob(str(ROOT / "runs/v2" / f"test_metrics_{c}_s*_cold.csv")))
    if fs:
        g = pd.concat([pd.read_csv(f) for f in fs])
        put(f"hitSpearmanTest{suf}", g["spearman"].mean(), "{:.2f}")
# ablation of the residue sum at inference (ablate_residue_sum.py): ligand term plus global descriptor term only
AB = load("ablation_residue_sum.json")
if AB:
    for c, suf in (("rsg_wt", "RsgWt"), ("rsg_int", "RsgInt"), ("rsgh_int", "RsghInt")):
        put("ablFull" + suf, AB[c]["ensemble_full_r"])
        put("ablNoRes" + suf, AB[c]["ensemble_no_residue_sum_r"])
        put("ablNoResSeedMin" + suf, min(AB[c]["no_residue_sum_r"]))
        put("ablNoResSeedMax" + suf, max(AB[c]["no_residue_sum_r"]))
# docking rate benchmark (dock_rate_bench.py): per-call record of Uni-Dock 1.2.0 on one RTX 4090
DOCK_RATE, FAST_RATE = 11.6, None
DTF = EV / "docking_timing.csv"
if DTF.exists():
    dt = pd.concat([pd.read_csv(f) for f in (DTF, EV / "docking_timing_big.csv", EV / "docking_timing_4k.csv") if f.exists()])
    dt = dt[dt.warmup == 0].copy()
    dt["rate"] = dt.n_ok / dt.sec
    put("dockNCalls", str(len(dt)))
    _rm = dt.groupby(["mode", "n", "target"]).rate.median().unstack("target")
    put("dockRecSpread", float(((_rm.max(axis=1) - _rm.min(axis=1)) / _rm.median(axis=1)).max() * 100), "{:.0f}")
    for mode, ns, tag in (("detail", [160, 256], "Detail"), ("detail", [1000], "DetailBig"), ("balance", [256], "BalSmall"), ("balance", [1000], "BalBig"), ("fast", [256], "FastSmall"), ("fast", [1000], "FastBig"), ("detail", [4000], "DetailHuge"), ("balance", [4000], "BalHuge"), ("fast", [4000], "FastHuge")):
        sub = dt[(dt["mode"] == mode) & dt.n.isin(ns)].rate
        if len(sub):
            put("dockRate" + tag, float(sub.median()), "{:.1f}")
            put("dockRatePLo" + tag, float(sub.quantile(0.1)), "{:.1f}")
            put("dockRatePHi" + tag, float(sub.quantile(0.9)), "{:.1f}")
    DOCK_RATE = float(dt[dt["mode"] == "detail"].groupby("n").rate.median().max())          # best batch size of the detail mode
    best = dt.groupby(["mode", "n"]).rate.median()
    FAST_RATE = float(best.max())
    put("dockModeFastest", " ".join(map(str, best.idxmax())))
    put("dockRateFastest", FAST_RATE, "{:.1f}")
    DSF = EV / "docking_scores.csv"
    if DSF.exists():
        ds = pd.concat([pd.read_csv(f) for f in (DSF, EV / "docking_scores_big.csv", EV / "docking_scores_4k.csv") if f.exists()])
        NAG = 4000 if (ds.n == 4000).any() else 256
        rho_rep, rho = [], {"balance": [], "fast": []}
        shift = {"balance": [], "fast": []}
        for t, g in ds.groupby("target"):
            dref = g[(g["mode"] == "detail") & (g.n == NAG)].set_index("k").score
            d256 = g[(g["mode"] == "detail") & (g.n == 256)].set_index("k").score
            d160 = g[(g["mode"] == "detail") & (g.n == 160)].set_index("k").score
            jj = d160.index.intersection(d256.index)
            rho_rep.append(float(d160[jj].corr(d256[jj], method="spearman")))
            for mode in ("balance", "fast"):
                b = g[(g["mode"] == mode) & (g.n == NAG)].set_index("k").score
                jj = dref.index.intersection(b.index)
                rho[mode].append(float(dref[jj].corr(b[jj], method="spearman")))
                shift[mode].append(float(np.median(dref[jj] - b[jj])))
        put("dockAgreeDetail", float(np.mean(rho_rep)), "{:.2f}")
        put("dockAgreeBatch", str(NAG))
        for mode, tag in (("balance", "Bal"), ("fast", "Fast")):
            put("dockAgree" + tag, float(np.mean(rho[mode])), "{:.2f}")
            put("dockShift" + tag, abs(float(np.mean(shift[mode]))), "{:.2f}")
PLF = EV / "prep4k.log"
if PLF.exists():
    import re as _re
    _m = _re.search(r'"ok": (\d+)', PLF.read_text())
    if _m:
        put("dockPrepOkFourK", thousands(int(_m.group(1))))
THR_TOT = {}
# network families timed on the same GPU class (dm_throughput.py): baseline dual encoders, registered geometry networks, hit-head networks
for fam, tag in (("c0", "Base"), ("rsg", "Geo"), ("hit", "Hit")):
    f = EV / "throughput" / f"thr_{fam}_run2.json"
    if f.exists():
        j = json.loads(f.read_text())
        pk, lb = j["pockets"], j["library"]
        tot = lb["sec_featurise_cpu"] + lb["sec_embed_gpu_all_models"] + pk["sec_state_encoding"] + pk["sec_scoring_all_models"]
        put("thrSec" + tag, tot, "{:.0f}")
        put("thrScoreSec" + tag, pk["sec_scoring_all_models"], "{:.1f}")
        put("thrPocketSec" + tag, pk["sec_scoring_all_models"] / pk["n_pockets"], "{:.2f}")
        ind = j["independence"]
        mx = max(ind["embedding_batch1_vs_big"], ind["score_batch1_vs_big"], ind["score_permuted_batches64_vs_big"])
        mant, ex = f"{mx:.1e}".split("e")
        defs["indepMax" + tag] = "$" + str(round(float(mant))) + "\\times10^{" + str(int(ex)) + "}$"
        put("indepRange" + tag, ind["score_range"], "{:.1f}")
        put("thrRate" + tag, pk["per_model_pairs_per_sec"] / 1e6, "{:.0f}" if pk["per_model_pairs_per_sec"] / 1e6 >= 10 else "{:.1f}")
        put("thrEmbSec" + tag, lb["sec_embed_gpu_all_models"], "{:.1f}")
        THR_TOT[tag] = tot
if THR_TOT:
    put("thrSecMin", min(THR_TOT.values()), "{:.0f}")
    put("thrSecMax", max(THR_TOT.values()), "{:.0f}")
# screening at several budgets (wt_budget_summary.py)
BU = load("wt_budget_summary.json")
if BU:
    CS = {"rsgh_int": "RsghInt", "rsgh_wt": "RsghWt", "C0": "CZero", "C0c": "CZeroC", "CB": "LigOnly", "rsg_wt": "RsgWt", "rsg_int": "RsgInt", "XA": "Xa", "PC": "Pc"}
    MS = {"recall_1@1": "One", "recall_1@5": "Five", "recall_1@10": "Ten", "recall_1@20": "Twenty"}
    for split, tag in (("test", "Test"), ("dev", "Dev")):
        for pk, pv in BU[split].get("paired_recall_1@10", {}).items():
            nm = {"C0_minus_XA": "DualXa", "C0_minus_PC": "DualPc"}[pk]
            put(f"pair{nm}{tag}", pv["mean"])
            put(f"pair{nm}Lo{tag}", pv["ci"][0])
            put(f"pair{nm}Hi{tag}", pv["ci"][1])
        for c, cs in CS.items():
            if c not in BU[split]:
                continue
            for m, mt in MS.items():
                put(f"bud{tag}{mt}{cs}", BU[split][c][m]["mean"])
            put(f"bud{tag}EfOne{cs}", BU[split][c]["ef1@1"]["mean"], "{:.0f}")
            put(f"bud{tag}TenLo{cs}", BU[split][c]["recall_1@10"]["ci"][0])
            put(f"bud{tag}TenHi{cs}", BU[split][c]["recall_1@10"]["ci"][1])
            pt10 = BU[split][c]["recall_1@10"].get("per_target")
            if pt10:
                put(f"bud{tag}TenMin{cs}", min(pt10.values()))
                put(f"bud{tag}TenMax{cs}", max(pt10.values()))
            put(f"bud{tag}NinetyFive{cs}", 100 * BU[split][c]["recall95_budget"]["mean"], "{:.0f}")
# joint networks and the cost model (addendum 8)
JT = load("joint_timing.json")
CM = load("cost_model.json")
REG3 = ROOT / "runs/v3"
if JT:
    put("jointGpu", JT["gpu"].replace("NVIDIA GeForce ", ""))
    put("jointNLig", thousands(JT["n_lig"]))
    put("jointNPockets", str(JT["n_pockets"]))
if CM:
    names = {"dual": "Dual", "rsgh": "Rsgh", "pooled": "Pc", "xattn_cached": "XaCached", "xattn_naive": "XaNaive", "unidock_fast": "DockFast", "unidock_detail": "DockDetail"}
    for P, tag in (("57", "Lo"), ("1000", "Mid"), ("10000", "Hi")):
        row = CM["seconds_by_pockets"][P]
        for k, nm in names.items():
            put(f"cost{nm}{tag}", row[k], "{:.0f}" if row[k] >= 10 else "{:.1f}")
        for k, nm in (("xattn_cached", "XaCached"), ("xattn_naive", "XaNaive"), ("pooled", "Pc"), ("rsgh", "Rsgh")):
            r = row[k] / row["dual"]
            put(f"ratio{nm}{tag}", thousands(round(r)) if r >= 1000 else r, "{:.0f}" if r >= 10 else "{:.1f}")
    for k, nm in (("dual", "Dual"), ("rsgh", "Rsgh"), ("pooled", "Pc"), ("xattn_cached", "XaCached"), ("xattn_naive", "XaNaive")):
        v = CM["scoring_pairs_per_s"][k] / 1e6
        put(f"rate{nm}J", v, "{:.0f}" if v >= 10 else "{:.2f}")
if CM:
    put("costXaCachedHiHours", CM["seconds_by_pockets"]["10000"]["xattn_cached"] / 3600, "{:.0f}")
    put("costDockFastHiDays", CM["seconds_by_pockets"]["10000"]["unidock_fast"] / 86400, "{:.0f}")
    put("costDockFastMidDays", CM["seconds_by_pockets"]["1000"]["unidock_fast"] / 86400, "{:.0f}")
    put("costDualMidFold", CM["seconds_by_pockets"]["1000"]["unidock_fast"] / CM["seconds_by_pockets"]["1000"]["dual"], "{:.0f}")
if JT and "xattn_amp" in JT:
    amp = JT["xattn_amp"]
    put("ampGain", JT["xattn"]["score_cached_s"] / amp["score_cached_s"], "{:.1f}")
    put("ampRateXaJ", JT["n_lig"] * JT["n_pockets"] / amp["score_cached_s"] / 1e6, "{:.2f}")
    put("ampRatioDualJ", thousands(round((JT["n_lig"] * JT["n_pockets"] / JT["dual"]["score_s"]) / (JT["n_lig"] * JT["n_pockets"] / amp["score_cached_s"]))))
if REG3.exists() and list(REG3.glob("registry_*.json")):
    rr = {f.stem: json.loads(f.read_text()) for f in sorted(REG3.glob("registry_*.json"))}
    for kind, nm in (("xattn", "Xa"), ("pooled", "Pc")):
        ps = [v["params"] for v in rr.values() if v["model"] == kind]
        if ps:
            put("params" + nm, thousands(ps[0]))
        ups = [v["updates"] for v in rr.values() if v["model"] == kind]
        if ups:
            put("updates" + nm + "Max", thousands(max(ups)))
# comparison of the screening workflows on the development targets (ml_guided_baseline.py)
ML = load("ml_guided_dev.json")
if ML:
    zs = ML["zero_shot"]["rsgh_int"]
    zw = ML["zero_shot"]["rsgh_wt"]
    for b, tag in (("0.10", "Ten"), ("0.20", "Twenty")):
        put("mlZero" + tag + "Wt", float(np.mean(zw[b])))
        if "C0" in ML["zero_shot"]:
            put("mlZero" + tag + "Base", float(np.mean(ML["zero_shot"]["C0"][b])))
    put("mlLibrary", thousands(ML["n_library"]))
    for b, tag in (("0.10", "Ten"), ("0.20", "Twenty")):
        put("mlZero" + tag, float(np.mean(zs[b])))
    for f, tag in (("0.01", "One"), ("0.02", "Two"), ("0.05", "Five")):
        e = ML["ml_guided"][f]
        put("mlSample" + tag, thousands(e["sample"]))
        for b, bt in (("0.10", "Ten"), ("0.20", "Twenty")):
            put(f"ml{bt}{tag}", float(np.mean(e[b])))
        if tag == "Five":
            d = np.asarray(zs["0.10"]) - np.asarray(e["0.10"])
            bs = d[np.random.default_rng(0).integers(0, len(d), size=(5000, len(d)))].mean(1)
            put("mlDiffTenFive", float(d.mean()))
            put("mlDiffLoTenFive", float(np.percentile(bs, 2.5)))
            put("mlDiffHiTenFive", float(np.percentile(bs, 97.5)))
            put("mlWinsTenFive", str(int((d > 0).sum())))
            dw = np.asarray(zw["0.10"]) - np.asarray(e["0.10"])
            bw = dw[np.random.default_rng(0).integers(0, len(dw), size=(5000, len(dw)))].mean(1)
            put("mlDiffTenFiveWt", float(dw.mean()))
            put("mlDiffLoTenFiveWt", float(np.percentile(bw, 2.5)))
            put("mlDiffHiTenFiveWt", float(np.percentile(bw, 97.5)))
            put("mlWinsTenFiveWt", str(int((dw > 0).sum())))
            exc = [k for k, v in enumerate(dw) if v <= 0]
            if len(exc) == 1:
                tnm = pd.read_parquet(ROOT / "data/splits/targets.parquet")
                tnm = tnm[tnm.split_role == "dev"].target_id.tolist()
                put("mlExcTarget", tnm[exc[0]])
                put("mlExcZero", float(np.asarray(zw["0.10"])[exc[0]]))
                put("mlExcFive", float(np.asarray(e["0.10"])[exc[0]]))
# throughput on one RTX 4090 (the GPU class of the docking runs); second run, after CUDA warm-up
T = load("throughput/thr_rsg_run2.json") or load("throughput/throughput_4090_run2.json")
if T:
    sc, pk, lib, ind = T["scan"], T["pockets"], T["library"], T["independence"]
    dock_rate = DOCK_RATE
    total = lib["sec_featurise_cpu"] + lib["sec_embed_gpu_all_models"] + sc["sec_build_edited_graphs_cpu"] + sc["sec_state_encoding_all_models"] + sc["sec_queries_all_models"]
    put("scanLig", thousands(lib["featurised"]))
    put("scanEdits", str(sc["edits"]))
    put("scanPairs", sc["pairs"] / 1e6, "{:.1f}")
    put("scanQuerySec", sc["sec_queries_all_models"], "{:.0f}")
    put("scanFeatSec", lib["sec_featurise_cpu"], "{:.0f}")
    put("scanEmbSec", lib["sec_embed_gpu_all_models"], "{:.0f}")
    put("scanModels", str(T["models"]))
    put("scanAllSec", total, "{:.0f}")
    put("scanRate", sc["pairs"] / total / 1e3, "{:.0f}")
    put("dockRate", dock_rate, "{:.1f}")
    put("trainLabelDays", 39 * 40000 / dock_rate / 86400, "{:.1f}")
    put("trainLabelDaysHit", int(__import__("re").search(r"nTrainTargets\}\{(\d+)", OUT.read_text(encoding="utf-8")).group(1)) * 40000 / dock_rate / 86400, "{:.1f}")
    put("dockDays", sc["pairs"] / dock_rate / 86400, "{:.0f}")
    put("dockBillionDays", thousands(round(1e9 / dock_rate / 86400, -2)))
    if FAST_RATE:
        put("dockBillionDaysFast", thousands(round(1e9 / FAST_RATE / 86400, -1)))
    put("speedup", thousands(round(sc["pairs"] / dock_rate / total, -3)))
    put("nPocketsBench", str(pk["n_pockets"]))
    put("pocketPairs", pk["pairs"] / 1e6, "{:.1f}")
    pk_total = lib["sec_featurise_cpu"] + lib["sec_embed_gpu_all_models"] + pk["sec_state_encoding"] + pk["sec_scoring_all_models"]
    put("pocketSec", pk_total, "{:.0f}")
    put("pocketPerSec", pk["sec_per_pocket"], "{:.2f}")
    put("pocketDockDays", pk["pairs"] / dock_rate / 86400, "{:.0f}")
    put("pocketDockDaysShortlist", 0.1 * pk["pairs"] / dock_rate / 86400, "{:.1f}")
    put("pocketSpeedup", thousands(round(pk["pairs"] / dock_rate / pk_total, -3)))
    if FAST_RATE:
        for fam, tg in (("hit", "Hit"), ("c0", "Base")):
            fj = EV / "throughput" / f"thr_{fam}_run2.json"
            if fj.exists():
                jj = json.loads(fj.read_text())
                ttot = jj["library"]["sec_featurise_cpu"] + jj["library"]["sec_embed_gpu_all_models"] + jj["pockets"]["sec_state_encoding"] + jj["pockets"]["sec_scoring_all_models"]
                put("pocketSpeedupFast" + tg, thousands(round(pk["pairs"] / FAST_RATE / ttot, -2)))
                put("pocketSpeedup" + tg, thousands(round(pk["pairs"] / dock_rate / ttot, -2)))
        put("pocketDockDaysFast", pk["pairs"] / FAST_RATE / 86400, "{:.1f}")
        put("pocketSpeedupFast", thousands(round(pk["pairs"] / FAST_RATE / pk_total, -2)))
    put("gpuName", T["gpu"].replace("NVIDIA GeForce ", ""))
    mx = max(ind["embedding_batch1_vs_big"], ind["score_batch1_vs_big"], ind["score_permuted_batches64_vs_big"])
    mant, ex = f"{mx:.1e}".split("e")
    defs["indepMax"] = "$" + str(round(float(mant))) + "\\times10^{" + str(int(ex)) + "}$"
    put("indepRange", ind["score_range"], "{:.1f}")
H = load("scan/DHFR_rsg_int_ens.timing.json")
if H:
    put("laptopScanSec", H["sec_featurization_cpu"] + H["sec_ligand_embedding"] * H.get("models", 1) + H["sec_queries_all_models"], "{:.0f}")

# registered outcome 2 of addendum 8: reversal accuracy of the joint networks (dockmut/v3_reversal.py), and end-to-end cost ratios with the CPU featurization
JR = load("joint_reversal.json")
if JR:
    for split, sp in (("dev", "Dev"), ("test", "Test")):
        for cond, nm in (("XA", "Xa"), ("PC", "Pc")):
            put(f"jointJ{sp}{nm}", JR[split][f"mean_{cond}"])
FS = {k: load(f"throughput/thr_{k}_run2.json") for k in ("c0", "hit", "rsg")}
if all(FS.values()):
    fv = {k: v["library"]["sec_featurise_cpu"] for k, v in FS.items()}
    put("featSecBase", fv["c0"], "{:.0f}")
    put("featSecLo", min(fv.values()), "{:.0f}")
    put("featSecHi", max(fv.values()), "{:.0f}")
if CM and FS["c0"]:
    feat = FS["c0"]["library"]["sec_featurise_cpu"]
    for P, tag in (("57", "Lo"), ("10000", "Hi")):
        row = CM["seconds_by_pockets"][P]
        r = (feat + row["xattn_cached"]) / (feat + row["dual"])
        put(f"ratioFullXaCached{tag}", r, "{:.0f}")

prev = OUT.read_text(encoding="utf-8").splitlines() if OUT.exists() else []
keep = []
for ln in prev:
    if ln.startswith(MARK):
        break
    keep.append(ln)
lines = keep + [MARK]
for k in sorted(defs):
    lines.append("\\newcommand{\\%s}{%s}" % (k, defs[k]))
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(defs), "model macros written")
