"""Write numbers.tex: every number quoted in the text is defined here from analysis outputs.

Re-run after any analysis changes. Macros for model results are added by make_numbers_models.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "dockmut/analysis"
OUT = Path(__file__).parent / "numbers.tex"

defs: dict[str, str] = {}


def put(name: str, value, fmt: str = "{:.2f}"):
    s = value if isinstance(value, str) else fmt.format(value)
    defs[name] = s.replace("-", "$-$")


def thousands(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


T = pd.read_csv(ANA / "per_target.csv")
M = pd.read_csv(ANA / "per_mutant.csv")
S = json.loads((ANA / "summary.json").read_text())
plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())

put("nTargets", str(len(T)))
put("nHeldTargets", str(int((T.role != "train").sum())))
put("nTrainTargets", str(int((T.role == "train").sum())))
put("nDock", thousands(S["n_dockings"]))
put("nPlanned", thousands(plan["total_dockings"]))
put("nEdits", thousands(len(M)))
put("sigmaMed", S["median_sigma_seed"])
put("sigmaMin", S["sigma_seed_min"])
put("sigmaMax", S["sigma_seed_max"])
put("dsigmaMed", S["delta_sigma_median"])
null = M[M["class"].isin(["shell control", "far control"])]
put("sdNull", null.sd_dS.mean())
put("fracNull", 100 * null.frac_changed.mean(), "{:.0f}")
for cls, key in (("single contact", "Single"), ("multi contact", "Multi")):
    g = M[M["class"] == cls]
    put("sd" + key, g.sd_dS.mean())
    put("frac" + key, 100 * g.frac_changed.mean(), "{:.0f}")
put("rNremoved", S["corr_sd_nremoved"])
put("rDmin", S["corr_sd_dmin"])
D = S["decomposition_mean"]
put("shareNoise", 100 * D["noise"], "{:.0f}")
put("shareMutant", 100 * D["mutant_mean"], "{:.0f}")
put("shareLigand", 100 * D["ligand_main"], "{:.0f}")
put("shareInter", 100 * D["interaction"], "{:.0f}")
put("shareSum", 100 * sum(D.values()), "{:.0f}")
put("ceilLig", float(np.sqrt((D["ligand_main"] + D["interaction"]) / (D["ligand_main"] + D["interaction"] + D["noise"]))))
G = pd.read_csv(ANA / "geometry/target_geometry.csv")
put("nNegSlope", str(int((G.slope_ha < 0).sum())))
put("nSlopeTargets", str(len(G)))
put("slopeMin", G.slope_ha.min(), "{:.3f}")
put("slopeMax", G.slope_ha.max(), "{:.3f}")
for name, col in (("rBuried", "buried_frac"), ("rAtomsTen", "n_atoms_r10")):
    put(name, float(np.corrcoef(G[col], G.slope_ha)[0, 1]))
geo_loo = json.loads((ANA / "geometry/geometry_loo.json").read_text())
put("geoLooR", geo_loo["loo_target_r2"])

C = pd.read_parquet(ANA / "contacts/contact_changes.parquet")
C["abs"] = C.dS.abs()
put("nChangesKept", thousands(len(C)))
sc = C[C.kind == "single_contact"]
put("contactNone", sc[sc.n_contact == 0]["abs"].mean())
put("contactMany", sc[sc.n_contact >= 6]["abs"].mean())
put("rPoseContact", float(np.corrcoef(sc.n_contact, sc["abs"])[0, 1]))
CM = M[M["class"].isin(["single contact", "multi contact"])]
put("nContactEdits", thousands(len(CM)))
SM = M[M["class"] == "single contact"]
put("rNremovedSingle", float(SM.sd_dS.corr(SM.n_removed)))
put("rDminSingle", float(SM.sd_dS.corr(SM.d_min)))
pt = sc.groupby(["target", sc.n_contact > 0])["abs"].mean().unstack()
put("nContactTargets", str(int((pt[True] > pt[False]).sum())))
put("nContactOf", str(int(pt.notna().all(axis=1).sum())))
ctrl = C[C.kind.isin(["single_shell", "single_far"]) & (C.n_contact == 0)]
put("controlNoContact", ctrl["abs"].mean())

P = pd.read_csv(ANA / "wt_parity.csv")
put("parityR", P.pearson.median())
put("parityMae", P.mae.median())
put("parityRMin", P.pearson.min())
put("parityRMax", P.pearson.max())
put("parityRMinTarget", str(P.loc[P.pearson.idxmin(), "target"]))
put("nParityBelow", str(int((P.pearson < 0.9).sum())))
put("parityBias", P.bias.median())
put("nParity", str(len(P)))
pilot = {"rLo": 0.97, "rHi": 0.98, "maeLo": 0.11, "maeHi": 0.13}
put("pilotRLo", pilot["rLo"])
put("pilotRHi", pilot["rHi"])
put("pilotMaeLo", pilot["maeLo"])
put("pilotMaeHi", pilot["maeHi"])

vina = ROOT / "dockmut/eval/vina_check/pairs.csv"
if vina.exists():
    V = pd.read_csv(vina)
    put("vinaR", float(np.corrcoef(V.dS_ud, V.dS_vina)[0, 1]))
    m = V.groupby(["target", "receptor"]).agg(ud=("dS_ud", "mean"), v=("dS_vina", "mean"))
    put("vinaEditR", float(np.corrcoef(m.ud, m.v)[0, 1]), "{:.3f}")
    put("vinaN", thousands(len(V)))
    put("vinaEdits", str(len(m)))
    V["ud_c"] = V.dS_ud - V.groupby(["target", "receptor"]).dS_ud.transform("mean")
    V["v_c"] = V.dS_vina - V.groupby(["target", "receptor"]).dS_vina.transform("mean")
    put("vinaWithinR", float(np.corrcoef(V.ud_c, V.v_c)[0, 1]))
    put("vinaLigPerEdit", str(int(V.groupby(["target", "receptor"]).size().median())))

# counts of the score table
RS = pd.concat([pd.read_csv(f, usecols=["target", "receptor", "score"]) for f in sorted((ROOT / "dockmut/results").glob("*.csv"))])
RS = RS[RS.score.notna()]
n_wt = int((RS.receptor == "wt").sum())
put("nScoresWt", thousands(n_wt))
put("nScoresChange", thousands(len(RS) - n_wt))
put("nNoScore", thousands(json.loads((ROOT / "dockmut/plan/plan.json").read_text())["total_dockings"] - len(RS)))
HOLD = M[(M["role"] != "train") & M["class"].isin(["single contact", "multi contact"])]
put("nPrimaryEdits", thousands(len(HOLD)))
put("nPrimaryPairs", thousands(int(HOLD.n.sum())))
# edits left for training after the 15% withhold, and the share of composite edits among the withheld contact edits
TRC = M[M["role"] == "train"].groupby("target").size()
put("editsLeftTwo", str(2 - int(round(0.15 * 2))))
put("editsLeftEight", str(8 - int(round(0.15 * 8))))
put("editsLeftTwelve", str(12 - int(round(0.15 * 12))))
put("editsLeftFull", str(int(round(np.mean([n - int(round(0.15 * n)) for n in TRC])))))
put("editsAllMean", str(int(round(TRC.mean()))))
TCM = M[(M["role"] == "train") & M["class"].isin(["single contact", "multi contact"])]
put("shareMultiTrain", 100 * (TCM["class"] == "multi contact").mean(), "{:.0f}")
# motivating numbers of the Introduction (result files of the preceding analysis)
ADD = json.loads((ROOT / "runs/v1/v1a_additive.json").read_text())
put("addVarShare", 100 * ADD["r2_insample"], "{:.0f}")
put("addTargetSd", ADD["target_effect_std"])
put("addLigandSd", ADD["ligand_effect_std"])
put("addRawSd", ADD["raw_std"])
SW = pd.read_csv(ROOT / "runs/v2/pocket_swap_eval.csv")
SW = SW[SW.condition == "C0"]
put("swapRhoMean", SW.spearman.mean(), "{:.3f}")
put("swapRhoMedian", SW.spearman.median(), "{:.3f}")
put("nSwapTargets", str(SW.target.nunique()))
# docking budget of a training set with 12 edits per target
PLAN = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
put("dockTwelveEdits", thousands(12 * PLAN["design"]["n_train_lig"]))
put("dockTwelveEditsTwentyfour", thousands(12 * PLAN["design"]["n_train_lig"] * 24))

lines = ["% Generated by make_numbers.py; do not edit."]
for k in sorted(defs):
    lines.append("\\newcommand{\\%s}{%s}" % (k, defs[k]))
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(defs), "macros written")
for k in sorted(defs):
    print(f"  {k} = {defs[k]}")
