"""Summary of the family-grouped cross-validation (v4): how much do the networks use the pocket?

Reads runs/v4/{registry,metrics,pred}_<tag>.* written by scripts/v4_cv_train.py and the fold file data/splits/v4_cv_folds.json.
A (condition, recipe) pair is analysed when it has predictions for all five folds; several seeds are averaged per target (and per
fold for the fold-level quantities). Everything is computed from the saved files, no label cell is opened.

Per held-out target (49 targets in total):
  recall_1@1/5/10/20, spearman  from metrics_<tag>.csv (Recall of the true top 1% inside the best 1/5/10/20% of the predictions)
  sel_pearson, sel_spearman     selectivity: within the fold, every ligand's mean over the held-out targets is subtracted from the
                                predicted logit and from the negated true score; the correlation of the two centred vectors over the
                                target's ligands. The fold-level value pools all (target, ligand) pairs of the fold.
  reversal_acc                  up to 20,000 quadruplets per fold (default_rng(0)): targets a != b, ligands i != j, |d_a| >= 1,
                                |d_b| >= 1 and opposite signs of d_a = s_a(i) - s_a(j) and d_b = s_b(i) - s_b(j). A quadruplet is
                                correct when the predicted logit differences have both signs (lower score = higher logit). The
                                per-target value is the accuracy over the quadruplets that involve the target; the fold-level value
                                uses all quadruplets of the fold. A model that ignores the pocket scores 0 by construction.
  oracle_recall_1@10            shared-ranking oracle of v4_cv_folds.json (mean docking score over the training targets of the fold)
A zero-variance prediction has no defined correlation and is given selectivity 0.

Macro means over the 49 targets carry 95% percentile bootstrap intervals (5,000 resamples of the targets, default_rng(0)), overall
and for the low-headroom targets (oracle recall < 0.5). Paired differences use the same resamples for both conditions.
Pocket gain is a real-pocket condition minus its constant-pocket counterpart.

Usage: python dockmut/v4_cv_analyze.py [--runs runs/v4] [--out dockmut/eval/v4_cv_summary.json] [--folds PATH] [--selftest]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

N_FOLDS = 5
N_BOOT = 5000
MAX_QUADS = 20000
LOW_HEADROOM = 0.5
METRICS = ["recall_1@1", "recall_1@5", "recall_1@10", "recall_1@20", "spearman", "sel_pearson", "sel_spearman", "reversal_acc"]
LABEL = {"recall_1@1": "R@1%", "recall_1@5": "R@5%", "recall_1@10": "R@10%", "recall_1@20": "R@20%", "spearman": "Spearman",
         "sel_pearson": "Sel (r)", "sel_spearman": "Sel (rho)", "reversal_acc": "Reversal"}
COND_ORDER = ["dual", "pooled", "xattn", "dual_const", "pooled_const", "xattn_const"]
ARCH_COMPARE = ["dual_R0", "pooled_R0", "xattn_R0", "dual_R1", "xattn_R1"]


# --------------------------------------------------------------------------- helpers
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() < 1e-6 or y.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spear(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() < 1e-6 or y.std() < 1e-6:
        return 0.0
    return corr(rankdata(x), rankdata(y))


def key_of(cond: str, recipe: str) -> str:
    return f"{cond}_{recipe}"


# --------------------------------------------------------------------------- discovery
def discover(runs: Path, include_smoke: bool) -> list[dict]:
    items = []
    for p in sorted(runs.glob("registry_*.json")):
        reg = json.loads(p.read_text())
        tag = p.stem[len("registry_"):]
        if reg.get("smoke") and not include_smoke:
            print(f"skipping smoke run {tag}")
            continue
        m = re.match(r"^f(\d+)_(.+)_(R\d)_s(\d+)$", tag)
        if m:
            reg.setdefault("fold", int(m.group(1)))
            reg.setdefault("cond", m.group(2))
            reg.setdefault("recipe", m.group(3))
            reg.setdefault("seed", int(m.group(4)))
        mp, pp = runs / f"metrics_{tag}.csv", runs / f"pred_{tag}.parquet"
        if not (mp.exists() and pp.exists()):
            print(f"skipping {tag}: metrics or predictions missing")
            continue
        items.append({"tag": tag, "cond": reg["cond"], "recipe": reg["recipe"], "fold": int(reg["fold"]), "seed": int(reg["seed"]),
                      "metrics": mp, "pred": pp})
    return items


# --------------------------------------------------------------------------- per-fold statistics
class FoldData:
    """True scores of a fold as a targets x ligands matrix and its fixed quadruplets."""

    def __init__(self, heldout: list[str], pred: pd.DataFrame):
        self.targets = sorted(heldout)
        self.ligands = sorted(pred["ligand_id"].unique())
        S = pred.pivot(index="target_id", columns="ligand_id", values="score").reindex(index=self.targets, columns=self.ligands)
        self.S = S.to_numpy(float)
        self.quads = self._sample_quads()

    def _sample_quads(self):
        T, N = self.S.shape
        rng = np.random.default_rng(0)
        got = [[], [], [], []]
        n = 0
        for _ in range(200):
            a = rng.integers(0, T, 400000)
            b = (a + 1 + rng.integers(0, T - 1, 400000)) % T
            i = rng.integers(0, N, 400000)
            j = rng.integers(0, N, 400000)
            with np.errstate(invalid="ignore"):
                da, db = self.S[a, i] - self.S[a, j], self.S[b, i] - self.S[b, j]
                ok = (np.abs(da) >= 1) & (np.abs(db) >= 1) & (np.sign(da) != np.sign(db)) & (i != j)
            for g, v in zip(got, (a, b, i, j)):
                g.append(v[ok])
            n += int(ok.sum())
            if n >= MAX_QUADS:
                break
        a, b, i, j = (np.concatenate(g)[:MAX_QUADS] for g in got)
        da, db = self.S[a, i] - self.S[a, j], self.S[b, i] - self.S[b, j]
        return {"a": a, "b": b, "i": i, "j": j, "da": da, "db": db}

    def matrix(self, pred: pd.DataFrame, col: str) -> np.ndarray:
        return pred.pivot(index="target_id", columns="ligand_id", values=col).reindex(index=self.targets,
                                                                                      columns=self.ligands).to_numpy(float)

    def stats(self, pred: pd.DataFrame) -> dict:
        P = self.matrix(pred, "logit")
        S2 = self.matrix(pred, "score")
        assert np.allclose(S2, self.S, equal_nan=True), "true scores differ between runs of the same fold"
        T = len(self.targets)
        M = ~np.isnan(self.S) & ~np.isnan(P)
        M &= M.sum(0) >= 2
        cnt = np.maximum(M.sum(0), 1)
        Pc = np.where(M, P - np.where(M, P, 0).sum(0) / cnt, np.nan)
        Yc = np.where(M, -self.S - np.where(M, -self.S, 0).sum(0) / cnt, np.nan)
        sp_t = np.array([corr(Pc[t, M[t]], Yc[t, M[t]]) for t in range(T)])
        ss_t = np.array([spear(Pc[t, M[t]], Yc[t, M[t]]) for t in range(T)])
        q = self.quads
        pa, pb = P[q["a"], q["i"]] - P[q["a"], q["j"]], P[q["b"], q["i"]] - P[q["b"], q["j"]]
        right = (np.sign(pa) == -np.sign(q["da"])) & (np.sign(pb) == -np.sign(q["db"]))
        rev_t = np.array([right[(q["a"] == t) | (q["b"] == t)].mean() if ((q["a"] == t) | (q["b"] == t)).any() else np.nan
                          for t in range(T)])
        return {"sel_pearson_t": sp_t, "sel_spearman_t": ss_t, "reversal_acc_t": rev_t,
                "sel_pearson": corr(Pc[M], Yc[M]), "sel_spearman": spear(Pc[M], Yc[M]),
                "reversal_acc": float(right.mean()), "n_quads": int(len(right))}


def collect(items: list[dict], folds: dict) -> tuple[dict, dict, dict]:
    """-> per_target[key] DataFrame (targets x METRICS), per_fold[key][fold] dict, seeds[key][fold] list."""
    by = {}
    for it in items:
        by.setdefault(key_of(it["cond"], it["recipe"]), {}).setdefault(it["fold"], []).append(it)
    complete = {k: v for k, v in by.items() if set(v) == set(range(N_FOLDS))}
    incomplete = {k: sorted(v) for k, v in by.items() if k not in complete}
    fd_cache: dict[int, FoldData] = {}
    per_target, per_fold, seeds = {}, {}, {}
    for key, fv in sorted(complete.items()):
        rows, per_fold[key], seeds[key] = [], {}, {}
        for fold, its in sorted(fv.items()):
            held = folds[fold]["heldout"]
            tgt_vals, fstats = [], []
            for it in its:
                pred = pd.read_parquet(it["pred"])
                if fold not in fd_cache:
                    fd_cache[fold] = FoldData(held, pred)
                fd = fd_cache[fold]
                st = fd.stats(pred)
                m = pd.read_csv(it["metrics"]).set_index("target_id").reindex(fd.targets)
                assert m["recall_1@10"].notna().all(), f"{it['tag']}: metrics missing for held-out targets"
                df = m[["recall_1@1", "recall_1@5", "recall_1@10", "recall_1@20", "spearman"]].copy()
                df["sel_pearson"], df["sel_spearman"], df["reversal_acc"] = st["sel_pearson_t"], st["sel_spearman_t"], st["reversal_acc_t"]
                tgt_vals.append(df)
                fstats.append({k: st[k] for k in ("sel_pearson", "sel_spearman", "reversal_acc", "n_quads")})
            rows.append(pd.concat(tgt_vals).groupby(level=0).mean())
            per_fold[key][fold] = {k: float(np.mean([s[k] for s in fstats])) for k in fstats[0]}
            seeds[key][fold] = sorted(it["seed"] for it in its)
        per_target[key] = pd.concat(rows).sort_index()
        assert len(per_target[key]) == 49 and per_target[key].index.is_unique, f"{key}: expected 49 targets"
    return per_target, per_fold, {"seeds": seeds, "incomplete": incomplete}


# --------------------------------------------------------------------------- bootstrap
class Boot:
    def __init__(self, targets: list[str], low: np.ndarray):
        self.targets, self.low = targets, low
        self.idx_all = np.random.default_rng(0).integers(0, len(targets), size=(N_BOOT, len(targets)))
        self.idx_low = np.random.default_rng(0).integers(0, max(len(low), 1), size=(N_BOOT, max(len(low), 1)))

    def ci(self, v: np.ndarray, subset: str) -> dict:
        x, idx = (v, self.idx_all) if subset == "overall" else (v[self.low], self.idx_low)
        if len(x) == 0:
            return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_pos": float("nan"), "n": 0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            b = np.nanmean(x[idx], axis=1)
            lo, hi = np.nanpercentile(b, [2.5, 97.5])
            return {"mean": float(np.nanmean(x)), "lo": float(lo), "hi": float(hi), "p_pos": float((b > 0).mean()), "n": int(len(x))}


def summarize(per_target: dict, boot: Boot, oracle: pd.DataFrame) -> dict:
    out = {}
    for subset in ("overall", "low_headroom"):
        out.setdefault("oracle", {})[subset] = {m: boot.ci(oracle[m].to_numpy(float), subset)
                                                for m in ("oracle_recall_1@10", "oracle_spearman")}
    for key, df in per_target.items():
        out[key] = {s: {m: boot.ci(df[m].to_numpy(float), s) for m in METRICS} for s in ("overall", "low_headroom")}
    return out


def paired(a: str, b: str, per_target: dict, boot: Boot) -> dict:
    return {s: {m: boot.ci(per_target[a][m].to_numpy(float) - per_target[b][m].to_numpy(float), s) for m in METRICS}
            for s in ("overall", "low_headroom")}


# --------------------------------------------------------------------------- report
def fmt(c: dict, ci: bool = True) -> str:
    if c is None or c.get("mean") is None:
        return "NA"
    return f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]" if ci else f"{c['mean']:.3f}"


def markdown(res: dict) -> str:
    L = []
    meta = res["meta"]
    L.append("# Family-grouped cross-validation: pocket use\n")
    L.append(f"Runs: {meta['runs']}. Fold assignment sha256: {meta['fold_assignment_sha256']}. "
             f"Bootstrap: {N_BOOT} resamples of the targets (seed 0), 95% percentile intervals. "
             f"Low-headroom targets (oracle Recall_1%@10% < {LOW_HEADROOM}): {', '.join(meta['low_headroom_targets'])}.\n")
    keys = [k for k in res["conditions"]]
    for subset, title in (("overall", "All 49 held-out targets"), ("low_headroom", "Low-headroom targets")):
        L.append(f"## {title}: macro means\n")
        L.append("| condition | seeds | " + " | ".join(LABEL[m] for m in METRICS) + " |")
        L.append("|---|---|" + "---|" * len(METRICS))
        o = res["oracle"][subset]
        L.append(f"| shared-ranking oracle | - | NA | NA | {fmt(o['oracle_recall_1@10'])} | NA | {fmt(o['oracle_spearman'])} | NA | NA | NA |")
        for k in keys:
            c = res["conditions"][k]
            L.append(f"| {k} | {c['n_seeds_min']}-{c['n_seeds_max']} | " + " | ".join(fmt(c[subset][m]) for m in METRICS) + " |")
        L.append("")
    L.append("## Per fold (mean over the held-out targets of the fold; selectivity and reversal pool the fold)\n")
    L.append("| condition | fold | targets | R@10% | Spearman | Sel (r) | Sel (rho) | Reversal | quadruplets |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        for f, v in res["conditions"][k]["per_fold"].items():
            L.append(f"| {k} | {f} | {v['n_targets']} | {v['recall_1@10']:.3f} | {v['spearman']:.3f} | {v['sel_pearson']:.3f} | "
                     f"{v['sel_spearman']:.3f} | {v['reversal_acc']:.3f} | {int(v['n_quads'])} |")
    L.append("")
    for name, title in (("pocket_gain", "Pocket gain (real pocket minus constant pocket)"),
                        ("architecture", "Architecture differences (first minus second)")):
        for subset, st in (("overall", "all targets"), ("low_headroom", "low-headroom targets")):
            rows = [p for p in res["paired"] if p["kind"] == name]
            if not rows:
                continue
            L.append(f"## {title}, {st}\n")
            L.append("| contrast | " + " | ".join(LABEL[m] for m in METRICS) + " |")
            L.append("|---|" + "---|" * len(METRICS))
            for p in rows:
                L.append(f"| {p['a']} - {p['b']} | " + " | ".join(fmt(p[subset][m]) for m in METRICS) + " |")
            L.append("")
    if meta["incomplete"]:
        L.append("## Not analysed (folds with predictions)\n")
        for k, v in meta["incomplete"].items():
            L.append(f"{k}: folds {v}")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- main analysis
def analyze(runs: Path, folds_path: Path, include_smoke: bool = False) -> dict:
    fj = json.loads(folds_path.read_text())
    folds = {f["fold"]: f for f in fj["folds"]}
    items = discover(runs, include_smoke)
    assert items, f"no finished jobs in {runs}"
    per_target, per_fold_raw, extra = collect(items, folds)
    assert per_target, f"no condition has all {N_FOLDS} folds; found {extra['incomplete']}"

    orc = pd.DataFrame(fj["oracle"]["per_target"]).T
    orc = orc.rename(columns={"recall_1@10": "oracle_recall_1@10", "spearman": "oracle_spearman"})[["fold", "oracle_recall_1@10",
                                                                                                  "oracle_spearman"]].astype(float)
    targets = sorted(next(iter(per_target.values())).index)
    orc = orc.reindex(targets)
    assert orc["oracle_recall_1@10"].notna().all()
    low = np.where(orc["oracle_recall_1@10"].to_numpy() < LOW_HEADROOM)[0]
    boot = Boot(targets, low)

    res = {"meta": {"runs": str(runs), "folds_file": str(folds_path), "fold_assignment_sha256": fj["fold_assignment_sha256"],
                    "n_targets": len(targets), "n_boot": N_BOOT, "bootstrap_seed": 0, "low_headroom_threshold": LOW_HEADROOM,
                    "low_headroom_targets": [targets[i] for i in low], "max_quads_per_fold": MAX_QUADS,
                    "incomplete": extra["incomplete"], "seeds": extra["seeds"]}}
    summ = summarize(per_target, boot, orc)
    res["oracle"] = summ.pop("oracle")
    res["conditions"] = {}
    order = lambda k: (k.rsplit("_", 1)[1], COND_ORDER.index(k.rsplit("_", 1)[0]) if k.rsplit("_", 1)[0] in COND_ORDER else 99)
    for key in sorted(per_target, key=order):
        df = per_target[key].join(orc)
        pf = {}
        for f in range(N_FOLDS):
            ts = folds[f]["heldout"]
            pf[f] = {"n_targets": len(ts), "recall_1@10": float(df.loc[ts, "recall_1@10"].mean()),
                     "spearman": float(df.loc[ts, "spearman"].mean()), "oracle_recall_1@10": float(df.loc[ts, "oracle_recall_1@10"].mean()),
                     **per_fold_raw[key][f]}
        ns = [len(v) for v in extra["seeds"][key].values()]
        res["conditions"][key] = {**summ[key], "n_seeds_min": min(ns), "n_seeds_max": max(ns), "seeds_per_fold": extra["seeds"][key],
                                  "per_fold": pf,
                                  "per_target": {t: {m: float(df.loc[t, m]) for m in METRICS} | {"oracle_recall_1@10":
                                                 float(df.loc[t, "oracle_recall_1@10"]), "fold": int(df.loc[t, "fold"])}
                                                 for t in targets}}
    res["paired"] = []
    for cond in ("dual", "pooled", "xattn"):
        for recipe in ("R0", "R1"):
            a, b = key_of(cond, recipe), key_of(cond + "_const", recipe)
            if a in per_target and b in per_target:
                res["paired"].append({"kind": "pocket_gain", "a": a, "b": b, **paired(a, b, per_target, boot)})
    avail = [k for k in ARCH_COMPARE if k in per_target]
    for a, b in combinations(avail, 2):
        res["paired"].append({"kind": "architecture", "a": a, "b": b, **paired(a, b, per_target, boot)})
    return res


# --------------------------------------------------------------------------- self test on synthetic predictions
def make_fake_runs(folds_path: Path, runs: Path, n_lig: int = 600) -> None:
    from pocketgate.evaluation.metrics import macro_eval
    fj = json.loads(folds_path.read_text())
    strength = {"dual_R0": 0.3, "pooled_R0": 0.5, "xattn_R0": 0.8, "dual_R1": 0.4, "xattn_R1": 0.9, "dual_const_R0": 0.0,
                "pooled_const_R0": 0.0, "xattn_const_R0": 0.0, "pooled_R1": 0.5}
    seeds = {"dual_R0": [11, 22], "pooled_R0": [11], "xattn_R0": [11, 22, 33]}
    lig = [f"L{i:04d}" for i in range(n_lig)]
    rng = np.random.default_rng(5)
    shared = rng.normal(size=n_lig)
    v = rng.normal(size=(n_lig, 3))
    for key, w in strength.items():
        cond, recipe = key.rsplit("_", 1)
        for f in fj["folds"]:
            if key == "pooled_R1" and f["fold"] > 2:
                continue   # incomplete on purpose
            for seed in seeds.get(key, [11]):
                rows = []
                lig_noise = np.random.default_rng(seed).normal(0, 0.7, n_lig)   # shared by all targets: no pocket use
                for t in f["heldout"]:
                    u = np.random.default_rng(sum(map(ord, t))).normal(size=3)
                    spec = v @ u
                    score = -8.0 - 1.0 * shared - 0.8 * spec + np.random.default_rng(sum(map(ord, t)) + 1).normal(0, 0.5, n_lig)
                    eps = np.random.default_rng(seed * 1000 + sum(map(ord, t))).normal(0, 1.0, n_lig)
                    logit = shared + lig_noise + w * spec + (1.0 * eps if w > 0 else 0.0)
                    rows.append(pd.DataFrame({"target_id": t, "ligand_id": lig, "logit": logit.astype(np.float32), "score": score}))
                pred = pd.concat(rows, ignore_index=True)
                pred["p_hit"] = 1 / (1 + np.exp(-pred["logit"].to_numpy(float)))
                pred["score_pred"] = 0.0
                tag = f"f{f['fold']}_{cond}_{recipe}_s{seed}"
                pred[["target_id", "ligand_id", "logit", "p_hit", "score_pred", "score"]].to_parquet(runs / f"pred_{tag}.parquet", index=False)
                met = macro_eval(pred)
                met["fold"] = f["fold"]
                met.to_csv(runs / f"metrics_{tag}.csv", index=False)
                (runs / f"registry_{tag}.json").write_text(json.dumps({"cond": cond, "fold": f["fold"], "recipe": recipe, "seed": seed}))


def selftest(folds_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "runs"
        runs.mkdir()
        make_fake_runs(folds_path, runs)
        res = analyze(runs, folds_path)
        out = Path(tmp) / "summary.json"
        out.write_text(json.dumps(clean(res), indent=1))
        md = markdown(res)
        (Path(tmp) / "summary.md").write_text(md)
        c = res["conditions"]
        assert set(c) == {"dual_R0", "pooled_R0", "xattn_R0", "dual_R1", "xattn_R1", "dual_const_R0", "pooled_const_R0",
                          "xattn_const_R0"}, set(c)
        assert "pooled_R1" in res["meta"]["incomplete"]
        for k in ("dual_const_R0", "pooled_const_R0", "xattn_const_R0"):
            assert c[k]["overall"]["reversal_acc"]["mean"] == 0.0, c[k]["overall"]["reversal_acc"]
            assert abs(c[k]["overall"]["sel_pearson"]["mean"]) < 1e-9
        gains = {(p["a"]): p["overall"]["sel_pearson"]["mean"] for p in res["paired"] if p["kind"] == "pocket_gain"}
        assert gains["xattn_R0"] > gains["pooled_R0"] > gains["dual_R0"] > 0, gains
        assert c["xattn_R0"]["overall"]["reversal_acc"]["mean"] > c["dual_R0"]["overall"]["reversal_acc"]["mean"] > 0
        assert c["xattn_R0"]["per_fold"][0]["n_quads"] == MAX_QUADS and c["xattn_R0"]["per_fold"][4]["n_targets"] == 3
        assert len([p for p in res["paired"] if p["kind"] == "architecture"]) == 10
        assert json.loads(out.read_text()) and "| xattn_R0 |" in md
        print(md[:2500])
        print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/v4")
    ap.add_argument("--out", default="dockmut/eval/v4_cv_summary.json")
    ap.add_argument("--folds", default=str(ROOT / "data" / "splits" / "v4_cv_folds.json"))
    ap.add_argument("--include-smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run on synthetic predictions in a temporary folder")
    a = ap.parse_args()
    if a.selftest:
        selftest(Path(a.folds))
        return
    runs = Path(a.runs) if Path(a.runs).is_absolute() else ROOT / a.runs
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    res = analyze(runs, Path(a.folds), a.include_smoke)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean(res), indent=1))
    out.with_suffix(".md").write_text(markdown(res))
    print(f"wrote {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
