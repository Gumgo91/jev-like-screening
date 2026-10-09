"""Summaries of the DockMut-1 re-docking data.

Outputs (JSON + CSV) used for Table 1, Figure 2 and the validation paragraphs:
  per target  : receptors, ligands, stable ligands, replicate noise sigma, reliability
  per class   : mean change, SD of change, signal-to-noise, fraction of ligands changed
  parity      : Uni-Dock wild type vs the DOCKSTRING table (train and dev targets only, so that no
                sealed test label is read)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
sys.path.insert(0, str(ROOT / "src"))
from dm_core import delta_table, read_results  # noqa: E402
from decomp import decompose  # noqa: E402


def cls(kind: str) -> str:
    if kind == "single_contact":
        return "single contact"
    if kind.startswith("multi"):
        return "multi contact"
    if kind == "single_shell":
        return "shell control"
    if kind == "single_far":
        return "far control"
    return kind


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "dockmut/results")
    p.add_argument("--plan", type=Path, default=ROOT / "dockmut/plan")
    p.add_argument("--out", type=Path, default=ROOT / "dockmut/analysis")
    p.add_argument("--parity", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((a.plan / "plan.json").read_text())
    rows_t, rows_m, parity = [], [], []
    labels = None
    if a.parity:
        from pocketgate.data.labels import load_labels
        labels = pd.concat([load_labels("train", "train", caller="dockmut_parity"),
                            load_labels("dev", "dev", caller="dockmut_parity"),
                            load_labels("train", "dev", caller="dockmut_parity")])
    for t, info in plan["targets"].items():
        f = a.results / f"{t}.csv"
        if not f.exists():
            continue
        df = read_results(a.results, t)
        man = json.loads((a.plan / "receptors" / t / "manifest.json").read_text())
        tab, sigma, n_stable = delta_table(df, man)
        nwt = df[df.receptor == "wt"].tag.nunique()
        dsig = sigma * np.sqrt(1 + 1.0 / max(nwt, 1)) if np.isfinite(sigma) else np.nan
        allv = np.concatenate([v[1] for v in tab.values()]) if tab else np.array([])
        rel = max(0.0, 1 - dsig ** 2 / allv.var()) if len(allv) and np.isfinite(dsig) else np.nan
        rows_t.append({"target": t, "role": info["role"], "n_receptors": len(man["receptors"]),
                       "n_ligands": info["n_ligands"], "n_stable": n_stable, "n_mutants": len(tab),
                       "sigma_seed": sigma, "delta_sigma": dsig, "reliability_all": rel,
                       "n_dockings_ok": int(len(df))})
        for rid, (ids, dS, kind) in tab.items():
            m = [r for r in man["receptors"] if r["id"] == rid][0]
            rows_m.append({"target": t, "role": info["role"], "receptor": rid, "class": cls(kind), "kind": kind,
                           "n_removed": m["n_removed"], "n_residues": len(m["residues"]),
                           "d_min": min([x["d_sc"] for x in m["residues"] if x["d_sc"] is not None], default=np.nan),
                           "n": len(ids), "mean_dS": float(dS.mean()), "sd_dS": float(dS.std(ddof=1)),
                           "sigma_delta": dsig, "snr": float(dS.std(ddof=1) / dsig) if dsig and np.isfinite(dsig) else np.nan,
                           "frac_changed": float((np.abs(dS) > 2 * dsig).mean()) if np.isfinite(dsig) else np.nan})
        if labels is not None and info["role"] != "test":
            wt = df[df.receptor == "wt"].groupby("inchikey").score.mean()
            ref = labels[labels.target_id == t].set_index("ligand_id").score
            j = wt.index.intersection(ref.index)
            if len(j) > 20:
                x, y = wt[j].to_numpy(), ref[j].to_numpy()
                ok = np.isfinite(x) & np.isfinite(y) & (y < 0)
                from scipy.stats import spearmanr
                parity.append({"target": t, "n": int(ok.sum()), "pearson": float(np.corrcoef(x[ok], y[ok])[0, 1]),
                               "spearman": float(spearmanr(x[ok], y[ok])[0]),
                               "bias": float((x[ok] - y[ok]).mean()), "mae": float(np.abs(x[ok] - y[ok]).mean())})
    T = pd.DataFrame(rows_t)
    M = pd.DataFrame(rows_m)
    T.to_csv(a.out / "per_target.csv", index=False)
    M.to_csv(a.out / "per_mutant.csv", index=False)
    agg = (M.groupby("class").agg(n_mutants=("receptor", "size"), mean_dS=("mean_dS", "mean"),
                                   sd_dS=("sd_dS", "mean"), snr=("snr", "mean"),
                                   frac_changed=("frac_changed", "mean"))).round(3)
    print(agg.to_string())
    print("targets", len(T), "dockings", int(T.n_dockings_ok.sum()), "median sigma", round(T.sigma_seed.median(), 3))
    summary = {"n_targets": int(len(T)), "n_dockings": int(T.n_dockings_ok.sum()),
               "median_sigma_seed": float(T.sigma_seed.median()), "by_class": agg.to_dict("index")}
    if parity:
        P = pd.DataFrame(parity)
        P.to_csv(a.out / "wt_parity.csv", index=False)
        summary["wt_parity"] = {"median_pearson": float(P.pearson.median()), "median_spearman": float(P.spearman.median()),
                                "median_bias": float(P.bias.median()), "median_mae": float(P.mae.median()),
                                "n_targets": int(len(P))}
        print(summary["wt_parity"])
    roles = {t: v["role"] for t, v in plan["targets"].items()}
    D = decompose(a.results, a.plan, roles)
    D.to_csv(a.out / "decomposition.csv", index=False)
    summary["decomposition_mean"] = D[["mutant_mean", "ligand_main", "interaction", "noise"]].mean().to_dict()
    summary["decomposition_by_group"] = {"train": D[D.role == "train"][["mutant_mean", "ligand_main", "interaction", "noise"]].mean().to_dict(),
                                         "heldout": D[D.role != "train"][["mutant_mean", "ligand_main", "interaction", "noise"]].mean().to_dict()}
    C = M[M["class"].isin(["single contact", "multi contact"])]
    summary["corr_sd_nremoved"] = float(C.sd_dS.corr(C.n_removed))
    summary["corr_sd_dmin"] = float(C.sd_dS.corr(C.d_min))
    summary["sigma_seed_min"] = float(T.sigma_seed.min())
    summary["sigma_seed_max"] = float(T.sigma_seed.max())
    summary["delta_sigma_median"] = float(T.delta_sigma.median())
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
