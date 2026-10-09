"""Pilot analysis: docking noise, mutant effect sizes, and signal-to-noise by residue class."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(noise_csv: Path, mut_csv: Path, rec_dir: Path, out_json: Path):
    noise = pd.read_csv(noise_csv).dropna(subset=["score"])
    mut = pd.read_csv(mut_csv).dropna(subset=["score"])
    kinds = {}
    for t in mut.target.unique():
        m = json.loads((rec_dir / t / "manifest.json").read_text())
        for r in m["receptors"]:
            kinds[(t, r["id"])] = (r["kind"], [x["d_sc"] for x in r["residues"]], r["n_removed"])
    res = {"targets": {}}
    allrows = []
    for t in sorted(mut.target.unique()):
        nz = noise[noise.target == t].pivot(index="k", columns="seed", values="score").dropna()
        sd_seed = nz.std(axis=1, ddof=1)
        sigma = float(np.sqrt((sd_seed ** 2).mean()))
        wt_mean = nz.mean(axis=1)
        wt_ref = mut[(mut.target == t) & (mut.receptor == "wt")].set_index("k").score
        rows = []
        for rid, g in mut[(mut.target == t) & (mut.receptor != "wt")].groupby("receptor"):
            s = g.set_index("k").score
            common = s.index.intersection(wt_mean.index)
            d = (s[common] - wt_mean[common])
            kind, dsc, nrem = kinds[(t, rid)]
            rows.append({"target": t, "receptor": rid, "kind": kind, "n_removed": nrem,
                         "d_sc_min": min([x for x in dsc if x is not None], default=np.nan),
                         "n": len(d), "mean_dS": d.mean(), "sd_dS": d.std(ddof=1),
                         "frac_worse_0.5": float((d > 0.5).mean()),
                         "frac_better_0.5": float((d < -0.5).mean()),
                         "snr": float(d.std(ddof=1) / (sigma * np.sqrt(1 + 1.0 / nz.shape[1])))})
        df = pd.DataFrame(rows)
        allrows.append(df)
        by = df.groupby("kind")[["mean_dS", "sd_dS", "frac_worse_0.5", "frac_better_0.5", "snr"]].mean().round(3)
        res["targets"][t] = {"sigma_seed": round(sigma, 3), "n_seeds": int(nz.shape[1]),
                             "delta_sigma": round(sigma * np.sqrt(1 + 1.0 / nz.shape[1]), 3),
                             "by_kind": by.to_dict("index")}
        print(f"\n{t}: per-score docking noise sigma = {sigma:.3f} kcal/mol "
              f"(seeds {nz.shape[1]}, ligands {len(nz)}); delta-noise {sigma*np.sqrt(1+1/nz.shape[1]):.3f}")
        print(by.to_string())
    full = pd.concat(allrows)
    print("\nAll receptors, sorted by effect SD (top 12):")
    print(full.sort_values("sd_dS", ascending=False).head(12)[
        ["target", "receptor", "kind", "n_removed", "d_sc_min", "mean_dS", "sd_dS", "snr"]].round(3).to_string(index=False))
    out_json.write_text(json.dumps(res, indent=1))
    full.to_csv(out_json.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
