"""Variance decomposition of measured score changes.

For the contact truncations of one target the measured change matrix X (mutants x ligands) is split
into a mutant mean, a ligand main effect, a mutant-by-ligand interaction and replicate noise.
Each component is corrected for the noise that its mean or residual carries (mutant noise, and the noise of the shared wild-type mean for the ligand component).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
from dm_core import delta_table, read_results  # noqa: E402


def decompose(results_dir: Path, plan_dir: Path, roles: dict) -> pd.DataFrame:
    rows = []
    for f in sorted(results_dir.glob("*.csv")):
        t = f.stem
        man = json.loads((plan_dir / "receptors" / t / "manifest.json").read_text())
        tab, sigma, _ = delta_table(read_results(results_dir, t), man)
        nwt = 3 if roles[t] in ("dev", "test") else 2
        s2 = sigma ** 2                         # replicate variance of one docking score
        dsig2 = s2 * (1 + 1.0 / nwt)            # variance of a change: mutant noise plus the noise of the wild-type mean
        cm = [r for r, v in tab.items() if v[2] == "single_contact" or v[2].startswith("multi")]
        if len(cm) < 4:
            continue
        common = sorted(set.intersection(*[set(tab[r][0]) for r in cm]))
        if len(common) < 20:
            continue
        X = np.array([[dict(zip(tab[r][0], tab[r][1]))[l] for l in common] for r in cm])
        M, L = X.shape
        mu, lam, g = X.mean(1, keepdims=True), X.mean(0, keepdims=True), X.mean()
        tot = X.var()
        # The wild-type mean is shared by all mutants of a ligand, so its noise (s2 / nwt) is an additive ligand term: it
        # inflates the ligand means and cancels in the residuals, which carry the mutant noise s2 only.
        v_mut = max(((mu - g) ** 2).mean() - s2 / L, 0)
        v_lig = max(((lam - g) ** 2).mean() - s2 / M - s2 / nwt, 0)
        resid = X - mu - lam + g
        v_int = max((resid ** 2).sum() / ((M - 1) * (L - 1)) - s2, 0)
        rows.append({"target": t, "role": roles[t], "n_mut": M, "n_lig": L, "var_total": tot,
                     "mutant_mean": v_mut / tot, "ligand_main": v_lig / tot, "interaction": v_int / tot,
                     "noise": dsig2 / tot})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
    roles = {t: v["role"] for t, v in plan["targets"].items()}
    df = decompose(Path(sys.argv[1]), ROOT / "dockmut/plan", roles)
    print(df.round(3).to_string(index=False))
    print(df[["mutant_mean", "ligand_main", "interaction", "noise"]].mean().round(3).to_dict())
