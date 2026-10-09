"""Cost model of a library-against-many-pockets screen from the per-stage times of dm_joint_timing.py.

For n ligands and p pockets the times are linear, so the time of L ligands against P pockets is
  Jev-like (dual, rsgh, pooled): L x embed + P x state + L x P x score        (the ligand embedding is shared by all pockets)
  cross-attention, cached ligand atom states: L x embed + P x state + L x P x score_cached
  cross-attention, naive: P x state + L x P x score_naive                     (the whole network for every pair)
  Uni-Dock: L x P / rate
Featurization of the ligands on the CPU is the same for every network and is reported separately (thr_*_run2.json).
Usage: python make_cost_model.py eval/joint_timing.json eval/cost_model.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "dockmut/eval"
T = json.loads(Path(sys.argv[1]).read_text())
nl, npk = T["n_lig"], T["n_pockets"]
L0, PS = 260155, [1, 10, 57, 100, 1000, 10000]


def jev(m, L, P):
    s = L / nl
    return s * m["embed_s"] + (P / npk) * m["state_s"] + s * (P / npk) * m["score_s"]


def xa_cached(m, L, P):
    s = L / nl
    return s * m["embed_s"] + (P / npk) * m["state_s"] + s * (P / npk) * m["score_cached_s"]


def xa_naive(m, L, P):
    s = L / nl
    return (P / npk) * m["state_s"] + s * (P / npk) * m["score_naive_s"]


dt = pd.concat([pd.read_csv(f) for f in (EV / "docking_timing.csv", EV / "docking_timing_big.csv", EV / "docking_timing_4k.csv") if f.exists()])
dt = dt[dt.warmup == 0].copy()
dt["rate"] = dt.n_ok / dt.sec
rate = {mode: float(dt[dt["mode"] == mode].groupby("n").rate.median().max()) for mode in ("detail", "fast")}

rows = {}
for P in PS:
    rows[str(P)] = {
        "dual": jev(T["dual"], L0, P), "rsgh": jev(T["rsgh"], L0, P), "pooled": jev(T["pooled"], L0, P),
        "xattn_cached": xa_cached(T["xattn"], L0, P), "xattn_naive": xa_naive(T["xattn"], L0, P),
        "unidock_fast": L0 * P / rate["fast"], "unidock_detail": L0 * P / rate["detail"],
    }
pairs_per_s = {
    "dual": nl * npk / T["dual"]["score_s"], "rsgh": nl * npk / T["rsgh"]["score_s"], "pooled": nl * npk / T["pooled"]["score_s"],
    "xattn_cached": nl * npk / T["xattn"]["score_cached_s"], "xattn_naive": nl * npk / T["xattn"]["score_naive_s"],
}
out = {"gpu": T["gpu"], "n_lig_timed": nl, "n_pockets_timed": npk, "library": L0, "seconds_by_pockets": rows, "scoring_pairs_per_s": pairs_per_s,
       "unidock_rate": rate, "stage_times": {k: T[k] for k in ("dual", "rsgh", "pooled", "xattn")}}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
for P in (57, 1000, 10000):
    r = rows[str(P)]
    print(P, {k: round(v, 1) for k, v in r.items()}, "xattn naive / dual =", round(r["xattn_naive"] / r["dual"]), "cached / dual =", round(r["xattn_cached"] / r["dual"], 1))
print({k: f"{v / 1e6:.2f}M pairs/s" for k, v in pairs_per_s.items()})
