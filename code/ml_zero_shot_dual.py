"""Zero-shot recall of the baseline dual encoders on the development library of the ML-guided comparison.

Adds zero_shot["C0"] (plain objective, hit logit, three seeds) to the json of ml_guided_baseline.py, on the same library
(26,009 development ligands with scores on all ten development targets) and with the same budgets and hit definition.
The development labels are read through the logged label interface (caller ml_zero_shot_dual); the test labels are not read.
Usage: python ml_zero_shot_dual.py eval/ml_guided_dev.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
sys.path.insert(0, str(ROOT / "src"))
from dm_baseline_logit import CKPT, load  # noqa: E402
from dm_core import lig_graphs_for, load_pockets  # noqa: E402
from dm_wt_eval import wt_scores  # noqa: E402
from pocketgate.data.labels import load_labels  # noqa: E402

BUDGETS = (0.05, 0.10, 0.20)
path = Path(sys.argv[1])
res = json.loads(path.read_text())

tg = pd.read_parquet(ROOT / "data/splits/targets.parquet")
targets = tg[tg.split_role == "dev"].target_id.tolist()
lsp = pd.read_parquet(ROOT / "data/splits/ligand_splits.parquet")
ids = lsp[(lsp.split_role == "dev") & (lsp.feature_status == "OK")].ligand_id.tolist()
smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
lab = load_labels("dev", "dev", caller="ml_zero_shot_dual")
lab = lab[lab.target_id.isin(targets) & lab.ligand_id.isin(set(ids))].dropna(subset=["score"])
Y = {t: g.set_index("ligand_id").score.reindex(ids) for t, g in lab.groupby("target_id")}
ids = [i for i in ids if all(not np.isnan(Y[t][i]) for t in targets)]
N = len(ids)
assert N == res["n_library"], (N, res["n_library"])
ys = {t: Y[t].reindex(ids).to_numpy() for t in targets}
hits = {t: ys[t] <= np.quantile(ys[t], 0.01) for t in targets}

dev = "cuda" if torch.cuda.is_available() else "cpu"
plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
pockets = load_pockets(ROOT / "dockmut/plan/pocket_graphs.pt", plan)
ligs = lig_graphs_for(ids, ROOT / "data/processed/g2_ligand_graphs.pt", smiles)
per_seed = {b: [] for b in BUDGETS}
for ck in CKPT["C0"]:
    model, _ = load(str(ROOT / ck))
    rec = {b: [] for b in BUDGETS}
    for t in targets:
        sc = wt_scores(model, pockets[t]["wt"], ligs, ids, dev)         # minus the hit logit, a lower value ranks first
        order = np.argsort(sc)
        for b in BUDGETS:
            sel = order[: int(round(b * N))]
            rec[b].append(float(hits[t][sel].sum() / hits[t].sum()))
    for b in BUDGETS:
        per_seed[b].append(rec[b])
res["zero_shot"]["C0"] = {f"{b:.2f}": np.mean(per_seed[b], axis=0).tolist() for b in BUDGETS}
print("C0", {f"{b:.2f}": round(float(np.mean(per_seed[b])), 3) for b in BUDGETS})
path.write_text(json.dumps(res, indent=1))
