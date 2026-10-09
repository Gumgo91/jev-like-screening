"""Zero-shot shared-state networks against a target-specific ML-guided docking workflow on the ten development targets.

Library: the 26,015 development ligands of every development target (their DOCKSTRING scores are the docking truth). The ML-guided
workflow docks a random sample, fits a CatBoost regressor on Morgan fingerprints to the sample scores, ranks the rest and docks the
top-ranked ligands until the total docked fraction (sample included) reaches the budget. The zero-shot networks need no docked
sample, so their whole budget goes to the top of their ranking. The metric is the top-1% recall of the docked set. The development
labels are read through the logged label interface (caller ml_guided_dev); the sealed test labels are not read.
Usage: python ml_guided_baseline.py out.json
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
from dm_core import lig_graphs_for, load_pockets  # noqa: E402
from dm_eval import load_model  # noqa: E402
from dm_wt_eval import wt_scores  # noqa: E402
from pocketgate.data.labels import load_labels  # noqa: E402

FRACS = (0.01, 0.02, 0.05)
BUDGETS = (0.05, 0.10, 0.20)
REPEATS = 5
rng = np.random.default_rng(0)

tg = pd.read_parquet(ROOT / "data/splits/targets.parquet")
targets = tg[tg.split_role == "dev"].target_id.tolist()
lsp = pd.read_parquet(ROOT / "data/splits/ligand_splits.parquet")
ids = lsp[(lsp.split_role == "dev") & (lsp.feature_status == "OK")].ligand_id.tolist()
smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
lab = load_labels("dev", "dev", caller="ml_guided_dev")
lab = lab[lab.target_id.isin(targets) & lab.ligand_id.isin(set(ids))].dropna(subset=["score"])
Y = {t: g.set_index("ligand_id").score.reindex(ids) for t, g in lab.groupby("target_id")}
ids = [i for i in ids if all(not np.isnan(Y[t][i]) for t in targets)]
N = len(ids)
print("library", N, "ligands,", len(targets), "targets", flush=True)


def recall_of(sel: np.ndarray, y: np.ndarray, h: np.ndarray) -> float:
    return float(h[sel].sum() / h.sum())


hits = {t: (Y[t].reindex(ids).to_numpy() <= np.quantile(Y[t].reindex(ids).to_numpy(), 0.01)) for t in targets}
ys = {t: Y[t].reindex(ids).to_numpy() for t in targets}

# zero-shot networks: hit logit of the hit-head networks, no docked sample
dev = "cuda" if torch.cuda.is_available() else "cpu"
pockets = load_pockets(ROOT / "dockmut/plan/pocket_graphs.pt")
ligs = lig_graphs_for(ids, ROOT / "data/processed/g2_ligand_graphs.pt", smiles)
res = {"n_library": N, "n_targets": len(targets), "zero_shot": {}, "ml_guided": {}}
for cond in ("rsgh_wt", "rsgh_int"):
    per_seed = {b: [] for b in BUDGETS}
    for s in (11, 22, 33):
        model, _ = load_model(f"dm:{ROOT / 'dockmut/runs_hit_pulled/runs_hit' / f'{cond}_s{s}.pt'}")
        rec = {b: [] for b in BUDGETS}
        for t in targets:
            sc = wt_scores(model, pockets[t]["wt"], ligs, ids, dev, head="hit")
            order = np.argsort(sc)                      # the hit logit is stored so that a lower value ranks first, as in dm_wt_eval
            for b in BUDGETS:
                rec[b].append(recall_of(order[: int(round(b * N))], ys[t], hits[t]))
        for b in BUDGETS:
            per_seed[b].append(rec[b])
    res["zero_shot"][cond] = {f"{b:.2f}": np.mean(per_seed[b], axis=0).tolist() for b in BUDGETS}     # per target, mean over seeds
    print(cond, {f"{b:.2f}": round(float(np.mean(per_seed[b])), 3) for b in BUDGETS}, flush=True)

# ML-guided workflow per target
from catboost import CatBoostRegressor  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

RDLogger.DisableLog("rdApp.*")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X = np.zeros((N, 2048), dtype=np.uint8)
for k, i in enumerate(ids):
    X[k] = gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles[i]))
for f in FRACS:
    s = int(round(f * N))
    per_t = {b: [] for b in BUDGETS}
    for t in targets:
        rr = {b: [] for b in BUDGETS}
        for r in range(REPEATS):
            samp = rng.choice(N, size=s, replace=False)
            m = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.1, loss_function="RMSE", verbose=0, thread_count=8, random_seed=r)
            m.fit(X[samp], ys[t][samp])
            rest = np.setdiff1d(np.arange(N), samp)
            pred = m.predict(X[rest])
            order = rest[np.argsort(pred)]
            for b in BUDGETS:
                k = max(int(round(b * N)) - s, 0)
                rr[b].append(recall_of(np.concatenate([samp, order[:k]]), ys[t], hits[t]))
        for b in BUDGETS:
            per_t[b].append(float(np.mean(rr[b])))
    res["ml_guided"][f"{f:.2f}"] = {"sample": s, **{f"{b:.2f}": per_t[b] for b in BUDGETS}}
    print(f"sample {f:.0%} ({s})", {f"{b:.2f}": round(float(np.mean(per_t[b])), 3) for b in BUDGETS}, flush=True)
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
