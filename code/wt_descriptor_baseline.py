"""Wild-type screening with descriptor models (exploratory, post hoc, development split only).

A boosting model on eight ligand descriptors, optionally with the ten geometric descriptors of the wild-type pocket, is
fitted to the DOCKSTRING scores of the training targets and evaluated on the development targets with the frozen frames and
reversal quadruplets. It shows how much of the target-specific ranking of the neural surrogates is explained by ligand
size and pocket geometry. The test split is not read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
sys.path.insert(0, str(ROOT / "src"))
from dm_baselines import lig_desc  # noqa: E402
from dm_wt_eval import joint_reversal  # noqa: E402
from pocketgate.data.labels import load_labels, train_labels  # noqa: E402
from pocketgate.evaluation.metrics import macro_eval  # noqa: E402

out = Path(sys.argv[1])
N_PER_TARGET = 6000
plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
train_t = [t for t, v in plan["targets"].items() if v["role"] == "train"]
tg = pd.read_parquet(ROOT / "data/splits/targets.parquet")
dev_t = tg[tg.split_role == "dev"].target_id.tolist()
smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
pockets = torch.load(ROOT / "dockmut/plan/pocket_graphs.pt", weights_only=False)
geo = {t: pockets[t]["wt"]["geo"].numpy() for t in pockets}
rng = np.random.default_rng(0)
lab = train_labels(train_t, caller="wt_descriptor_baseline").dropna(subset=["score"])
lab = lab.groupby("target_id", group_keys=False).apply(lambda g: g.sample(min(len(g), N_PER_TARGET), random_state=0))
cache: dict = {}


def desc(ids):
    for i in ids:
        if i not in cache:
            cache[i] = lig_desc(smiles[i])
    return np.array([cache[i] for i in ids])


Xl = desc(lab.ligand_id.tolist())
Xg = np.stack([geo[t] for t in lab.target_id])
y = lab.score.to_numpy()
print("training rows", len(y), flush=True)
cfg = json.loads((ROOT / "configs/g2_config.json").read_text())
dev_ids = list(cfg["dev_ligands"])
dev_lab = load_labels("dev", "dev", caller="wt_descriptor_baseline_dev")
dev_lab = dev_lab[dev_lab.target_id.isin(dev_t) & dev_lab.ligand_id.isin(set(dev_ids))]
quads = pd.read_parquet(ROOT / "runs/v1/eval_quads_dev.parquet")
Xd = desc(dev_ids)
res = {}
for name, use_geo in (("wt_gbm_ligand_only", False), ("wt_gbm_geometry", True)):
    feats = np.hstack([Xl, Xg]) if use_geo else Xl
    m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=0).fit(feats, y)
    frames = []
    for t in dev_t:
        f = np.hstack([Xd, np.repeat(geo[t][None], len(dev_ids), 0)]) if use_geo else Xd
        frames.append(pd.DataFrame({"target_id": t, "ligand_id": dev_ids, "s_hat": m.predict(f)}))
    pred = pd.concat(frames)
    pred["p_hit"] = 1.0 / (1.0 + np.exp((pred.s_hat - pred.s_hat.mean()) / 1.0))
    df = pred.merge(dev_lab, on=["target_id", "ligand_id"]).dropna(subset=["score"])
    me = macro_eval(df)
    jr = joint_reversal(pred, quads)
    res[name] = {"recall_1@10": float(me["recall_1@10"].mean()), "spearman": float(me["spearman"].mean()), **jr}
    print(f"{name:20s} R1@10={res[name]['recall_1@10']:.3f} spearman={res[name]['spearman']:.3f} joint={jr['joint']:.3f} (n={jr['n_quads']})", flush=True)
out.write_text(json.dumps(res, indent=1))
