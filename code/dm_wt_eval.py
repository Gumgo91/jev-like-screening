"""Wild-type screening evaluation of DockMut models on unseen targets.

Uses the frozen evaluation frames and reversal quads of the original study so that numbers are
comparable with its baselines: development = 10 development targets x 10,000 development ligands,
test = 9 test targets x 26,015 test ligands (opened through the guarded label API and logged).
The ranking score is the predicted docking score (lower is better) mapped monotonically to (0, 1).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from dm_core import SCORE_STD, collate_pockets, lig_graphs_for, load_pockets, to_dev  # noqa: E402
from dm_eval import load_model  # noqa: E402
from pocketgate.data.graphbatch import collate_ligands  # noqa: E402
from pocketgate.evaluation.metrics import macro_eval  # noqa: E402


@torch.no_grad()
def wt_scores(model, pocket, ligs, ids, dev, chunk=1024, head="score"):
    model.eval().to(dev)
    z = []
    for s in range(0, len(ids), 1024):
        z.append(model.lig_embed(to_dev(collate_ligands([ligs[i] for i in ids[s:s + 1024]]), dev)))
    z = torch.cat(z)
    ctx = model.poc_embed(to_dev(collate_pockets([pocket]), dev))
    out = []
    for s in range(0, len(ids), chunk):
        li = torch.arange(s, min(s + chunk, len(ids)), device=dev)
        if head == "hit":       # minus the hit logit, so that a lower value is a better molecule like a docking score
            out.append((-model.hit_pairs(ctx, torch.zeros_like(li), z, li)).float().cpu())
        else:
            out.append(model.score_pairs(ctx, torch.zeros_like(li), z, li).float().cpu())
    return torch.cat(out).numpy() * (1.0 if head == "hit" else SCORE_STD)


def joint_reversal(pred: pd.DataFrame, quads: pd.DataFrame) -> dict:
    key = pred.set_index(["target_id", "ligand_id"])["p_hit"]
    q = quads.copy()
    for tgt, lig, col in [("target_a", "ligand_i", "fAi"), ("target_a", "ligand_j", "fAj"),
                          ("target_b", "ligand_i", "fBi"), ("target_b", "ligand_j", "fBj")]:
        q[col] = key.reindex(pd.MultiIndex.from_frame(q[[tgt, lig]])).to_numpy()
    q = q.dropna(subset=["fAi", "fAj", "fBi", "fBj"])
    qa, qb = q.fAi - q.fAj, q.fBi - q.fBj
    ok_a, ok_b = np.sign(qa) == -np.sign(q.d_a), np.sign(qb) == -np.sign(q.d_b)
    return {"joint": float((ok_a & ok_b).mean()), "n_quads": int(len(q))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--pockets", type=Path, default=ROOT / "dockmut/plan/pocket_graphs.pt")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--head", choices=["score", "hit"], default="score", help="readout used for ranking (hit: models with a hit head)")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads((ROOT / "configs/g2_config.json").read_text())
    tg = pd.read_parquet(ROOT / "data/splits/targets.parquet")
    pockets = load_pockets(a.pockets)
    smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
    if a.split == "dev":
        from pocketgate.data.labels import load_labels
        targets = tg[tg.split_role == "dev"].target_id.tolist()
        ids = [i for i in cfg["dev_ligands"]]
        lab = load_labels("dev", "dev", caller="dm_wt_eval_dev")
        quads = pd.read_parquet(ROOT / "runs/v1/eval_quads_dev.parquet")
    else:
        from pocketgate.data.labels import load_labels
        targets = tg[tg.split_role == "test"].target_id.tolist()
        lsp = pd.read_parquet(ROOT / "data/splits/ligand_splits.parquet")
        ids = lsp[(lsp.split_role == "test") & (lsp.feature_status == "OK")].ligand_id.tolist()
        lab = load_labels("test", "test", allow_test=True, caller="dm_wt_eval_test")
        quads = pd.read_parquet(ROOT / "runs/v2/test_quads_cold.parquet")
        full = pd.read_parquet(ROOT / "data/processed/ligands_full.parquet") if (ROOT / "data/processed/ligands_full.parquet").exists() else None
        smiles = pd.read_parquet(ROOT / "data/processed/ligand_full.parquet").set_index("ligand_id").standardized_smiles
    ligs = lig_graphs_for(ids, ROOT / "data/processed/g2_ligand_graphs.pt", smiles)
    ids = [i for i in ids if i in ligs and ligs[i] is not None]
    lab = lab[lab.target_id.isin(targets) & lab.ligand_id.isin(set(ids))]
    results, per_target = {}, []
    for spec in a.models.split(","):
        tag, ms = spec.split("=", 1)
        model, use_const = load_model(ms)
        if use_const:
            from train_dm import make_constant_pocket
            const = make_constant_pocket(pockets, [t for t in pockets if tg.set_index("target_id").loc[t, "split_role"] == "train"])
        frames = []
        for t in targets:
            pk = const if use_const else pockets[t]["wt"]
            s = wt_scores(model, pk, ligs, ids, dev, head=a.head)
            frames.append(pd.DataFrame({"target_id": t, "ligand_id": ids, "s_hat": s}))
        pred = pd.concat(frames)
        pred["p_hit"] = 1.0 / (1.0 + np.exp((pred.s_hat - pred.s_hat.mean()) / 1.0))
        df = pred.merge(lab, on=["target_id", "ligand_id"]).dropna(subset=["score"])
        me = macro_eval(df)
        me["model"] = tag
        per_target.append(me)
        jr = joint_reversal(pred, quads)
        results[tag] = {"recall_1@10": float(me["recall_1@10"].mean()), "recall_1@5": float(me["recall_1@5"].mean()),
                        "recall_1@1": float(me["recall_1@1"].mean()), "spearman": float(me["spearman"].mean()),
                        **jr}
        print(f"{tag:14s} R1@10={results[tag]['recall_1@10']:.3f} R1@5={results[tag]['recall_1@5']:.3f} "
              f"spearman={results[tag]['spearman']:.3f} joint={jr['joint']:.3f} (n={jr['n_quads']})", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1))
    pd.concat(per_target).to_csv(a.out.with_suffix(".per_target.csv"), index=False)


if __name__ == "__main__":
    main()
