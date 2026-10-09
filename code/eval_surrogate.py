"""Interventional evaluation of a pocket-conditioned docking surrogate.

For every receptor variant of a target the surrogate predicts the change in docking
score relative to the wild-type pocket. The prediction is compared with the change
measured by re-docking the same ligands into the truncated receptor.

Reported quantities
  pooled r      : Pearson r over all (mutant, ligand) pairs
  mutant-level r: Pearson r between predicted and true mean change per mutant
  within r      : mean over mutants of the ligand-level correlation after removing
                  the mutant mean, the part that needs ligand-specific pocket reading
Ligands whose wild-type replicates disagree by more than UNSTABLE_SD or that fail to dock
(score >= 0) are dropped from the truth table, and changes are winsorised at +-4 kcal/mol.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dockmut"))
from pocketgate.data.features import featurize_ligand  # noqa: E402
from pocketgate.data.graphbatch import collate_ligands, collate_pockets  # noqa: E402
from pocketgate.engine import move  # noqa: E402
from pocketgate.models.pocketgate import DualEncoderNet  # noqa: E402

UNSTABLE_SD = 1.0
WINS = 4.0


def truth_table(noise_csv: Path, mut_csv: Path, target: str, man: dict) -> pd.DataFrame:
    noise = pd.read_csv(noise_csv).dropna(subset=["score"])
    mut = pd.read_csv(mut_csv).dropna(subset=["score"])
    wt = noise[noise.target == target].pivot(index="k", columns="seed", values="score")
    sd = wt.std(axis=1, ddof=1)
    stable = sd[(sd < UNSTABLE_SD) & (wt.mean(axis=1) < 0)].index
    wtm = wt.mean(axis=1)
    rows = []
    for rid, g in mut[(mut.target == target) & (mut.receptor != "wt")].groupby("receptor"):
        s = g.set_index("k").score
        for k in s.index.intersection(stable):
            rows.append((target, rid, man[rid]["kind"], int(k), float(np.clip(s[k] - wtm[k], -WINS, WINS))))
    return pd.DataFrame(rows, columns=["target", "receptor", "kind", "k", "dS"])


@torch.no_grad()
def predict(model, pocket_graphs: dict, lig_graphs: list, dev: str, sstd: float, head: str = "score"):
    model.eval().to(dev)
    lb = move(collate_ligands(lig_graphs), dev)
    out = {}
    for rid, pg in pocket_graphs.items():
        ctx = model.encode_pocket(move(collate_pockets([pg]), dev))
        o = model(ctx, lb)
        out[rid] = (o["score"] * sstd if head == "score" else o["logit"]).float().cpu().numpy()
    return out


def metrics(pred: dict, truth: pd.DataFrame, head: str):
    tr = truth.copy()
    tr["pred"] = [pred[r][k] - pred["wt"][k] for r, k in zip(tr.receptor, tr.k)]
    if head == "logit":
        tr["pred"] = -tr["pred"]  # higher logit = better = lower score
    res = {}
    for name, g in [("contact_single", tr[tr.kind == "single_contact"]),
                    ("contact_multi", tr[tr.kind.str.startswith("multi")]),
                    ("null_shell_far", tr[tr.kind.isin(["single_shell", "single_far"])])]:
        if len(g) < 10:
            continue
        pooled = float(np.corrcoef(g.pred, g.dS)[0, 1]) if g.pred.std() > 1e-9 else 0.0
        mm = g.groupby("receptor").agg(p=("pred", "mean"), t=("dS", "mean"))
        mlev = float(np.corrcoef(mm.p, mm.t)[0, 1]) if len(mm) > 2 and mm.p.std() > 1e-9 else float("nan")
        within = []
        for _, h in g.groupby("receptor"):
            hp, ht = h.pred - h.pred.mean(), h.dS - h.dS.mean()
            if hp.std() > 1e-9 and ht.std() > 1e-9:
                within.append(float(np.corrcoef(hp, ht)[0, 1]))
        res[name] = {"pooled_r": round(pooled, 3), "mutant_level_r": round(mlev, 3),
                     "within_r": round(float(np.mean(within)), 3) if within else float("nan"),
                     "pred_sd": round(float(g.pred.std()), 4), "true_sd": round(float(g.dS.std()), 3),
                     "n_pairs": int(len(g)), "n_mutants": int(g.receptor.nunique())}
    return res


def main():
    from pocketgate.common import PROCESSED_DIR  # noqa: F401
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads((ROOT / "dockmut/pilot_results/eval_cfg.json").read_text())
    lig = pd.read_csv(ROOT / "dockmut/ligands/pilot_dev96.csv")
    std = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
    graphs = [featurize_ligand(std[i]) for i in lig.inchikey]
    results = {}
    for tag, path in cfg["checkpoints"].items():
        model = DualEncoderNet(dim=128)
        model.load_state_dict(torch.load(ROOT / path, map_location="cpu"))
        results[tag] = {}
        for t in cfg["targets"]:
            from pocket_graphs import build_target_graphs
            man = {r["id"]: r for r in json.loads((ROOT / f"dockmut/receptors_pilot/{t}/manifest.json").read_text())["receptors"]}
            pg = build_target_graphs(t, ROOT / "dockmut/receptors_pilot", ROOT / "data/raw/targets")
            truth = truth_table(ROOT / "dockmut/pilot_results/pilot_noise.csv",
                                ROOT / "dockmut/pilot_results/pilot_mut.csv", t, man)
            for head in ("score", "logit"):
                pred = predict(model, pg, graphs, dev, cfg["sstd"], head)
                results[tag].setdefault(t, {})[head] = metrics(pred, truth, head)
    out = ROOT / "dockmut/pilot_results/eval_v2_baselines.json"
    out.write_text(json.dumps(results, indent=1))
    for tag, r in results.items():
        for t, hh in r.items():
            for head, m in hh.items():
                s = " | ".join(f"{k}: r={v['pooled_r']}, mut-level={v['mutant_level_r']}, within={v['within_r']}, predSD={v['pred_sd']}"
                               for k, v in m.items() if k != "null_shell_far")
                n = m.get("null_shell_far", {})
                print(f"{tag:8s} {t:6s} {head:5s} {s} | null predSD={n.get('pred_sd')}")


if __name__ == "__main__":
    main()
