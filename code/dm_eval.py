"""Benchmark evaluation on held-out targets.

For every model the script predicts the docking-score change of each truncated receptor relative to
the wild-type pocket and compares it with the change measured by re-docking. Uncertainty comes
from bootstrapping targets, since targets, not ligand-mutant pairs, are the unit of generalisation.

Checkpoint kinds
  v2:<path>    checkpoint of the original study (DualEncoderNet)
  dm:<path>    checkpoint written by train_dm.py
  const:<path> v2 checkpoint evaluated with the constant synthetic pocket for every receptor
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
from dm_core import (SCORE_STD, delta_table, lig_graphs_for, load_pockets, metric_block, predict_delta,  # noqa: E402
                     read_results)
from dm_models import build  # noqa: E402
from train_dm import make_constant_pocket  # noqa: E402


class V2Adapter(torch.nn.Module):
    """Expose the original DualEncoderNet through the shared-state interface."""
    kind = "v2"

    def __init__(self, net):
        super().__init__()
        self.net = net

    def lig_embed(self, lb):
        from pocketgate.data.graphbatch import pad_flat
        from pocketgate.models.mpnn import masked_mean
        H = self.net.ligand_encoder(lb["x"], lb["edge_index"], lb["edge_attr"])
        Hp, m = pad_flat(H, lb["batch"], lb["n_atoms"], lb["n_graphs"])
        return masked_mean(Hp, m)

    def poc_embed(self, pb):
        from pocketgate.data.graphbatch import pad_flat
        from pocketgate.models.mpnn import masked_mean
        H = self.net.pocket_encoder(pb["x"], pb["edge_index"], pb["edge_attr"])
        Hp, m = pad_flat(H, pb["batch"], pb["n_res"], pb["n_graphs"])
        return {"pv": self.net.W_p(masked_mean(Hp, m))}

    def score_pairs(self, ctx, pidx, z, lidx):
        return self.net.score_head(z[lidx] * ctx["pv"][pidx]).squeeze(-1)


def load_model(spec: str):
    kind, path = spec.split(":", 1)
    path = Path(path)
    if kind in ("v2", "const"):
        from pocketgate.models.pocketgate import DualEncoderNet
        net = DualEncoderNet(dim=128)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        return V2Adapter(net), kind == "const"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    kname = ("rse" if "We.weight" in blob["state"] else ("rsg" if "Wg.weight" in blob["state"] else "rs")) if any(k.startswith("lig_off") for k in blob["state"]) else "b5"
    hit = any(k.startswith("lig_off_h") for k in blob["state"])
    m = build(kname, hit=True) if hit else build(kname)
    m.load_state_dict(blob["state"])
    return m, blob["cfg"].get("const_pocket") is not None


def groups(df: pd.DataFrame) -> dict:
    return {"contact_single": df[df.kind == "single_contact"],
            "contact_multi": df[df.kind.str.startswith("multi")],
            "contact_all": df[df.kind.isin(["single_contact"]) | df.kind.str.startswith("multi")],
            "null": df[df.kind.isin(["single_shell", "single_far"])]}


def detect_auc(df: pd.DataFrame) -> float:
    """AUROC of mean |predicted change| separating contact mutants from shell/far controls."""
    m = df.assign(a=df.pred.abs()).groupby(["target", "receptor", "kind"]).a.mean().reset_index()
    pos = m[m.kind.isin(["single_contact"]) | m.kind.str.startswith("multi")].a.to_numpy()
    neg = m[m.kind.isin(["single_shell", "single_far"])].a.to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def boot(df: pd.DataFrame, n=1000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    ts = df.target.unique()
    by = {t: g for t, g in df.groupby("target")}
    out = {k: [] for k in ("pooled_r", "mutant_r", "within_r", "gain")}
    for _ in range(n):
        pick = rng.choice(ts, size=len(ts), replace=True)
        g = pd.concat([by[t].assign(target=f"{t}_{i}") for i, t in enumerate(pick)])
        mb = metric_block(g)
        for k in out:
            out[k].append(mb[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True, help="tag=kind:path,tag=kind:path")
    p.add_argument("--targets", default="")
    p.add_argument("--roles", default="dev,test")
    p.add_argument("--results", type=Path, default=ROOT / "dockmut/results")
    p.add_argument("--plan", type=Path, default=ROOT / "dockmut/plan")
    p.add_argument("--pockets", type=Path, default=ROOT / "dockmut/plan/pocket_graphs.pt")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    plan = json.loads((a.plan / "plan.json").read_text())
    tg = [t for t, v in plan["targets"].items() if v["role"] in a.roles.split(",")]
    if a.targets:
        tg = [t for t in tg if t in a.targets.split(",")]
    pockets = load_pockets(a.pockets, plan)
    tables, sigma = {}, {}
    for t in tg:
        man = json.loads((a.plan / "receptors" / t / "manifest.json").read_text())
        tables[t], sigma[t], _ = delta_table(read_results(a.results, t), man)
    ids = sorted({i for t in tables for ids_, _, _ in tables[t].values() for i in ids_})
    smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
    ligs = lig_graphs_for(ids, ROOT / "data/processed/g2_ligand_graphs.pt", smiles)
    results, frames = {}, []
    for spec in a.models.split(","):
        tag, ms = spec.split("=", 1)
        model, use_const = load_model(ms)
        pk = pockets
        if use_const:
            const = make_constant_pocket(pockets, [t for t, v in plan["targets"].items() if v["role"] == "train"])
            pk = {t: {r: const for r in pockets[t]} for t in tg}
        df = pd.concat([predict_delta(model, pk, ligs, tables[t], t, dev) for t in tg])
        df["model"] = tag
        frames.append(df)
        res = {"n_targets": len(tg)}
        for name, g in groups(df).items():
            if len(g) < 20:
                continue
            res[name] = metric_block(g)
            if name == "contact_all":
                res[name]["ci95"] = boot(g, a.boot)
        res["detect_auc"] = detect_auc(df)
        results[tag] = res
        c = res["contact_all"]
        print(f"{tag:14s} pooled r={c['pooled_r']:.3f} mutant r={c['mutant_r']:.3f} within r={c['within_r']:.3f} "
              f"gain={c['gain']:.2f} predSD={c['pred_sd']:.3f} (true {c['true_sd']:.3f}) detectAUC={res['detect_auc']:.3f}",
              flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_parquet(a.out.with_suffix(".parquet"), index=False)
    # noise ceiling: reliability of the measured change
    allt = pd.concat(frames)[lambda d: d.model == next(iter(results))]
    cont = groups(allt)["contact_all"]
    sig = np.nanmean([sigma[t] for t in tg])
    dsig = sig * np.sqrt(1 + 1.0 / 3.0)
    rel = max(0.0, 1.0 - dsig ** 2 / max(cont.dS.var(), 1e-9))
    results["_noise"] = {"sigma_seed": float(sig), "delta_sigma": float(dsig), "reliability": float(rel),
                         "r_max": float(np.sqrt(rel))}
    a.out.write_text(json.dumps(results, indent=1))
    print("noise ceiling", results["_noise"])


if __name__ == "__main__":
    main()
