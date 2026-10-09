"""Virtual alanine scan with a cached pocket state.

Every standard residue with a side chain is truncated to alanine in silico, and the trained surrogate predicts the
change of the score of a library of ligands for each edit. Ligand embeddings are computed once, every edited pocket
is a cheap re-encoding of the state, and all queries are independent. The script reports per-residue mean change,
measured residues for comparison, and throughput.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
sys.path.insert(0, str(ROOT / "src"))
from dm_core import SCORE_STD, collate_pockets, lig_graphs_for, load_pockets, to_dev  # noqa: E402
from dm_eval import load_model  # noqa: E402
from geometry import pocket_geometry  # noqa: E402
from make_mutants import STANDARD_AA, NO_TRUNCATE, read_box, read_receptor, residue_table, truncate  # noqa: E402
from pocket_graphs import graph_from_pdbqt  # noqa: E402
from pocketgate.data.graphbatch import collate_ligands  # noqa: E402


def all_single_edits(target: str, wt_dir: Path, conf_dir: Path, tmp: Path, geo_stats, max_d: float = 12.0, wt_geo=None):
    """Graph of every single-residue alanine truncation of a receptor (all standard residues with side chains)."""
    lines, atoms = read_receptor(wt_dir / f"{target}_target.pdbqt")
    box = read_box(conf_dir / f"{target}_conf.txt")
    rows, center = residue_table(atoms, box)
    graphs, meta = {}, []
    tmp.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r["res"] not in STANDARD_AA or r["res"] in NO_TRUNCATE or r["n_sc_heavy"] < 1 or not r["inside"] or not (r["d_sc"] <= max_d):
            continue
        f = tmp / f"{target}_{r['res']}{r['resid']}.pdbqt"
        f.write_text("".join(truncate(lines, r["remove_lines"])), newline="\n")
        g = graph_from_pdbqt(f, conf_dir / f"{target}_conf.txt", target, [(r["res"], r["resid"])])
        g = {"x": g["x"].float(), "edge_index": g["edge_index"], "edge_attr": g["edge_attr"].float(), "geo": g["geo"].float(),
             "n_residues": int(g["n_residues"])}
        if geo_stats is not None and wt_geo is not None:
            g["edit"] = torch.cat([(g["geo"] - wt_geo) / geo_stats[1],
                                   torch.tensor([len(r["remove_lines"]) / 20.0, 1 / 4.0, float(r["d_sc"]) / 15.0])]).float()
        if geo_stats is not None:
            g["geo"] = (g["geo"] - geo_stats[0]) / geo_stats[1]
        graphs[f"{r['res']}{r['resid']}"] = g
        meta.append({"residue": f"{r['res']}{r['resid']}", "d_sc": float(r["d_sc"]), "n_sc_heavy": r["n_sc_heavy"]})
    return graphs, pd.DataFrame(meta)


@torch.no_grad()
def scan(model, wt_graph, edit_graphs: dict, ligs: dict | None, ids: list[str], dev: str, chunk=16384, z=None, t_lig=0.0):
    model.eval().to(dev)
    if z is None:
        t0 = time.time()
        zs = []
        for s in range(0, len(ids), 2048):
            zs.append(model.lig_embed(to_dev(collate_ligands([ligs[i] for i in ids[s:s + 2048]]), dev)))
        z = torch.cat(zs)
        if dev == "cuda":
            torch.cuda.synchronize()
        t_lig = time.time() - t0
    names = list(edit_graphs)
    t1 = time.time()
    ctx = model.poc_embed(to_dev(collate_pockets([wt_graph] + [edit_graphs[n] for n in names]), dev))
    if dev == "cuda":
        torch.cuda.synchronize()
    t_state = time.time() - t1
    t2 = time.time()
    base = torch.cat([model.score_pairs(ctx, torch.zeros(min(chunk, len(ids) - s), dtype=torch.long, device=dev), z,
                                        torch.arange(s, min(s + chunk, len(ids)), device=dev)) for s in range(0, len(ids), chunk)])
    mean, absmean = [], []
    for k, n in enumerate(names):
        d = torch.cat([model.score_pairs(ctx, torch.full((min(chunk, len(ids) - s),), k + 1, dtype=torch.long, device=dev), z,
                                         torch.arange(s, min(s + chunk, len(ids)), device=dev)) for s in range(0, len(ids), chunk)])
        delta = ((d - base) * SCORE_STD).clamp(-4, 4)
        mean.append(float(delta.mean()))
        absmean.append(float(delta.abs().mean()))
    if dev == "cuda":
        torch.cuda.synchronize()
    t_query = time.time() - t2
    n_pairs = len(names) * len(ids)
    timing = {"ligands": len(ids), "edits": len(names), "pairs": n_pairs, "sec_ligand_embedding": t_lig,
              "sec_state_encoding": t_state, "sec_queries": t_query,
              "pairs_per_sec": n_pairs / max(t_query, 1e-9)}
    return pd.DataFrame({"residue": names, "mean_dS": mean, "mean_abs_dS": absmean}), timing



def _feat(item):
    from pocketgate.data.features import featurize_ligand
    return featurize_ligand(item)


@torch.no_grad()
def embed_library(models: list, smiles: list[str], dev: str, procs=10, chunk=16384):
    """Featurise a whole library once (CPU) and embed it with every model (GPU). Returns one embedding matrix per
    model and the timings of featurisation and of embedding for the whole list of models."""
    import multiprocessing as mp
    for m in models:
        m.eval().to(dev)
    t_feat = t_emb = 0.0
    zs = [[] for _ in models]
    n_fail = 0
    with mp.Pool(procs) as pool:
        for s in range(0, len(smiles), chunk):
            t0 = time.time()
            graphs = pool.map(_feat, smiles[s:s + chunk], chunksize=256)
            ok = [g for g in graphs if g is not None]
            n_fail += len(graphs) - len(ok)
            t_feat += time.time() - t0
            t1 = time.time()
            for q in range(0, len(ok), 4096):
                batch = to_dev(collate_ligands(ok[q:q + 4096]), dev)
                for k, m in enumerate(models):
                    zs[k].append(m.lig_embed(batch))
            if dev == "cuda":
                torch.cuda.synchronize()
            t_emb += time.time() - t1
    return [torch.cat(z) for z in zs], t_feat, t_emb, n_fail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="comma-separated model specs; predictions are averaged")
    p.add_argument("--target", required=True)
    p.add_argument("--n-ligands", type=int, default=2000)
    p.add_argument("--all-ligands", action="store_true")
    p.add_argument("--max-d", type=float, default=12.0)
    p.add_argument("--pockets", type=Path, default=ROOT / "dockmut/plan/pocket_graphs.pt")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
    pockets = load_pockets(a.pockets, plan)
    train = [t for t, v in plan["targets"].items() if v["role"] == "train"]
    raw = torch.load(a.pockets, weights_only=False)
    G = torch.stack([raw[t]["wt"]["geo"] for t in train])
    geo_stats = (G.mean(0), G.std(0).clamp(min=1e-6))
    models = [load_model(s)[0] for s in a.model.split(",")]
    wt_dir = ROOT / "dockmut/plan/wt_raw"
    graphs, meta = all_single_edits(a.target, wt_dir, ROOT / "data/raw/targets", ROOT / "dockmut/scan_tmp", geo_stats, a.max_d, raw[a.target]["wt"]["geo"])
    cfg = json.loads((ROOT / "configs/g2_config.json").read_text())
    smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
    preds, timings = [], []
    if a.all_ligands:
        sm = pd.read_parquet(ROOT / "data/processed/ligands.parquet", columns=["ligand_id", "standardized_smiles"])
        zs, t_feat, t_emb, n_fail = embed_library(models, sm.standardized_smiles.tolist(), dev)
        for m, z in zip(models, zs):
            pr, tm = scan(m, pockets[a.target]["wt"], graphs, None, list(range(len(z))), dev, z=z, t_lig=t_emb / len(models))
            preds.append(pr)
            timings.append(tm)
        extra = {"sec_featurization_cpu": t_feat, "featurization_failures": n_fail, "models": len(models)}
    else:
        ids = cfg["dev_ligands"][: a.n_ligands]
        ligs = lig_graphs_for(ids, ROOT / "data/processed/g2_ligand_graphs.pt", smiles)
        for m in models:
            pr, tm = scan(m, pockets[a.target]["wt"], graphs, ligs, ids, dev)
            preds.append(pr)
            timings.append(tm)
        extra = {"models": len(models)}
    pred = preds[0][["residue"]].copy()
    pred["mean_dS"] = np.mean([q.mean_dS.to_numpy() for q in preds], axis=0)
    pred["mean_abs_dS"] = np.mean([q.mean_abs_dS.to_numpy() for q in preds], axis=0)
    for k, q in enumerate(preds):
        pred[f"mean_dS_m{k}"] = q.mean_dS.to_numpy()
    pred = pred.merge(meta, on="residue")
    timing = dict(timings[0])
    timing.update(extra)
    timing["sec_queries_all_models"] = float(sum(t["sec_queries"] for t in timings))
    timing["edits_per_ligand_library"] = float(timing["edits"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.out, index=False)
    (a.out.with_suffix(".timing.json")).write_text(json.dumps(timing, indent=1))
    print(json.dumps(timing))


if __name__ == "__main__":
    main()
