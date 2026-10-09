"""Throughput of the Jev-like shared-state screener on one GPU (no labels are read).

Stage 1  independence: the score of a ligand does not depend on the other candidates in a batch (batch size 1, a random
         permutation in batches of 64 and one large batch give the same scores up to float rounding).
Stage 2  library x pockets: the 260,155 ligands are featurised and embedded once, then scored against the wild-type
         pocket of every target. The ligand embedding is shared by all pockets.
Stage 3  library x edits: virtual alanine scan of one target (every residue with a side chain within 14 A of the box center)
         over the whole library, with the ensemble of the given checkpoints.
Usage: python dm_throughput.py --models dm:a.pt,dm:b.pt,dm:c.pt --library lib.parquet --target DHFR --out out.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
sys.path.insert(0, str(ROOT / "src"))
from dm_core import collate_pockets, load_pockets, to_dev  # noqa: E402
from dm_eval import load_model  # noqa: E402
from dm_scan import _feat, all_single_edits, scan  # noqa: E402
from pocketgate.data.graphbatch import collate_ligands  # noqa: E402


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def embed(models, graphs, dev, chunk=4096):
    zs = [[] for _ in models]
    for q in range(0, len(graphs), chunk):
        batch = to_dev(collate_ligands(graphs[q:q + chunk]), dev)
        for k, m in enumerate(models):
            zs[k].append(m.lig_embed(batch))
    sync(dev)
    return [torch.cat(z) for z in zs]


HEAD = "score"


@torch.no_grad()
def score_all(model, ctx, z, dev, chunk):
    out = []
    for s in range(0, len(z), chunk):
        li = torch.arange(s, min(s + chunk, len(z)), device=dev)
        fn = model.hit_pairs if HEAD == "hit" else model.score_pairs
        out.append(fn(ctx, torch.zeros_like(li), z, li))
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--target", default="DHFR")
    p.add_argument("--n-pockets", type=int, default=0)
    p.add_argument("--n-library", type=int, default=0)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--head", choices=["score", "hit"], default="score", help="readout that is timed (hit needs a hit-head network)")
    p.add_argument("--skip-scan", action="store_true", help="time the library x pockets stages only (baselines have no edit graphs)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    global HEAD
    HEAD = a.head
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load_model(s)[0].eval().to(dev) for s in a.models.split(",")]
    plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
    pockets = load_pockets(ROOT / "dockmut/plan/pocket_graphs.pt", plan)
    res = {"gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "cpus": os.cpu_count(), "models": len(models), "specs": a.models.split(",")}

    lib = pd.read_parquet(a.library, columns=["ligand_id", "standardized_smiles"])
    if a.n_library:
        lib = lib.iloc[: a.n_library]
    smiles = lib.standardized_smiles.tolist()
    t0 = time.time()
    with mp.Pool(min(os.cpu_count() or 8, 32)) as pool:
        graphs = pool.map(_feat, smiles, chunksize=256)
    ok = [g for g in graphs if g is not None]
    res["library"] = {"smiles": len(smiles), "featurised": len(ok), "failures": len(smiles) - len(ok), "sec_featurise_cpu": time.time() - t0}
    print("featurised", res["library"], flush=True)

    # stage 1: independence of candidates
    n = min(4096, len(ok))
    tgt = a.target
    ctxs = [m.poc_embed(to_dev(collate_pockets([pockets[tgt]["wt"]]), dev)) for m in models]
    sub = ok[:n]
    diffs = {}
    with torch.no_grad():
        m0, c0 = models[0], ctxs[0]
        z_big = embed([m0], sub, dev)[0]
        s_big = score_all(m0, c0, z_big, dev, a.chunk)
        z1 = torch.cat([m0.lig_embed(to_dev(collate_ligands([g]), dev)) for g in sub[:256]])
        diffs["embedding_batch1_vs_big"] = float((z1 - z_big[:256]).abs().max())
        s1 = score_all(m0, c0, z1, dev, a.chunk)
        diffs["score_batch1_vs_big"] = float((s1 - s_big[:256]).abs().max())
        perm = np.random.default_rng(0).permutation(n)
        zp = torch.cat([embed([m0], [sub[i] for i in perm[q:q + 64]], dev)[0] for q in range(0, n, 64)])
        inv = np.argsort(perm)
        sp = score_all(m0, c0, zp, dev, a.chunk)[torch.as_tensor(inv, device=dev)]
        diffs["score_permuted_batches64_vs_big"] = float((sp - s_big).abs().max())
        diffs["score_range"] = float(s_big.max() - s_big.min())
    res["independence"] = diffs
    print("independence", diffs, flush=True)

    # stage 2: library embedded once, scored against every pocket
    sync(dev)
    t1 = time.time()
    zs = embed(models, ok, dev)
    sync(dev)
    res["library"]["sec_embed_gpu_all_models"] = time.time() - t1
    targets = [t for t in plan["targets"] if t in pockets]
    if a.n_pockets:
        targets = targets[: a.n_pockets]
    per_pocket = []
    with torch.no_grad():
        for t in targets:
            sync(dev)
            t2 = time.time()
            ctx_t = [m.poc_embed(to_dev(collate_pockets([pockets[t]["wt"]]), dev)) for m in models]
            sync(dev)
            t3 = time.time()
            for m, c, z in zip(models, ctx_t, zs):
                score_all(m, c, z, dev, a.chunk)
            sync(dev)
            per_pocket.append({"target": t, "sec_state": t3 - t2, "sec_score_all_models": time.time() - t3})
    tot_state = sum(x["sec_state"] for x in per_pocket)
    tot_score = sum(x["sec_score_all_models"] for x in per_pocket)
    pairs = len(ok) * len(targets)
    res["pockets"] = {"n_pockets": len(targets), "pairs": pairs, "sec_state_encoding": tot_state, "sec_scoring_all_models": tot_score,
                      "pairs_per_sec_scoring": pairs / max(tot_score, 1e-9), "per_model_pairs_per_sec": pairs * len(models) / max(tot_score, 1e-9),
                      "sec_per_pocket": (tot_state + tot_score) / max(len(targets), 1)}
    res["pockets"]["sec_end_to_end"] = res["library"]["sec_featurise_cpu"] + res["library"]["sec_embed_gpu_all_models"] + tot_state + tot_score
    print("pockets", res["pockets"], flush=True)

    if a.skip_scan:
        res["head"] = a.head
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        return
    # stage 3: virtual alanine scan of one target
    train = [t for t, v in plan["targets"].items() if v["role"] == "train"]
    raw = torch.load(ROOT / "dockmut/plan/pocket_graphs.pt", weights_only=False)
    G = torch.stack([raw[t]["wt"]["geo"] for t in train])
    geo_stats = (G.mean(0), G.std(0).clamp(min=1e-6))
    t4 = time.time()
    graphs_e, meta = all_single_edits(tgt, ROOT / "data/raw/targets", ROOT / "data/raw/targets", ROOT / "dockmut/scan_tmp", geo_stats, 14.0, raw[tgt]["wt"]["geo"])
    res["scan"] = {"target": tgt, "edits": len(graphs_e), "sec_build_edited_graphs_cpu": time.time() - t4}
    t5 = time.time()
    q_sec, state_sec = 0.0, 0.0
    for m, z in zip(models, zs):
        _, tm = scan(m, pockets[tgt]["wt"], graphs_e, None, list(range(len(z))), dev, chunk=a.chunk, z=z, t_lig=0.0)
        q_sec += tm["sec_queries"]
        state_sec += tm["sec_state_encoding"]
    res["scan"].update({"ligands": len(ok), "pairs": len(ok) * len(graphs_e), "sec_state_encoding_all_models": state_sec, "sec_queries_all_models": q_sec,
                        "pairs_per_sec_queries": len(ok) * len(graphs_e) / max(q_sec / len(models), 1e-9)})
    res["scan"]["sec_end_to_end"] = res["library"]["sec_featurise_cpu"] + res["library"]["sec_embed_gpu_all_models"] + state_sec + q_sec
    res["scan"]["pairs_per_sec_end_to_end"] = res["scan"]["pairs"] / res["scan"]["sec_end_to_end"]
    print("scan", res["scan"], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
