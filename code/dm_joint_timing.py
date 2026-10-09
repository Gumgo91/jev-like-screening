"""Cost of scoring a ligand library against many pockets: Jev-like networks against joint (non-factorized) networks on one GPU.

All networks are timed on the same ligands and the same pockets in float32, without labels. Ligand graphs come from the
cache and are on the GPU (featurization on the CPU is identical for every network and is timed in dm_throughput.py).
  dual    baseline dual encoder (Jev-like): ligand embedding once, pocket state once per pocket, dot-product readout
  rsgh    residue-sum network with hit head (Jev-like): same stages with the residue-sum readout
  pooled  PooledConcatNet (shared ligand embedding, pooled concatenation read by an MLP)
  xattn   PocketGate cross-attention network (joint): ligand atoms attend to the pocket residues
            naive   : the whole network is evaluated for every ligand-pocket pair
            cached  : the pocket-independent ligand atom states are computed once, the interaction layers and the head
                      are evaluated per pair (the best implementation of a joint network)
Every stage is run twice and the shorter time is kept (the first run absorbs kernel compilation and allocator warm-up).
Times are linear in the number of ligands and of pockets, so that the time of a library of 260,155 ligands against any number
of pockets follows from the per-stage times (make_cost_model.py).
Usage: python dm_joint_timing.py --n-lig 50000 --n-pockets 5 --xattn runs/v3/ckpt_v3_xattn_s11.pt --pooled runs/v3/ckpt_v3_pooled_s11.pt --out eval/joint_timing.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
sys.path.insert(0, str(ROOT / "src"))
from dm_core import collate_pockets, load_pockets, to_dev  # noqa: E402
from dm_eval import V2Adapter, load_model  # noqa: E402
from pocketgate.common import PROCESSED_DIR  # noqa: E402
from pocketgate.data.graphbatch import collate_ligands  # noqa: E402
from pocketgate.models.mpnn import masked_mean  # noqa: E402
from pocketgate.models.pocketgate import DualEncoderNet, PocketGate, PooledConcatNet, _ligand_pool  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


def best(fn, reps=2):
    """Shortest wall time of reps runs of fn (the result of the last run is returned with it)."""
    times, out = [], None
    for _ in range(reps):
        sync()
        t = time.perf_counter()
        out = fn()
        sync()
        times.append(time.perf_counter() - t)
    return min(times), out


def chunks(ids, ligs, size):
    return [to_dev(collate_ligands([ligs[i] for i in ids[q:q + size]]), dev) for q in range(0, len(ids), size)]


def _load_net(cls, path):
    net = cls(dim=128)
    if path:
        net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval().to(dev)


@torch.no_grad()
def time_dual(path, batches, pockets):
    m = V2Adapter(_load_net(DualEncoderNet, path)).eval().to(dev)
    t_emb, z = best(lambda: torch.cat([m.lig_embed(b) for b in batches]))
    t_state, states = best(lambda: [m.poc_embed(to_dev(collate_pockets([pk]), dev)) for pk in pockets])

    def score():
        li = torch.arange(len(z), device=dev)
        for st in states:
            m.score_pairs(st, torch.zeros_like(li), z, li)
    t_sc, _ = best(score)
    return {"embed_s": t_emb, "state_s": t_state, "score_s": t_sc}


@torch.no_grad()
def time_pooled(path, batches, pockets):
    net = _load_net(PooledConcatNet, path)

    def emb(b):
        H = net.ligand_encoder(b["x"], b["edge_index"], b["edge_attr"])
        return _ligand_pool(H, b)[2]
    t_emb, z = best(lambda: torch.cat([emb(b) for b in batches]))
    t_state, pools = best(lambda: [net.encode_pocket(to_dev(collate_pockets([pk]), dev))["pocket_pool"] for pk in pockets])

    def score():
        for pp in pools:
            for q in range(0, len(z), 65536):
                zz = z[q:q + 65536]
                net.head(torch.cat([zz, pp.expand(len(zz), -1)], dim=-1))
    t_sc, _ = best(score)
    return {"embed_s": t_emb, "state_s": t_state, "score_s": t_sc}


@torch.no_grad()
def time_xattn(path, batches, pockets):
    net = _load_net(PocketGate, path)
    t_state, ctxs = best(lambda: [net.encode_pocket(to_dev(collate_pockets([pk]), dev)) for pk in pockets])

    def naive():                                           # the whole network for every pair
        for ctx in ctxs:
            for b in batches:
                net(ctx, b)
    t_naive, _ = best(naive)

    def embed():                                           # pocket-independent ligand atom states
        out = []
        for b in batches:
            H = net.ligand_encoder(b["x"], b["edge_index"], b["edge_attr"])
            Hp, mask, _ = _ligand_pool(H, b)
            out.append((Hp, mask))
        return out
    t_emb, states = best(embed)

    def cached():                                          # interaction layers and head per pair
        for ctx in ctxs:
            KVs = [(K.squeeze(0), V.squeeze(0)) for K, V in ctx["kv"]]
            for Hp, mask in states:
                H2 = net.interaction(Hp, KVs, pocket_mask=ctx["pocket_mask"].squeeze(0), ligand_mask=mask)
                pool = masked_mean(H2, mask)
                net.head(torch.cat([pool, ctx["pocket_pool"].expand(len(pool), -1), pool], dim=-1))
    t_cached, _ = best(cached)
    return {"embed_s": t_emb, "state_s": t_state, "score_cached_s": t_cached, "score_naive_s": t_naive}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-lig", type=int, default=50000)
    p.add_argument("--n-pockets", type=int, default=5)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--dual", default=str(ROOT / "runs/v1/ckpt_v1_C0.pt"))
    p.add_argument("--xattn", default=None)
    p.add_argument("--pooled", default=None)
    p.add_argument("--rsgh", default=str(ROOT / "dockmut/runs_hit_pulled/runs_hit/rsgh_wt_s11.pt"))
    p.add_argument("--only", choices=["dual", "pooled", "xattn", "rsgh"], default=None, help="time one network per process (GPU memory of one network must not slow the next)")
    p.add_argument("--amp", action="store_true", help="time the cross-attention network under float16 autocast (key xattn_amp)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ligs = torch.load(PROCESSED_DIR / "g2_ligand_graphs.pt", weights_only=False)
    ids = sorted(ligs)[:a.n_lig]
    batches = chunks(ids, ligs, a.chunk)
    pk_all = torch.load(PROCESSED_DIR / "v1_pocket_graphs.pt", weights_only=False)
    pockets = [pk_all[t] for t in sorted(pk_all)[:a.n_pockets]]
    res = {"gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "n_lig": len(ids), "n_pockets": len(pockets), "chunk": a.chunk}
    if a.only in (None, "dual"):
        res["dual"] = time_dual(a.dual, batches, pockets)
        print("dual", res["dual"], flush=True)
    if a.only in (None, "pooled"):
        res["pooled"] = time_pooled(a.pooled, batches, pockets)
        print("pooled", res["pooled"], flush=True)
    if a.only in (None, "xattn"):
        if a.amp:
            with torch.autocast("cuda", dtype=torch.float16):
                res["xattn_amp"] = time_xattn(a.xattn, batches, pockets)
            print("xattn_amp", res["xattn_amp"], flush=True)
        else:
            res["xattn"] = time_xattn(a.xattn, batches, pockets)
            print("xattn", res["xattn"], flush=True)
    if a.only not in (None, "rsgh"):
        a.out.write_text(json.dumps(res, indent=1))
        return
    plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
    dm_pockets = load_pockets(ROOT / "dockmut/plan/pocket_graphs.pt", plan)
    model = load_model("dm:" + a.rsgh)[0].eval().to(dev)
    tn = sorted(dm_pockets)[:a.n_pockets]
    with torch.no_grad():
        t_emb, z = best(lambda: torch.cat([model.lig_embed(b) for b in batches]))
        t_state, ctxs = best(lambda: [model.poc_embed(to_dev(collate_pockets([dm_pockets[t]["wt"]]), dev)) for t in tn])

        def score():
            for ctx in ctxs:
                for s in range(0, len(z), 16384):
                    li = torch.arange(s, min(s + 16384, len(z)), device=dev)
                    model.hit_pairs(ctx, torch.zeros_like(li), z, li)
        t_sc, _ = best(score)
    res["rsgh"] = {"embed_s": t_emb, "state_s": t_state, "score_s": t_sc}
    print("rsgh", res["rsgh"], flush=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
