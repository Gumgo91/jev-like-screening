"""Full-scale, end-to-end cost of screening the whole library against many pockets: Jev-like dual encoder against joint networks.

Nothing is extrapolated except the one labelled field of the original padded implementation (see below). Every block screens the
whole library (260,155 ligands, standardized SMILES) against P pockets and reports, for the same ligands and pockets:
  stage 1  CPU featurization of all SMILES with a process pool (measured once per process) and CPU collation into ligand groups
  stage 2  host-to-device transfer and ligand embedding (dual / pooled: ligand vector; xattn: ragged flat atom states of the whole
           library kept on the GPU)
  stage 3  pocket states of all pockets, encoded in batches of pockets (dual: pocket vector, pooled: pocket projection, xattn: per-pocket K/V)
  stage 4  scoring of all ligand-pocket pairs, then selection of the top 10% of the library per pocket with torch.topk (same for every network)
Networks (all networks use the same ligand groups, the same pockets and fp32 ligand encoders):
  dual                 DualEncoderNet, hit logit = (W_l pool_l . W_p pool_p) / logit_scale as one GEMM per chunk of pockets
  pooled               PooledConcatNet, first layer of the head split exactly into a ligand and a pocket projection (computed once),
                       combined per pair (relu of the sum, then the logit row of the second layer)
  xattn_padded_fp32    original implementation of dm_joint_timing.py (padded chunks of 1024 ligands, fp32), timed on a subset only
  xattn_fp32           PocketGate, ragged flat atom states, scaled_dot_product_attention, fp32 (TF32 off)
  xattn_fp16           PocketGate, ragged flat atom states, scaled_dot_product_attention, pure half weights and states for the
                       interaction layers and the head (the audit shows it is faster than autocast and rank-identical); --fp16-mode autocast
                       switches to autocast
P = 1000 is obtained by cycling the 57 wild-type pockets (repeated pockets, used only for timing; every pocket is encoded and
scored again, so the cost equals the cost of 1000 distinct pockets).
A correctness block (2,000 ligands x 5 pockets, every implementation against net(ctx, lb)['logit']) runs first and the script stops with
exit code 2 if fp32 max abs diff > 1e-3, fp32 Spearman < 0.9999 or fp16 Spearman < 0.999.
GPU stages are CUDA-synchronized before and after and the best of 2 repetitions is kept. end_to_end_s is the sum of the CPU stages
(featurization, collation) and the GPU stages; wallclock_s is one extra pass of the whole pipeline from SMILES with one timer.
The result JSON is appended after every (network, P) block, so an interrupted run is resumed with the same command (finished blocks are
skipped; --redo recomputes them, --fresh starts a new file). No labels are read.
Usage (pod):  python dockmut/v4_fair_timing.py --out dockmut/eval/v4_fair_timing.json
Small pocket file for the pod (16 MB instead of 386 MB):  python dockmut/v4_fair_timing.py --export-wt-pockets dockmut/plan/pocket_graphs_wt.pt
Smoke test:   python dockmut/v4_fair_timing.py --device cpu --n-lig 300 --n-pockets 3 --procs 2 --out tmp.json
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
sys.path.insert(0, str(ROOT / "src"))
from dm_core import collate_pockets, to_dev  # noqa: E402
from pocketgate.data.graphbatch import collate_ligands  # noqa: E402
from pocketgate.models.mpnn import masked_mean  # noqa: E402
from pocketgate.models.pocketgate import DualEncoderNet, PocketGate, PooledConcatNet, _ligand_pool  # noqa: E402

NETWORKS = ["dual", "pooled", "xattn_padded_fp32", "xattn_fp32", "xattn_fp16"]
LARGE_P = 1000
TOL_FP32_DIFF, TOL_FP32_RHO, TOL_FP16_RHO = 1e-3, 0.9999, 0.999


# --------------------------------------------------------------------------- helpers
def _feat(smiles):
    from pocketgate.data.features import featurize_ligand
    return featurize_ligand(smiles)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def timed(fn, dev, reps=2):
    """Shortest synchronized wall time of reps runs of fn, all rep times and the result of the last run."""
    times, out = [], None
    for _ in range(reps):
        out = None
        sync(dev)
        t = time.perf_counter()
        out = fn()
        sync(dev)
        times.append(time.perf_counter() - t)
    return min(times), times, out


def seg_mean(H, bidx, n_atoms, ng):
    """Mean of flat rows per graph (fp32 accumulation), (sumA,d) -> (ng,d)."""
    pool = torch.zeros(ng, H.shape[1], device=H.device, dtype=torch.float32)
    pool.index_add_(0, bidx, H.float())
    return pool / n_atoms.float().unsqueeze(1)


def _ranks(x):
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv.reshape(-1)]


def spearman(a, b):
    ra, rb = _ranks(np.asarray(a, np.float64)), _ranks(np.asarray(b, np.float64))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def sanitize(o):
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (Path, torch.device, torch.dtype)):
        return str(o)
    return o


def make_groups(graphs, size):
    return [collate_ligands(graphs[q:q + size]) for q in range(0, len(graphs), size)]


def make_pocket_batches(pockets, size):
    return [collate_pockets(pockets[q:q + size]) for q in range(0, len(pockets), size)]


def load_wt_pockets(path, n=0):
    """Wild-type pocket graphs (x, edge_index, edge_attr) of the targets in sorted order; accepts the full cache or an exported file."""
    raw = torch.load(path, weights_only=False)
    names = sorted(raw)
    if n:
        names = names[:n]
    out = []
    for t in names:
        g = raw[t]["wt"] if "wt" in raw[t] else raw[t]
        out.append({k: g[k] for k in ("x", "edge_index", "edge_attr")})
    return names, out


def load_net(cls, path, dev, allow_random=False):
    """Network with the checkpoint weights (or seeded random weights if allowed and the checkpoint is missing), eval mode."""
    if path is not None and Path(path).exists():
        net = cls(dim=128)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        src = "checkpoint"
    elif allow_random:
        torch.manual_seed(0)
        net = cls(dim=128)
        src = "random"
    else:
        raise FileNotFoundError(f"checkpoint {path} not found (use --allow-random-weights for a smoke test)")
    net = net.eval().to(dev)
    net.weights_source = src
    return net


# --------------------------------------------------------------------------- scorers
class Scorer:
    """embed(groups) -> ligand state, pocket_states(pocket_batches) -> pocket state, score(ls, ps, p0, p1) -> (p1-p0, N) fp32 logits."""
    name = ""
    kind = "fp32"            # tolerance class in the correctness block
    group_size = None        # ligands per group (None: --group)
    pocket_batch = None      # pockets per encoder call (None: --pocket-batch)

    def __init__(self, net, dev):
        self.net, self.dev = net, dev

    def ligand_pool(self, b):
        H = self.net.ligand_encoder(b["x"], b["edge_index"], b["edge_attr"])
        return seg_mean(H, b["batch"], b["n_atoms"], b["n_graphs"])


class DualGemm(Scorer):
    name = "dual_gemm"

    def embed(self, groups):
        return torch.cat([self.net.W_l(self.ligand_pool(to_dev(b, self.dev))) for b in groups])

    def pocket_states(self, pbatches):
        scale = self.net.logit_scale.clamp(min=1e-3)
        pv = torch.cat([self.net.encode_pocket(to_dev(pb, self.dev))["pocket_vec"] for pb in pbatches])
        return pv / scale

    def score(self, ls, ps, p0, p1):
        return ps[p0:p1] @ ls.T


class PooledDecomp(Scorer):
    """head([lig, pocket]) with the first layer split exactly: W1 [lig; pocket] + b1 = W1[:, :d] lig + (W1[:, d:] pocket + b1)."""
    name = "pooled_decomp"
    tile_pairs = 2 ** 20

    def __init__(self, net, dev):
        super().__init__(net, dev)
        lin1, lin2 = net.head.mlp[0], net.head.mlp[2]
        d = lin1.in_features // 2
        self.A, self.B = lin1.weight[:, :d].contiguous(), lin1.weight[:, d:].contiguous()
        self.b1, self.w2, self.b2 = lin1.bias, lin2.weight[0].contiguous(), lin2.bias[0]

    def embed(self, groups):
        return torch.cat([self.ligand_pool(to_dev(b, self.dev)) @ self.A.T for b in groups])

    def pocket_states(self, pbatches):
        pool = torch.cat([self.net.encode_pocket(to_dev(pb, self.dev))["pocket_pool"] for pb in pbatches])
        return pool @ self.B.T + self.b1

    def score(self, ls, ps, p0, p1):
        Zp = ps[p0:p1]
        out = torch.empty(p1 - p0, ls.shape[0], device=self.dev, dtype=torch.float32)
        step = max(1, self.tile_pairs // (p1 - p0))
        for l0 in range(0, ls.shape[0], step):
            H = ls[l0:l0 + step].unsqueeze(0) + Zp.unsqueeze(1)          # (Pc, Lc, hidden)
            out[:, l0:l0 + step] = torch.matmul(H.relu_(), self.w2) + self.b2
        return out


def _attn_layer(lyr, H, K, V):
    """CrossAttnLayer.forward (eval) on flat atom rows (N,d) of one pocket: K,V (heads,M,dh); no padding, no masks."""
    N = H.shape[0]
    h, dh = lyr.heads, lyr.dh
    Q = lyr.W_q(H).view(N, h, dh).transpose(0, 1)                      # (h,N,dh)
    C = F.scaled_dot_product_attention(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)).squeeze(0)
    C = C.transpose(0, 1).reshape(N, h * dh)
    out = lyr.ln1(H + lyr.W_o(C))
    return lyr.ln2(out + lyr.ffn(out))


class XattnRagged(Scorer):
    """mode: fp32 | fp16 (pure half weights, states, K/V) | fp16_autocast. The ligand and pocket encoders always run in fp32."""

    def __init__(self, net, dev, mode):
        super().__init__(net, dev)
        self.mode = mode
        self.autocast = mode == "fp16_autocast"
        self.kind = "fp32" if mode == "fp32" else "fp16"
        self.name = {"fp32": "xattn_fp32", "fp16": "xattn_fp16_pure", "fp16_autocast": "xattn_fp16_autocast"}[mode]
        self.model = copy.deepcopy(net).half().eval() if mode == "fp16" else net
        self.cdtype = torch.float16 if mode == "fp16" else torch.float32
        self.store_half = mode == "fp16"
        self.fallback = False

    def configure_states(self, total_atoms, budget_gb):
        """Atom states of the whole library are kept on the GPU in fp32, or in fp16 if they would exceed the budget."""
        need = total_atoms * self.net.ligand_encoder.inp.out_features * 4
        if self.mode != "fp16":
            self.store_half = need > budget_gb * 1e9
            self.fallback = self.store_half

    def embed(self, groups):
        out, n = [], 0
        for b in groups:
            b = to_dev(b, self.dev)
            H = self.net.ligand_encoder(b["x"], b["edge_index"], b["edge_attr"])
            if self.store_half:
                H = H.half()
            out.append({"H": H, "bidx": b["batch"], "na": b["n_atoms"].float(), "ng": b["n_graphs"]})
            n += b["n_graphs"]
        return {"groups": out, "n": n}

    def pocket_states(self, pbatches):
        out = []
        for pb in pbatches:
            n_res = pb["n_res"].tolist()
            ctx = self.net.encode_pocket(to_dev(pb, self.dev))
            for i, n in enumerate(n_res):
                kv = [(K[i, :, :n].to(self.cdtype).contiguous(), V[i, :, :n].to(self.cdtype).contiguous()) for K, V in ctx["kv"]]
                out.append({"kv": kv, "pool": ctx["pocket_pool"][i].to(self.cdtype)})
        return out

    def score(self, ls, ps, p0, p1):
        layers, head = self.model.interaction.layers, self.model.head
        out = torch.empty(p1 - p0, ls["n"], device=self.dev, dtype=torch.float32)
        ctx = torch.autocast(self.dev.type, dtype=torch.float16) if self.autocast else contextlib.nullcontext()
        with ctx:
            for j in range(p0, p1):
                pk, off = ps[j], 0
                for g in ls["groups"]:
                    x = g["H"] if self.autocast or g["H"].dtype == self.cdtype else g["H"].to(self.cdtype)
                    for lyr, (K, V) in zip(layers, pk["kv"]):
                        x = _attn_layer(lyr, x, K, V)
                    pool = seg_mean(x, g["bidx"], g["na"], g["ng"]).to(self.cdtype)
                    feats = torch.cat([pool, pk["pool"].expand(g["ng"], -1), pool], dim=-1)
                    out[j - p0, off:off + g["ng"]] = head(feats)["logit"]
                    off += g["ng"]
        return out


class XattnPadded(Scorer):
    """The implementation of dm_joint_timing.py (cached padded ligand states, chunks of 1024 ligands, fp32), one pocket at a time."""
    name = "xattn_padded_fp32"
    group_size = 1024
    pocket_batch = 1

    def embed(self, groups):
        states = []
        for b in groups:
            b = to_dev(b, self.dev)
            H = self.net.ligand_encoder(b["x"], b["edge_index"], b["edge_attr"])
            Hp, mask, _ = _ligand_pool(H, b)
            states.append((Hp, mask))
        return states

    def pocket_states(self, pbatches):
        return [self.net.encode_pocket(to_dev(pb, self.dev)) for pb in pbatches]

    def score(self, ls, ps, p0, p1):
        net, rows = self.net, []
        for ctx in ps[p0:p1]:
            KVs = [(K.squeeze(0), V.squeeze(0)) for K, V in ctx["kv"]]
            res = []
            for Hp, mask in ls:
                H2 = net.interaction(Hp, KVs, pocket_mask=ctx["pocket_mask"].squeeze(0), ligand_mask=mask)
                pool = masked_mean(H2, mask)
                res.append(net.head(torch.cat([pool, ctx["pocket_pool"].expand(len(pool), -1), pool], dim=-1))["logit"])
            rows.append(torch.cat(res))
        return torch.stack(rows)


def build_scorers(nets, dev):
    """nets: dict with keys dual / pooled / xattn (modules in eval mode) -> dict of all scorer variants."""
    s = {}
    if "dual" in nets:
        s["dual"] = DualGemm(nets["dual"], dev)
    if "pooled" in nets:
        s["pooled"] = PooledDecomp(nets["pooled"], dev)
    if "xattn" in nets:
        s["xattn_padded_fp32"] = XattnPadded(nets["xattn"], dev)
        s["xattn_fp32"] = XattnRagged(nets["xattn"], dev, "fp32")
        s["xattn_fp16_pure"] = XattnRagged(nets["xattn"], dev, "fp16")
        s["xattn_fp16_autocast"] = XattnRagged(nets["xattn"], dev, "fp16_autocast")
    return s


def score_all(scorer, ls, ps, P, N, frac, chunk, dev, keep=False):
    """Scores of all pairs in chunks of pockets, then top-frac selection per pocket (frac None: no selection).
    -> (seconds scoring, seconds selection, indices (or scores if frac is None) if keep)."""
    k = None if frac is None else max(1, math.ceil(frac * N))
    t_sc = t_tk = 0.0
    kept = []
    for p0 in range(0, P, chunk):
        p1 = min(P, p0 + chunk)
        sync(dev)
        t = time.perf_counter()
        S = scorer.score(ls, ps, p0, p1)
        sync(dev)
        t_sc += time.perf_counter() - t
        if k is not None:
            t = time.perf_counter()
            _, idx = torch.topk(S, k, dim=1)
            sync(dev)
            t_tk += time.perf_counter() - t
            if keep:
                kept.append(idx)
        elif keep:
            kept.append(S)
        del S
    return t_sc, t_tk, (torch.cat(kept) if keep else None)


# --------------------------------------------------------------------------- correctness
@torch.no_grad()
def reference_logits(net, graphs, pockets, dev, chunk=500):
    """net(ctx, lb)['logit'] of the module for every pocket (single-pocket ctx) and every ligand: (P,N) float64."""
    rows = []
    for pk in pockets:
        ctx = net.encode_pocket(to_dev(collate_pockets([pk]), dev))
        r = [net(ctx, to_dev(collate_ligands(graphs[q:q + chunk]), dev))["logit"].float().cpu() for q in range(0, len(graphs), chunk)]
        rows.append(torch.cat(r))
    return torch.stack(rows).double().numpy()


def compare(S, ref, frac=0.1):
    k = max(1, math.ceil(frac * ref.shape[1]))
    rhos = [spearman(S[p], ref[p]) for p in range(ref.shape[0])]
    ov = [len(set(np.argsort(-S[p], kind="stable")[:k]) & set(np.argsort(-ref[p], kind="stable")[:k])) / k for p in range(ref.shape[0])]
    return {"max_abs_diff": float(np.abs(S - ref).max()), "spearman_min": float(np.min(rhos)), "spearman_mean": float(np.mean(rhos)),
            "spearman_pooled": spearman(S.ravel(), ref.ravel()), "top10_overlap_min": float(np.min(ov)),
            "finite": bool(np.isfinite(S).all()), "logit_min": float(ref.min()), "logit_max": float(ref.max())}


def verdict(kind, r):
    if kind == "fp32":
        ok = r["finite"] and r["max_abs_diff"] <= TOL_FP32_DIFF and r["spearman_min"] >= TOL_FP32_RHO
        tol = {"max_abs_diff": TOL_FP32_DIFF, "spearman_min": TOL_FP32_RHO}
    else:
        ok = r["finite"] and r["spearman_min"] >= TOL_FP16_RHO
        tol = {"spearman_min": TOL_FP16_RHO}
    return bool(ok), tol


@torch.no_grad()
def check_scorers(scorers, graphs, pockets, dev, group=500, pocket_batch=2, frac=0.1, gate=None):
    """Every scorer against the module forward on the given ligands x pockets. -> (results, names of failed gating scorers)."""
    refs, results, failed = {}, {}, []
    for name, sc in scorers.items():
        try:
            if id(sc.net) not in refs:
                refs[id(sc.net)] = reference_logits(sc.net, graphs, pockets, dev)
            ref = refs[id(sc.net)]
            ls = sc.embed(make_groups(graphs, sc.group_size or group))
            ps = sc.pocket_states(make_pocket_batches(pockets, sc.pocket_batch or pocket_batch))
            S = sc.score(ls, ps, 0, len(pockets)).float().cpu().numpy().astype(np.float64)
            r = compare(S, ref, frac)
            r["pass"], r["tolerance"] = verdict(sc.kind, r)
            r["kind"] = sc.kind
            # the same scores through the chunked scoring + top-k path used in the timing blocks
            if name != "xattn_padded_fp32":
                ls_n = ls.shape[0] if torch.is_tensor(ls) else ls["n"]
                _, _, idx = score_all(sc, ls, ps, len(pockets), ls_n, frac, 2, dev, keep=True)
                k = idx.shape[1]
                sel = [len(set(idx[p].tolist()) & set(np.argsort(-S[p], kind="stable")[:k].tolist())) / k for p in range(len(pockets))]
                r["topk_path_overlap_min"] = float(min(sel))
        except Exception as e:                                           # an implementation that cannot run here is reported
            r = {"pass": False, "error": f"{type(e).__name__}: {e}", "kind": sc.kind}
        results[name] = r
        if gate is None or name in gate:
            if not r["pass"]:
                failed.append(name)
    return results, failed


# --------------------------------------------------------------------------- benchmark state
class Bench:
    def __init__(self, a, dev, pool, smiles, pockets):
        self.a, self.dev, self.pool, self.smiles, self.pockets = a, dev, pool, smiles, pockets
        self.graphs, self.feat_s, self.feat_fail, self.library_size = None, None, 0, 0
        self._groups, self.atoms = {}, None

    def featurize(self):
        t = time.perf_counter()
        graphs = [_feat(s) for s in self.smiles] if self.pool is None else self.pool.map(_feat, self.smiles, chunksize=256)
        dt = time.perf_counter() - t
        ok = [g for g in graphs if g is not None]
        return ok, dt, len(graphs) - len(ok)

    def run_featurization(self):
        self.graphs, self.feat_s, self.feat_fail = self.featurize()
        self.atoms = np.array([g["x"].shape[0] for g in self.graphs])

    def groups(self, size, n):
        """CPU collation of the first n ligands into groups of `size` (built once per (size, n), seconds recorded)."""
        key = (size, n)
        if key not in self._groups:
            t = time.perf_counter()
            g = make_groups(self.graphs[:n], size)
            self._groups[key] = (g, time.perf_counter() - t)
        return self._groups[key]

    def wallclock(self, scorer, P):
        """One pass of the whole pipeline from SMILES with a single timer (featurization, collation, transfer, embedding, states, scoring, top-k)."""
        a, dev = self.a, self.dev
        sync(dev)
        t0 = time.perf_counter()
        graphs, _, _ = self.featurize()
        groups = make_groups(graphs, scorer.group_size or a.group)
        pbs = make_pocket_batches([self.pockets[i % len(self.pockets)] for i in range(P)], scorer.pocket_batch or a.pocket_batch)
        ls = scorer.embed(groups)
        ps = scorer.pocket_states(pbs)
        score_all(scorer, ls, ps, P, len(graphs), a.topk_frac, a.pocket_chunk, dev)
        sync(dev)
        return time.perf_counter() - t0


def run_block(bench, name, scorer, P):
    a, dev = bench.a, bench.dev
    padded = name == "xattn_padded_fp32"
    n_lig = min(a.padded_n_lig, len(bench.graphs)) if padded else len(bench.graphs)
    groups, collate_s = bench.groups(scorer.group_size or a.group, n_lig)
    unique = len(bench.pockets)
    t = time.perf_counter()
    pbs = make_pocket_batches([bench.pockets[i % unique] for i in range(P)], scorer.pocket_batch or a.pocket_batch)
    pocket_collate_s = time.perf_counter() - t
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    total_atoms = int(bench.atoms[:n_lig].sum())
    if isinstance(scorer, XattnRagged):
        scorer.configure_states(total_atoms, a.state_budget_gb)

    def embed():
        return scorer.embed(groups)
    try:
        t_emb, r_emb, ls = timed(embed, dev, a.reps)
    except torch.cuda.OutOfMemoryError:
        if not isinstance(scorer, XattnRagged) or scorer.store_half:
            raise
        print("[warn] out of memory with fp32 atom states, falling back to fp16 states", flush=True)
        scorer.store_half, scorer.fallback = True, True
        ls = None
        gc.collect()
        torch.cuda.empty_cache()
        t_emb, r_emb, ls = timed(embed, dev, a.reps)
    t_state, r_state, ps = timed(lambda: scorer.pocket_states(pbs), dev, a.reps)
    frac = None if padded else a.topk_frac
    reps = []
    for _ in range(a.reps):
        ts, tk, _ = score_all(scorer, ls, ps, P, n_lig, frac, a.pocket_chunk, dev)
        reps.append((ts, tk))
    t_sc, t_tk = min(reps, key=lambda r: r[0] + r[1])
    peak = torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else None
    model_s = t_emb + t_state + t_sc + t_tk
    n_pairs = n_lig * P
    blk = {"network": name, "P": P, "unique_pockets": min(P, unique), "pockets_repeated": P > unique, "n_lig": n_lig, "n_pairs": n_pairs,
           "subset": padded, "gpu": str(torch.cuda.get_device_name(dev)) if dev.type == "cuda" else "cpu",
           "stages_s": {"featurize_cpu": None if padded else bench.feat_s, "collate_cpu": collate_s, "pocket_collate_cpu": pocket_collate_s,
                        "embed_gpu": t_emb, "pocket_states_gpu": t_state, "score_gpu": t_sc, "topk_gpu": t_tk if frac else None},
           "reps_s": {"embed": r_emb, "pocket_states": r_state, "score_plus_topk": [r[0] + r[1] for r in reps]},
           "model_time_s": model_s, "scoring_pairs_per_s": n_pairs / (t_sc + t_tk), "scoring_only_pairs_per_s": n_pairs / t_sc,
           "model_pairs_per_s": n_pairs / model_s, "peak_gpu_mem_gb": peak, "over_20gb": bool(peak is not None and peak > 20.0),
           "group_size": scorer.group_size or a.group, "pocket_batch": scorer.pocket_batch or a.pocket_batch, "pocket_chunk": a.pocket_chunk,
           "topk_fraction": frac, "total_atoms": total_atoms}
    if isinstance(scorer, XattnRagged):
        blk.update({"state_dtype": "fp16" if scorer.store_half else "fp32", "state_fallback_to_fp16": scorer.fallback, "mode": scorer.mode})
    if padded:
        blk["end_to_end_s"] = None
        blk["s_per_pair_scoring"] = t_sc / n_pairs
        blk["extrapolated_score_s_per_pocket_full_library"] = t_sc / n_pairs * bench.library_size
        blk["note"] = "subset timing of the original padded implementation; not used in totals"
    else:
        prep = bench.feat_s + collate_s + pocket_collate_s
        blk["cpu_prep_s"] = prep
        blk["end_to_end_s"] = prep + model_s
        blk["end_to_end_pairs_per_s"] = n_pairs / (prep + model_s)
        del ls, ps
        gc.collect()
        if a.wallclock:
            blk["wallclock_s"] = bench.wallclock(scorer, P)
    ls = ps = None
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return blk


# --------------------------------------------------------------------------- results file
def environment(dev, procs):
    env = {"device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu", "torch": torch.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "cpu_count": os.cpu_count(), "featurization_processes": procs,
           "python": platform.python_version(), "platform": platform.platform(), "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}
    if dev.type == "cuda":
        pr = torch.cuda.get_device_properties(dev)
        env.update(gpu_memory_gb=pr.total_memory / 1e9, gpu_capability=f"{pr.major}.{pr.minor}")
    try:
        import rdkit
        env["rdkit"] = rdkit.__version__
    except Exception:
        pass
    return env


def compute_ratios(blocks):
    """xattn/dual and pooled/dual. End-to-end ratios use the CPU preparation (featurization, collation) of the dual block for both networks."""
    out = {}
    for P, d in blocks.get("dual", {}).items():
        for name in ("pooled", "xattn_fp32", "xattn_fp16"):
            x = blocks.get(name, {}).get(P)
            if not x:
                continue
            prep = d["cpu_prep_s"]
            out.setdefault(P, {})[name] = {
                "model_time": x["model_time_s"] / d["model_time_s"],
                "scoring_only": (x["stages_s"]["score_gpu"] + x["stages_s"]["topk_gpu"]) / (d["stages_s"]["score_gpu"] + d["stages_s"]["topk_gpu"]),
                "end_to_end": (prep + x["model_time_s"]) / (prep + d["model_time_s"]),
                "end_to_end_own_cpu_prep": x["end_to_end_s"] / d["end_to_end_s"],
                "shared_cpu_prep_s": prep}
            if x.get("wallclock_s") and d.get("wallclock_s"):
                out[P][name]["wallclock"] = x["wallclock_s"] / d["wallclock_s"]
    pad = blocks.get("xattn_padded_fp32", {})
    for P, p in pad.items():
        r = blocks.get("xattn_fp32", {}).get(P)
        if r:
            out.setdefault("padded_vs_ragged", {})[P] = {"ragged_fp32_over_padded_scoring_pairs_per_s": r["scoring_only_pairs_per_s"] / p["scoring_only_pairs_per_s"]}
            r16 = blocks.get("xattn_fp16", {}).get(P)
            if r16:
                out["padded_vs_ragged"][P]["ragged_fp16_over_padded_scoring_pairs_per_s"] = r16["scoring_only_pairs_per_s"] / p["scoring_only_pairs_per_s"]
    return out


def fmt(x, nd=3):
    """Table number: thousands separator from 1000, nd decimals from 1, nd significant digits below 1."""
    if x is None:
        return "-"
    a = abs(x)
    if a >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}" if a >= 1 else f"{x:.{nd}g}"


def write_markdown(res, path):
    env, lib = res["environment"], res.get("library", {})
    feat = (res.get("featurization") or {}).get("seconds")
    L = [f"# v4 fair timing: {env.get('gpu')} ({env.get('date')})", "",
         f"Library {lib.get('size', '-')} ligands, {lib.get('n_atoms', '-')} atoms; featurization {fmt(feat, 2)} s "
         f"with {env.get('featurization_processes')} processes ({env.get('cpu_count')} CPUs). torch {env.get('torch')}, CUDA {env.get('cuda')}.",
         "P = 1000 cycles the 57 wild-type pockets (repeated pockets, timing only). All times in seconds, GPU stages best of 2, "
         "top 10% of the library selected per pocket with torch.topk for every network.", ""]
    if res.get("correctness", {}).get("results"):
        c = res["correctness"]
        L += [f"## Correctness ({c['n_lig']} ligands x {c['n_pockets']} pockets, against net(ctx, lb)['logit'])", "",
              "| implementation | max abs diff | Spearman (min over pockets) | top-10% overlap (min) | pass |", "|---|---|---|---|---|"]
        for k, r in c["results"].items():
            L.append(f"| {k} | {fmt(r.get('max_abs_diff'), 6)} | {fmt(r.get('spearman_min'), 6)} | {fmt(r.get('top10_overlap_min'), 4)} | {r.get('pass')} |")
        L.append("")
    for P in sorted({int(p) for n in res["blocks"].values() for p in n}):
        L += [f"## P = {P}", "", "| network | featurize | collate | embed | pocket states | score | top-k | model time | end to end | wall clock | pairs/s (score) | pairs/s (model) | peak GB |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in NETWORKS:
            b = res["blocks"].get(n, {}).get(str(P))
            if not b:
                continue
            s = b["stages_s"]
            tag = n + (" (subset)" if b["subset"] else "") + (" [fp16 states]" if b.get("state_fallback_to_fp16") else "")
            L.append(f"| {tag} | {fmt(s['featurize_cpu'])} | {fmt(s['collate_cpu'])} | {fmt(s['embed_gpu'])} | {fmt(s['pocket_states_gpu'])} | {fmt(s['score_gpu'])} | "
                     f"{fmt(s['topk_gpu'])} | {fmt(b['model_time_s'])} | {fmt(b['end_to_end_s'])} | {fmt(b.get('wallclock_s'))} | {fmt(b['scoring_pairs_per_s'], 0)} | "
                     f"{fmt(b['model_pairs_per_s'], 0)} | {fmt(b['peak_gpu_mem_gb'], 1)} |")
        L.append("")
    rat = {k: v for k, v in res.get("ratios", {}).items() if k != "padded_vs_ragged"}
    if rat:
        L += ["## Ratio to the dual encoder (model time: embedding + pocket states + scoring + top-k; end to end adds the CPU featurization and collation of the dual block)", "",
              "| P | network | model time | scoring only | end to end | wall clock |", "|---|---|---|---|---|---|"]
        for P in sorted(rat, key=int):
            for n, r in rat[P].items():
                L.append(f"| {P} | {n} / dual | {fmt(r['model_time'], 1)} | {fmt(r['scoring_only'], 1)} | {fmt(r['end_to_end'], 1)} | {fmt(r.get('wallclock'), 1)} |")
        L.append("")
    pv = res.get("ratios", {}).get("padded_vs_ragged")
    if pv:
        L += ["## Original padded implementation (subset) against the ragged implementations", ""]
        for P, r in pv.items():
            L.append(f"- P = {P}: ragged fp32 scoring is {fmt(r.get('ragged_fp32_over_padded_scoring_pairs_per_s'), 2)}x and ragged fp16 {fmt(r.get('ragged_fp16_over_padded_scoring_pairs_per_s'), 2)}x the pairs/s of the padded fp32 path.")
        L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def save(res, out):
    res["ratios"] = compute_ratios(res["blocks"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(sanitize(res), indent=1), encoding="utf-8")
    os.replace(tmp, out)
    write_markdown(res, out.with_suffix(".md"))


def plan_blocks(a, n_unique):
    sel = [n for n in NETWORKS if n in a.only.split(",")]
    if a.P == "auto":
        Ps = sorted({p for p in (1, 5, 20, 57) if p <= n_unique} | {n_unique} | ({LARGE_P} if n_unique >= 57 else set()))
    else:
        Ps = sorted(int(p) for p in a.P.split(","))
    p_pad = min(5, n_unique)
    blocks, skipped = [], []
    for n in sel:
        if n == "xattn_padded_fp32":
            if a.P == "auto" or p_pad in Ps:
                blocks.append((n, p_pad))
            continue
        for P in Ps:
            if n == "xattn_fp32" and P >= LARGE_P and not a.xattn_fp32_1000:
                skipped.append((n, P))
                continue
            blocks.append((n, P))
    return blocks, skipped


# --------------------------------------------------------------------------- main
@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None, help="result JSON (a markdown summary is written next to it)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--library", type=Path, default=ROOT / "data/processed/ligand_full.parquet")
    p.add_argument("--smiles-col", default="standardized_smiles")
    p.add_argument("--pockets", type=Path, default=None, help="full pocket cache or the file written by --export-wt-pockets "
                   "(default: dockmut/plan/pocket_graphs_wt.pt if it exists, else dockmut/plan/pocket_graphs.pt)")
    p.add_argument("--export-wt-pockets", type=Path, default=None, help="write the wild-type pocket graphs only (small file for the pod) and exit")
    p.add_argument("--dual", type=Path, default=ROOT / "runs/v1/ckpt_v1_C0.pt")
    p.add_argument("--xattn", type=Path, default=ROOT / "runs/v3/ckpt_v3_xattn_s11.pt")
    p.add_argument("--pooled", type=Path, default=ROOT / "runs/v3/ckpt_v3_pooled_s11.pt")
    p.add_argument("--allow-random-weights", action="store_true", help="smoke test without checkpoints")
    p.add_argument("--n-lig", type=int, default=0, help="first n SMILES of the library (0: all)")
    p.add_argument("--n-pockets", type=int, default=0, help="number of distinct wild-type pockets (0: all 57); P above it cycles them")
    p.add_argument("--P", default="auto", help="comma list of pocket counts; auto: 1,5,20,57 and 1000 (cycled) for the full run")
    p.add_argument("--only", default=",".join(NETWORKS), help="comma list of: " + ",".join(NETWORKS))
    p.add_argument("--xattn-fp32-1000", action="store_true", help="also time the ragged fp32 cross-attention network at P = 1000")
    p.add_argument("--fp16-mode", choices=["pure", "autocast"], default="pure")
    p.add_argument("--procs", type=int, default=0, help="featurization processes (0: min(32, cpu count))")
    p.add_argument("--group", type=int, default=4096, help="ligands per group (collation, embedding, ragged scoring)")
    p.add_argument("--pocket-batch", type=int, default=64, help="pockets per encoder call")
    p.add_argument("--pocket-chunk", type=int, default=128, help="pockets scored and selected per step")
    p.add_argument("--topk-frac", type=float, default=0.10)
    p.add_argument("--padded-n-lig", type=int, default=20000)
    p.add_argument("--state-budget-gb", type=float, default=8.0, help="fp32 atom states above this size are stored in fp16 (reported)")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--n-check", type=int, default=2000)
    p.add_argument("--n-check-pockets", type=int, default=5)
    p.add_argument("--skip-correctness", action="store_true")
    p.add_argument("--no-wallclock", dest="wallclock", action="store_false")
    p.add_argument("--redo", action="store_true", help="recompute blocks that are already in the output file")
    p.add_argument("--fresh", action="store_true", help="ignore an existing output file")
    a = p.parse_args()
    if a.pockets is None:
        a.pockets = ROOT / "dockmut/plan/pocket_graphs_wt.pt"
        if not a.pockets.exists():
            a.pockets = ROOT / "dockmut/plan/pocket_graphs.pt"
    if a.export_wt_pockets:
        names, pk = load_wt_pockets(a.pockets)
        torch.save(dict(zip(names, pk)), a.export_wt_pockets)
        print(f"wrote {len(pk)} wild-type pocket graphs to {a.export_wt_pockets}", flush=True)
        return
    if a.out is None:
        p.error("--out is required")
    dev = torch.device(a.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available")
    for n in a.only.split(","):
        if n not in NETWORKS:
            sys.exit(f"unknown network {n}")
    torch.backends.cuda.matmul.allow_tf32 = False       # strict fp32 for the fp32 implementations
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    res = {"schema": "v4_fair_timing/1", "environment": {}, "invocations": [], "library": {}, "featurization_runs": [], "correctness": {}, "blocks": {}, "ratios": {},
           "notes": ["P = 1000 cycles the 57 wild-type pockets (repeated pockets, used only for timing); every copy is encoded and scored again.",
                     "Pockets are the wild-type graphs of the targets in sorted order; P pockets are the first P.",
                     "end_to_end_s = featurization + CPU collation + pocket collation + embedding + pocket states + scoring + top-k; "
                     "wallclock_s = one pass of the full pipeline from SMILES with one timer.",
                     "Ligand and pocket encoders run in fp32 for every network; fp16 applies to the interaction layers and the head only."]}
    if a.out.exists() and not a.fresh:
        res.update(json.loads(a.out.read_text(encoding="utf-8")))
    procs = a.procs or min(32, os.cpu_count() or 8)

    # blocks to run (decided before any large object exists, so that the process pool is forked from a small process)
    n_unique_req = a.n_pockets or 57
    blocks, skipped = plan_blocks(a, n_unique_req)
    pending = [b for b in blocks if a.redo or str(b[1]) not in res["blocks"].get(b[0], {})]
    for n, P in skipped:
        print(f"[skip] {n} P={P} (give --xattn-fp32-1000 to time it)", flush=True)
    print(f"blocks: {len(blocks)} planned, {len(pending)} pending: {pending}", flush=True)
    if not pending:
        save(res, a.out)
        print("nothing to do; summary rewritten", flush=True)
        return
    pool = mp.Pool(procs) if procs > 1 else None

    names, pockets = load_wt_pockets(a.pockets, a.n_pockets)
    lib = pd.read_parquet(a.library, columns=["ligand_id", a.smiles_col])
    if a.n_lig:
        lib = lib.iloc[: a.n_lig]
    smiles = lib[a.smiles_col].tolist()
    res["environment"] = environment(dev, procs if pool else 1)
    res["invocations"].append({"date": res["environment"]["date"], "gpu": res["environment"]["gpu"], "argv": sys.argv[1:], "pending": [list(b) for b in pending]})
    gpus = {i["gpu"] for i in res["invocations"]}
    if len(gpus) > 1:
        print(f"[WARNING] this file was written on different devices: {sorted(gpus)}; blocks carry their own 'gpu' field", flush=True)
    print(f"device {res['environment']['gpu']}, {len(pockets)} distinct pockets, {len(smiles)} SMILES, {res['environment']['featurization_processes']} featurization processes", flush=True)

    # weights and scorers
    need = {"dual": "dual" in a.only.split(","), "pooled": "pooled" in a.only.split(","), "xattn": any(n.startswith("xattn") for n in a.only.split(","))}
    nets = {}
    for key, cls, path in (("dual", DualEncoderNet, a.dual), ("pooled", PooledConcatNet, a.pooled), ("xattn", PocketGate, a.xattn)):
        if need[key]:
            nets[key] = load_net(cls, path, dev, a.allow_random_weights)
    res["weights"] = {k: {"path": str(getattr(a, k)), "source": v.weights_source} for k, v in nets.items()}
    scorers = build_scorers(nets, dev)
    timing_scorer = {"dual": "dual", "pooled": "pooled", "xattn_padded_fp32": "xattn_padded_fp32", "xattn_fp32": "xattn_fp32",
                     "xattn_fp16": "xattn_fp16_pure" if a.fp16_mode == "pure" else "xattn_fp16_autocast"}

    # correctness block (first)
    if not a.skip_correctness:
        n_chk = min(a.n_check, len(smiles))
        chk = [g for g in (pool.map(_feat, smiles[:n_chk], chunksize=64) if pool else [_feat(s) for s in smiles[:n_chk]]) if g is not None]
        chk_pk = pockets[: min(a.n_check_pockets, len(pockets))]
        print(f"correctness: {len(chk)} ligands x {len(chk_pk)} pockets", flush=True)
        gate = {timing_scorer[n] for n in a.only.split(",")}
        use = {k: v for k, v in scorers.items() if k in gate or k.startswith("xattn_fp16")}
        results, failed = check_scorers(use, chk, chk_pk, dev, group=500, pocket_batch=2, frac=a.topk_frac, gate=gate)
        res["correctness"] = {"n_lig": len(chk), "n_pockets": len(chk_pk), "reference": "net(ctx, lb)['logit'] (module forward, fp32, eval)",
                              "tolerances": {"fp32_max_abs_diff": TOL_FP32_DIFF, "fp32_spearman": TOL_FP32_RHO, "fp16_spearman": TOL_FP16_RHO},
                              "gating": sorted(gate), "results": results, "failed": failed}
        for k, r in results.items():
            print(f"  {k:22s} maxdiff {r.get('max_abs_diff', float('nan')):.2e} rho_min {r.get('spearman_min', float('nan')):.6f} "
                  f"top10 overlap {r.get('top10_overlap_min', float('nan')):.4f} pass {r['pass']} {r.get('error', '')}", flush=True)
        save(res, a.out)
        if failed:
            print(f"CORRECTNESS FAILED for {failed}; no timing was run", flush=True)
            if pool:
                pool.terminate()
            sys.exit(2)

    # stage 1: featurization of the whole library
    bench = Bench(a, dev, pool, smiles, pockets)
    print(f"featurizing {len(smiles)} SMILES with {res['environment']['featurization_processes']} processes ...", flush=True)
    bench.run_featurization()
    bench.library_size = len(bench.graphs)
    lib_info = {"size": bench.library_size, "n_smiles": len(smiles), "featurization_failures": bench.feat_fail, "n_atoms": int(bench.atoms.sum()),
                "mean_atoms": float(bench.atoms.mean()), "source": str(a.library), "column": a.smiles_col}
    if res["library"] and res["library"].get("size") != lib_info["size"]:
        sys.exit(f"output file holds blocks for a library of {res['library'].get('size')} ligands, this run has {lib_info['size']}; use --fresh or another --out")
    res["library"] = lib_info
    res["library_size"] = lib_info["size"]
    res["featurization_runs"].append({"seconds": bench.feat_s, "processes": res["environment"]["featurization_processes"], "date": res["environment"]["date"]})
    res["featurization"] = res["featurization_runs"][0]
    print(f"featurized: {lib_info} in {bench.feat_s:.2f} s", flush=True)

    for name, P in pending:
        print(f"--- block {name} P={P} ---", flush=True)
        t = time.perf_counter()
        blk = run_block(bench, name, scorers[timing_scorer[name]], P)
        if name == "xattn_fp16":
            blk["fp16_mode"] = a.fp16_mode
        res["blocks"].setdefault(name, {})[str(P)] = blk
        save(res, a.out)
        s = blk["stages_s"]
        print(f"[{name} P={P}] embed {s['embed_gpu']:.3f} states {s['pocket_states_gpu']:.3f} score {s['score_gpu']:.3f} topk {fmt(s['topk_gpu'])} "
              f"model {blk['model_time_s']:.3f} e2e {fmt(blk['end_to_end_s'])} wall {fmt(blk.get('wallclock_s'))} s; {blk['scoring_pairs_per_s']:,.0f} pairs/s; "
              f"peak {fmt(blk['peak_gpu_mem_gb'], 1)} GB; block took {time.perf_counter() - t:.0f} s", flush=True)
        if blk["over_20gb"]:
            print(f"[WARNING] peak GPU memory {blk['peak_gpu_mem_gb']:.1f} GB exceeds 20 GB", flush=True)
    save(res, a.out)
    print(f"wrote {a.out} and {a.out.with_suffix('.md')}", flush=True)
    if pool:
        pool.terminate()


if __name__ == "__main__":
    main()
