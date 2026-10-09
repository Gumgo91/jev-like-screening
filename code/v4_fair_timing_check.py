"""CPU-only test of the optimized scorers of v4_fair_timing.py (run it before the timing run on the pod).

Every scorer (dual GEMM, pooled first-layer decomposition, padded and ragged cross-attention in fp32 and fp16, pure half and autocast) is
compared with the module forward net(ctx, lb)['logit'] on 200 ligands x 3 pockets, through the same code path as the timing blocks
(ligand groups, batches of pockets, chunks of pockets, torch.topk selection). Checkpoints are used when they exist, otherwise the weights
are random (seeded). The script never touches a GPU. Exit code 1 if a scorer misses its tolerance (fp32: max abs diff 1e-3 and Spearman
0.9999; fp16: Spearman 0.999).
Usage: python dockmut/v4_fair_timing_check.py [--n-lig 200] [--n-pockets 3]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # CPU only, whatever the machine offers

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
sys.path.insert(0, str(ROOT / "src"))
from pocketgate.data.features import POCKET_EDGE_DIM, POCKET_NODE_DIM  # noqa: E402
from pocketgate.models.pocketgate import DualEncoderNet, PocketGate, PooledConcatNet  # noqa: E402
from v4_fair_timing import _feat, build_scorers, check_scorers, load_net, load_wt_pockets  # noqa: E402

FALLBACK_SMILES = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "CN1CCC[C@H]1c1cccnc1", "O=C(O)c1ccccc1OC(C)=O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
                   "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "CC(C)NCC(O)c1ccc(O)c(O)c1", "Clc1ccc2c(c1)C(=NCC(=O)N2C)c1ccccc1"]


def synthetic_pockets(n, seed=0):
    """Random residue graphs with the feature sizes of the pocket encoder (used when the pocket cache is missing)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        m = 40 + 25 * i
        src = torch.arange(m).repeat_interleave(8)
        dst = torch.randint(0, m, (m * 8,), generator=g)
        out.append({"x": torch.randn(m, POCKET_NODE_DIM, generator=g), "edge_index": torch.stack([src, dst]),
                    "edge_attr": torch.rand(m * 8, POCKET_EDGE_DIM, generator=g)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-lig", type=int, default=200)
    p.add_argument("--n-pockets", type=int, default=3)
    p.add_argument("--library", type=Path, default=ROOT / "data/processed/ligand_full.parquet")
    p.add_argument("--pockets", type=Path, default=None, help="default: dockmut/plan/pocket_graphs_wt.pt if it exists, else pocket_graphs.pt")
    p.add_argument("--synthetic-pockets", action="store_true", help="do not load the pocket cache")
    p.add_argument("--dual", type=Path, default=ROOT / "runs/v1/ckpt_v1_C0.pt")
    p.add_argument("--xattn", type=Path, default=ROOT / "runs/v3/ckpt_v3_xattn_s11.pt")
    p.add_argument("--pooled", type=Path, default=ROOT / "runs/v3/ckpt_v3_pooled_s11.pt")
    a = p.parse_args()
    if a.pockets is None:
        a.pockets = ROOT / "dockmut/plan/pocket_graphs_wt.pt"
        if not a.pockets.exists():
            a.pockets = ROOT / "dockmut/plan/pocket_graphs.pt"
    dev = torch.device("cpu")
    assert not torch.cuda.is_available(), "CUDA must be hidden for this test"

    if a.library.exists():
        smiles = pd.read_parquet(a.library, columns=["standardized_smiles"]).standardized_smiles.tolist()[: a.n_lig]
    else:
        smiles = (FALLBACK_SMILES * (a.n_lig // len(FALLBACK_SMILES) + 1))[: a.n_lig]
    graphs = [g for g in map(_feat, smiles) if g is not None]
    if a.pockets.exists() and not a.synthetic_pockets:
        names, pockets = load_wt_pockets(a.pockets, a.n_pockets)
    else:
        names, pockets = [f"synthetic{i}" for i in range(a.n_pockets)], synthetic_pockets(a.n_pockets)
    print(f"{len(graphs)} ligands x {len(pockets)} pockets ({', '.join(names)})", flush=True)

    nets = {k: load_net(cls, path, dev, allow_random=True) for k, cls, path in (("dual", DualEncoderNet, a.dual), ("pooled", PooledConcatNet, a.pooled),
                                                                               ("xattn", PocketGate, a.xattn))}
    print("weights:", {k: v.weights_source for k, v in nets.items()}, flush=True)
    scorers = build_scorers(nets, dev)
    results, failed = check_scorers(scorers, graphs, pockets, dev, group=64, pocket_batch=2, frac=0.1)   # several groups, several pocket batches and chunks
    bad = []
    for k, r in results.items():
        ok = r["pass"] and r.get("topk_path_overlap_min", 1.0) >= (0.99 if r["kind"] == "fp32" else 0.8)
        print(f"{k:22s} maxdiff {r.get('max_abs_diff', float('nan')):.2e}  rho_min {r.get('spearman_min', float('nan')):.6f}  "
              f"top10 overlap {r.get('top10_overlap_min', float('nan')):.3f}  topk path overlap {r.get('topk_path_overlap_min', float('nan')):.3f}  "
              f"{'ok' if ok else 'FAIL'} {r.get('error', '')}", flush=True)
        if not ok:
            bad.append(k)
    if bad:
        print("FAILED:", bad, flush=True)
        sys.exit(1)
    print("all scorers agree with the module forward", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
