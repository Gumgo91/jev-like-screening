"""V1-B: four conditions C0-C3 on the B5 backbone, seed 11 each.

- Train cells: train_train only (39 targets x 40,000 G2 train ligands).
- Dev eval: cold = 10 dev targets x 10k dev ligands; warm = 39 train
  targets x 10k. Checkpoint selection on cold macro recall_1@10.
- C0: L_base. C1: +within-target margin pairs. C2: +reversal quads.
  C3: C2 loss with ONE constant synthetic pocket for every target.
- Preserves: loss/grad curves, eval curves, best ckpt + optimizer state.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pocketgate.common import PROCESSED_DIR, set_seed  # noqa: E402
from pocketgate.data import labels as L  # noqa: E402
from pocketgate.data.features import POCKET_NODE_DIM  # noqa: E402
from pocketgate.engine_v1 import TrainerV1, TrainConfigV1  # noqa: E402
from pocketgate.evaluation.metrics import macro_eval  # noqa: E402
from pocketgate.inference.screen import Screener  # noqa: E402
from pocketgate.models.pocketgate import DualEncoderNet  # noqa: E402

OUT = Path("runs/v1")
OUT.mkdir(parents=True, exist_ok=True)
DIM = 128
DELTA = 1.0


def make_synthetic_pocket(pockets_train: dict, seed: int = 11) -> dict:
    """One fixed graph for every target: 256 nodes with the per-feature
    mean of train-pocket node features, ring-lattice edges k=16,
    constant RBF edge features. Deterministic; hash recorded."""
    xs = np.concatenate([g["x"].numpy() if torch.is_tensor(g["x"])
                         else g["x"] for g in pockets_train.values()])
    mean_x = xs.mean(0).astype(np.float32)
    n = 256
    x = np.repeat(mean_x[None, :], n, axis=0)
    src, dst = [], []
    for i in range(n):
        for k in range(1, 17):
            src.append(i)
            dst.append((i + k) % n)
    edge_index = np.asarray([src, dst], dtype=np.int64)
    from pocketgate.data.features import _rbf
    edge_attr = _rbf(np.full(len(src), 2.0)).astype(np.float32)
    g = {"x": torch.as_tensor(x), "edge_index": torch.as_tensor(edge_index),
         "edge_attr": torch.as_tensor(edge_attr)}
    g["hash"] = hashlib.sha256(
        x.tobytes() + edge_index.tobytes() + edge_attr.tobytes()
    ).hexdigest()
    return g


def main():
    cfg = json.loads(Path("configs/g2_config.json").read_text())
    tg = pd.read_parquet("data/splits/targets.parquet")
    train_t = tg[tg.split_role == "train"]["target_id"].tolist()
    dev_t = tg[tg.split_role == "dev"]["target_id"].tolist()
    fam_of = dict(zip(tg.target_id, tg.family))
    print("train family dist:",
          pd.Series([fam_of[t] for t in train_t]).value_counts().to_dict(),
          flush=True)

    ligs = torch.load(PROCESSED_DIR / "g2_ligand_graphs.pt",
                      weights_only=False)
    pockets = torch.load(PROCESSED_DIR / "v1_pocket_graphs.pt",
                         weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- labels: train only -------------------------------------------
    lab = L.load_labels("train", "train", caller="v1b_train")
    lab = lab[lab.target_id.isin(train_t)
              & lab.ligand_id.isin(cfg["train_ligands"])]
    tau = lab.dropna().groupby("target_id")["score"].quantile(0.01)
    lab["y"] = (lab["score"] <= lab["target_id"].map(tau)).astype(int)
    scaler = {"mean": float(lab["score"].mean()),
              "std": float(lab["score"].std())}
    print("train pairs", len(lab), "pos rate", lab["y"].mean(),
          flush=True)

    # ---- dev eval labels (eval role) -----------------------------------
    dev_ligs = [l for l in cfg["dev_ligands"] if l in ligs]
    dev_lab = pd.concat([
        L.load_labels("train", "dev", caller="v1b_eval"),
        L.load_labels("dev", "dev", caller="v1b_eval")])
    dev_lab = dev_lab[dev_lab["ligand_id"].isin(dev_ligs)]
    warm_ev = pd.DataFrame(
        [(t, l) for t in train_t for l in dev_ligs],
        columns=["target_id", "ligand_id"])
    cold_ev = pd.DataFrame(
        [(t, l) for t in dev_t for l in dev_ligs],
        columns=["target_id", "ligand_id"])
    warm_lab = dev_lab.merge(warm_ev, on=["target_id", "ligand_id"])
    cold_lab = dev_lab.merge(cold_ev, on=["target_id", "ligand_id"])

    # ---- ranking pools -------------------------------------------------
    res = pd.read_parquet(OUT / "quadruplet_reservoir.parquet")
    res = res[res.target_a.isin(train_t) & res.target_b.isin(train_t)]
    # within-target pool for C1: same margin delta, stratified by target
    rng = np.random.default_rng(20260925)
    within_rows = []
    mat = lab.pivot(index="target_id", columns="ligand_id",
                    values="score")
    for t in train_t:
        s = mat.loc[t].to_numpy()
        ok = ~np.isnan(s)
        pool_idx = np.where(ok)[0]
        got, tries = 0, 0
        seen = set()
        target_cap = 6000
        while got < target_cap and tries < 200_000:
            i = rng.choice(pool_idx, 4096)
            j = rng.choice(pool_idx, 4096)
            d = s[i] - s[j]
            hit = np.abs(d) >= DELTA
            for k in np.where(hit)[0]:
                key = (int(i[k]), int(j[k]))
                if key in seen:
                    continue
                seen.add(key)
                within_rows.append({"target_a": t,
                                    "ligand_i": mat.columns[i[k]],
                                    "ligand_j": mat.columns[j[k]],
                                    "d_a": float(d[k])})
                got += 1
                if got >= target_cap:
                    break
            tries += 4096
    within = pd.DataFrame(within_rows)
    print("within pool", len(within), flush=True)

    synthetic = make_synthetic_pocket(
        {t: pockets[t] for t in train_t})
    print("synthetic pocket hash:", synthetic["hash"], flush=True)

    def eval_fn_factory(model, pock):
        def fn():
            scr = Screener(model, pock, ligs, device,
                           cache_mode="per_target")
            pc = scr.score(cold_ev, batch_size=512)
            mc = macro_eval(pc.merge(cold_lab,
                                     on=["target_id", "ligand_id"])
                            .dropna(subset=["score", "p_hit"]))
            pw = scr.score(warm_ev, batch_size=512)
            mw = macro_eval(pw.merge(warm_lab,
                                     on=["target_id", "ligand_id"])
                            .dropna(subset=["score", "p_hit"]))
            return {"cold_recall_1@10": float(mc["recall_1@10"].mean()),
                    "warm_recall_1@10": float(mw["recall_1@10"].mean())}
        return fn

    registry = {"seed": 11, "delta": DELTA, "lambda": 0.1,
                "batch_size": 256, "quads_per_update": 64,
                "within_pairs_per_update": 128,
                "max_updates": 60000, "eval_every": 6094,
                "patience_evals": 3, "lr": 3e-4, "weight_decay": 1e-4,
                "aux_weight": 0.2,
                "train_targets": train_t, "dev_targets": dev_t,
                "n_train_pairs": int(len(lab)),
                "train_pos_rate": float(lab["y"].mean()),
                "family_dist_train":
                    pd.Series([fam_of[t] for t in train_t])
                    .value_counts().to_dict(),
                "reservoir_quads": int(len(res)),
                "within_pool": int(len(within)),
                "synthetic_pocket_hash": synthetic["hash"],
                "conditions": {}}
    pockets_c3 = {t: synthetic for t in list(pockets.keys())}

    all_conds = {"C0": ("none", pockets), "C1": ("within", pockets),
                 "C2": ("reversal", pockets), "C3": ("reversal", pockets_c3)}
    wanted = sys.argv[1].split(",") if len(sys.argv) > 1 \
        else list(all_conds)
    for cond in wanted:
        mode, pock = all_conds[cond]
        set_seed(cfg["seed"])
        model = DualEncoderNet(dim=DIM)
        n_params = sum(p.numel() for p in model.parameters())
        cfg_t = TrainConfigV1(
            dim=DIM, seed=cfg["seed"], device=device,
            ckpt_path=str(OUT / f"ckpt_v1_{cond}.pt"),
            optim_path=str(OUT / f"optim_v1_{cond}.pt"))
        tr = TrainerV1(model, cfg_t, pock, ligs)
        rb = {"none": None, "within": within,
              "reversal": res}[mode]
        t0 = time.time()
        log = tr.train(lab, scaler, rb, mode, eval_fn_factory(model, pock))
        wall = time.time() - t0
        print(cond, "done", log["updates"], "upd",
              f"{wall/60:.0f}min best={log['best_metric']:.3f}",
              flush=True)

        pd.DataFrame(log["loss_curve"]).assign(condition=cond) \
            .to_csv(OUT / f"loss_curve_{cond}.csv", index=False)
        pd.DataFrame(log["eval_curve"]).assign(condition=cond) \
            .to_csv(OUT / f"eval_curve_{cond}.csv", index=False)
        # active-gradient params: pocket encoder gets grads in all conds
        registry["conditions"][cond] = {
            "mode": mode, "params": int(n_params),
            "updates": log["updates"], "epochs": log["epochs"],
            "train_examples": log["train_examples"],
            "rank_forward_pairs": log["rank_forward_pairs"],
            "best_cold_recall_1@10": log["best_metric"],
            "best_update": log["best_update"],
            "stop_reason": log["stop_reason"],
            "wall_seconds": log["wall_seconds"],
            "peak_vram_mb": log["peak_vram_mb"],
            "ckpt_sha256": hashlib.sha256(
                Path(cfg_t.ckpt_path).read_bytes()).hexdigest()
            if Path(cfg_t.ckpt_path).exists() else None,
        }
        # merge with existing registry (conditions may run in separate
        # invocations)
        reg_path = OUT / "v1_registry.json"
        if reg_path.exists():
            old = json.loads(reg_path.read_text())
            old.setdefault("conditions", {}).update(
                registry["conditions"])
            for k, v in registry.items():
                if k != "conditions":
                    old[k] = v
            registry = old
        reg_path.write_text(json.dumps(registry, indent=2, default=str))

    print(json.dumps(registry["conditions"], indent=2, default=str))


if __name__ == "__main__":
    main()
