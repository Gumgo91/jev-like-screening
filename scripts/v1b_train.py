"""Dual-encoder baseline C0 (real pockets), seed 11.

- Train cells: train_train only (39 targets x 40,000 train ligands of
  configs/g2_config.json).
- Dev eval: cold = 10 dev targets x 10k dev ligands; warm = 39 train
  targets x 10k. Checkpoint selection on cold macro recall_1@10.
- C0 loss: BCE on the top-1% hit label + 0.2 * Huber on the standardized
  docking score.
- Outputs in runs/v1/: ckpt_v1_C0.pt (best cold evaluation),
  loss_curve_C0.csv, eval_curve_C0.csv, v1_registry.json.

The constant-pocket dual encoders (C0c) and the other seeds are trained by
scripts/v2_train.py.

Usage: python scripts/v1b_train.py [C0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketgate.common import set_seed  # noqa: E402
from pocketgate.data import labels as L  # noqa: E402
# make_synthetic_pocket lives in the library; the name stays importable from
# this module for older scripts.
from pocketgate.data.synthetic import make_synthetic_pocket  # noqa: E402,F401
from pocketgate.engine_v1 import TrainerV1, TrainConfigV1  # noqa: E402
from pocketgate.evaluation.metrics import macro_eval  # noqa: E402
from pocketgate.inference.screen import Screener  # noqa: E402
from pocketgate.models.pocketgate import DualEncoderNet  # noqa: E402

OUT = ROOT / "runs" / "v1"
PROCESSED = ROOT / "data" / "processed"
DIM = 128
CONDITIONS = ("C0",)


def main():
    ap = argparse.ArgumentParser(
        description="Train the dual-encoder baseline C0 (real pockets, "
                    "seed 11) into runs/v1/.")
    ap.add_argument("conditions", nargs="?", default="C0",
                    help="condition to train; only C0 is defined (default)")
    ap.add_argument("--device", default=None,
                    help="default: cuda if available else cpu")
    a = ap.parse_args()
    wanted = a.conditions.split(",")
    for cond in wanted:
        if cond not in CONDITIONS:
            ap.error(f"unknown condition {cond!r}; defined: "
                     f"{', '.join(CONDITIONS)}")
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((ROOT / "configs" / "g2_config.json").read_text())
    tg = pd.read_parquet(ROOT / "data" / "splits" / "targets.parquet")
    train_t = tg[tg.split_role == "train"]["target_id"].tolist()
    dev_t = tg[tg.split_role == "dev"]["target_id"].tolist()
    fam_of = dict(zip(tg.target_id, tg.family))
    print("train family dist:",
          pd.Series([fam_of[t] for t in train_t]).value_counts().to_dict(),
          flush=True)

    ligs = torch.load(PROCESSED / "g2_ligand_graphs.pt",
                      weights_only=False)
    pockets = torch.load(PROCESSED / "v1_pocket_graphs.pt",
                         weights_only=False)
    # device_count() guards against an empty CUDA_VISIBLE_DEVICES, for which
    # is_available() can still report True
    device = a.device or ("cuda" if torch.cuda.is_available()
                          and torch.cuda.device_count() > 0 else "cpu")

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

    registry = {"seed": 11,
                "batch_size": 256,
                "max_updates": 60000, "eval_every": 6094,
                "patience_evals": 3, "lr": 3e-4, "weight_decay": 1e-4,
                "aux_weight": 0.2,
                "train_targets": train_t, "dev_targets": dev_t,
                "n_train_pairs": int(len(lab)),
                "train_pos_rate": float(lab["y"].mean()),
                "family_dist_train":
                    pd.Series([fam_of[t] for t in train_t])
                    .value_counts().to_dict(),
                "conditions": {}}
    all_conds = {"C0": pockets}

    for cond in wanted:
        pock = all_conds[cond]
        set_seed(cfg["seed"])
        model = DualEncoderNet(dim=DIM)
        n_params = sum(p.numel() for p in model.parameters())
        cfg_t = TrainConfigV1(
            dim=DIM, seed=cfg["seed"], device=device,
            ckpt_path=str(OUT / f"ckpt_v1_{cond}.pt"))
        tr = TrainerV1(model, cfg_t, pock, ligs)
        t0 = time.time()
        log = tr.train(lab, scaler, eval_fn_factory(model, pock))
        wall = time.time() - t0
        print(cond, "done", log["updates"], "upd",
              f"{wall/60:.0f}min best={log['best_metric']:.3f}",
              flush=True)

        pd.DataFrame(log["loss_curve"]).assign(condition=cond) \
            .to_csv(OUT / f"loss_curve_{cond}.csv", index=False)
        pd.DataFrame(log["eval_curve"]).assign(condition=cond) \
            .to_csv(OUT / f"eval_curve_{cond}.csv", index=False)
        registry["conditions"][cond] = {
            "params": int(n_params),
            "updates": log["updates"], "epochs": log["epochs"],
            "train_examples": log["train_examples"],
            "best_cold_recall_1@10": log["best_metric"],
            "best_update": log["best_update"],
            "stop_reason": log["stop_reason"],
            "wall_seconds": log["wall_seconds"],
            "peak_vram_mb": log["peak_vram_mb"],
            "ckpt_sha256": hashlib.sha256(
                Path(cfg_t.ckpt_path).read_bytes()).hexdigest()
            if Path(cfg_t.ckpt_path).exists() else None,
        }
        (OUT / "v1_registry.json").write_text(
            json.dumps(registry, indent=2, default=str))

    print(json.dumps(registry["conditions"], indent=2, default=str))


if __name__ == "__main__":
    main()
