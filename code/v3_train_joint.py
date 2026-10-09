"""Joint (non-factorized) surrogates trained with the recipe of the Jev-like baselines (runs/v2, C0).

xattn : PocketGate, ligand atoms attend to the residues of the pocket (the ligand representation depends on the pocket)
pooled: PooledConcatNet, pooled ligand and pocket vectors concatenated and read by an MLP

Same data, split, loss (BCE on top-1% hits + 0.2 Huber on scores), optimizer, batch (256), evaluation points (every 6,094
updates), patience (3 evaluations), maximum of 60,000 updates / 10 epochs and checkpoint rule (cold-development macro
Recall_1%@10%) as scripts/v2_train.py. Outputs go to runs/v3/ only; the sealed test labels are not read here.

Usage: python scripts/v3_train_joint.py xattn:11,xattn:22,xattn:33 [--smoke]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pocketgate.common import PROCESSED_DIR, set_seed  # noqa: E402
from pocketgate.data import labels as L  # noqa: E402
from pocketgate.engine import TrainConfig, Trainer  # noqa: E402
from pocketgate.evaluation.metrics import macro_eval  # noqa: E402
from pocketgate.inference.screen import Screener  # noqa: E402
from pocketgate.models.pocketgate import PocketGate, PooledConcatNet  # noqa: E402

DIM = 128
SMOKE = "--smoke" in sys.argv
OUT = Path("runs/v3_smoke" if SMOKE else "runs/v3")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    jobs = []
    for spec in sys.argv[1].split(","):
        kind, seed = spec.split(":")
        jobs.append((kind, int(seed)))
    cfg = json.loads(Path("configs/g2_config.json").read_text())
    tg = pd.read_parquet("data/splits/targets.parquet")
    train_t = tg[tg.split_role == "train"]["target_id"].tolist()
    dev_t = tg[tg.split_role == "dev"]["target_id"].tolist()
    ligs = torch.load(PROCESSED_DIR / "g2_ligand_graphs.pt", weights_only=False)
    pockets = torch.load(PROCESSED_DIR / "v1_pocket_graphs.pt", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lab = L.load_labels("train", "train", caller="v3_train_joint")
    lab = lab[lab.target_id.isin(train_t) & lab.ligand_id.isin(cfg["train_ligands"])]
    tau = lab.dropna().groupby("target_id")["score"].quantile(0.01)
    lab["y"] = (lab["score"] <= lab["target_id"].map(tau)).astype(int)
    scaler = {"mean": float(lab["score"].mean()), "std": float(lab["score"].std())}

    dev_ligs = [l for l in cfg["dev_ligands"] if l in ligs]
    dev_lab = pd.concat([L.load_labels("train", "dev", caller="v3_train_joint"), L.load_labels("dev", "dev", caller="v3_train_joint")])
    dev_lab = dev_lab[dev_lab["ligand_id"].isin(dev_ligs)]
    cold_ev = pd.DataFrame([(t, l) for t in dev_t for l in dev_ligs], columns=["target_id", "ligand_id"])
    cold_lab = dev_lab.merge(cold_ev, on=["target_id", "ligand_id"])

    def eval_fn_factory(model):
        def fn():
            scr = Screener(model, pockets, ligs, device, cache_mode="per_target")
            pc = scr.score(cold_ev, batch_size=512)
            mc = macro_eval(pc.merge(cold_lab, on=["target_id", "ligand_id"]).dropna(subset=["score", "p_hit"]))
            return float(mc["recall_1@10"].mean())
        return fn

    for kind, seed in jobs:
        tag = f"{kind}_s{seed}"
        if (OUT / f"ckpt_v3_{tag}.pt").exists() and (OUT / f"eval_curve_{tag}.json").exists():
            print(tag, "already complete, skipping", flush=True)
            continue
        set_seed(seed)
        model = PocketGate(dim=DIM) if kind == "xattn" else PooledConcatNet(dim=DIM)
        n_params = sum(p.numel() for p in model.parameters())
        tcfg = TrainConfig(dim=DIM, seed=seed, batch_size=256, lr=3e-4, weight_decay=1e-4,
                           max_updates=300 if SMOKE else 60000, max_epochs=10, eval_every_updates=100 if SMOKE else 6094,
                           patience_rounds=3, aux_weight=0.2, device=device, ckpt_path=str(OUT / f"ckpt_v3_{tag}.pt"))
        tr = Trainer(model, tcfg, pockets, ligs)
        t0 = time.time()
        log = tr.train(lab, dev_eval_fn=eval_fn_factory(model), scaler=scaler)
        wall = time.time() - t0
        print(tag, "done", log.updates, "updates", f"{wall / 60:.0f} min", f"best cold recall@10% = {log.best_metric:.3f} at update {log.best_update}", log.stop_reason, flush=True)
        (OUT / f"eval_curve_{tag}.json").write_text(json.dumps(log.evals, indent=1))
        (OUT / f"registry_{tag}.json").write_text(json.dumps({
            "model": kind, "seed": seed, "params": int(n_params), "updates": log.updates, "epochs": log.epochs,
            "best_cold_recall_1@10": log.best_metric, "best_update": log.best_update, "stop_reason": log.stop_reason,
            "wall_seconds": log.wall_seconds, "peak_vram_mb": log.peak_vram_mb,
            "ckpt_sha256": hashlib.sha256(Path(tcfg.ckpt_path).read_bytes()).hexdigest() if Path(tcfg.ckpt_path).exists() else None,
        }, indent=2, default=str))
    print("V3 TRAIN DONE")


if __name__ == "__main__":
    main()
