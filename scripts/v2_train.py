"""Dual-encoder networks at further seeds, and the constant-pocket dual
encoder: C0 (real pockets) at seeds 22 and 33, C0c (one constant synthetic
pocket for every target) at seeds 11, 22 and 33. Data, split, loss, batch,
optimizer and checkpoint rule are those of scripts/v1b_train.py (which trains
C0 at seed 11). Outputs go to runs/v2/ (ckpt_v1_<cond>_s<seed>.pt,
loss_curve_*.csv, eval_curve_*.csv, v2_registry.json).

Usage: python scripts/v2_train.py "C0:22,C0:33,C0c:11,C0c:22,C0c:33"
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
from pocketgate.data.synthetic import make_synthetic_pocket  # noqa: E402
from pocketgate.engine_v1 import TrainerV1, TrainConfigV1  # noqa: E402
from pocketgate.evaluation.metrics import macro_eval  # noqa: E402
from pocketgate.inference.screen import Screener  # noqa: E402
from pocketgate.models.pocketgate import DualEncoderNet  # noqa: E402

OUT = ROOT / "runs" / "v2"
PROCESSED = ROOT / "data" / "processed"
DIM = 128
CONDITIONS = ("C0", "C0c")
SYNTH_HASH = ("1cb5081f2d82f7695252d3c809d636acd3cb150"
              "20aeb249d67d42bb56b0f4c81")


def main():
    ap = argparse.ArgumentParser(
        description="Train dual-encoder networks into runs/v2/. A job is "
                    "<condition>:<seed>, condition C0 (real pockets) or C0c "
                    "(one constant synthetic pocket).")
    ap.add_argument("jobs",
                    help='comma separated condition:seed, e.g. '
                         '"C0:22,C0:33,C0c:11,C0c:22,C0c:33"')
    ap.add_argument("--device", default=None,
                    help="default: cuda if available else cpu")
    a = ap.parse_args()
    jobs = []
    for spec in a.jobs.split(","):
        cond, seed = spec.split(":")
        if cond not in CONDITIONS:
            ap.error(f"unknown condition {cond!r}; defined: "
                     f"{', '.join(CONDITIONS)}")
        jobs.append((cond, int(seed)))
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((ROOT / "configs" / "g2_config.json").read_text())
    tg = pd.read_parquet(ROOT / "data" / "splits" / "targets.parquet")
    train_t = tg[tg.split_role == "train"]["target_id"].tolist()
    dev_t = tg[tg.split_role == "dev"]["target_id"].tolist()
    ligs = torch.load(PROCESSED / "g2_ligand_graphs.pt",
                      weights_only=False)
    pockets = torch.load(PROCESSED / "v1_pocket_graphs.pt",
                         weights_only=False)
    # device_count() guards against an empty CUDA_VISIBLE_DEVICES, for which
    # is_available() can still report True
    device = a.device or ("cuda" if torch.cuda.is_available()
                          and torch.cuda.device_count() > 0 else "cpu")

    lab = L.load_labels("train", "train", caller="v2_train")
    lab = lab[lab.target_id.isin(train_t)
              & lab.ligand_id.isin(cfg["train_ligands"])]
    tau = lab.dropna().groupby("target_id")["score"].quantile(0.01)
    lab["y"] = (lab["score"] <= lab["target_id"].map(tau)).astype(int)
    scaler = {"mean": float(lab["score"].mean()),
              "std": float(lab["score"].std())}

    dev_ligs = [l for l in cfg["dev_ligands"] if l in ligs]
    dev_lab = pd.concat([
        L.load_labels("train", "dev", caller="v2_train"),
        L.load_labels("dev", "dev", caller="v2_train")])
    dev_lab = dev_lab[dev_lab["ligand_id"].isin(dev_ligs)]
    warm_ev = pd.DataFrame([(t, l) for t in train_t for l in dev_ligs],
                           columns=["target_id", "ligand_id"])
    cold_ev = pd.DataFrame([(t, l) for t in dev_t for l in dev_ligs],
                           columns=["target_id", "ligand_id"])
    warm_lab = dev_lab.merge(warm_ev, on=["target_id", "ligand_id"])
    cold_lab = dev_lab.merge(cold_ev, on=["target_id", "ligand_id"])

    synth = make_synthetic_pocket({t: pockets[t] for t in train_t})
    pockets_const = {t: synth for t in pockets}
    print("synthetic hash:", synth["hash"], flush=True)
    assert synth["hash"] == SYNTH_HASH

    cond_map = {"C0": pockets, "C0c": pockets_const}

    reg_path = OUT / "v2_registry.json"
    registry = json.loads(reg_path.read_text()) if reg_path.exists() \
        else {"conditions": {}}

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

    for cond, seed in jobs:
        pock = cond_map[cond]
        tag = f"{cond}_s{seed}"
        done_marker = OUT / f"eval_curve_{tag}.csv"
        if done_marker.exists() and \
                (OUT / f"ckpt_v1_{tag}.pt").exists():
            print(tag, "already complete - skipping", flush=True)
            continue
        set_seed(seed)
        model = DualEncoderNet(dim=DIM)
        n_params = sum(p.numel() for p in model.parameters())
        cfg_t = TrainConfigV1(
            dim=DIM, seed=seed, device=device,
            ckpt_path=str(OUT / f"ckpt_v1_{tag}.pt"))
        tr = TrainerV1(model, cfg_t, pock, ligs)
        t0 = time.time()
        log = tr.train(lab, scaler, eval_fn_factory(model, pock))
        wall = time.time() - t0
        print(tag, "done", log["updates"], "upd",
              f"{wall/60:.0f}min best={log['best_metric']:.3f}",
              flush=True)
        pd.DataFrame(log["loss_curve"]).assign(run=tag) \
            .to_csv(OUT / f"loss_curve_{tag}.csv", index=False)
        pd.DataFrame(log["eval_curve"]).assign(run=tag) \
            .to_csv(OUT / f"eval_curve_{tag}.csv", index=False)
        registry["conditions"][tag] = {
            "condition": cond, "seed": seed,
            "params": int(n_params), "updates": log["updates"],
            "epochs": log["epochs"],
            "train_examples": log["train_examples"],
            "best_cold_recall_1@10": log["best_metric"],
            "best_update": log["best_update"],
            "stop_reason": log["stop_reason"],
            "wall_seconds": log["wall_seconds"],
            "peak_vram_mb": log["peak_vram_mb"],
            "ckpt_sha256": hashlib.sha256(
                Path(cfg_t.ckpt_path).read_bytes()).hexdigest()}
        reg_path.write_text(json.dumps(registry, indent=2,
                                       default=str))
    print("V2 TRAIN DONE")


if __name__ == "__main__":
    main()
