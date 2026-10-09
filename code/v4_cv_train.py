"""Family-grouped cross-validation of pocket use (v4): dual encoder, pooled concat MLP and cross-attention network, each with the
real pockets and with one constant synthetic pocket.

A job is fold:cond:recipe:seed. Fold k (data/splits/v4_cv_folds.json, written by dockmut/v4_make_folds.py) holds out whole target
families H_k and trains on the other targets R_k. cond is one of dual, dual_const, pooled, pooled_const, xattn, xattn_const; the
'_const' conditions map every target to a single synthetic pocket graph built from the pocket graphs of R_k, so the gap to the
matching real-pocket condition measures how much the network uses the pocket.

Training data: labels_train_train and labels_dev_train (the 39 train-role and 10 dev-role targets x the 40,000 training ligands of
configs/g2_config.json) restricted to R_k; hit label y = score <= the 1% quantile of the target's training-ligand scores. Loss, batch
(256 ligands of one target), AdamW and evaluation points follow scripts/v3_train_joint.py. Early stopping and the checkpoint choice
use macro Recall_1%@10% over six validation targets V_k (drawn from R_k) x 10,000 probcal ligands (labels_train_probcal +
labels_dev_probcal). The best checkpoint then predicts H_k x the 10,000 development ligands (labels_train_dev + labels_dev_dev).
Only these six label cells are read; the sealed test cells are never touched.

Recipe R0: lr 3e-4, patience 3, max 60,000 updates / 10 epochs, evaluation every 6,094 updates.
Recipe R1 (dual and xattn only): lr 1e-4, patience 8, max 150,000 updates / 30 epochs, evaluation every 6,094 updates.

Outputs in <out>/ for tag f<fold>_<cond>_<recipe>_s<seed>: ckpt_v4_<tag>.pt, eval_curve_<tag>.json, pred_<tag>.parquet,
metrics_<tag>.csv, registry_<tag>.json (written last; a job whose registry exists is skipped) and one line per finished job in
jobs_done.log. pred_<tag>.parquet also carries the true development score ('score') so that the analysis needs no label access.

Usage: python scripts/v4_cv_train.py "0:xattn:R0:11,0:dual_const:R0:11" [--smoke] [--root PATH] [--out runs/v4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--root", default=None)
_root_arg, _ = _pre.parse_known_args()
ROOT = Path(_root_arg.root).resolve() if _root_arg.root else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pocketgate import common  # noqa: E402
from pocketgate.common import set_seed  # noqa: E402
from pocketgate.data import labels as L  # noqa: E402
from pocketgate.engine import TrainConfig, Trainer  # noqa: E402
from pocketgate.evaluation.metrics import macro_eval  # noqa: E402
from pocketgate.inference.screen import Screener  # noqa: E402
from pocketgate.models.pocketgate import DualEncoderNet, PocketGate, PooledConcatNet  # noqa: E402

DIM = 128
ALLOWED_CELLS = {"train_train", "dev_train", "train_dev", "dev_dev", "train_probcal", "dev_probcal"}
CONDS = ["dual", "dual_const", "pooled", "pooled_const", "xattn", "xattn_const"]
ARCH = {"dual": DualEncoderNet, "pooled": PooledConcatNet, "xattn": PocketGate}
RECIPES = {
    "R0": dict(lr=3e-4, patience_rounds=3, max_updates=60000, max_epochs=10, eval_every_updates=6094),
    "R1": dict(lr=1e-4, patience_rounds=8, max_updates=150000, max_epochs=30, eval_every_updates=6094),
}
SMOKE = dict(max_updates=40, eval_every_updates=20, n_val_ligands=400, n_dev_ligands=400, n_val_targets=2)


def load_cell(target_role: str, ligand_role: str) -> pd.DataFrame:
    assert f"{target_role}_{ligand_role}" in ALLOWED_CELLS, f"label cell {target_role}_{ligand_role} is not allowed"
    return L.load_labels(target_role, ligand_role, caller="v4_cv_train")


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def import_make_synthetic_pocket():
    """scripts/v1b_train.py creates runs/v1 relative to the working directory when imported; do that in a scratch directory."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            from v1b_train import make_synthetic_pocket
        finally:
            os.chdir(cwd)
    return make_synthetic_pocket


def parse_jobs(spec: str) -> list[tuple[int, str, str, int]]:
    jobs = []
    for s in spec.split(","):
        fold, cond, recipe, seed = s.strip().split(":")
        assert cond in CONDS, f"unknown condition {cond}"
        assert recipe in RECIPES, f"unknown recipe {recipe}"
        assert recipe == "R0" or cond in ("dual", "xattn"), "recipe R1 is only defined for dual and xattn"
        jobs.append((int(fold), cond, recipe, int(seed)))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", help='comma separated fold:cond:recipe:seed, e.g. "0:xattn:R0:11,0:dual_const:R0:11"')
    ap.add_argument("--smoke", action="store_true", help="40 updates, 400 validation/dev ligands, 2 validation targets")
    ap.add_argument("--root", default=None, help="repository root (src, configs, data); default: parent of scripts/")
    ap.add_argument("--out", default=None, help="output directory, relative paths are relative to the root (default runs/v4)")
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    a = ap.parse_args()
    jobs = parse_jobs(a.jobs)
    assert Path(common.PROJECT_ROOT).resolve() == ROOT, f"pocketgate was imported from {common.PROJECT_ROOT}, not from {ROOT}"
    out = Path(a.out if a.out else ("runs/v4_smoke" if a.smoke else "runs/v4"))
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    # device_count() guards against an empty CUDA_VISIBLE_DEVICES, for which is_available() can still report True
    device = a.device or ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    make_synthetic_pocket = import_make_synthetic_pocket()

    cfg = json.loads((ROOT / "configs" / "g2_config.json").read_text())
    fj = json.loads((ROOT / "data" / "splits" / "v4_cv_folds.json").read_text())
    assert canonical_sha256(fj["fold_assignment"]) == fj["fold_assignment_sha256"], "fold file does not match its hash"
    fold_sha = fj["fold_assignment_sha256"]
    folds = {f["fold"]: f for f in fj["folds"]}
    for f in fj["folds"]:
        assert fj["fold_assignment"][str(f["fold"])] == {k: f[k] for k in ("heldout", "train", "validation")}
    val_ids_all = fj["validation_ligands"]["ids"]
    assert canonical_sha256(val_ids_all) == fj["validation_ligands"]["sha256"]

    print(f"root {ROOT}  out {out}  device {device}  smoke {a.smoke}  fold sha256 {fold_sha[:12]}", flush=True)
    t0 = time.time()
    ligs = torch.load(common.PROCESSED_DIR / "g2_ligand_graphs.pt", weights_only=False)
    val_graphs = torch.load(common.PROCESSED_DIR / "v4_probcal_ligand_graphs.pt", weights_only=False)
    assert not set(ligs) & set(val_graphs), "probcal and g2 ligand graphs overlap"
    ligs.update(val_graphs)
    pockets = torch.load(common.PROCESSED_DIR / "v1_pocket_graphs.pt", weights_only=False)
    assert len(pockets) == 49

    train_ligs = cfg["train_ligands"]
    dev_ligs = [l for l in cfg["dev_ligands"] if l in ligs]
    val_ligs = [l for l in val_ids_all if l in ligs]
    assert len(train_ligs) == 40000 and len(dev_ligs) == 10000 and len(val_ligs) == 10000
    if a.smoke:
        dev_ligs, val_ligs = dev_ligs[:SMOKE["n_dev_ligands"]], val_ligs[:SMOKE["n_val_ligands"]]

    # training labels: 39 train-role + 10 dev-role targets x training ligands; hit threshold per target (fold independent)
    lab_all = pd.concat([load_cell("train", "train"), load_cell("dev", "train")], ignore_index=True)
    lab_all = lab_all[lab_all.ligand_id.isin(set(train_ligs))].reset_index(drop=True)
    tau = lab_all.dropna(subset=["score"]).groupby("target_id")["score"].quantile(0.01)
    lab_all["y"] = (lab_all["score"] <= lab_all["target_id"].map(tau)).astype(int)
    assert lab_all.target_id.nunique() == 49, lab_all.target_id.nunique()
    val_lab_all = pd.concat([load_cell("train", "probcal"), load_cell("dev", "probcal")], ignore_index=True)
    val_lab_all = val_lab_all[val_lab_all.ligand_id.isin(set(val_ligs))]
    dev_lab_all = pd.concat([load_cell("train", "dev"), load_cell("dev", "dev")], ignore_index=True)
    dev_lab_all = dev_lab_all[dev_lab_all.ligand_id.isin(set(dev_ligs))]
    print(f"data loaded in {time.time() - t0:.0f} s: train rows {len(lab_all)}, validation rows {len(val_lab_all)}, "
          f"dev rows {len(dev_lab_all)}", flush=True)

    for fold, cond, recipe, seed in jobs:
        tag = f"f{fold}_{cond}_{recipe}_s{seed}"
        if (out / f"registry_{tag}.json").exists():
            print(tag, "already complete, skipping", flush=True)
            continue
        f = folds[fold]
        arch, const = (cond[:-6], True) if cond.endswith("_const") else (cond, False)
        rec = dict(RECIPES[recipe])
        val_t = f["validation"][:SMOKE["n_val_targets"]] if a.smoke else f["validation"]
        held_t = f["heldout"]

        lab = lab_all[lab_all.target_id.isin(f["train"])].reset_index(drop=True)
        scaler = {"mean": float(lab["score"].mean()), "std": float(lab["score"].std())}
        pockets_used, const_hash = pockets, None
        if const:
            synth = make_synthetic_pocket({t: pockets[t] for t in f["train"]})
            const_hash = synth["hash"]
            pockets_used = {t: synth for t in pockets}
        val_ev = pd.DataFrame([(t, l) for t in val_t for l in val_ligs], columns=["target_id", "ligand_id"])
        val_lab = val_lab_all.merge(val_ev, on=["target_id", "ligand_id"])
        held_ev = pd.DataFrame([(t, l) for t in held_t for l in dev_ligs], columns=["target_id", "ligand_id"])
        held_lab = dev_lab_all.merge(held_ev, on=["target_id", "ligand_id"])
        print(f"{tag}: R_k {len(f['train'])} targets, {len(lab)} training pairs (hit rate {lab['y'].mean():.4f}), "
              f"validation {len(val_t)} targets x {len(val_ligs)} ligands, held out {len(held_t)} targets x {len(dev_ligs)} ligands, "
              f"const pocket {const_hash}", flush=True)

        set_seed(seed)
        model = ARCH[arch](dim=DIM)
        n_params = sum(p.numel() for p in model.parameters())
        ckpt = out / f"ckpt_v4_{tag}.pt"
        tcfg = TrainConfig(dim=DIM, seed=seed, batch_size=256, lr=rec["lr"], weight_decay=1e-4,
                           max_updates=SMOKE["max_updates"] if a.smoke else rec["max_updates"], max_epochs=rec["max_epochs"],
                           eval_every_updates=SMOKE["eval_every_updates"] if a.smoke else rec["eval_every_updates"],
                           patience_rounds=rec["patience_rounds"], aux_weight=0.2, device=device, ckpt_path=str(ckpt))
        if ckpt.exists():
            ckpt.unlink()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tr = Trainer(model, tcfg, pockets_used, ligs)
        n_eval = [0]
        t_job = time.time()

        def val_fn():
            scr = Screener(model, pockets_used, ligs, device, cache_mode="per_target")
            pc = scr.score(val_ev, batch_size=512)
            mc = macro_eval(pc.merge(val_lab, on=["target_id", "ligand_id"]).dropna(subset=["score", "p_hit"]))
            n_eval[0] += 1
            r = float(mc["recall_1@10"].mean())
            print(f"  {tag} eval {n_eval[0]}: validation macro recall_1@10 {r:.4f} ({(time.time() - t_job) / 60:.1f} min)", flush=True)
            return r

        log = tr.train(lab[["target_id", "ligand_id", "score", "y"]], dev_eval_fn=val_fn, scaler=scaler)
        assert ckpt.exists(), "no checkpoint was written (validation metric undefined at every evaluation)"
        wall = time.time() - t_job

        model.load_state_dict(torch.load(ckpt, map_location=device))
        scr = Screener(model, pockets_used, ligs, device, cache_mode="per_target")
        pc = scr.score(held_ev, batch_size=512)[["target_id", "ligand_id", "logit", "p_hit", "score_pred"]]
        pred = pc.merge(held_lab, on=["target_id", "ligand_id"], how="left")
        pred.to_parquet(out / f"pred_{tag}.parquet", index=False)
        met = macro_eval(pred.dropna(subset=["score", "p_hit"]))
        met["fold"] = fold
        met.to_csv(out / f"metrics_{tag}.csv", index=False)
        (out / f"eval_curve_{tag}.json").write_text(json.dumps(log.evals, indent=1))
        held_recall = float(met["recall_1@10"].mean())
        reg = {
            "model": arch, "cond": cond, "fold": fold, "recipe": recipe, "seed": seed, "params": int(n_params),
            "updates": log.updates, "epochs": log.epochs, "best_update": log.best_update, "best_val_recall": log.best_metric,
            "stop_reason": log.stop_reason, "wall_seconds": wall, "peak_vram_mb": log.peak_vram_mb,
            "ckpt_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            "val_targets": val_t, "heldout_targets": held_t,
            "heldout_macro_recall_1@10": held_recall, "n_train_pairs": int(len(lab)), "train_hit_rate": float(lab["y"].mean()),
            "n_val_ligands": len(val_ligs), "n_dev_ligands": len(dev_ligs), "const_pocket_hash": const_hash,
            "fold_assignment_sha256": fold_sha, "smoke": bool(a.smoke), "device": device,
        }
        (out / f"registry_{tag}.json").write_text(json.dumps(reg, indent=2, default=str))
        with open(out / "jobs_done.log", "a") as fh:
            fh.write(json.dumps({"tag": tag, "ts": time.time(), "updates": log.updates, "best_update": log.best_update,
                                 "best_val_recall": log.best_metric, "heldout_macro_recall_1@10": held_recall,
                                 "wall_seconds": wall, "stop_reason": log.stop_reason}) + "\n")
        print(f"{tag} done: {log.updates} updates, {wall / 60:.1f} min, best validation recall {log.best_metric:.4f} at update "
              f"{log.best_update}, held-out macro recall_1@10 {held_recall:.4f}, {log.stop_reason}", flush=True)
    print("V4 TRAIN DONE")


if __name__ == "__main__":
    main()
