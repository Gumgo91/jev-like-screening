"""Train one surrogate under a DockMut condition.

conditions
  rs_wt        ResidueSumNet, wild-type scores only
  rs_int       ResidueSumNet + intervention loss (measured changes of truncated receptors)
  rs_int_shl   intervention labels permuted across ligands inside each mutant (control)
  rs_int_shm   intervention labels permuted across mutants (control)
  rs_const     ResidueSumNet with one constant synthetic pocket, wild-type scores only
  b5_wt        dual encoder, wild-type scores only
  b5_int       dual encoder + intervention loss
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
sys.path.insert(0, str(ROOT / "src"))
from dm_core import (DMTrainer, delta_table, lig_graphs_for, load_pockets, metric_block, predict_delta,  # noqa: E402
                     read_results, to_matrix)
from dm_models import build  # noqa: E402

COND = {"rse_wt": ("rse", 0.0, None), "rse_int": ("rse", 1.0, None), "rse_int_shl": ("rse", 1.0, "ligand"), "rse_int_shm": ("rse", 1.0, "mutant"),
        "rse_int_sha": ("rse", 1.0, "all"), "rse_const": ("rse", 0.0, None), "rsg_wt": ("rsg", 0.0, None), "rsg_int": ("rsg", 1.0, None), "rsg_int_shl": ("rsg", 1.0, "ligand"),
        "rsg_int_shm": ("rsg", 1.0, "mutant"), "rsg_const": ("rsg", 0.0, None), "rs_wt": ("rs", 0.0, None), "rs_int": ("rs", 1.0, None), "rs_int_shl": ("rs", 1.0, "ligand"),
        "rs_int_shm": ("rs", 1.0, "mutant"), "rs_int_sha": ("rs", 1.0, "all"), "rsg_int_sha": ("rsg", 1.0, "all"), "rs_const": ("rs", 0.0, None),
        "b5_wt": ("b5", 0.0, None), "b5_int": ("b5", 1.0, None),
        "rsg_int_shm2": ("rsg", 1.0, "mutant_d"), "rsg_int_sha2": ("rsg", 1.0, "all_d"),
        "rsgh_wt": ("rsg", 0.0, None), "rsgh_int": ("rsg", 1.0, None)}   # rsgh: ResidueSum-Geo with a hit head


def make_constant_pocket(pockets: dict, targets: list) -> dict:
    xs = torch.cat([pockets[t]["wt"]["x"] for t in targets]).mean(0, keepdim=True).repeat(256, 1)
    src = [i for i in range(256) for k in range(1, 17)]
    dst = [(i + k) % 256 for i in range(256) for k in range(1, 17)]
    from pocketgate.data.features import _rbf
    ea = torch.from_numpy(_rbf(np.full(len(src), 2.0)).astype(np.float32))
    geo = torch.stack([pockets[t]["wt"]["geo"] for t in targets]).mean(0) if "geo" in pockets[targets[0]]["wt"] else None
    out = {"x": xs, "edge_index": torch.tensor([src, dst]), "edge_attr": ea, "n_residues": 256}
    if geo is not None:
        out["geo"] = geo
    if "edit" in pockets[targets[0]]["wt"]:
        out["edit"] = torch.zeros_like(pockets[targets[0]]["wt"]["edit"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", required=True, choices=list(COND))
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--updates", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--n-mut", type=int, default=4)
    p.add_argument("--mut-batch", type=int, default=64)
    p.add_argument("--results", type=Path, default=ROOT / "dockmut/results")
    p.add_argument("--plan", type=Path, default=ROOT / "dockmut/plan")
    p.add_argument("--pockets", type=Path, default=ROOT / "dockmut/plan/pocket_graphs.pt")
    p.add_argument("--out", type=Path, default=ROOT / "dockmut/runs")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--targets", default="")
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--family-balance", action="store_true")
    p.add_argument("--val-targets", default="", help="comma list of individual training targets withheld for validation")
    p.add_argument("--heldout-eval", action="store_true", help="final runs only: save predictions on the 19 held-out targets")
    p.add_argument("--n-train-targets", type=int, default=0, help="random subset of training targets (scaling study)")
    p.add_argument("--mut-keep", type=int, default=0, help="keep at most K edited receptors per target")
    p.add_argument("--val-families", default="", help="comma list of families withheld from training for validation")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mu", type=float, default=None, help="weight of the hit-logit BCE loss (rsgh conditions: 1.0)")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    kind, lam, shuffle = COND[a.cond]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    plan = json.loads((a.plan / "plan.json").read_text())
    train_t = [t for t, v in plan["targets"].items() if v["role"] == "train"]
    if a.targets:
        train_t = [t for t in train_t if t in a.targets.split(",")]
    tgt = pd.read_parquet(ROOT / "data/splits/targets.parquet").set_index("target_id")
    val_fams = [f for f in a.val_families.split(",") if f]
    val_t = [t for t in train_t if tgt.loc[t, "family"] in val_fams or t in a.val_targets.split(",")]
    train_t = [t for t in train_t if t not in val_t]
    if a.n_train_targets and a.n_train_targets < len(train_t):
        r2 = np.random.default_rng(1000 + a.seed)
        train_t = sorted(r2.choice(train_t, size=a.n_train_targets, replace=False).tolist())
    cfg_g2 = json.loads((ROOT / "configs/g2_config.json").read_text())
    pockets = load_pockets(a.pockets, plan)

    # wild-type labels for the training ligands
    sys.path.insert(0, str(ROOT / "src"))
    from pocketgate.data.labels import load_labels
    lab = load_labels("train", "train", caller=f"dockmut_train_{a.cond}")
    lab = lab[lab.target_id.isin(train_t) & lab.ligand_id.isin(cfg_g2["train_ligands"])]
    wt = {t: (g.ligand_id.to_numpy(), g.score.to_numpy(np.float32)) for t, g in lab.dropna().groupby("target_id")}

    deltas, val_deltas, train_tabs = {}, {}, {}
    rng = np.random.default_rng(17)
    for t in train_t:
        man = json.loads((a.plan / "receptors" / t / "manifest.json").read_text())
        tab, sigma, n_st = delta_table(read_results(a.results, t), man)
        rids = sorted(tab)
        if a.mut_keep and len(rids) > a.mut_keep:
            keep = set(np.random.default_rng(29 + a.seed).choice(rids, size=a.mut_keep, replace=False).tolist())
            tab = {r: v for r, v in tab.items() if r in keep}
            rids = sorted(tab)
        hold = set(rng.choice(rids, size=int(round(a.val_frac * len(rids))), replace=False)) if rids else set()
        deltas[t] = to_matrix({r: v for r, v in tab.items() if r not in hold}) if len(tab) > len(hold) else {}
        val_deltas[t] = {r: v for r, v in tab.items() if r in hold}
        train_tabs[t] = {r: v for r, v in list(tab.items()) if r not in hold}
    fam_val = {}
    for t in val_t:
        man = json.loads((a.plan / "receptors" / t / "manifest.json").read_text())
        fam_val[t] = delta_table(read_results(a.results, t), man)[0]
    need = {i for t in deltas if deltas[t] for i in deltas[t]["ligs"]}
    need |= {i for t in fam_val for r, (ids, _, _) in fam_val[t].items() for i in ids}
    need |= {i for t in val_deltas for r, (ids, _, _) in val_deltas[t].items() for i in ids}
    need |= {i for t in wt for i in wt[t][0]}
    smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
    ligs = lig_graphs_for(sorted(need), ROOT / "data/processed/g2_ligand_graphs.pt", smiles)

    const = make_constant_pocket(pockets, train_t) if a.cond.endswith("_const") else None
    hit = a.cond.startswith("rsgh")
    mu = a.mu if a.mu is not None else (1.0 if hit else 0.0)
    model = build(kind, hit=True) if hit else build(kind)
    cfg = {"seed": a.seed, "updates": a.updates, "batch": a.batch, "n_mut": a.n_mut, "mut_batch": a.mut_batch,
           "lam": a.lam if a.lam is not None else lam, "mu": mu, "shuffle": shuffle, "lr": a.lr, "wd": 1e-4, "const_pocket": const,
           "family_balance": a.family_balance, "family_of": {t: tgt.loc[t, "family"] for t in train_t}}
    tag = f"{a.cond}{a.tag}_s{a.seed}"
    a.out.mkdir(parents=True, exist_ok=True)
    trainer = DMTrainer(model, pockets, ligs, wt, deltas, cfg, dev)
    hist = trainer.train(ckpt=a.out / f"{tag}.pt")
    pd.DataFrame(hist).to_csv(a.out / f"{tag}_loss.csv", index=False)
    diag = {}
    for name, tabs in (("val_mutants", val_deltas), ("train_mutants", {t: dict(list(v.items())[:6]) for t, v in train_tabs.items()}), ("val_families", fam_val)):
        frames = [predict_delta(model, pockets if const is None else {t: {r: const for r in pockets[t]} for t in pockets},
                                ligs, tabs[t], t, dev) for t in tabs if tabs[t]]
        if frames:
            g = pd.concat(frames)
            g = g[g.kind.isin(["single_contact"]) | g.kind.str.startswith("multi")]
            diag[name] = metric_block(g)
    (a.out / f"{tag}_diag.json").write_text(json.dumps(diag, indent=1))
    if a.heldout_eval:
        held = [t for t, v in plan["targets"].items() if v["role"] in ("dev", "test")]
        tabs = {}
        for t in held:
            man = json.loads((a.plan / "receptors" / t / "manifest.json").read_text())
            tabs[t] = delta_table(read_results(a.results, t), man)[0]
        ids = sorted({i for t in tabs for ids_, _, _ in tabs[t].values() for i in ids_})
        ligs_h = lig_graphs_for(ids, ROOT / "data/processed/g2_ligand_graphs.pt", smiles)
        pk = pockets if const is None else {t: {r: const for r in pockets[t]} for t in pockets}
        frames = [predict_delta(model, pk, ligs_h, tabs[t], t, dev) for t in held]
        pd.concat(frames).to_parquet(a.out / f"{tag}_heldout.parquet", index=False)
        print("heldout predictions saved", flush=True)
    print("diag", json.dumps({k: {m: round(v, 3) for m, v in d.items() if m in ("pooled_r", "mutant_r", "within_r", "gain", "pred_sd", "true_sd")} for k, d in diag.items()}))
    print("done", tag)


if __name__ == "__main__":
    main()
