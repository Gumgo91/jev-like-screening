"""Hit-probability head of the baseline surrogates (robustness check, post hoc).

The baseline dual encoders have two outputs: a regression score and a hit logit trained on top-1% hits, which is the
readout used for screening in the preceding analysis. The main text evaluates edits and wild-type screening with the
regression score. This script repeats both with the hit logit (sign flipped, so that a lower value is a better molecule):
(1) the response to the truncations of the 19 held-out targets, and (2) wild-type screening on the development split.
The sealed test labels are not read here; test recall of the hit head comes from the preceding analysis.

Usage: python dm_baseline_logit.py out_prefix
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
sys.path.insert(0, str(ROOT / "src"))
from dm_core import delta_table, lig_graphs_for, load_pockets, metric_block, predict_delta, read_results  # noqa: E402
from dm_eval import V2Adapter, detect_auc  # noqa: E402
from dm_wt_eval import joint_reversal, wt_scores  # noqa: E402
from pocketgate.data.graphbatch import collate_ligands  # noqa: E402
from pocketgate.evaluation.metrics import macro_eval  # noqa: E402
from pocketgate.models.pocketgate import DualEncoderNet  # noqa: E402
from train_dm import make_constant_pocket  # noqa: E402


class V2LogitAdapter(V2Adapter):
    """Score = minus the hit logit of the dual encoder, in the standardized units of the other models (arbitrary scale)."""

    def score_pairs(self, ctx, pidx, z, lidx):
        lv = self.net.W_l(z[lidx])
        return -(lv * ctx["pv"][pidx]).sum(-1) / self.net.logit_scale.clamp(min=1e-3)


def load(path: str, const: bool = False):
    net = DualEncoderNet(dim=128)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return V2LogitAdapter(net), const


CKPT = {"C0": ["runs/v1/ckpt_v1_C0.pt", "runs/v2/ckpt_v1_C0_s22.pt", "runs/v2/ckpt_v1_C0_s33.pt"],
        "C1": ["runs/v1/ckpt_v1_C1.pt", "runs/v2/ckpt_v1_C1_s22.pt", "runs/v2/ckpt_v1_C1_s33.pt"],
        "C2": ["runs/v1/ckpt_v1_C2.pt", "runs/v2/ckpt_v1_C2_s22.pt", "runs/v2/ckpt_v1_C2_s33.pt"],
        "C0c": ["runs/v2/ckpt_v1_C0c_s11.pt", "runs/v2/ckpt_v1_C0c_s22.pt", "runs/v2/ckpt_v1_C0c_s33.pt"]}
SEEDS = [11, 22, 33]


def main():
    out = Path(sys.argv[1])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
    pockets = load_pockets(ROOT / "dockmut/plan/pocket_graphs.pt", plan)
    held = [t for t, v in plan["targets"].items() if v["role"] in ("dev", "test")]
    tables = {}
    for t in held:
        man = json.loads((ROOT / "dockmut/plan/receptors" / t / "manifest.json").read_text())
        tables[t] = delta_table(read_results(ROOT / "dockmut/results", t), man)[0]
    ids = sorted({i for t in tables for ids_, _, _ in tables[t].values() for i in ids_})
    smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
    ligs = lig_graphs_for(ids, ROOT / "data/processed/g2_ligand_graphs.pt", smiles)
    frames, summary = [], {}
    for cond, paths in CKPT.items():
        for seed, path in zip(SEEDS, paths):
            model, use_const = load(str(ROOT / path), const=cond == "C0c")
            pk = pockets
            if use_const:
                const = make_constant_pocket(pockets, [t for t, v in plan["targets"].items() if v["role"] == "train"])
                pk = {t: {r: const for r in pockets[t]} for t in held}
            df = pd.concat([predict_delta(model, pk, ligs, tables[t], t, dev) for t in held])
            df["model"] = f"{cond}L_s{seed}"
            frames.append(df)
            c = df[df.kind.isin(["single_contact"]) | df.kind.str.startswith("multi")]
            mb = metric_block(c) if c.pred.std() > 1e-9 else {"pooled_r": float("nan"), "mutant_r": float("nan"), "within_r": float("nan"), "pred_sd": 0.0}
            summary[f"{cond}L_s{seed}"] = {**mb, "detect_auc": detect_auc(df) if c.pred.std() > 1e-9 else 0.5}
            print(f"{cond}L_s{seed} edit response with the hit logit: pooled r={mb['pooled_r']:+.3f} edit r={mb['mutant_r']:+.3f} "
                  f"within r={mb['within_r']:+.3f} pred SD={mb['pred_sd']:.3f} AUC={summary[f'{cond}L_s{seed}']['detect_auc']:.2f}", flush=True)
    pd.concat(frames).to_parquet(out.with_suffix(".parquet"), index=False)
    # wild-type screening on the development split with the hit logit
    tg = pd.read_parquet(ROOT / "data/splits/targets.parquet")
    dev_t = tg[tg.split_role == "dev"].target_id.tolist()
    cfg = json.loads((ROOT / "configs/g2_config.json").read_text())
    from pocketgate.data.labels import load_labels
    dev_ids = list(cfg["dev_ligands"])
    lab = load_labels("dev", "dev", caller="dm_baseline_logit_dev")
    quads = pd.read_parquet(ROOT / "runs/v1/eval_quads_dev.parquet")
    ligs_d = lig_graphs_for(dev_ids, ROOT / "data/processed/g2_ligand_graphs.pt", smiles)
    dev_ids = [i for i in dev_ids if i in ligs_d and ligs_d[i] is not None]
    lab = lab[lab.target_id.isin(dev_t) & lab.ligand_id.isin(set(dev_ids))]
    wt = {}
    for cond, paths in CKPT.items():
        for seed, path in zip(SEEDS, paths):
            model, use_const = load(str(ROOT / path), const=cond == "C0c")
            if use_const:
                const = make_constant_pocket(pockets, [t for t, v in plan["targets"].items() if v["role"] == "train"])
            rows = []
            for t in dev_t:
                pk = const if use_const else pockets[t]["wt"]
                rows.append(pd.DataFrame({"target_id": t, "ligand_id": dev_ids, "s_hat": wt_scores(model, pk, ligs_d, dev_ids, dev)}))
            pred = pd.concat(rows)
            pred["p_hit"] = 1.0 / (1.0 + np.exp((pred.s_hat - pred.s_hat.mean()) / 1.0))
            df = pred.merge(lab, on=["target_id", "ligand_id"]).dropna(subset=["score"])
            me = macro_eval(df)
            jr = joint_reversal(pred, quads)
            wt[f"{cond}L_s{seed}"] = {"recall_1@10": float(me["recall_1@10"].mean()), "recall_1@5": float(me["recall_1@5"].mean()),
                                      "recall_1@1": float(me["recall_1@1"].mean()), "spearman": float(me["spearman"].mean()), **jr}
            print(f"{cond}L_s{seed} dev screening with the hit logit: R1@10={wt[f'{cond}L_s{seed}']['recall_1@10']:.3f} "
                  f"joint reversal={jr['joint']:.3f}", flush=True)
    out.with_suffix(".json").write_text(json.dumps({"edit": summary, "wt_dev": wt}, indent=1))


if __name__ == "__main__":
    main()
