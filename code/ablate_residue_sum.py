"""Ablation of the residue sum at inference: predicted change of ResidueSum-Geo networks with and without the residue terms.

For every checkpoint the predicted change of a contact edit is computed with the full score and with the score that keeps only the
ligand-only term b(z_l) and the global descriptor term gamma(z_l, g_P). The pooled correlation with the measured changes of the
19 held-out targets is reported per seed and for the seed ensemble. No sealed label is read (DockMut results only).
Usage: python ablate_residue_sum.py out.json
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
from dm_core import SCORE_STD, WINS, collate_pockets, delta_table, lig_graphs_for, load_pockets, read_results  # noqa: E402
from dm_eval import load_model  # noqa: E402
from pocketgate.data.graphbatch import collate_ligands  # noqa: E402

plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
pockets = load_pockets(ROOT / "dockmut/plan/pocket_graphs.pt", plan)
held = [t for t, v in plan["targets"].items() if v["role"] in ("dev", "test")]
smiles = pd.read_parquet(ROOT / "data/processed/ligands.parquet").set_index("ligand_id").standardized_smiles
tabs = {t: delta_table(read_results(ROOT / "dockmut/results", t), json.loads((ROOT / f"dockmut/plan/receptors/{t}/manifest.json").read_text()))[0] for t in held}
shared = pd.read_csv(ROOT / "dockmut/plan/ligands/heldout_shared.csv").inchikey.tolist()
ligs = lig_graphs_for(shared, ROOT / "data/processed/g2_ligand_graphs.pt", smiles)
lb = collate_ligands([ligs[i] for i in shared])
index = {k: i for i, k in enumerate(shared)}
SETS = {"rsg_wt": (ROOT / "dockmut/runs_collected/final_ckpt", (11, 22, 33, 44, 55, 66)),
        "rsg_int": (ROOT / "dockmut/runs_collected/final_ckpt", (11, 22, 33, 44, 55, 66)),
        "rsgh_int": (ROOT / "dockmut/runs_hit_pulled/runs_hit", (11, 22, 33))}


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


out = {}
for cond, (folder, seeds) in SETS.items():
    full, nores, y_ref = {}, {}, None
    for s in seeds:
        f = folder / f"{cond}_s{s}.pt"
        if not f.exists():
            continue
        m, _ = load_model("dm:" + str(f))
        m.eval()
        pf, pn, ys = [], [], []
        with torch.no_grad():
            z = m.lig_embed(lb)
            for t in held:
                cs = [r for r, v in tabs[t].items() if v[2] == "single_contact" or v[2].startswith("multi")]
                ctx = m.poc_embed(collate_pockets([pockets[t]["wt"]] + [pockets[t][r] for r in cs]))
                li = torch.arange(len(shared))

                def score(j, use_res):
                    pidx = torch.full_like(li, j)
                    sc = m.lig_off(z).squeeze(-1) + m.glob(torch.cat([z, ctx["geo"][pidx]], -1)).squeeze(-1)
                    if use_res:
                        sc = sc + (m.out(m._u(ctx, pidx, z, li)).squeeze(-1) * ctx["mask"][pidx]).sum(-1)
                    return sc

                f0, n0 = score(0, True), score(0, False)
                for j, r in enumerate(cs, 1):
                    ids, dS, _ = tabs[t][r]
                    sel = [index[i] for i in ids if i in index]
                    d = np.array([x for i, x in zip(ids, dS) if i in index])
                    pf.append(((score(j, True) - f0)[sel] * SCORE_STD).clamp(-WINS, WINS).numpy())
                    pn.append(((score(j, False) - n0)[sel] * SCORE_STD).clamp(-WINS, WINS).numpy())
                    ys.append(d)
        full[s], nores[s] = np.concatenate(pf), np.concatenate(pn)
        y_ref = np.concatenate(ys)
        print(f"{cond}_s{s}: full {corr(full[s], y_ref):.3f} | no residue sum {corr(nores[s], y_ref):.3f}", flush=True)
    sd = sorted(full)
    out[cond] = {"seeds": sd, "full_r": [corr(full[s], y_ref) for s in sd], "no_residue_sum_r": [corr(nores[s], y_ref) for s in sd],
                 "ensemble_full_r": corr(np.mean([full[s] for s in sd], 0), y_ref),
                 "ensemble_no_residue_sum_r": corr(np.mean([nores[s] for s in sd], 0), y_ref)}
    print(cond, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[cond].items() if k.startswith("ensemble")}, flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
