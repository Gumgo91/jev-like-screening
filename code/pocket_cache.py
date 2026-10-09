"""Featurize every wild-type and truncated receptor of the plan into one cache file.

Output: dockmut/plan/pocket_graphs.pt  {target: {receptor_id: graph dict}}
The featurizer is the one used in the original study, so wild-type graphs are identical to
the stored V1 graphs (verified in pocket_graphs.py).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
from pocket_graphs import build_target_graphs  # noqa: E402


def main():
    plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
    out, t0 = {}, time.time()
    for i, t in enumerate(plan["targets"]):
        g = build_target_graphs(t, ROOT / "dockmut/plan/receptors", ROOT / "data/raw/targets")
        slim = {}
        for rid, gr in g.items():
            slim[rid] = {"x": gr["x"].float(), "edge_index": gr["edge_index"], "edge_attr": gr["edge_attr"].float(),
                         "n_residues": int(gr["n_residues"]), "residue_ids": gr["residue_ids"], "geo": gr["geo"].float(), "edit_meta": gr["edit_meta"].float()}
        out[t] = slim
        print(f"{i+1}/{len(plan['targets'])} {t} {len(slim)} receptors {time.time()-t0:.0f}s", flush=True)
    torch.save(out, ROOT / "dockmut/plan/pocket_graphs.pt")
    print("saved", sum(len(v) for v in out.values()), "graphs")


if __name__ == "__main__":
    main()
