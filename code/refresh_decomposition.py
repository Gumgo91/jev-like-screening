"""Recompute the variance decomposition with the corrected noise accounting and update analysis/decomposition.csv and summary.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockmut"))
from decomp import decompose  # noqa: E402

plan = json.loads((ROOT / "dockmut/plan/plan.json").read_text())
roles = {t: v["role"] for t, v in plan["targets"].items()}
D = decompose(ROOT / "dockmut/results", ROOT / "dockmut/plan", roles)
cols = ["mutant_mean", "ligand_main", "interaction", "noise"]
D.to_csv(ROOT / "dockmut/analysis/decomposition.csv", index=False)
f = ROOT / "dockmut/analysis/summary.json"
S = json.loads(f.read_text())
S.pop("decomposition_by_group", None)
S["decomposition_mean"] = D[cols].mean().to_dict()
S["decomposition_by_group"] = {"train": D[D.role == "train"][cols].mean().to_dict(), "heldout": D[D.role != "train"][cols].mean().to_dict()}
f.write_text(json.dumps(S, indent=1))
print("mean shares", {k: round(v, 3) for k, v in S["decomposition_mean"].items()}, "sum", round(sum(S["decomposition_mean"].values()), 3))
print("old shares", json.loads((ROOT / "dockmut/analysis/summary.before_fix.json").read_text())["decomposition_mean"])
