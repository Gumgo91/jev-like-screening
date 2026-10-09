"""Family-grouped cross-validation folds for the pocket-use experiment (v4).

The 49 non-test targets (train and dev roles of data/splits/targets.parquet) are split into five fixed folds that hold out
whole target families. For every fold k the script fixes the held-out targets H_k, the training targets R_k (all others) and six
validation targets V_k drawn from R_k. Early stopping of a fold uses V_k x 10,000 probcal ligands, which are featurized here
with the project featurizer (same code path as scripts/g2_featurize.py) and saved to data/processed/v4_probcal_ligand_graphs.pt.
The final evaluation of a fold is H_k x the 10,000 development ligands of configs/g2_config.json.

For every held-out target the script also computes the shared-ranking oracle recall: the 10,000 development ligands are ranked
by their mean docking score over the training targets R_k (lower is better) and Recall_1%@10% and the Spearman correlation of that
single, target-independent ranking are measured against the held-out target. A network that ignores the pocket cannot beat this
reference in expectation; targets whose oracle recall is high leave little room for pocket-specific learning.

Only the label cells train_dev and dev_dev are read here (through the guarded API pocketgate.data.labels); no sealed test cell is
opened. Outputs: data/splits/v4_cv_folds.json, data/processed/v4_probcal_ligand_graphs.pt and the pod bundle directory.

Usage: python dockmut/v4_make_folds.py [--bundle PATH] [--no-bundle] [--bundle-only] [--n-val-ligands 10000]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketgate.common import PROCESSED_DIR, SPLITS_DIR, sha256_file  # noqa: E402
from pocketgate.data import labels as L  # noqa: E402
from pocketgate.data.features import featurize_ligand  # noqa: E402
from pocketgate.evaluation.metrics import recall_at  # noqa: E402

SEED = 20261009
N_VAL_TARGETS = 6
FOLD_FAMILIES = [
    ["kinase"],
    ["nuclear_receptor"],
    ["gpcr_classA", "cyp450", "maob"],
    ["aspartyl_protease", "metalloprotease", "serine_protease_S1", "nos1", "parp1"],
    ["pde5a", "ptgs2", "ptpn1"],
]
ALLOWED_CELLS = ["train_train", "dev_train", "train_dev", "dev_dev", "train_probcal", "dev_probcal"]
FOLDS_PATH = SPLITS_DIR / "v4_cv_folds.json"
PROBCAL_GRAPHS = PROCESSED_DIR / "v4_probcal_ligand_graphs.pt"
BUNDLE_SCRIPTS = ["v4_cv_train.py", "v1b_train.py"]
POD_REQUIREMENTS = "\n".join(["numpy>=1.26", "pandas>=2.2", "pyarrow>=14", "scipy>=1.11", "scikit-learn>=1.3", "torch>=2.4",
                              "rdkit>=2023.9.1"]) + "\n"


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_cell(target_role: str, ligand_role: str) -> pd.DataFrame:
    assert f"{target_role}_{ligand_role}" in ALLOWED_CELLS, f"label cell {target_role}_{ligand_role} is not allowed"
    return L.load_labels(target_role, ligand_role, caller="v4_make_folds")


# --------------------------------------------------------------------------- folds
def build_folds(tg: pd.DataFrame) -> list[dict]:
    non_test = tg[tg.split_role != "test"]
    fam_of = dict(zip(non_test.target_id, non_test.family))
    assert len(non_test) == 49 and non_test.target_id.is_unique, len(non_test)
    folds, seen = [], []
    for k, fams in enumerate(FOLD_FAMILIES):
        held = sorted(t for t, f in fam_of.items() if f in fams)
        assert all((non_test.family == f).any() for f in fams), f"unknown family in fold {k}: {fams}"
        train = sorted(set(fam_of) - set(held))
        val = sorted(np.random.default_rng(SEED).choice(train, size=N_VAL_TARGETS, replace=False).tolist())
        assert not set(held) & set(train) and set(val) <= set(train)
        folds.append({"fold": k, "families": fams, "heldout": held, "train": train, "validation": val})
        seen += held
    assert sorted(seen) == sorted(fam_of), "folds do not cover the 49 targets exactly once"
    return folds


# --------------------------------------------------------------------------- validation ligands
def featurize_probcal(n: int) -> tuple[dict, list[str]]:
    sp = pd.read_parquet(SPLITS_DIR / "ligand_splits.parquet")
    cand = sorted(sp[(sp.split_role == "probcal") & (sp.feature_status == "OK")].ligand_id)
    assert len(cand) == 26015, len(cand)
    lig = pd.read_parquet(PROCESSED_DIR / "ligands.parquet", columns=["ligand_id", "standardized_smiles"])
    smi_of = dict(zip(lig.ligand_id, lig.standardized_smiles))
    order = np.random.default_rng(SEED).permutation(len(cand))
    graphs, failed, t0 = {}, [], time.time()
    for i in order:
        lid = cand[i]
        g = featurize_ligand(smi_of[lid])
        if g is None:
            failed.append(lid)
            continue
        graphs[lid] = {k: torch.from_numpy(np.asarray(v)) for k, v in g.items()}
        if len(graphs) % 2500 == 0:
            print(f"  probcal ligands {len(graphs)}/{n} ({(time.time() - t0) / len(graphs) * 1000:.1f} ms/mol)", flush=True)
        if len(graphs) == n:
            break
    assert len(graphs) == n, f"only {len(graphs)} probcal ligands could be featurized"
    return {k: graphs[k] for k in sorted(graphs)}, failed


def check_featurizer_matches_g2(n_check: int = 300) -> int:
    """Re-featurize ligands that are in g2_ligand_graphs.pt and require tensor-identical graphs."""
    g2 = torch.load(PROCESSED_DIR / "g2_ligand_graphs.pt", weights_only=False)
    lig = pd.read_parquet(PROCESSED_DIR / "ligands.parquet", columns=["ligand_id", "standardized_smiles"])
    smi_of = dict(zip(lig.ligand_id, lig.standardized_smiles))
    ids = np.random.default_rng(0).choice(sorted(g2), size=n_check, replace=False)
    for lid in ids:
        g = featurize_ligand(smi_of[lid])
        for k, v in g.items():
            ref = g2[lid][k]
            assert torch.equal(torch.from_numpy(np.asarray(v)), ref), f"featurizer differs from g2 graphs for {lid}.{k}"
    return len(ids)


# --------------------------------------------------------------------------- oracle
def oracle_table(folds: list[dict], dev_ligands: list[str]) -> dict:
    lab = pd.concat([load_cell("train", "dev"), load_cell("dev", "dev")], ignore_index=True)
    lab = lab[lab.ligand_id.isin(set(dev_ligands))]
    ids = sorted(dev_ligands)
    mat = lab.pivot(index="target_id", columns="ligand_id", values="score").reindex(columns=ids)
    tb = np.arange(len(ids))   # ids are sorted, so the ligand_id tie-break of the project metrics is the column index
    per_target, n_missing_rank = {}, {}
    for f in folds:
        S = mat.loc[f["train"]].to_numpy(float)
        cnt = (~np.isnan(S)).sum(0)
        mean = np.where(cnt > 0, np.nansum(S, 0) / np.maximum(cnt, 1), np.nan)
        n_missing_rank[f["fold"]] = int(np.isnan(mean).sum())
        mean = np.where(np.isnan(mean), np.nanmax(mean) + 1.0, mean)   # ligands without any training-target score rank last
        pred = -mean
        for t in f["heldout"]:
            s = mat.loc[t].to_numpy(float)
            m = ~np.isnan(s)
            thr = np.quantile(s[m], 0.01)
            per_target[t] = {
                "fold": f["fold"],
                "recall_1@10": recall_at(pred, s, tb, 0.01, 0.10),
                "spearman": float(spearmanr(mean[m], s[m]).statistic),
                "n_ligands": int(m.sum()),
                "n_hits": int((s[m] <= thr).sum()),
            }
    return {"per_target": per_target, "ligands_without_ranking_score": n_missing_rank,
            "n_scores_missing_in_heldout_or_train": int(mat.isna().to_numpy().sum())}


# --------------------------------------------------------------------------- bundle
def script_deps(entry: Path, scripts_dir: Path) -> list[str]:
    """Names of the files in scripts/ that `entry` imports, followed transitively."""
    have = {p.stem: p.name for p in scripts_dir.glob("*.py")}
    out, todo = [], [entry.name]
    while todo:
        name = todo.pop()
        if name in out:
            continue
        out.append(name)
        for node in ast.walk(ast.parse((scripts_dir / name).read_text(encoding="utf-8"))):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
            todo += [have[m.split(".")[0]] for m in mods if m.split(".")[0] in have]
    return out


def build_bundle(bundle: Path) -> None:
    if bundle.exists() and any(bundle.iterdir()):
        assert (bundle / "bundle_manifest.json").exists(), f"{bundle} exists and is not a bundle, refusing to delete it"
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    shutil.copytree(ROOT / "src", bundle / "src", ignore=ignore)
    (bundle / "scripts").mkdir()
    needed = []
    for s in BUNDLE_SCRIPTS:
        needed += [d for d in script_deps(ROOT / "scripts" / s, ROOT / "scripts") if d not in needed]
    for s in needed:
        shutil.copy2(ROOT / "scripts" / s, bundle / "scripts" / s)
    (bundle / "configs").mkdir()
    shutil.copy2(ROOT / "configs" / "g2_config.json", bundle / "configs" / "g2_config.json")
    for name in ["targets.parquet", "ligand_splits.parquet", "v4_cv_folds.json"]:
        (bundle / "data" / "splits").mkdir(parents=True, exist_ok=True)
        shutil.copy2(SPLITS_DIR / name, bundle / "data" / "splits" / name)
    for name in ["g2_ligand_graphs.pt", "v1_pocket_graphs.pt", "v4_probcal_ligand_graphs.pt"]:
        (bundle / "data" / "processed").mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROCESSED_DIR / name, bundle / "data" / "processed" / name)
    vault = bundle / "data" / "label_vault"
    vault.mkdir(parents=True, exist_ok=True)
    for c in ALLOWED_CELLS:
        shutil.copy2(L.LABEL_VAULT_DIR / f"labels_{c}.parquet", vault / f"labels_{c}.parquet")
    names = sorted(p.name for p in vault.iterdir())
    assert not any("test" in n.lower() for n in names), f"test cell in bundle label_vault: {names}"
    assert names == sorted(f"labels_{c}.parquet" for c in ALLOWED_CELLS), names
    (bundle / "requirements_v4.txt").write_text(POD_REQUIREMENTS)
    files = sorted(p for p in bundle.rglob("*") if p.is_file())
    assert not any("test" in p.name.lower() for p in files), [p.name for p in files if "test" in p.name.lower()]
    manifest = {"files": {str(p.relative_to(bundle)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                          for p in files}, "label_vault_files": names}
    (bundle / "bundle_manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())
    print(f"bundle {bundle}: {len(files) + 1} files, scripts {needed}, label_vault {names}")
    print(f"bundle size {total / 1e6:.1f} MB ({total / 2 ** 30:.2f} GiB)")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(ROOT / "tmp" / "bundle_v4"))
    ap.add_argument("--no-bundle", action="store_true")
    ap.add_argument("--bundle-only", action="store_true", help="rebuild the bundle from the existing folds file and graphs")
    ap.add_argument("--n-val-ligands", type=int, default=10000)
    a = ap.parse_args()

    if not a.bundle_only:
        cfg = json.loads((ROOT / "configs" / "g2_config.json").read_text())
        tg = pd.read_parquet(SPLITS_DIR / "targets.parquet")
        folds = build_folds(tg)
        for f in folds:
            print(f"fold {f['fold']}: held out {len(f['heldout'])} ({'+'.join(f['families'])}), train {len(f['train'])}, "
                  f"validation {f['validation']}")
        assignment = {str(f["fold"]): {"heldout": f["heldout"], "train": f["train"], "validation": f["validation"]} for f in folds}
        assignment_sha = canonical_sha256(assignment)

        t0 = time.time()
        graphs, failed = featurize_probcal(a.n_val_ligands)
        torch.save(graphs, PROBCAL_GRAPHS)
        val_ids = sorted(graphs)
        print(f"probcal graphs: {len(graphs)} saved to {PROBCAL_GRAPHS} ({time.time() - t0:.0f} s), featurization failures skipped: "
              f"{len(failed)}", flush=True)
        n_chk = check_featurizer_matches_g2()
        print(f"featurizer reproduces {n_chk} stored g2 ligand graphs exactly", flush=True)

        dev_ligands = cfg["dev_ligands"]
        assert len(dev_ligands) == 10000
        orc = oracle_table(folds, dev_ligands)
        per_fold = {}
        for f in folds:
            r = [orc["per_target"][t]["recall_1@10"] for t in f["heldout"]]
            s = [orc["per_target"][t]["spearman"] for t in f["heldout"]]
            per_fold[str(f["fold"])] = {"mean_recall_1@10": float(np.mean(r)), "mean_spearman": float(np.mean(s)), "n_targets": len(r)}
            print(f"fold {f['fold']}: oracle Recall_1%@10% mean {np.mean(r):.3f} (min {np.min(r):.3f}, max {np.max(r):.3f}), "
                  f"Spearman mean {np.mean(s):.3f}")
        all_r = np.array([v["recall_1@10"] for v in orc["per_target"].values()])
        print(f"oracle over 49 targets: mean {all_r.mean():.3f}, targets below 0.5: {(all_r < 0.5).sum()}")
        print("ligands without any training-target score, per fold:", orc["ligands_without_ranking_score"])

        out = {
            "version": "v4_cv_folds_1",
            "seed": SEED,
            "sampling": "validation targets: default_rng(seed).choice(sorted R_k, 6, replace=False) per fold; validation ligands: "
                        "first n of default_rng(seed).permutation(sorted probcal ligand ids) that featurize",
            "fold_assignment_sha256": assignment_sha,
            "fold_assignment_note": "sha256 of json.dumps(fold_assignment, sort_keys=True, separators=(',', ':'))",
            "fold_assignment": assignment,
            "folds": folds,
            "validation_ligands": {"role": "probcal", "n": len(val_ids), "sha256": canonical_sha256(val_ids),
                                   "featurization_failures_skipped": failed, "ids": val_ids},
            "development_ligands": {"n": len(dev_ligands), "sha256": canonical_sha256(sorted(dev_ligands))},
            "label_cells_read": ["train_dev", "dev_dev"],
            "oracle": {"definition": "rank the 10,000 development ligands by the mean docking score over the training targets R_k "
                                     "(train_dev + dev_dev cells, lower is better); Recall_1%@10% of the held-out target's lowest 1% "
                                     "scores inside the best 10% of that ranking; Spearman between the mean score and the held-out "
                                     "score",
                       "per_target": orc["per_target"], "per_fold": per_fold,
                       "ligands_without_ranking_score": orc["ligands_without_ranking_score"],
                       "mean_recall_1@10_all_targets": float(all_r.mean())},
        }
        FOLDS_PATH.write_text(json.dumps(out, indent=1))
        print(f"wrote {FOLDS_PATH}")
        print(f"FOLD ASSIGNMENT SHA256 {assignment_sha}")
        print(f"validation ligand ids sha256 {out['validation_ligands']['sha256']}")
    else:
        assert FOLDS_PATH.exists() and PROBCAL_GRAPHS.exists(), "run without --bundle-only first"
        print("fold assignment sha256", json.loads(FOLDS_PATH.read_text())["fold_assignment_sha256"])

    if not a.no_bundle:
        build_bundle(Path(a.bundle))


if __name__ == "__main__":
    main()
