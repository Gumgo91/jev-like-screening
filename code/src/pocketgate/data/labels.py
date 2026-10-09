"""Split-aware label access with a test guard (plan 5.3).

Label files live in data/label_vault/labels_<trole>_<lrole>.parquet.
Callers must declare which cells they may read; any request touching
test cells (target_role == 'test' or ligand_role == 'test') requires
`allow_test=True`, which only the final evaluator passes. Accesses are
appended to runs/label_access.log.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from pocketgate.common import LABEL_VAULT_DIR, RUNS_DIR


def _access_log() -> Path:
    return RUNS_DIR / "label_access.log"


def load_labels(target_role: str, ligand_role: str,
                allow_test: bool = False,
                caller: str = "unknown") -> pd.DataFrame:
    if target_role == "test" or ligand_role == "test":
        if not allow_test:
            raise PermissionError(
                f"sealed cell {target_role}x{ligand_role}: "
                "final evaluator only")
    path = LABEL_VAULT_DIR / f"labels_{target_role}_{ligand_role}.parquet"
    df = pd.read_parquet(path)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_access_log(), "a") as f:
        f.write(json.dumps({
            "ts": time.time(), "caller": caller,
            "cell": f"{target_role}_{ligand_role}", "n": len(df),
        }) + "\n")
    return df


def train_labels(target_ids: list[str] | None = None,
                 caller: str = "train") -> pd.DataFrame:
    """P_train x L_train only."""
    df = load_labels("train", "train", caller=caller)
    if target_ids is not None:
        df = df[df["target_id"].isin(target_ids)]
    return df.reset_index(drop=True)


def dev_labels(target_ids: list[str] | None = None,
               ligand_role: str = "dev",
               caller: str = "dev") -> pd.DataFrame:
    """P_* x L_dev for model selection (train or dev targets)."""
    frames = []
    for tr in ("train", "dev"):
        try:
            frames.append(load_labels(tr, ligand_role, caller=caller))
        except FileNotFoundError:
            pass
    df = pd.concat(frames, ignore_index=True)
    if target_ids is not None:
        df = df[df["target_id"].isin(target_ids)]
    return df.reset_index(drop=True)


def probcal_labels(caller: str = "calibration") -> pd.DataFrame:
    return load_labels("train", "probcal", caller=caller)


def policycal_labels(caller: str = "policy") -> pd.DataFrame:
    return load_labels("train", "policycal", caller=caller)
