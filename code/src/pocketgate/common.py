"""Shared constants and small utilities for PocketGate.

Conventions:
- Docking score direction: LOWER is better (DOCKSTRING / Vina convention).
- hit label: y = 1[score <= tau_P], tau_P = 1% quantile over train ligands per target.
- All hashes are sha256 hex unless noted.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
LABEL_VAULT_DIR = DATA_DIR / "label_vault"
TARGETS_DIR = RAW_DIR / "targets"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"
RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"

FEATURIZER_VERSION = "pg_feat_v1"
PREPARATION_VERSION = "dockstring_pdbqt_b539832"

# Quantile used to define a hit on TRAIN ligands per target (plan 3.2)
TRAIN_HIT_QUANTILE = 0.01


def sha256_file(path: os.PathLike | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def list_target_names() -> list[str]:
    """Target names = stems of *_target.pdbqt files (matches TSV columns)."""
    names = sorted(
        p.name[: -len("_target.pdbqt")]
        for p in TARGETS_DIR.glob("*_target.pdbqt")
    )
    return names
