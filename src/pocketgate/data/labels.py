"""Access to the per-cell DOCKSTRING label files.

A cell holds the labels of one target role x one ligand role and lives in
LABELS_DIR (pocketgate.common) as labels_<trole>_<lrole>.parquet, written by
pocketgate.data.split. No experiment of the paper reads a cell with the
test role: load_labels refuses such a cell unless allow_test=True is passed
explicitly.
"""
from __future__ import annotations

import pandas as pd

from pocketgate.common import LABELS_DIR  # noqa: F401


def load_labels(target_role: str, ligand_role: str,
                allow_test: bool = False,
                caller: str = "unknown") -> pd.DataFrame:
    """Read the label cell (target_role, ligand_role) as a DataFrame with
    the columns target_id, ligand_id, score.

    `caller` names the calling script at the call site; it has no effect.
    """
    if target_role == "test" or ligand_role == "test":
        if not allow_test:
            raise PermissionError(
                f"label cell {target_role}x{ligand_role} has the test role; "
                "pass allow_test=True to read it")
    path = LABELS_DIR / f"labels_{target_role}_{ligand_role}.parquet"
    return pd.read_parquet(path)
