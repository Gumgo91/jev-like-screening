"""Shared paths and helpers of the manuscript generators.

Every generator in this folder imports this module. All paths are built from ROOT, the repository root (the parent of
manuscript/), so the scripts run from any working directory.

Inputs (all shipped in the repository)
  eval/v4_fair_timing.json         full-scale timing on the RTX 4090 pod (the primary timing, "A" in the macros)
  eval/v4_fair_timing_L40S.json    second full-scale timing on an L40S pod ("B")
  eval/docking_timing*.csv         per-call record of the Uni-Dock benchmark
  eval/v5_known_cv_summary.json, eval/v5_known_pockets_summary.json, eval/v4_cv_summary.json   recall summaries
  runs/v3/registry_*.json, runs/v4/registry_*.json                                           run records
  data/splits/targets.parquet, data/splits/v4_cv_folds.json                                   split definitions
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
FIGURES = MS / "figures"
EVAL = ROOT / "eval"
RUNS = ROOT / "runs"
RUNS_V3 = RUNS / "v3"
RUNS_V4 = RUNS / "v4"
SPLITS = ROOT / "data" / "splits"

# Full-scale timing records. The first is the primary one (the RTX 4090 pod); there is no fallback between the two files.
TIMING_PRIMARY = EVAL / "v4_fair_timing.json"
TIMING_SECOND = EVAL / "v4_fair_timing_L40S.json"

DOCKING_TIMING_FILES = tuple(EVAL / n for n in ("docking_timing.csv", "docking_timing_big.csv", "docking_timing_4k.csv"))
DOCKING_SCORE_FILES = tuple(EVAL / n for n in ("docking_scores.csv", "docking_scores_big.csv", "docking_scores_4k.csv"))
PREP_LOG_4K = EVAL / "prep4k.log"

KNOWN_CV_SUMMARY = EVAL / "v5_known_cv_summary.json"
KNOWN_POCKETS_SUMMARY = EVAL / "v5_known_pockets_summary.json"
HELDOUT_FAMILY_SUMMARY = EVAL / "v4_cv_summary.json"
TARGETS_PARQUET = SPLITS / "targets.parquet"
FOLDS_JSON = SPLITS / "v4_cv_folds.json"

# Pocket counts of the timing blocks, and the number of pockets of the Uni-Dock comparison.
N_POCKETS = 57


def load_timing(path: Path) -> dict:
    """Load a full-scale timing file. Raises if the file is missing or lacks the cross-attention float16 block for 1,000 pockets."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"timing file not found: {path}")
    T = json.loads(path.read_text(encoding="utf-8"))
    if "1000" not in T.get("blocks", {}).get("xattn_fp16", {}):
        raise ValueError(f"{path.name} has no xattn_fp16 block for 1000 pockets; the timing record is incomplete "
                         "(there is no fallback to another timing file)")
    return T


def gpu_name(T: dict) -> str:
    """Short GPU name of a timing record, for example 'RTX 4090'."""
    return T["environment"]["gpu"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")


def thousands(x) -> str:
    """Integer with LaTeX thousands separators, for example 4{,}093."""
    return f"{int(round(x)):,}".replace(",", "{,}")


def docking_timing():
    """Timed calls of the Uni-Dock benchmark (warm-up calls removed) with the column rate = ligands docked per second."""
    import pandas as pd
    for f in DOCKING_TIMING_FILES:
        if not f.exists():
            raise FileNotFoundError(f"docking timing file not found: {f}")
    dt = pd.concat([pd.read_csv(f) for f in DOCKING_TIMING_FILES])
    dt = dt[dt.warmup == 0].copy()
    dt["rate"] = dt.n_ok / dt.sec
    return dt


def unidock_rates() -> dict:
    """Median Uni-Dock rates in ligands per second, pooled over the receptors, from the timed calls of eval/docking_timing*.csv.

    detail:  best number of ligands per call in the detail mode (median rate over the calls of that setting)
    fast:    the same for the fast mode (42.4646 ligands per second at 4,000 ligands per call)
    fastest: the highest median rate over all modes and numbers of ligands per call
    fastest_config: (mode, ligands per call) of that setting
    """
    dt = docking_timing()
    out = {mode: float(dt[dt["mode"] == mode].groupby("n").rate.median().max()) for mode in ("detail", "fast")}
    best = dt.groupby(["mode", "n"]).rate.median()
    out["fastest"] = float(best.max())
    mode, n = best.idxmax()
    out["fastest_config"] = (str(mode), int(n))
    return out


def used_macros(exclude=("numbers.tex", "numbers_v5.tex")) -> set[str]:
    """Names of all control sequences that occur in the LaTeX sources of the manuscript (comments removed).
    The number generators write only the macros that the text uses."""
    bs = chr(92)
    seq = re.compile(re.escape(bs) + r"([A-Za-z]+)")
    comment = re.compile(r"(?<!" + re.escape(bs) + r")%.*")
    names: set[str] = set()
    for f in sorted(MS.glob("*.tex")):
        if f.name in exclude:
            continue
        names |= set(seq.findall(comment.sub("", f.read_text(encoding="utf-8"))))
    return names


def defined_macros(path: Path) -> set[str]:
    """Names defined by \\newcommand lines of a numbers file (empty if the file does not exist)."""
    path = Path(path)
    if not path.exists():
        return set()
    bs = chr(92)
    return set(re.findall(re.escape(bs + "newcommand{" + bs) + r"([A-Za-z]+)\}", path.read_text(encoding="utf-8")))


def write_macros(out: Path, defs: dict[str, str], keep=None, header: str = "") -> int:
    """Write \\newcommand lines for the macros in defs (sorted by name). With keep, only names in keep are written.
    Returns the number of macros written."""
    names = sorted(k for k in defs if keep is None or k in keep)
    text = (header + "\n" if header else "") + "\n".join("\\newcommand{\\%s}{%s}" % (k, defs[k]) for k in names) + "\n"
    Path(out).write_text(text, encoding="utf-8")
    return len(names)
