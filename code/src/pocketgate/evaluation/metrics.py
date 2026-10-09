"""Screening metrics (plan section 10).

Conventions:
- score: LOWER docking score is better.
- model output: HIGHER p/logit = more likely hit -> rank descending.
- H_q: ligands with score <= q-quantile of the eval set (ties included).
- A_b: top max(1, round(b*N)) predictions, deterministic tie-break by
  (pred desc, ligand_id asc).
- All metrics computed per target, then macro-averaged by the caller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Q_LIST = (0.01, 0.05)
B_LIST = (0.01, 0.05, 0.10, 0.20)


def _rank_desc(pred: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Indices sorted by pred desc, tiebreak asc."""
    return np.lexsort((tiebreak, -pred))


def recall_at(pred: np.ndarray, scores: np.ndarray, tiebreak: np.ndarray,
              q: float, b: float) -> float:
    """Recall of true top-q% hits within predicted top-b%."""
    mask = ~np.isnan(scores)
    pred, scores, tb = pred[mask], scores[mask], tiebreak[mask]
    n = len(scores)
    if n == 0:
        return float("nan")
    thr = np.quantile(scores, q)
    H = scores <= thr
    if H.sum() == 0:
        return float("nan")
    k = max(1, int(round(b * n)))
    top = _rank_desc(pred, tb)[:k]
    return float(H[top].sum() / H.sum())


def recall95_budget(pred: np.ndarray, scores: np.ndarray,
                    tiebreak: np.ndarray, q: float = 0.01) -> float:
    """Smallest fraction of library to select to reach 95% of top-q% hits.

    np.inf if unreachable. Descriptive statistic on a finite library.
    """
    mask = ~np.isnan(scores)
    pred, scores, tb = pred[mask], scores[mask], tiebreak[mask]
    n = len(scores)
    if n == 0:
        return float("nan")
    thr = np.quantile(scores, q)
    H = scores <= thr
    need = int(np.ceil(0.95 * H.sum()))
    if H.sum() == 0:
        return float("nan")
    order = _rank_desc(pred, tb)
    cum = np.cumsum(H[order])
    idx = np.searchsorted(cum, need)
    if idx >= n:
        return float("inf")
    return float((idx + 1) / n)


def enrichment_factor(pred, scores, tiebreak, q=0.01, b=0.01) -> float:
    mask = ~np.isnan(scores)
    pred, scores, tb = pred[mask], scores[mask], tiebreak[mask]
    n = len(scores)
    if n == 0:
        return float("nan")
    thr = np.quantile(scores, q)
    H = scores <= thr
    k = max(1, int(round(b * n)))
    top = _rank_desc(pred, tb)[:k]
    base = H.sum() / n
    return float(H[top].sum() / k / base) if base > 0 else float("nan")


def per_target_metrics(df: pd.DataFrame, q_list=Q_LIST, b_list=B_LIST,
                       pred_col="p_hit", score_col="score",
                       id_col="ligand_id") -> dict:
    """df: predictions+labels for ONE target. Returns metric dict."""
    pred = df[pred_col].to_numpy(float)
    scores = df[score_col].to_numpy(float)
    # deterministic tiebreak: stable ligand_id ordering
    tb = df[id_col].map(
        {v: i for i, v in enumerate(sorted(df[id_col].unique()))}).to_numpy()
    out = {}
    for q in q_list:
        for b in b_list:
            out[f"recall_{int(q*100)}@{int(b*100)}"] = recall_at(
                pred, scores, tb, q, b)
    out["recall95_budget"] = recall95_budget(pred, scores, tb)
    out["ef1@1"] = enrichment_factor(pred, scores, tb, 0.01, 0.01)
    m = ~np.isnan(scores)
    if m.sum() > 2 and np.unique(scores[m]).size > 1:
        from scipy.stats import spearmanr
        out["spearman"] = float(spearmanr(-pred[m], scores[m]).statistic)
    else:
        out["spearman"] = float("nan")
    # AUPRC/NLL/Brier on the eval-set hit label (top-1% as hit)
    if m.sum() > 2:
        thr = np.quantile(scores[m], 0.01)
        y = (scores[m] <= thr).astype(int)
        p = np.clip(pred[m], 1e-9, 1 - 1e-9)
        from sklearn.metrics import average_precision_score, brier_score_loss
        out["auprc"] = float(average_precision_score(y, p)) \
            if y.sum() > 0 else float("nan")
        out["nll"] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        out["brier"] = float(brier_score_loss(y, p))
        out["hit_rate_eval"] = float(y.mean())
    out["n_ligands"] = int(m.sum())
    return out


def macro_eval(df: pd.DataFrame, target_col="target_id", **kw) -> pd.DataFrame:
    rows = []
    for t, sub in df.groupby(target_col):
        r = per_target_metrics(sub, **kw)
        r[target_col] = t
        rows.append(r)
    return pd.DataFrame(rows)
