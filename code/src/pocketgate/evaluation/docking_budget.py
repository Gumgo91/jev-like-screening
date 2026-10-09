"""Query hit recovery with support observations charged to the docking budget.

Support and query pools are disjoint. Support hits are not credited toward the
fixed query hit set, but all attempted support observations consume budget.
This is query recovery under a total-cost constraint, not recovery over the
union of the support and query libraries.
"""
import numpy as np
import pandas as pd


def support_cost_screening(scores, predictions, support_queries, *,
                           hit_quantile=.01, budget_fraction=.1, ligand_ids=None):
    """Scores are lower-better, predictions higher-better, both (T,N).

    Budget is round(budget_fraction * N_valid_query), consistent with the
    original query-screening metric. Spend support_queries first, then dock
    max(0, budget-support_queries) predicted query leaders. Missing predictions
    are not selected and never reduce the reference hit denominator.
    support_queries is a scalar or one nonnegative integer count per target;
    it includes failed docking attempts, not just finite observed labels.
    """
    score, pred = np.asarray(scores, float), np.asarray(predictions, float)
    if score.ndim != 2 or pred.shape != score.shape:
        raise ValueError('scores and predictions must have equal (T,N) shape')
    if np.isinf(score).any() or np.isinf(pred).any():
        raise ValueError('Infinity is invalid; use NaN for missing data')
    if not 0 < hit_quantile <= 1 or not 0 < budget_fraction <= 1:
        raise ValueError('Fractions must be in (0,1]')
    costs = np.asarray(support_queries, float)
    if costs.ndim == 0:
        costs = np.full(len(score), costs)
    if costs.shape != (len(score),) or not np.isfinite(costs).all() or (costs < 0).any() or (costs != np.floor(costs)).any():
        raise ValueError('Support query counts must be nonnegative integers per target')
    ids = np.arange(score.shape[1]).astype(str) if ligand_ids is None else np.asarray(ligand_ids).astype(str)
    if ids.shape != (score.shape[1],) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique ligand_ids must match query columns')
    rows = []
    metric_name = f'recall_{100*hit_quantile:g}_at_total{100*budget_fraction:g}'
    for t in range(len(score)):
        valid = np.isfinite(score[t])
        n = int(valid.sum())
        total = max(1, int(round(budget_fraction*n))) if n else 0
        cost = int(costs[t])
        remaining = max(0, total-cost)
        threshold = float(np.quantile(score[t, valid], hit_quantile)) if n else np.nan
        hits = valid & (score[t] <= threshold)
        eligible = np.flatnonzero(valid & np.isfinite(pred[t]))
        ranked = eligible[np.lexsort((ids[eligible], -pred[t, eligible]))]
        found = int(hits[ranked[:remaining]].sum())
        n_hits = int(hits.sum())
        rows.append({'target_id': t, 'n_valid_query': n, 'n_query_hits': n_hits,
                     'n_support_queries': cost, 'n_total_docking_budget': total,
                     'n_query_docks': min(remaining, len(ranked)),
                     'support_exceeds_budget': cost > total, 'n_found_query_hits': found,
                     metric_name: found/n_hits if n_hits else np.nan,
                     'hit_quantile': hit_quantile, 'total_budget_fraction': budget_fraction})
    return pd.DataFrame(rows)
