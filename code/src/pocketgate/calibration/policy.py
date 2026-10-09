"""Selection policies (plan 9.1, 9.3).

- FixedBudgetPolicy: dock top-b% by predicted score.
- EmpiricalThresholdPolicy: threshold on calibrated p_hit chosen on
  L_policycal to hit a target recall level (per-target if enough data,
  else pooled - flagged).
- ConformalPolicy (G3 extension, class-conditional, alpha=0.05):
  a_i = 1 - p_i on positive examples; q_+ = k-th smallest with
  k = ceil((n_+ + 1)(1 - alpha)); keep ligand if 1 - p_new <= q_+.
  n_+ = 0 -> policy unset -> keep all (documented fallback).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


class FixedBudgetPolicy:
    policy_id = "fixed_budget"

    def __init__(self, budget: float = 0.10):
        self.budget = budget

    def select(self, pred: pd.DataFrame) -> pd.Series:
        out = pd.Series(False, index=pred.index)
        for _, sub in pred.groupby("target_id"):
            n = len(sub)
            k = max(1, int(round(self.budget * n)))
            order = sub["p_hit"].sort_values(ascending=False).index
            out.loc[order[:k]] = True
        return out


class EmpiricalThresholdPolicy:
    """Threshold on p_hit estimated on policy-calibration data to reach
    target_recall of tau_P hits there (pooled across targets)."""
    policy_id = "empirical_threshold"

    def __init__(self, target_recall: float = 0.95):
        self.target_recall = target_recall
        self.threshold_ = None

    def fit(self, cal: pd.DataFrame) -> "EmpiricalThresholdPolicy":
        """cal: p_hit + hit (label) on L_policycal x P_train."""
        pos = cal.loc[cal["hit"] == 1, "p_hit"].to_numpy()
        if len(pos) == 0:
            self.threshold_ = None
            return self
        self.threshold_ = float(np.quantile(pos, 1 - self.target_recall))
        return self

    def select(self, pred: pd.DataFrame) -> pd.Series:
        if self.threshold_ is None:
            return pd.Series(True, index=pred.index)  # keep-all fallback
        return pred["p_hit"] >= self.threshold_


class ConformalPolicy:
    policy_id = "class_conditional_conformal"

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_plus = None

    def fit(self, cal: pd.DataFrame) -> "ConformalPolicy":
        pos = cal.loc[cal["hit"] == 1, "p_hit"].to_numpy()
        n = len(pos)
        if n == 0:
            self.q_plus = None
            return self
        a = np.sort(1.0 - pos)
        k = math.ceil((n + 1) * (1 - self.alpha))
        self.q_plus = float(a[k - 1]) if k <= n else float("inf")
        return self

    def select(self, pred: pd.DataFrame) -> pd.Series:
        if self.q_plus is None or not np.isfinite(self.q_plus):
            return pd.Series(True, index=pred.index)
        return (1.0 - pred["p_hit"]) <= self.q_plus
