"""Probability calibration (plan 9.2).

Frozen model + L_probcal -> scalar temperature (positive-slope affine
logistic: p = sigmoid((a - b)/T) reduced to intercept+scale fit).
Method pre-selected: single-temperature scaling fit by NLL on the
probcal split, natural prevalence preserved (no resampling).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TemperatureCalibrator:
    def __init__(self):
        self.temperature = 1.0
        self.bias = 0.0
        self.fitted = False

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        mask = ~np.isnan(y)
        a = torch.from_numpy(logits[mask].astype(np.float64))
        t = torch.from_numpy(y[mask].astype(np.float64))
        log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_T, b], max_iter=200)
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = bce((a - b) / torch.exp(log_T), t)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(torch.exp(log_T).item())
        self.bias = float(b.item())
        self.fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        z = (np.asarray(logits, dtype=np.float64) - self.bias) / self.temperature
        return 1.0 / (1.0 + np.exp(-z))
