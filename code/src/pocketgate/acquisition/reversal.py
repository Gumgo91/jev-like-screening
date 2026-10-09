"""Label-blind utilities for selecting the next docking observation.

Train-derived reversal weights are computed separately from acquisition.
The selector takes predictions, covariance, and observed flags only; it does
not receive held-out query labels, target hit thresholds, or evaluation quads.

The pair-order utility is a heuristic under a Gaussian residual-surrogate
assumption. Its ordering probability/entropy is not an empirically calibrated
probability of correct docking order. ``mean`` MUST be the full raw docking
score prediction, including the ligand prior, rather than residual mean.
Covariance/noise are in squared raw docking-score units. Covariance is latent
residual uncertainty; observation noise is added only in the update denominator.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def _real_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error


def _pairs(value, n_ligands):
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("pairs must have shape (P,2)")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("pairs must contain integer indices")
    if (array < 0).any() or (array >= n_ligands).any():
        raise ValueError("pair index outside the ligand pool")
    if (array[:, 0] == array[:, 1]).any():
        raise ValueError("a pair must contain two distinct ligand indices")
    return array.astype(np.int64, copy=False)


def train_reversal_weights(train_scores, pairs, margin=1.0):
    """Fraction of eligible unordered train-target pairs with opposite orders.

    ``train_scores`` is (T,N), raw lower-is-better scores; ``pairs`` is (P,2).
    A target is eligible for a ligand pair when both scores are finite, their
    difference is nonzero, and its magnitude is >= margin. With m eligible
    targets, p positive and q negative differences, return 2*p*q/(m*(m-1)).
    Fewer than two eligible targets gives weight zero. NaN denotes unavailable
    train observations and is excluded; infinities are rejected. Ties are
    excluded even at margin=0. Returns float64 (P,), between zero and one.
    """
    score = _real_array(train_scores, "train_scores")
    if score.ndim != 2 or score.shape[1] == 0:
        raise ValueError("train_scores must have shape (T,N) with N > 0")
    if np.isinf(score).any():
        raise ValueError("train_scores may contain missing NaN, but not infinity")
    if not np.isscalar(margin) or not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    pair = _pairs(pairs, score.shape[1])
    a, b = score[:, pair[:, 0]], score[:, pair[:, 1]]
    available = np.isfinite(a) & np.isfinite(b)
    difference = np.where(available, a, 0.0) - np.where(available, b, 0.0)
    if not np.isfinite(difference).all():
        raise ValueError("finite train scores produced an overflowing difference")
    eligible = available & (difference != 0) & (np.abs(difference) >= margin)
    positive = (eligible & (difference > 0)).sum(axis=0).astype(np.float64)
    negative = (eligible & (difference < 0)).sum(axis=0).astype(np.float64)
    count = positive + negative
    denominator = count * (count - 1)
    weight = np.zeros(len(pair), dtype=np.float64)
    np.divide(2 * positive * negative, denominator, out=weight,
              where=denominator > 0)
    return weight


def _ordering_entropy(mean_difference, pair_variance):
    """Bernoulli ordering entropy in bits under a Gaussian surrogate."""
    entropy = np.zeros_like(pair_variance)
    uncertain = pair_variance > 0
    z = mean_difference[uncertain] / np.sqrt(pair_variance[uncertain])
    probability = ndtr(z)
    complement = ndtr(-z)  # More accurate than 1-p in the right tail.
    h = np.zeros_like(probability)
    positive = probability > 0
    h[positive] -= probability[positive] * np.log2(probability[positive])
    positive = complement > 0
    h[positive] -= complement[positive] * np.log2(complement[positive])
    entropy[uncertain] = h
    # Degenerate latent order has no uncertainty, including an exact known tie.
    return entropy


def acquisition_scores(policy, mean, covariance, pairs, weights, observed_mask,
                       noise_variance, *, pair_chunk_size=256):
    """Return (N,) utilities, higher-is-better, observed cells set to -inf.

    Policies:
      greedy: -full_mean, since lower docking score is better;
      max_variance: diagonal latent residual covariance;
      pair_order: sum pair ordering entropy times expected variance reduction;
      reversal_order: the same sum multiplied by train-derived pair weights.
    Random selection belongs to the caller's seeded RNG.

    For pair (i,j), observing x reduces latent difference variance by
    Cov(x,r_i-r_j)^2 / (Var(r_x)+noise_variance). Pair-order entropy uses the
    FULL prediction mean_i-mean_j; the fixed prior's difference does not cancel.
    Pair chunks bound temporary memory to O(N*pair_chunk_size). Empty pairs or
    zero reversal weights give zero pair-policy utilities on unobserved cells.

    Covariance must be symmetric and positive semidefinite. This routine checks
    finite values, symmetry, nonnegative marginal/pair variances and pairwise
    update bounds, without an O(N^3) full eigendecomposition on every call.
    """
    if policy not in {"greedy", "max_variance", "pair_order", "reversal_order"}:
        raise ValueError("unknown policy; random acquisition is caller-owned")
    mu = _real_array(mean, "mean")
    cov = _real_array(covariance, "covariance")
    if mu.ndim != 1 or len(mu) == 0 or not np.isfinite(mu).all():
        raise ValueError("mean must be a finite nonempty (N,) full-score vector")
    n = len(mu)
    if cov.shape != (n, n) or not np.isfinite(cov).all():
        raise ValueError("covariance must be finite with shape (N,N)")
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(cov))))
    if not np.allclose(cov, cov.T, rtol=1e-8, atol=tolerance):
        raise ValueError("covariance must be symmetric")
    cov = (cov + cov.T) * 0.5
    diagonal = cov.diagonal().copy()
    if (diagonal < -tolerance).any():
        raise ValueError("covariance has a negative marginal variance")
    diagonal = np.maximum(diagonal, 0.0)
    pair = _pairs(pairs, n)
    weight = _real_array(weights, "weights")
    if weight.shape != (len(pair),) or not np.isfinite(weight).all():
        raise ValueError("weights must be finite with shape (P,)")
    if (weight < 0).any() or (weight > 1).any():
        raise ValueError("reversal weights must be in [0,1]")
    observed = np.asarray(observed_mask)
    if observed.dtype != np.bool_ or observed.shape != (n,):
        raise ValueError("observed_mask must be bool with shape (N,)")
    if (not np.isscalar(noise_variance) or not np.isfinite(noise_variance)
            or noise_variance <= 0):
        raise ValueError("noise_variance must be finite and positive")
    if (isinstance(pair_chunk_size, (bool, np.bool_))
            or not isinstance(pair_chunk_size, (int, np.integer))
            or pair_chunk_size <= 0):
        raise ValueError("pair_chunk_size must be a positive integer")

    if policy == "greedy":
        utility = -mu.copy()
    elif policy == "max_variance":
        utility = diagonal.copy()
    else:
        utility = np.zeros(n, dtype=np.float64)
        denominator = diagonal + float(noise_variance)
        for start in range(0, len(pair), pair_chunk_size):
            chunk = pair[start:start + pair_chunk_size]
            i, j = chunk.T
            pair_variance = diagonal[i] + diagonal[j] - 2 * cov[i, j]
            if (pair_variance < -4 * tolerance).any():
                raise ValueError("covariance has a negative pair-difference variance")
            pair_variance = np.maximum(pair_variance, 0.0)
            difference = mu[i] - mu[j]
            if not np.isfinite(difference).all():
                raise ValueError("full mean produced an overflowing difference")
            entropy = _ordering_entropy(difference, pair_variance)
            if policy == "reversal_order":
                entropy *= weight[start:start + pair_chunk_size]
            cross = cov[:, i] - cov[:, j]
            # Division before squaring avoids unnecessary overflow in cov^2.
            reduction = (cross / np.sqrt(denominator[:, None])) ** 2
            if not np.isfinite(reduction).all():
                raise ValueError("covariance produced a non-finite variance reduction")
            if (reduction > pair_variance[None, :] + 8 * tolerance).any():
                raise ValueError("covariance update exceeds pair variance; check PSD")
            utility += reduction @ entropy
    if not np.isfinite(utility).all():
        raise ValueError("acquisition utility is non-finite")
    utility[observed] = -np.inf
    return utility
