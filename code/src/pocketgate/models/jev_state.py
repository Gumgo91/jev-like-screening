"""Support-state regressors over precomputed ligand and pocket vectors.

This is a research adaptation of option-query attention, not an implementation
of TypeSafe's private Jev. Support-conditioned molecular prediction has prior
art, including conditional neural processes. The attention, pooled CNP, and
kernel-ridge controls below share the same fixed-prior/centering convention.

Inputs are tensors; this module neither reads labels nor fits the external
ligand prior. ``residualize=True`` uses y - support_baseline as state scores
and adds candidate_baseline to the prediction. ``False`` uses raw scores and
ignores both priors. State scores are centered/scaled over valid support only.
Support/candidate priors are treated as fixed (detached) inputs.

Build state once per target in eval/no_grad mode and score candidate chunks.
Rebuild it after any parameter update. Queries never attend to other queries;
there is no softmax over candidates. Variance is a model estimate, not a
coverage guarantee or a calibrated confidence interval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class StatePrediction(TypedDict):
    score: Tensor
    variance: Tensor


@dataclass(frozen=True)
class StateCache:
    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    mask: Tensor
    pooled: Tensor
    location: Tensor
    scale: Tensor
    support_count: Tensor
    residualize: bool


@dataclass(frozen=True)
class KernelStateCache:
    support: Tensor
    support_mask: Tensor
    coefficients: Tensor
    cholesky: Tensor
    location: Tensor
    scale: Tensor
    residualize: bool


def _mask(mask: Tensor | None, shape: tuple[int, int], device) -> Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if mask.dtype != torch.bool or tuple(mask.shape) != shape:
        raise ValueError(f"mask must be bool with shape {shape}")
    if mask.device != device:
        raise ValueError("mask and vectors must be on the same device")
    return mask


def _fixed_prior(prior: Tensor | None, reference: Tensor) -> Tensor:
    if prior is None:
        return torch.zeros_like(reference)
    if prior.shape != reference.shape or prior.device != reference.device:
        raise ValueError("baseline must match score shape and device")
    return prior.detach().to(reference.dtype)


def _finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains a non-finite unmasked value")


def _support_inputs(support: Tensor, scores: Tensor, support_mask: Tensor | None,
                    baseline: Tensor | None, residualize: bool,
                    ligand_dim: int, scale_floor: float):
    if support.ndim == 2:
        support = support.unsqueeze(0)
        scores = scores.unsqueeze(0)
        if support_mask is not None:
            support_mask = support_mask.unsqueeze(0)
        if baseline is not None:
            baseline = baseline.unsqueeze(0)
    if support.ndim != 3 or support.shape[-1] != ligand_dim:
        raise ValueError("support_vectors must have shape (B,S,ligand_dim)")
    if scores.shape != support.shape[:2] or scores.device != support.device:
        raise ValueError("support_scores must have shape (B,S) on vector device")
    if not support.is_floating_point() or not scores.is_floating_point():
        raise ValueError("vectors and scores must be floating point")
    mask = _mask(support_mask, tuple(scores.shape), support.device)
    support = torch.where(mask[..., None], support, torch.zeros_like(support))
    scores = torch.where(mask, scores, torch.zeros_like(scores)).to(support.dtype)
    _finite(support, "support_vectors")
    _finite(scores, "support_scores")
    if residualize:
        prior = _fixed_prior(baseline, scores)
        prior = torch.where(mask, prior, torch.zeros_like(prior))
        _finite(prior, "support_baseline")
        scores = scores - prior
    count = mask.sum(dim=1, keepdim=True).to(scores.dtype)
    location = scores.sum(dim=1, keepdim=True) / count.clamp_min(1)
    centered = torch.where(mask, scores - location, torch.zeros_like(scores))
    # A positive floor also avoids undefined sqrt gradients on constant state.
    scale = (centered.square().sum(1, keepdim=True)
             / count.clamp_min(1) + scale_floor ** 2).sqrt()
    scale = torch.where(count > 0, scale, torch.ones_like(scale))
    return support, centered, mask, location, scale, count


def _candidate_inputs(candidates: Tensor, baseline: Tensor | None,
                      batch_size: int, ligand_dim: int, residualize: bool):
    single = candidates.ndim == 2
    if single:
        candidates = candidates.unsqueeze(0)
        if baseline is not None:
            baseline = baseline.unsqueeze(0)
    if (candidates.ndim != 3 or candidates.shape[0] != batch_size
            or candidates.shape[-1] != ligand_dim):
        raise ValueError("candidate_vectors must have shape (B,N,ligand_dim)")
    _finite(candidates, "candidate_vectors")
    reference = candidates.new_zeros(candidates.shape[:2])
    prior = _fixed_prior(baseline, reference) if residualize else reference
    _finite(prior, "candidate_baseline")
    return candidates, prior, single


class _StateRegressor(nn.Module):
    def __init__(self, ligand_dim: int, pocket_dim: int, hidden_dim: int,
                 residualize: bool, scale_floor: float):
        super().__init__()
        if min(ligand_dim, pocket_dim, hidden_dim) <= 0 or scale_floor <= 0:
            raise ValueError("dimensions and scale_floor must be positive")
        self.ligand_dim = ligand_dim
        self.pocket_dim = pocket_dim
        self.hidden_dim = hidden_dim
        self.residualize = residualize
        self.scale_floor = scale_floor
        self.query = nn.Sequential(nn.Linear(ligand_dim, hidden_dim), nn.SiLU())
        self.support = nn.Sequential(
            nn.Linear(ligand_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.pocket = nn.Sequential(nn.Linear(pocket_dim, hidden_dim), nn.SiLU())
        self.token_type = nn.Parameter(torch.zeros(2, hidden_dim))
        self.score_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1))
        self.variance_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1))
        # Start from the external prior plus observed target residual mean.
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def _tokens(self, pocket_vectors, support_vectors, support_scores,
                support_mask, support_baseline, pocket_mask):
        support, centered, smask, location, scale, count = _support_inputs(
            support_vectors, support_scores, support_mask, support_baseline,
            self.residualize, self.ligand_dim, self.scale_floor)
        if pocket_vectors.ndim == 1:
            pocket_vectors = pocket_vectors[None, None, :]
        elif pocket_vectors.ndim == 2:
            pocket_vectors = pocket_vectors[:, None, :]
        if (pocket_vectors.ndim != 3
                or pocket_vectors.shape[0] != support.shape[0]
                or pocket_vectors.shape[-1] != self.pocket_dim
                or pocket_vectors.device != support.device):
            raise ValueError("pocket_vectors must have shape (B,P,pocket_dim)")
        pmask = _mask(pocket_mask, tuple(pocket_vectors.shape[:2]), support.device)
        pocket_vectors = torch.where(pmask[..., None], pocket_vectors,
                                     torch.zeros_like(pocket_vectors))
        _finite(pocket_vectors, "pocket_vectors")
        p = self.pocket(pocket_vectors) + self.token_type[0]
        s = self.support(torch.cat([support, (centered / scale)[..., None]], -1))
        s = s + self.token_type[1]
        tokens = torch.cat([p, s], dim=1)
        mask = torch.cat([pmask, smask], dim=1)
        if not mask.any(dim=1).all():
            raise ValueError("each state needs a valid pocket or support token")
        # Separate means prevent support count from diluting the pocket token.
        pp = (p * pmask[..., None]).sum(1) / pmask.sum(1, keepdim=True).clamp_min(1)
        sp = (s * smask[..., None]).sum(1) / count.clamp_min(1)
        pooled = torch.cat([pp, sp], dim=-1)
        return tokens, mask, pooled, location, scale, count

    def _prediction(self, state, candidates, prior, initial, context, single):
        count = torch.log1p(state.support_count).expand(-1, candidates.shape[1])
        scale = torch.log(state.scale).expand(-1, candidates.shape[1])
        features = torch.cat([initial, context, count[..., None], scale[..., None]], -1)
        score = (prior + state.location
                 + state.scale * self.score_head(features).squeeze(-1))
        variance = state.scale.square() * (
            F.softplus(self.variance_head(features).squeeze(-1)) + 1e-4)
        if single:
            score, variance = score.squeeze(0), variance.squeeze(0)
        return StatePrediction(score=score, variance=variance)


class _QueryAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.out = nn.Linear(dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))

    def kv(self, tokens):
        b, s, _ = tokens.shape
        shape = (b, s, self.heads, self.head_dim)
        return self.k(tokens).view(shape).transpose(1, 2), self.v(tokens).view(shape).transpose(1, 2)

    def forward(self, query, key, value, mask):
        b, n, d = query.shape
        q = self.q(query).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        logits = (q @ key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weights = logits.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        context = (weights @ value).transpose(1, 2).reshape(b, n, d)
        query = self.norm1(query + self.out(context))
        return self.norm2(query + self.ff(query))


class JevStateScorer(_StateRegressor):
    """Each candidate independently queries cacheable pocket/support state.

    ``pocket_vectors`` accepts (B,P,Dp), (B,Dp), or one (Dp,) vector.
    Support accepts (B,S,Dl)/(B,S), or (S,Dl)/(S,) for one target.
    Candidate accepts (B,N,Dl), or (N,Dl) for one target; output preserves
    those leading dimensions. Masks are True for valid tokens. Masked NaN
    padding is permitted. Empty support is valid when pocket is available.
    """
    def __init__(self, ligand_dim: int = 128, pocket_dim: int = 34,
                 hidden_dim: int = 128, heads: int = 4, layers: int = 2,
                 residualize: bool = True, scale_floor: float = 0.1):
        super().__init__(ligand_dim, pocket_dim, hidden_dim, residualize, scale_floor)
        if heads <= 0 or hidden_dim % heads or layers <= 0:
            raise ValueError("heads must divide hidden_dim and layers must be positive")
        self.layers = nn.ModuleList([_QueryAttention(hidden_dim, heads) for _ in range(layers)])

    def encode_state(self, pocket_vectors: Tensor, support_vectors: Tensor,
                     support_scores: Tensor, support_mask: Tensor | None = None,
                     support_baseline: Tensor | None = None,
                     pocket_mask: Tensor | None = None) -> StateCache:
        tokens, mask, pooled, location, scale, count = self._tokens(
            pocket_vectors, support_vectors, support_scores, support_mask,
            support_baseline, pocket_mask)
        kv = [layer.kv(tokens) for layer in self.layers]
        return StateCache(tuple(k for k, _ in kv), tuple(v for _, v in kv),
                          mask, pooled, location, scale, count, self.residualize)

    def forward(self, state: StateCache, candidate_vectors: Tensor,
                candidate_baseline: Tensor | None = None) -> StatePrediction:
        if state.residualize != self.residualize or len(state.keys) != len(self.layers):
            raise ValueError("state was encoded with an incompatible model")
        candidates, prior, single = _candidate_inputs(
            candidate_vectors, candidate_baseline, state.mask.shape[0],
            self.ligand_dim, self.residualize)
        initial = self.query(candidates)
        context = initial
        for layer, key, value in zip(self.layers, state.keys, state.values):
            context = layer(context, key, value, state.mask)
        return self._prediction(state, candidates, prior, initial, context, single)


class PooledCNPStateScorer(_StateRegressor):
    """Permutation-invariant pooled CNP-style control with the same priors.

    Pocket and support token means are pooled separately, then supplied to
    each candidate decoder. It has no candidate-specific state attention.
    """
    def __init__(self, ligand_dim: int = 128, pocket_dim: int = 34,
                 hidden_dim: int = 128, residualize: bool = True,
                 scale_floor: float = 0.1):
        super().__init__(ligand_dim, pocket_dim, hidden_dim, residualize, scale_floor)
        self.pool_projection = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU())

    def encode_state(self, pocket_vectors: Tensor, support_vectors: Tensor,
                     support_scores: Tensor, support_mask: Tensor | None = None,
                     support_baseline: Tensor | None = None,
                     pocket_mask: Tensor | None = None) -> StateCache:
        _, mask, pooled, location, scale, count = self._tokens(
            pocket_vectors, support_vectors, support_scores, support_mask,
            support_baseline, pocket_mask)
        return StateCache((), (), mask, self.pool_projection(pooled), location,
                          scale, count, self.residualize)

    def forward(self, state: StateCache, candidate_vectors: Tensor,
                candidate_baseline: Tensor | None = None) -> StatePrediction:
        if state.residualize != self.residualize or state.keys:
            raise ValueError("state was encoded with an incompatible model")
        candidates, prior, single = _candidate_inputs(
            candidate_vectors, candidate_baseline, state.mask.shape[0],
            self.ligand_dim, self.residualize)
        initial = self.query(candidates)
        context = state.pooled[:, None, :].expand(-1, candidates.shape[1], -1)
        return self._prediction(state, candidates, prior, initial, context, single)


class KernelRidgeStateScorer(nn.Module):
    """Deterministic RBF kernel ridge on centered support scores/residuals.

    L2-normalized ligand vectors define the kernel. The pocket argument is
    accepted for API parity but is unused. Empty support falls back to the
    fixed candidate prior (zero in raw mode). Variance is a kernel-distance
    heuristic scaled by observed support variation, not calibrated coverage.
    """
    def __init__(self, ligand_dim: int = 128, pocket_dim: int = 34,
                 ridge: float = 0.1, length_scale: float = 0.5,
                 residualize: bool = True, scale_floor: float = 0.1):
        super().__init__()
        if min(ligand_dim, pocket_dim) <= 0 or min(ridge, length_scale, scale_floor) <= 0:
            raise ValueError("dimensions and kernel hyperparameters must be positive")
        self.ligand_dim, self.pocket_dim = ligand_dim, pocket_dim
        self.ridge, self.length_scale = ridge, length_scale
        self.residualize, self.scale_floor = residualize, scale_floor

    def _kernel(self, a, b):
        distance = (a.square().sum(-1, keepdim=True)
                    + b.square().sum(-1)[:, None, :] - 2 * (a @ b.transpose(-1, -2)))
        return torch.exp(-distance.clamp_min(0) / (2 * self.length_scale ** 2))

    def encode_state(self, pocket_vectors: Tensor, support_vectors: Tensor,
                     support_scores: Tensor, support_mask: Tensor | None = None,
                     support_baseline: Tensor | None = None,
                     pocket_mask: Tensor | None = None) -> KernelStateCache:
        support, centered, mask, location, scale, _ = _support_inputs(
            support_vectors, support_scores, support_mask, support_baseline,
            self.residualize, self.ligand_dim, self.scale_floor)
        support = F.normalize(support, dim=-1)
        kernel = self._kernel(support, support) * (mask[:, :, None] & mask[:, None, :])
        eye = torch.eye(support.shape[1], dtype=kernel.dtype, device=kernel.device)
        chol = torch.linalg.cholesky(kernel + self.ridge * eye)
        coefficients = torch.cholesky_solve(centered[..., None], chol).squeeze(-1)
        return KernelStateCache(support, mask, coefficients, chol, location, scale,
                                self.residualize)

    def forward(self, state: KernelStateCache, candidate_vectors: Tensor,
                candidate_baseline: Tensor | None = None) -> StatePrediction:
        if state.residualize != self.residualize:
            raise ValueError("state was encoded with an incompatible model")
        candidates, prior, single = _candidate_inputs(
            candidate_vectors, candidate_baseline, state.support.shape[0],
            self.ligand_dim, self.residualize)
        kernel = self._kernel(F.normalize(candidates, dim=-1), state.support)
        kernel = kernel * state.support_mask[:, None, :]
        score = prior + state.location + (kernel @ state.coefficients[..., None]).squeeze(-1)
        solved = torch.cholesky_solve(kernel.transpose(1, 2), state.cholesky)
        remaining = (1 - (kernel * solved.transpose(1, 2)).sum(-1)).clamp_min(0)
        variance = state.scale.square() * (remaining + self.ridge)
        if single:
            score, variance = score.squeeze(0), variance.squeeze(0)
        return StatePrediction(score=score, variance=variance)
