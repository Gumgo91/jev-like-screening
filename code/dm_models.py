"""Surrogates for interventional docking-score prediction.

Both models follow the shared-state, independent-query pattern. The pocket is encoded once
into a state, every ligand queries that state on its own, and no ligand sees another ligand.

ResidueSumNet  : S(l, P) = b(z_l) + sum_r phi(h_r, z_l). The ligand-only term b carries the
                 universal ranking, the residue sum carries pocket-specific interaction.
                 Because the pocket enters through a sum, changing one residue changes only
                 that residue's term (plus its graph neighbours through the encoder).
DualEncoderDM  : the dual encoder of the original study (mean-pooled pocket vector), used as
                 an architecture control.
Outputs are standardised scores; multiply by the score std to obtain kcal/mol.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pocketgate.data.graphbatch import pad_flat  # noqa: E402
from pocketgate.models.encoders import LigandEncoder, PocketEncoder  # noqa: E402
from pocketgate.models.mpnn import masked_mean  # noqa: E402


class ResidueSumNet(nn.Module):
    kind = "rs"

    def __init__(self, dim=128, pocket_layers=3, ligand_layers=3, dropout=0.1, hidden=64, geo_dim=0, edit_dim=0, hit=False):
        super().__init__()
        self.geo_dim = geo_dim
        self.edit_dim = edit_dim
        self.hit = hit
        # no dropout on the pocket side: wild-type and edited pockets must differ only through the edit
        self.pocket_encoder = PocketEncoder(dim, pocket_layers, 0.0)
        self.ligand_encoder = LigandEncoder(dim, ligand_layers, dropout)
        self.lig_off = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.Wp = nn.Linear(dim, hidden)
        self.Wl = nn.Linear(dim, hidden)
        self.Gp = nn.Linear(dim, hidden, bias=False)
        self.Gl = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Linear(hidden, 1)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)
        if geo_dim:
            self.Wg = nn.Linear(geo_dim, hidden, bias=False)
            self.glob = nn.Sequential(nn.Linear(dim + geo_dim + edit_dim, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, 1))
            nn.init.normal_(self.glob[2].weight, std=0.02)
        if edit_dim:
            self.We = nn.Linear(edit_dim, hidden, bias=False)
        if hit:
            # second readout of the same residue interaction features: a hit logit trained on top-1% hits
            self.lig_off_h = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
            self.out_h = nn.Linear(hidden, 1)
            nn.init.normal_(self.out_h.weight, std=0.02)
            nn.init.zeros_(self.out_h.bias)
            if geo_dim:
                self.glob_h = nn.Sequential(nn.Linear(dim + geo_dim + edit_dim, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, 1))
                nn.init.normal_(self.glob_h[2].weight, std=0.02)

    def lig_embed(self, lb):
        H = self.ligand_encoder(lb["x"], lb["edge_index"], lb["edge_attr"])
        Hp, m = pad_flat(H, lb["batch"], lb["n_atoms"], lb["n_graphs"])
        return masked_mean(Hp, m)

    def poc_embed(self, pb):
        H = self.pocket_encoder(pb["x"], pb["edge_index"], pb["edge_attr"])
        Hp, m = pad_flat(H, pb["batch"], pb["n_res"], pb["n_graphs"])
        ctx = {"P": self.Wp(Hp), "G": self.Gp(Hp), "mask": m}
        if self.geo_dim:
            ctx["geo"] = pb["geo"]
            ctx["Wg"] = self.Wg(pb["geo"])
        if self.edit_dim:
            ctx["edit"] = pb["edit"]
            ctx["We"] = self.We(pb["edit"])
        return ctx

    def _u(self, ctx, pidx, z, lidx):
        zl = z[lidx]
        u = ctx["P"][pidx] + self.Wl(zl)[:, None, :] + ctx["G"][pidx] * self.Gl(zl)[:, None, :]
        if self.geo_dim:
            u = u + ctx["Wg"][pidx][:, None, :]
        if self.edit_dim:
            u = u + ctx["We"][pidx][:, None, :]
        return F.relu(u)

    def residue_terms(self, ctx, pidx, z, lidx):
        """Per-residue contributions (N, M) for N (pocket, ligand) pairs."""
        return self.out(self._u(ctx, pidx, z, lidx)).squeeze(-1) * ctx["mask"][pidx]

    def heads_pairs(self, ctx, pidx, z, lidx):
        """Score and hit logit of N (pocket, ligand) pairs from one pass over the residue features."""
        u = self._u(ctx, pidx, z, lidx)
        mask = ctx["mask"][pidx]
        zl = z[lidx]
        s = self.lig_off(zl).squeeze(-1) + (self.out(u).squeeze(-1) * mask).sum(-1)
        h = self.lig_off_h(zl).squeeze(-1) + (self.out_h(u).squeeze(-1) * mask).sum(-1)
        if self.geo_dim:
            parts = [zl, ctx["geo"][pidx]] + ([ctx["edit"][pidx]] if self.edit_dim else [])
            cat = torch.cat(parts, dim=-1)
            s = s + self.glob(cat).squeeze(-1)
            h = h + self.glob_h(cat).squeeze(-1)
        return s, h

    def hit_pairs(self, ctx, pidx, z, lidx):
        return self.heads_pairs(ctx, pidx, z, lidx)[1]

    def score_pairs(self, ctx, pidx, z, lidx):
        c = self.residue_terms(ctx, pidx, z, lidx)
        s = self.lig_off(z[lidx]).squeeze(-1) + c.sum(-1)
        if self.geo_dim:
            parts = [z[lidx], ctx["geo"][pidx]] + ([ctx["edit"][pidx]] if self.edit_dim else [])
            s = s + self.glob(torch.cat(parts, dim=-1)).squeeze(-1)
        return s


class DualEncoderDM(nn.Module):
    kind = "b5"

    def __init__(self, dim=128, pocket_layers=3, ligand_layers=3, dropout=0.1):
        super().__init__()
        self.pocket_encoder = PocketEncoder(dim, pocket_layers, 0.0)
        self.ligand_encoder = LigandEncoder(dim, ligand_layers, dropout)
        self.W_p = nn.Linear(dim, dim)
        self.score_head = nn.Linear(dim, 1)

    def lig_embed(self, lb):
        H = self.ligand_encoder(lb["x"], lb["edge_index"], lb["edge_attr"])
        Hp, m = pad_flat(H, lb["batch"], lb["n_atoms"], lb["n_graphs"])
        return masked_mean(Hp, m)

    def poc_embed(self, pb):
        H = self.pocket_encoder(pb["x"], pb["edge_index"], pb["edge_attr"])
        Hp, m = pad_flat(H, pb["batch"], pb["n_res"], pb["n_graphs"])
        return {"pv": self.W_p(masked_mean(Hp, m))}

    def score_pairs(self, ctx, pidx, z, lidx):
        return self.score_head(z[lidx] * ctx["pv"][pidx]).squeeze(-1)


def build(kind: str, **kw):
    if kind == "rsg":
        return ResidueSumNet(geo_dim=10, **kw)
    if kind == "rse":
        return ResidueSumNet(geo_dim=10, edit_dim=13, **kw)
    return {"rs": ResidueSumNet, "b5": DualEncoderDM}[kind](**kw)
