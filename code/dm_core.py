"""Data handling, interventional training and evaluation for DockMut.

Training uses, for every training target, the DOCKSTRING wild-type scores of the 40k training
ligands and the re-docked changes of its truncated receptors. A step draws one target,
one wild-type ligand batch and a few mutants with their own ligand batches. The wild-type term
is a Huber regression on standardised scores. The intervention term is a Huber regression of
the predicted change S(l, P_mut) - S(l, P_wt) on the measured change.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dockmut"))
from pocketgate.data.features import featurize_ligand  # noqa: E402
from pocketgate.data.graphbatch import collate_ligands  # noqa: E402
from pocketgate.data.graphbatch import collate_pockets as _collate_pockets  # noqa: E402

UNSTABLE_SD = 1.0
WINS = 4.0
SCORE_MEAN, SCORE_STD = -8.4289948407346, 1.200659407938392   # train_train, 38+DRD2 targets, 40k ligands


# --------------------------------------------------------------------------- data
def collate_pockets(graphs):
    b = _collate_pockets(graphs)
    if "geo" in graphs[0]:
        b["geo"] = torch.stack([g["geo"] for g in graphs]).float()
    if "edit" in graphs[0]:
        b["edit"] = torch.stack([g["edit"] for g in graphs]).float()
    return b


def load_pockets(path: Path, plan: dict | None = None):
    """Load the pocket graph cache and standardise the geometry vector with statistics of the
    wild-type pockets of the training targets (receptor coordinates only, no labels)."""
    pockets = torch.load(path, weights_only=False)
    if plan is None:
        plan = json.loads((Path(path).parent / "plan.json").read_text())
    train = [t for t, v in plan["targets"].items() if v["role"] == "train" and t in pockets]
    if train and "geo" in pockets[train[0]]["wt"]:
        G = torch.stack([pockets[t]["wt"]["geo"] for t in train])
        mu, sd = G.mean(0), G.std(0).clamp(min=1e-6)
        for t in pockets:
            wt_geo = pockets[t]["wt"]["geo"].clone()
            for rid, g in pockets[t].items():
                if "edit_meta" in g:
                    m = g["edit_meta"]
                    g["edit"] = torch.cat([(g["geo"] - wt_geo) / sd, torch.stack([m[0] / 20.0, m[1] / 4.0, m[2] / 15.0])])                         if rid != "wt" else torch.zeros(len(sd) + 3)
                g["geo"] = (g["geo"] - mu) / sd
    return pockets


def to_dev(batch: dict, dev: str) -> dict:
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


def read_results(results_dir: Path, target: str) -> pd.DataFrame:
    df = pd.read_csv(results_dir / f"{target}.csv")
    df = df.drop_duplicates(["receptor", "tag", "inchikey"], keep="last")
    return df.dropna(subset=["score"])


def delta_table(df: pd.DataFrame, manifest: dict):
    """-> (per-mutant dict rid -> (ids, dS, kind), noise sigma, n_stable)."""
    wt = df[df.receptor == "wt"].pivot_table(index="inchikey", columns="seed", values="score")
    wt_mean = wt.mean(axis=1)
    if wt.shape[1] >= 2:
        wt_sd = wt.std(axis=1, ddof=1)
        stable = wt_sd[(wt_sd < UNSTABLE_SD) & (wt_mean.reindex(wt_sd.index) < 0)].index
        sigma = float(np.sqrt((wt_sd.loc[stable] ** 2).mean()))
    else:
        stable, sigma = wt_mean[wt_mean < 0].index, float("nan")
    kinds = {r["id"]: r["kind"] for r in manifest["receptors"]}
    out = {}
    for rid, g in df[(df.receptor != "wt") & (df.tag == "mut_s1")].groupby("receptor"):
        s = g.set_index("inchikey").score
        ids = s.index.intersection(stable)
        if len(ids) < 20:
            continue
        d = (s[ids] - wt_mean[ids]).clip(-WINS, WINS)
        out[rid] = (np.array(ids), d.to_numpy(np.float32), kinds[rid])
    return out, sigma, len(stable)


def to_matrix(tab: dict):
    """rid -> (ids, dS, kind) into a mutants x ligands matrix (NaN where a ligand was not docked/stable)."""
    rids = sorted(tab)
    ligs = sorted({i for r in rids for i in tab[r][0]})
    col = {i: k for k, i in enumerate(ligs)}
    D = np.full((len(rids), len(ligs)), np.nan, dtype=np.float32)
    for a, r in enumerate(rids):
        ids, dS, _ = tab[r]
        D[a, [col[i] for i in ids]] = dS
    return {"rids": rids, "ligs": np.array(ligs), "D": D, "kinds": [tab[r][2] for r in rids]}


def lig_graphs_for(ids, graphs_path: Path | None, smiles: pd.Series | None):
    have = torch.load(graphs_path, weights_only=False) if graphs_path and graphs_path.exists() else {}
    out = {}
    for i in ids:
        if i in have:
            out[i] = have[i]
        else:
            out[i] = featurize_ligand(smiles[i])
    return out


# --------------------------------------------------------------------------- training
class DMTrainer:
    def __init__(self, model, pockets, ligs, wt, deltas, cfg, dev="cuda"):
        self.m, self.pockets, self.ligs, self.wt, self.deltas = model.to(dev), pockets, ligs, wt, deltas
        self.cfg, self.dev = cfg, dev
        self.targets = sorted(wt)
        self.rng = np.random.default_rng(cfg["seed"])
        fam = cfg.get("family_of")
        if cfg.get("family_balance") and fam:
            n = pd.Series([fam[t] for t in self.targets]).value_counts()
            w = np.array([1.0 / n[fam[t]] for t in self.targets])
            self.tp = w / w.sum()
        else:
            self.tp = None
        torch.manual_seed(cfg["seed"])
        self.const = None
        if cfg.get("const_pocket"):
            self.const = cfg["const_pocket"]
        # top-1% hit labels of the wild-type training ligands of every target (hit head only)
        self.hit_thr = {t: float(np.quantile(wt[t][1], 0.01)) for t in wt} if cfg.get("mu", 0) > 0 else {}
        self.last_hit = 0.0

    def _pocket(self, t, rid):
        return self.const if self.const is not None else self.pockets[t][rid]

    def step(self, opt):
        c, rng = self.cfg, self.rng
        t = self.targets[rng.choice(len(self.targets), p=self.tp) if self.tp is not None else rng.integers(len(self.targets))]
        ids_t, sc_t = self.wt[t]
        bi = rng.integers(0, len(ids_t), size=c["batch"])
        wt_ids = [ids_t[i] for i in bi]
        y_wt = torch.from_numpy((sc_t[bi] - SCORE_MEAN) / SCORE_STD).float().to(self.dev)
        muts = []
        if c["lam"] > 0 and self.deltas.get(t):
            mat = self.deltas[t]
            k = min(c["n_mut"], len(mat["rids"]))
            rows = rng.choice(len(mat["rids"]), size=k, replace=False)
            ok = np.where(~np.isnan(mat["D"][rows]).any(0))[0]
            if len(ok) >= 8:
                cols = rng.choice(ok, size=min(c["mut_batch"], len(ok)), replace=False)
                ids = [mat["ligs"][j] for j in cols]
                for a in rows:
                    muts.append((mat["rids"][a], ids, mat["D"][a, cols]))
        lig_index, lig_list = {}, []
        for i in wt_ids + [x for _, ids, _ in muts for x in ids]:
            if i not in lig_index:
                lig_index[i] = len(lig_list)
                lig_list.append(i)
        lb = to_dev(collate_ligands([self.ligs[i] for i in lig_list]), self.dev)
        pg = [self._pocket(t, "wt")] + [self._pocket(t, rid) for rid, _, _ in muts]
        pb = to_dev(collate_pockets(pg), self.dev)
        z = self.m.lig_embed(lb)
        ctx = self.m.poc_embed(pb)
        dev = self.dev
        li_wt = torch.tensor([lig_index[i] for i in wt_ids], device=dev)
        loss_hit = torch.zeros((), device=dev)
        if c.get("mu", 0) > 0 and getattr(self.m, "hit", False):
            s_wt, h_wt = self.m.heads_pairs(ctx, torch.zeros_like(li_wt), z, li_wt)
            y_hit = torch.from_numpy((sc_t[bi] <= self.hit_thr[t]).astype(np.float32)).to(dev)
            loss_hit = F.binary_cross_entropy_with_logits(h_wt, y_hit)
        else:
            s_wt = self.m.score_pairs(ctx, torch.zeros_like(li_wt), z, li_wt)
        loss_wt = F.huber_loss(s_wt, y_wt, delta=1.0)
        loss_d = torch.zeros((), device=dev)
        if muts:
            P, L, Y = [], [], []
            for j, (_, ids, dS) in enumerate(muts):
                P.append(torch.full((len(ids),), j + 1, device=dev))
                L.append(torch.tensor([lig_index[i] for i in ids], device=dev))
                Y.append(torch.from_numpy(dS / SCORE_STD).float().to(dev))
            P, L, Y = torch.cat(P), torch.cat(L), torch.cat(Y)
            if c.get("shuffle") == "ligand":          # break ligand-specific response, keep mutant mean
                off = 0
                for _, ids, _ in muts:
                    n = len(ids)
                    Y[off:off + n] = Y[off:off + n][torch.randperm(n, device=dev)]
                    off += n
            elif c.get("shuffle") == "mutant":        # assign each edit the measured row of another edit of the same target
                k = len(muts)
                Y = Y.view(k, -1)[torch.randperm(k, device=dev)].reshape(-1)
            elif c.get("shuffle") == "all":           # break both ligand and edit structure, keep the marginal
                Y = Y[torch.randperm(len(Y), device=dev)]
            elif c.get("shuffle") in ("mutant_d", "all_d"):
                # derangement of the edit rows: every edit receives the row of a different edit of the same target
                # (a uniform permutation leaves an edit's own row in place with probability 1/k)
                k = len(muts)
                order = torch.randperm(k, device=dev)
                shift = int(torch.randint(1, k, (1,))) if k > 1 else 0
                src = torch.empty(k, dtype=torch.long, device=dev)
                src[order] = order.roll(-shift)
                Yr = Y.view(k, -1)[src]
                if c.get("shuffle") == "all_d":     # additionally permute the ligands independently within each row
                    Yr = torch.gather(Yr, 1, torch.argsort(torch.rand(Yr.shape, device=dev), dim=1))
                Y = Yr.reshape(-1)
            d_hat = self.m.score_pairs(ctx, P, z, L) - self.m.score_pairs(ctx, torch.zeros_like(P), z, L)
            loss_d = F.huber_loss(d_hat, Y, delta=1.0)
        loss = loss_wt + c["lam"] * loss_d + c.get("mu", 0.0) * loss_hit
        self.last_hit = float(loss_hit.detach())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.m.parameters(), 5.0)
        opt.step()
        return float(loss_wt.detach()), float(loss_d.detach())

    def train(self, log_every=500, ckpt=None):
        c = self.cfg
        opt = torch.optim.AdamW(self.m.parameters(), lr=c["lr"], weight_decay=c["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c["updates"], eta_min=c["lr"] * 0.05)
        self.m.train()
        hist, t0 = [], time.time()
        a, b, h = [], [], []
        for u in range(1, c["updates"] + 1):
            lw, ld = self.step(opt)
            sched.step()
            a.append(lw); b.append(ld); h.append(self.last_hit)
            if u % log_every == 0:
                hist.append({"update": u, "loss_wt": float(np.mean(a)), "loss_delta": float(np.mean(b)), "loss_hit": float(np.mean(h)),
                             "min": round((time.time() - t0) / 60, 2)})
                print(json.dumps(hist[-1]), flush=True)
                a, b, h = [], [], []
        if ckpt:
            torch.save({"state": self.m.state_dict(), "cfg": {k: v for k, v in c.items() if k != "const_pocket"}}, ckpt)
        return hist


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict_delta(model, pockets, ligs, tables, target, dev="cuda", chunk=64):
    """Predicted change (kcal/mol) for every (mutant, ligand) in tables[target]."""
    model.eval().to(dev)
    all_ids = sorted({i for ids, _, _ in tables.values() for i in ids})
    index = {i: k for k, i in enumerate(all_ids)}
    zs = []
    for s in range(0, len(all_ids), 512):
        lb = to_dev(collate_ligands([ligs[i] for i in all_ids[s:s + 512]]), dev)
        zs.append(model.lig_embed(lb))
    z = torch.cat(zs)
    rows = []
    rids = ["wt"] + list(tables.keys())
    pred = {}
    for s in range(0, len(rids), chunk):
        sub = rids[s:s + chunk]
        pb = to_dev(collate_pockets([pockets[target][r] for r in sub]), dev)
        ctx = model.poc_embed(pb)
        for j, r in enumerate(sub):
            ids = all_ids if r == "wt" else tables[r][0]
            li = torch.tensor([index[i] for i in ids], device=dev)
            parts = []
            for q in range(0, len(li), 256):
                lq = li[q:q + 256]
                parts.append(model.score_pairs(ctx, torch.full_like(lq, j), z, lq))
            pred[r] = dict(zip(ids, (torch.cat(parts) * SCORE_STD).float().cpu().numpy()))
    for r, (ids, dS, kind) in tables.items():
        for i, d in zip(ids, dS):
            rows.append((target, r, kind, i, float(d), float(np.clip(pred[r][i] - pred["wt"][i], -WINS, WINS))))
    return pd.DataFrame(rows, columns=["target", "receptor", "kind", "ligand", "dS", "pred"])


def metric_block(g: pd.DataFrame) -> dict:
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 1e-9 and np.std(b) > 1e-9 else float("nan")
    keys = ["target", "receptor"] if "target" in g.columns else ["receptor"]   # receptor ids such as m00 repeat across targets
    mm = g.groupby(keys).agg(p=("pred", "mean"), t=("dS", "mean"))
    within = []
    for _, h in g.groupby(keys):
        within.append(corr(h.pred - h.pred.mean(), h.dS - h.dS.mean()))
    slope = float(np.polyfit(g.pred, g.dS, 1)[0]) if g.pred.std() > 1e-9 else 0.0
    return {"pooled_r": corr(g.pred, g.dS), "mutant_r": corr(mm.p, mm.t),
            "within_r": float(np.nanmean(within)) if within else float("nan"),
            "gain": slope, "pred_sd": float(g.pred.std()), "true_sd": float(g.dS.std()), "n": int(len(g))}
