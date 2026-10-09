"""Tabular baselines: B0 random, B1 descriptors, B2 ECFP+CatBoost.

B1: per-target CatBoost on RDKit descriptors (MW, logP, HBD, HBA, RotB).
B2: per-target CatBoost on ECFP4 2048-bit - the strong ligand-only
docking surrogate family (plan section 8, [R3]).
Both train on y = 1[score <= tau_P] per target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

RDLogger.DisableLog("rdApp.*")

_fpgen = GetMorganGenerator(radius=2, fpSize=2048)


def ecfp4(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _fpgen.GetFingerprintAsNumPy(mol).astype(np.float32)


DESCR_NAMES = ["MW", "logP", "HBD", "HBA", "RotB"]


def descriptors(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
    ], dtype=np.float32)


def build_matrix(smiles_list: list[str], kind: str):
    fn = ecfp4 if kind == "ecfp" else descriptors
    feats, ok = [], []
    for i, s in enumerate(smiles_list):
        f = fn(s)
        ok.append(f is not None)
        feats.append(f if f is not None else
                     np.zeros(2048 if kind == "ecfp" else 5, np.float32))
    return np.stack(feats), np.asarray(ok)


class RandomBaseline:
    """B0: seeded random ranking."""

    def __init__(self, seed: int = 11):
        self.rng = np.random.default_rng(seed)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["p_hit"] = self.rng.random(len(df))
        out["logit"] = np.log(out["p_hit"] / (1 - out["p_hit"]))
        return out


class PerTargetCatBoost:
    """B1/B2: one CatBoost classifier per train target."""

    def __init__(self, kind: str = "ecfp", seed: int = 11,
                 iterations: int = 500):
        self.kind = kind
        self.seed = seed
        self.iterations = iterations
        self.models = {}
        self.train_time = {}

    def featurize(self, smiles_list):
        return build_matrix(smiles_list, self.kind)

    def fit(self, train_pairs: pd.DataFrame, smiles_of: dict):
        """train_pairs: target_id, ligand_id, y."""
        from catboost import CatBoostClassifier
        import time
        for t, sub in train_pairs.groupby("target_id"):
            t0 = time.time()
            smi = [smiles_of[l] for l in sub["ligand_id"]]
            X, ok = self.featurize(smi)
            y = sub["y"].to_numpy(int)
            if len(np.unique(y)) < 2:
                self.models[t] = ("const", float(y.mean()))
            else:
                m = CatBoostClassifier(
                    iterations=self.iterations, learning_rate=0.1,
                    depth=6, loss_function="Logloss", random_seed=self.seed,
                    verbose=False, allow_writing_files=False)
                m.fit(X[ok], y[ok])
                self.models[t] = ("model", m)
            self.train_time[t] = time.time() - t0
        return self

    def predict(self, df: pd.DataFrame, smiles_of: dict) -> pd.DataFrame:
        out_frames = []
        for t, sub in df.groupby("target_id"):
            sub = sub.copy()
            entry = self.models.get(t)
            if entry is None:
                sub["p_hit"] = np.nan  # unseen target: N/A per plan
            elif entry[0] == "const":
                sub["p_hit"] = entry[1]
            else:
                smi = [smiles_of[l] for l in sub["ligand_id"]]
                X, ok = self.featurize(smi)
                p = np.full(len(sub), np.nan)
                if ok.any():
                    p[ok] = entry[1].predict_proba(X[ok])[:, 1]
                sub["p_hit"] = p
            sub["logit"] = np.log(
                np.clip(sub["p_hit"], 1e-9, 1 - 1e-9) /
                (1 - np.clip(sub["p_hit"], 1e-9, 1 - 1e-9)))
            out_frames.append(sub)
        return pd.concat(out_frames, ignore_index=True)
