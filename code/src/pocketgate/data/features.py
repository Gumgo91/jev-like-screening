"""Featurizers (plan 6.2 pocket, 6.3 ligand).

Ligand: 2D atom/bond graph from RDKit. Stereochemistry is preserved in
atom chiral-tag and bond stereo features.

Pocket: residue graph. Residues with >=1 heavy atom inside
box+4A margin; if >256 keep the 256 closest to the box center
(deterministic, recorded). Node features: amino-acid identity +
predefined charge/aromatic/donor/acceptor aggregates + distance of the
residue representative (CA, else heavy-atom centroid) to the box center.
Edges: kNN k=16 over representative coordinates; edge features are a
radial basis expansion of distance only -> translation/rotation
invariant by construction.

Returns plain dicts of numpy arrays; batching lives in
`pocketgate.data.graphbatch`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pocketgate.common import TARGETS_DIR  # noqa: E402
from pocketgate.data.audit import parse_conf, parse_pdbqt_atoms  # noqa: E402

RDLogger.DisableLog("rdApp.*")

BOX_MARGIN = 4.0
MAX_POCKET_RESIDUES = 256
POCKET_KNN = 16
N_RBF = 16
RBF_CUTOFF = 20.0  # Angstrom

# ---------------------------------------------------------------- ligand
ATOM_LIST = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
ATOM_DIM = (
    len(ATOM_LIST) + 1  # atomic number (+ other)
    + 5                 # formal charge {-2..2}
    + 2                 # aromaticity
    + 6                 # hybridization
    + 5                 # total num H {0..3,4+}
    + 4                 # chirality tag
    + 2                 # in ring
    + 6                 # min ring size {0,3,4,5,6,7+}
    + 5                 # degree {0..3,4+}
)
BOND_DIM = 4 + 2 + 2 + 6  # type, conjugated, ring, stereo

_HYB = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]
_CHI = [
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
]
_BOND_TYPE = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
_STEREO = [
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOATROPCW,
]


def _onehot(i: int, n: int) -> list[float]:
    v = [0.0] * n
    if 0 <= i < n:
        v[i] = 1.0
    return v


def atom_features(atom: Chem.Atom) -> np.ndarray:
    sym = atom.GetSymbol()
    feats = _onehot(ATOM_LIST.index(sym) if sym in ATOM_LIST else len(ATOM_LIST),
                    len(ATOM_LIST) + 1)
    fc = int(np.clip(atom.GetFormalCharge(), -2, 2))
    feats += _onehot(fc + 2, 5)
    feats += _onehot(1 if atom.GetIsAromatic() else 0, 2)
    hyb = atom.GetHybridization()
    feats += _onehot(_HYB.index(hyb) if hyb in _HYB else len(_HYB), 6)
    nh = min(atom.GetTotalNumHs(), 4)
    feats += _onehot(nh, 5)
    chi = atom.GetChiralTag()
    feats += _onehot(_CHI.index(chi) if chi in _CHI else len(_CHI), 4)
    feats += _onehot(1 if atom.IsInRing() else 0, 2)
    rs = 0
    if atom.IsInRing():
        for r in range(3, 8):
            if atom.IsInRingSize(r):
                rs = r
                break
        else:
            rs = 7
    feats += _onehot(min(rs, 7) - 3 + 1 if rs else 0, 6)
    feats += _onehot(min(atom.GetTotalDegree(), 4), 5)
    return np.asarray(feats, dtype=np.float32)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    bt = bond.GetBondType()
    feats = _onehot(_BOND_TYPE.index(bt) if bt in _BOND_TYPE else len(_BOND_TYPE) - 1, 4)
    feats += _onehot(1 if bond.GetIsConjugated() else 0, 2)
    feats += _onehot(1 if bond.IsInRing() else 0, 2)
    st = bond.GetStereo()
    feats += _onehot(_STEREO.index(st) if st in _STEREO else len(_STEREO), 6)
    return np.asarray(feats, dtype=np.float32)


def featurize_ligand(smiles: str) -> dict | None:
    """-> {x:(A,ATOM_DIM) f32, edge_index:(2,E) i64, edge_attr:(E,BOND_DIM) f32}.

    Directed edges both ways. Returns None for unparseable SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = np.stack([atom_features(a) for a in mol.GetAtoms()])
    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]
        dst += [j, i]
        eattr += [f, f]
    edge_index = np.asarray([src, dst], dtype=np.int64)
    edge_attr = (np.stack(eattr) if eattr
                 else np.zeros((0, BOND_DIM), dtype=np.float32))
    return {"x": x.astype(np.float32), "edge_index": edge_index,
            "edge_attr": edge_attr.astype(np.float32)}


# ---------------------------------------------------------------- pocket
AA21 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL"]  # +1 unknown
# AutoDock-type aggregate buckets on the receptor side
AD_BUCKETS = ["C", "A", "N", "NA", "OA", "SA", "HD", "OTHER"]
POCKET_NODE_DIM = (len(AA21) + 1) + 1 + 8 + 4  # aa(+unk) + charge + AD + geo
POCKET_EDGE_DIM = N_RBF  # RBF distance only


def _rbf(d: np.ndarray) -> np.ndarray:
    centers = np.linspace(0.0, RBF_CUTOFF, N_RBF, dtype=np.float32)
    gamma = 1.0 / (2 * (RBF_CUTOFF / N_RBF) ** 2)
    return np.exp(-gamma * (d[..., None] - centers) ** 2)


def featurize_pocket(target_id: str, targets_dir: Path | None = None,
                     margin: float = BOX_MARGIN,
                     max_res: int = MAX_POCKET_RESIDUES) -> dict:
    """Build the ligand-independent pocket graph for one target.

    Returns dict with x, edge_index, edge_attr, n_residues,
    truncated(bool), box dict, residue_ids.
    """
    tdir = Path(targets_dir) if targets_dir else TARGETS_DIR
    conf = parse_conf(tdir / f"{target_id}_conf.txt")
    atoms = parse_pdbqt_atoms(tdir / f"{target_id}_target.pdbqt")
    return featurize_pocket_atoms(atoms, conf, target_id=target_id,
                                  margin=margin, max_res=max_res)


def featurize_pocket_atoms(atoms, conf: dict, target_id: str = "",
                           margin: float = BOX_MARGIN,
                           max_res: int = MAX_POCKET_RESIDUES) -> dict:
    """Same as featurize_pocket but from a parsed atom list (test hook)."""
    cx, cy, cz = conf["center_x"], conf["center_y"], conf["center_z"]
    hx = conf["size_x"] / 2 + margin
    hy = conf["size_y"] / 2 + margin
    hz = conf["size_z"] / 2 + margin

    res_order: list[tuple] = []
    res_atoms: dict[tuple, list] = {}
    for atom_name, res_name, res_seq, x, y, z, charge, adt in atoms:
        key = (res_name, res_seq)
        if key not in res_atoms:
            res_atoms[key] = []
            res_order.append(key)
        res_atoms[key].append((atom_name, x, y, z, charge, adt))

    members = []
    for key in res_order:
        alist = res_atoms[key]
        heavy = [a for a in alist if not a[5].startswith("H")]
        if any(abs(a[1] - cx) <= hx and abs(a[2] - cy) <= hy
               and abs(a[3] - cz) <= hz for a in heavy):
            members.append((key, alist))

    truncated = len(members) > max_res
    if truncated:
        def dmin(item):
            _, alist = item
            return min(float(np.sqrt((a[1] - cx) ** 2 + (a[2] - cy) ** 2
                                     + (a[3] - cz) ** 2))
                       for a in alist if not a[5].startswith("H"))
        members = sorted(members, key=lambda it: (dmin(it), it[0]))[:max_res]
    members.sort(key=lambda it: it[0][1])

    reps, node_feats, residue_ids = [], [], []
    for (res_name, res_seq), alist in members:
        heavy = [a for a in alist if not a[5].startswith("H")]
        xyz = np.array([[a[1], a[2], a[3]] for a in heavy], dtype=np.float32)
        ca = [a for a in heavy if a[0].strip() == "CA"]
        rep = np.asarray(ca[0][1:4], dtype=np.float32) if ca else xyz.mean(axis=0)
        reps.append(rep)
        residue_ids.append(f"{res_name}{res_seq}")

        rn = res_name.upper()
        aa = _onehot(AA21.index(rn) if rn in AA21 else len(AA21), len(AA21) + 1)
        mean_charge = float(np.mean([a[4] for a in alist]))
        counts = [0] * len(AD_BUCKETS)
        for a in heavy:
            t = a[5]
            t = t if t in AD_BUCKETS[:-1] else "OTHER"
            counts[AD_BUCKETS.index(t)] += 1
        counts = [c / max(len(heavy), 1) for c in counts]
        box_scale = float(np.mean([hx, hy, hz]))
        d_center = float(np.linalg.norm(rep - np.array([cx, cy, cz])) /
                         box_scale)
        spread = float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum()
                               / max(len(xyz), 1)))
        geo = [d_center,
               float(np.linalg.norm(rep - np.array([cx, cy, cz]))) / 50.0,
               spread,
               float(len(heavy)) / 20.0]
        node_feats.append(np.asarray(aa + [mean_charge] + counts + geo,
                                     dtype=np.float32))

    reps = np.stack(reps)  # (M,3)
    M = len(members)
    k = min(POCKET_KNN, M - 1)
    src, dst = [], []
    if k > 0:
        dm = np.sqrt(((reps[:, None, :] - reps[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(dm, np.inf)
        nbr = np.argpartition(dm, k, axis=1)[:, :k]
        for i in range(M):
            for j in nbr[i]:
                src.append(i)
                dst.append(int(j))
    edge_index = np.asarray([src, dst], dtype=np.int64)
    if len(src):
        dvec = np.sqrt(((reps[src] - reps[dst]) ** 2).sum(-1))
        edge_attr = _rbf(dvec).astype(np.float32)
    else:
        edge_attr = np.zeros((0, N_RBF), dtype=np.float32)

    return {
        "x": np.stack(node_feats).astype(np.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "reps": reps.astype(np.float32),
        "n_residues": M,
        "truncated": truncated,
        "residue_ids": residue_ids,
        "box": conf,
        "target_id": target_id,
    }
