"""G0 data audit for the DOCKSTRING dataset.

Checks (plan section 4.1, G0 exit criteria):
- TSV integrity: row/col counts, header = 58 targets + inchikey + smiles.
- Missing / non-numeric score cells per target (kept as failed state, not
  replaced by pessimistic scores - plan 3.1).
- Duplicate inchikey / smiles.
- Receptor mapping: every TSV score column has *_target.pdbqt + *_conf.txt
  and vice versa; conf box parses; pocket residue count within box+4A margin.
- cluster_split.tsv consistency with the main TSV.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pocketgate.common import (  # noqa: E402
    RAW_DIR,
    TARGETS_DIR,
    list_target_names,
    sha256_file,
)

BOX_MARGIN = 4.0  # Angstrom, plan 6.2

AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "MSE", "SEC", "PYL",
}


def parse_conf(conf_path: Path) -> dict[str, float]:
    vals = {}
    for line in conf_path.read_text().splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        vals[k.strip()] = float(v.strip())
    return vals


def parse_pdbqt_atoms(pdbqt_path: Path):
    """Yield (atom_name, res_name, res_seq, x, y, z, ad_type, charge)."""
    records = []
    with open(pdbqt_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            # pdbqt fixed-width-ish; whitespace split is sufficient for these
            # files (dockstring resources have no chain id column)
            parts = line.split()
            atom_name = parts[2]
            res_name = parts[3]
            tok = parts[4]
            m = re.fullmatch(r"[A-Za-z]?(\d+)[A-Za-z]?", tok)
            if m:
                # plain resid OR fused chain+resid (e.g. 'U1002' when resid >= 1000)
                res_seq = int(m.group(1))
                off = 4
            else:
                # separate chain-id token: ... resname chain resid x y z ...
                res_seq = int(re.fullmatch(r"(\d+)[A-Za-z]?", parts[5]).group(1))
                off = 5
            x, y, z = (float(parts[off + 1]), float(parts[off + 2]),
                       float(parts[off + 3]))
            # pdbqt tail columns: occupancy, bfactor, partial_charge, autodock_type
            charge = float(parts[-2])
            ad_type = parts[-1]
            records.append((atom_name, res_name, res_seq, x, y, z, charge, ad_type))
    return records


def pocket_residue_count(pdbqt_path: Path, conf: dict, margin: float = BOX_MARGIN):
    """Residues with >=1 heavy atom inside the docking box expanded by margin."""
    cx = conf["center_x"]
    cy = conf["center_y"]
    cz = conf["center_z"]
    hx = conf["size_x"] / 2 + margin
    hy = conf["size_y"] / 2 + margin
    hz = conf["size_z"] / 2 + margin
    residues = set()
    total_res = set()
    n_atoms = 0
    for atom_name, res_name, res_seq, x, y, z, charge, adt in parse_pdbqt_atoms(pdbqt_path):
        n_atoms += 1
        total_res.add((res_name, res_seq))
        if adt.startswith("H"):  # skip hydrogens for pocket membership
            continue
        if abs(x - cx) <= hx and abs(y - cy) <= hy and abs(z - cz) <= hz:
            residues.add((res_name, res_seq))
    return len(residues), len(total_res), n_atoms


def main() -> dict:
    tsv = RAW_DIR / "dockstring-dataset.tsv"
    cluster_tsv = RAW_DIR / "cluster_split.tsv"
    audit = {"files": {}, "matrix": {}, "receptors": {}, "issues": []}

    audit["files"]["dockstring-dataset.tsv"] = {
        "path": str(tsv),
        "sha256": sha256_file(tsv),
        "md5_expected": "d0553abd8563bfbbe862eb1d7ec6be6d",
        "bytes": tsv.stat().st_size,
    }
    audit["files"]["cluster_split.tsv"] = {
        "path": str(cluster_tsv),
        "sha256": sha256_file(cluster_tsv),
        "md5_expected": "46660336fbc138d5429c3137f081a423",
        "bytes": cluster_tsv.stat().st_size,
    }

    df = pd.read_csv(tsv, sep="\t")
    audit["matrix"]["n_rows"] = int(len(df))
    audit["matrix"]["n_cols"] = int(df.shape[1])
    audit["matrix"]["id_cols"] = ["inchikey", "smiles"]
    score_cols = [c for c in df.columns if c not in ("inchikey", "smiles")]
    audit["matrix"]["n_target_cols"] = len(score_cols)
    audit["matrix"]["target_cols_sorted"] = sorted(score_cols)
    audit["matrix"]["n_unique_inchikey"] = int(df["inchikey"].nunique())
    audit["matrix"]["n_unique_smiles"] = int(df["smiles"].nunique())
    audit["matrix"]["dup_inchikey_rows"] = int(df["inchikey"].duplicated().sum())
    audit["matrix"]["dup_smiles_rows"] = int(df["smiles"].duplicated().sum())

    scores = df[score_cols].apply(pd.to_numeric, errors="coerce")
    per_target = {}
    for c in score_cols:
        s = scores[c]
        per_target[c] = {
            "n_valid": int(s.notna().sum()),
            "n_missing": int(s.isna().sum()),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
        }
    audit["matrix"]["per_target"] = per_target
    audit["matrix"]["total_cells"] = int(scores.size)
    audit["matrix"]["total_missing"] = int(scores.isna().sum().sum())
    audit["matrix"]["score_range_global"] = [
        float(scores.min().min()),
        float(scores.max().max()),
    ]

    # receptor mapping
    target_files = set(list_target_names())
    conf_files = {
        p.name[: -len("_conf.txt")] for p in TARGETS_DIR.glob("*_conf.txt")
    }
    audit["receptors"]["n_pdbqt"] = len(target_files)
    audit["receptors"]["n_conf"] = len(conf_files)
    audit["receptors"]["tsv_cols_without_pdbqt"] = sorted(set(score_cols) - target_files)
    audit["receptors"]["pdbqt_without_tsv_col"] = sorted(target_files - set(score_cols))
    audit["receptors"]["conf_without_pdbqt"] = sorted(conf_files - target_files)

    pocket = {}
    for t in sorted(target_files):
        pdbqt = TARGETS_DIR / f"{t}_target.pdbqt"
        conf = parse_conf(TARGETS_DIR / f"{t}_conf.txt")
        n_pocket, n_res, n_atoms = pocket_residue_count(pdbqt, conf)
        pocket[t] = {
            "pocket_residues_box+4A": n_pocket,
            "total_residues": n_res,
            "n_atoms": n_atoms,
            "box": conf,
            "pdbqt_sha256": sha256_file(pdbqt),
            "conf_sha256": sha256_file(TARGETS_DIR / f"{t}_conf.txt"),
        }
    audit["receptors"]["per_target"] = pocket
    n_over = sum(1 for v in pocket.values() if v["pocket_residues_box+4A"] > 256)
    audit["receptors"]["targets_over_256_residues"] = n_over

    # cluster_split.tsv consistency
    cs = pd.read_csv(cluster_tsv, sep="\t")
    audit["cluster_split"] = {
        "n_rows": int(len(cs)),
        "cols": list(cs.columns),
        "n_clusters": int(cs["cluster"].nunique()),
        "split_counts": cs["split"].value_counts().to_dict(),
        "inchikey_set_equals_main": bool(set(cs["inchikey"]) == set(df["inchikey"])),
    }

    out = Path(audit.get("out", "manifests/data_audit.json"))
    return audit


if __name__ == "__main__":
    audit = main()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("manifests/data_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2))
    print(f"wrote {out}")
    print(
        f"rows={audit['matrix']['n_rows']} targets={audit['matrix']['n_target_cols']} "
        f"missing={audit['matrix']['total_missing']}"
    )
    print(
        f"receptors: pdbqt={audit['receptors']['n_pdbqt']} "
        f"unmatched_cols={audit['receptors']['tsv_cols_without_pdbqt']}"
    )
