"""In-silico residue truncation of DOCKSTRING receptors (PDBQT) for interventional docking.

For every target the script writes the wild-type receptor, single and multi
residue alanine truncations and a manifest that records which atoms were removed
and where each residue sits relative to the docking box. Atoms are removed in
place, all other PDBQT lines are kept byte-for-byte, so the grid box, atom
typing and charges of the untouched receptor stay identical.

Residue classes (distance d = smallest distance between any side-chain heavy
atom and the box center):
  contact : d <= CONTACT_MAX, at least MIN_SC_ATOMS heavy atoms beyond CB
  shell   : SHELL_MIN <= d <= SHELL_MAX
  far     : every atom outside the search box, truncation must not change a score
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BACKBONE = {"N", "CA", "C", "O", "OXT", "HN", "H", "HN1", "HN2", "HN3",
            "H1", "H2", "H3", "HA"}
KEEP_ALA = BACKBONE | {"CB"}
NO_TRUNCATE = {"GLY", "ALA", "PRO"}
STANDARD_AA = {"ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU",
               "LYS", "MET", "PHE", "SER", "THR", "TRP", "TYR", "VAL",
               "HID", "HIE", "HIP", "CYX", "ASH", "GLH"}
CONTACT_MAX = 8.0
SHELL_MIN, SHELL_MAX = 10.0, 14.0
MIN_SC_ATOMS = 2


def parse_line(line: str):
    name = line[12:16].strip()
    res = line[17:20].strip()
    resid = line[22:27].strip()  # residue number plus insertion code
    chain = line[21].strip()
    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
    adt = line[77:79].strip() if len(line) >= 79 else line.split()[-1]
    return name, res, chain, resid, np.array([x, y, z]), adt


def read_receptor(path: Path):
    lines = path.read_text().splitlines(keepends=True)
    atoms = []
    for i, line in enumerate(lines):
        if line.startswith(("ATOM", "HETATM")):
            name, res, chain, resid, xyz, adt = parse_line(line)
            atoms.append({"line": i, "name": name, "res": res, "chain": chain,
                          "resid": resid, "xyz": xyz, "adt": adt})
    return lines, atoms


def read_box(path: Path):
    v = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, val = line.split("=", 1)
            v[k.strip()] = float(val)
    return v


def residue_table(atoms, box):
    center = np.array([box["center_x"], box["center_y"], box["center_z"]])
    half = np.array([box["size_x"], box["size_y"], box["size_z"]]) / 2.0
    residues = {}
    for a in atoms:
        key = (a["chain"], a["resid"], a["res"])
        residues.setdefault(key, []).append(a)
    rows = []
    for key, al in residues.items():
        chain, resid, res = key
        sc = [a for a in al if a["name"] not in BACKBONE and a["name"] != "CB"
              and not a["adt"].startswith("H")]
        sc_all = [a for a in al if a["name"] not in KEEP_ALA]
        d_sc = (min(np.linalg.norm(a["xyz"] - center) for a in sc)
                if sc else float("nan"))
        inside = any(np.all(np.abs(a["xyz"] - center) <= half + 8.0) for a in al)
        rows.append({"key": key, "chain": chain, "resid": resid, "res": res,
                     "n_sc_heavy": len(sc), "d_sc": d_sc, "inside": bool(inside),
                     "atoms": al, "remove_lines": [a["line"] for a in sc_all]})
    return rows, center


def truncate(lines, remove_lines):
    drop = set(remove_lines)
    return [l for i, l in enumerate(lines) if i not in drop]


def choose(rows, rng, n_contact, n_shell, n_far):
    ok = [r for r in rows if r["res"] in STANDARD_AA and r["res"] not in NO_TRUNCATE
          and r["n_sc_heavy"] >= MIN_SC_ATOMS]
    contact = sorted([r for r in ok if r["d_sc"] <= CONTACT_MAX], key=lambda r: r["d_sc"])
    shell = [r for r in ok if SHELL_MIN <= r["d_sc"] <= SHELL_MAX]
    far = [r for r in ok if not r["inside"]]
    if len(contact) > n_contact:
        idx = np.linspace(0, len(contact) - 1, n_contact).round().astype(int)
        contact = [contact[i] for i in idx]
    shell = [shell[i] for i in rng.permutation(len(shell))[:n_shell]]
    far = [far[i] for i in rng.permutation(len(far))[:n_far]]
    return contact, shell, far


def label(r):
    return f"{r['res']}{r['resid']}"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build_target(target: str, src: Path, out: Path, seed: int, n_contact: int,
                 n_shell: int, n_far: int, n_multi: int):
    lines, atoms = read_receptor(src / f"{target}_target.pdbqt")
    box = read_box(src / f"{target}_conf.txt")
    rows, center = residue_table(atoms, box)
    rng = np.random.default_rng(seed)
    contact, shell, far = choose(rows, rng, n_contact, n_shell, n_far)
    tdir = out / target
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "wt.pdbqt").write_text("".join(lines))
    manifest = {"target": target, "box": box, "n_atoms": len(atoms),
                "receptors": [{"id": "wt", "kind": "wt", "residues": [], "n_removed": 0}]}

    def emit(rid, kind, rs):
        rem = sorted({ln for r in rs for ln in r["remove_lines"]})
        (tdir / f"{rid}.pdbqt").write_text("".join(truncate(lines, rem)))
        manifest["receptors"].append({
            "id": rid, "kind": kind, "n_removed": len(rem),
            "residues": [{"label": label(r), "res": r["res"], "resid": r["resid"],
                          "d_sc": None if np.isnan(r["d_sc"]) else round(float(r["d_sc"]), 2),
                          "n_sc_heavy": r["n_sc_heavy"], "class": cls}
                         for r, cls in [(r, _cls(r, contact, shell, far)) for r in rs]]})

    for r in contact:
        emit(f"c_{label(r)}", "single_contact", [r])
    for r in shell:
        emit(f"s_{label(r)}", "single_shell", [r])
    for r in far:
        emit(f"f_{label(r)}", "single_far", [r])
    for k in range(n_multi):
        m = int(rng.integers(2, 5))
        pick = [contact[i] for i in rng.permutation(len(contact))[:m]] if len(contact) >= m else contact
        emit(f"m{k:02d}", f"multi_contact_{len(pick)}", pick)
    (tdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _cls(r, contact, shell, far):
    if r in contact:
        return "contact"
    if r in shell:
        return "shell"
    return "far"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("data/raw/targets"))
    p.add_argument("--out", type=Path, default=Path("dockmut/receptors"))
    p.add_argument("--targets", default="")
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--contact", type=int, default=12)
    p.add_argument("--shell", type=int, default=4)
    p.add_argument("--far", type=int, default=2)
    p.add_argument("--multi", type=int, default=0)
    a = p.parse_args()
    names = ([t for t in a.targets.split(",") if t] if a.targets else
             sorted(f.name[: -len("_target.pdbqt")] for f in a.src.glob("*_target.pdbqt")))
    summary = []
    for t in names:
        m = build_target(t, a.src, a.out, a.seed + sum(map(ord, t)), a.contact,
                         a.shell, a.far, a.multi)
        kinds = {}
        for r in m["receptors"]:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        summary.append((t, kinds))
        print(t, kinds)
    (a.out / "summary.json").write_text(json.dumps(dict(summary), indent=1))


if __name__ == "__main__":
    main()
