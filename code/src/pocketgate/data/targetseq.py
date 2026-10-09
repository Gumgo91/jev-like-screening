"""Target sequence extraction and family clustering (plan 5.1).

- Sequence: ordered unique residues (CA atom per residue) parsed from the
  prepared receptor pdbqt. pdbqt has no chain column; residues are taken
  in file order, deduplicated by (res_name, res_seq). This approximates
  the construct sequence - documented limitation.
- Pairwise similarity: Biopython local aligner (BLOSUM62). Link rule:
  identity >= 30% AND coverage of the shorter sequence >= 80%.
- Known-family links (recorded in DATA_AUDIT, applied as an explicit
  conservative merge list FAMILY_MAP below): targets sharing a
  well-established ligand-binding family / pocket fold are merged even
  if pairwise identity is below the link threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pocketgate.common import TARGETS_DIR  # noqa: E402
from pocketgate.data.audit import parse_pdbqt_atoms  # noqa: E402

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "ASH": "D",
    "GLH": "E", "MSE": "M",
}

# Curated ligand-binding family annotation (standard pharmacology
# families; used as an explicit merge layer on top of sequence links).
FAMILY_MAP = {
    # protein kinase ATP-site fold
    "ABL1": "kinase", "AKT1": "kinase", "AKT2": "kinase", "CDK2": "kinase",
    "CSF1R": "kinase", "EGFR": "kinase", "FGFR1": "kinase",
    "IGF1R": "kinase", "JAK2": "kinase", "KDR": "kinase", "KIT": "kinase",
    "LCK": "kinase", "MAP2K1": "kinase", "MAPK1": "kinase",
    "MAPK14": "kinase", "MAPKAPK2": "kinase", "MET": "kinase",
    "PLK1": "kinase", "PTK2": "kinase", "ROCK1": "kinase", "SRC": "kinase",
    # nuclear receptor ligand-binding domain
    "AR": "nuclear_receptor", "ESR1": "nuclear_receptor",
    "ESR2": "nuclear_receptor", "NR3C1": "nuclear_receptor",
    "PGR": "nuclear_receptor", "PPARA": "nuclear_receptor",
    "PPARD": "nuclear_receptor", "PPARG": "nuclear_receptor",
    "THRB": "nuclear_receptor",
    # class-A aminergic/adenosine GPCR orthosteric site
    "ADORA2A": "gpcr_classA", "ADRB1": "gpcr_classA", "ADRB2": "gpcr_classA",
    "DRD2": "gpcr_classA", "DRD3": "gpcr_classA",
    # trypsin-like serine proteases (S1)
    "F2": "serine_protease_S1", "F10": "serine_protease_S1",
    # pepsin-like aspartyl proteases
    "BACE1": "aspartyl_protease", "REN": "aspartyl_protease",
    # metzincin metalloproteases
    "ADAM17": "metalloprotease", "MMP13": "metalloprotease",
    # cytochrome P450
    "CYP2C9": "cyp450", "CYP3A4": "cyp450",
    # singletons (own family)
    "ACHE": "ache", "CA2": "ca2", "CASP3": "casp3", "DHFR": "dhfr",
    "DPP4": "dpp4", "GBA": "gba", "HMGCR": "hmgcr", "HSD11B1": "hsd11b1",
    "HSP90AA1": "hsp90aa1", "MAOB": "maob", "NOS1": "nos1",
    "PARP1": "parp1", "PDE5A": "pde5a", "PTGS2": "ptgs2",
    "PTPN1": "ptpn1",
}


def extract_sequence(pdbqt_path: Path) -> str:
    seen = set()
    seq = []
    for atom_name, res_name, res_seq, *_ in parse_pdbqt_atoms(pdbqt_path):
        key = (res_name, res_seq)
        if key in seen:
            continue
        if atom_name.strip() not in ("CA",):
            continue
        seen.add(key)
        seq.append(AA3_TO_1.get(res_name.upper(), "X"))
    return "".join(seq)


def main():
    from Bio import Align
    from Bio.Align import substitution_matrices

    targets = sorted(
        p.name[: -len("_target.pdbqt")]
        for p in TARGETS_DIR.glob("*_target.pdbqt")
    )
    seqs = {t: extract_sequence(TARGETS_DIR / f"{t}_target.pdbqt") for t in targets}
    for t in targets:
        assert len(seqs[t]) > 0, t

    aligner = Align.PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5

    n = len(targets)
    seq_links = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = targets[i], targets[j]
            aln = aligner.align(seqs[a], seqs[b])[0]
            blocks_a, blocks_b = aln.aligned
            matches = aligned_len = span_a = span_b = 0
            for (a0, a1), (b0, b1) in zip(blocks_a, blocks_b):
                la, lb = a1 - a0, b1 - b0
                aligned_len += max(la, lb)
                span_a += la
                span_b += lb
                for k in range(min(la, lb)):
                    if seqs[a][a0 + k] == seqs[b][b0 + k]:
                        matches += 1
            ident = matches / aligned_len if aligned_len else 0.0
            cov_short = min(span_a / len(seqs[a]), span_b / len(seqs[b]))
            if ident >= 0.30 and cov_short >= 0.80:
                seq_links.append((a, b, round(ident, 3), round(cov_short, 3)))

    return targets, seqs, seq_links


if __name__ == "__main__":
    targets, seqs, links = main()
    print("targets:", len(targets))
    print("seq lengths:", {t: len(s) for t, s in list(seqs.items())[:5]}, "...")
    print(f"seq links (>=30% id, >=80% cov): {len(links)}")
    for a, b, i, c in links:
        fam_a, fam_b = FAMILY_MAP.get(a), FAMILY_MAP.get(b)
        tag = "samefam" if fam_a == fam_b else "CROSS-FAMILY"
        print(f"  {a}-{b}: id={i} cov={c} {tag} ({fam_a}/{fam_b})")
