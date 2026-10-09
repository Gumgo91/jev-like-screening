"""Write job files for the validation, final, scaling and within-family runs.

  python make_jobs.py val  > jobs_val.txt
  python make_jobs.py final --arch rsg --lam 10 --lr 5e-4 [--family-balance] > jobs_final.txt
  python make_jobs.py scale --arch rsg --lam 10 --lr 5e-4 [--family-balance] > jobs_scale.txt
"""
from __future__ import annotations

import argparse

COMMON = "--updates 20000 --n-mut 6 --mut-batch 96"
VAL_FAM = "--val-families gpcr_classA,cyp450,serine_protease_S1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["val", "val2", "final", "scale"])
    p.add_argument("--arch", default="rsg")
    p.add_argument("--lam", type=float, default=10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--family-balance", action="store_true")
    a = p.parse_args()
    fb = " --family-balance" if a.family_balance else ""
    if a.mode == "val":
        cands = [("rs_int", "_V1", 3, 5e-4, ""), ("rs_int", "_V2", 10, 5e-4, ""), ("rsg_int", "_V3", 3, 5e-4, ""),
                 ("rsg_int", "_V4", 10, 5e-4, ""), ("rsg_int", "_V5", 30, 5e-4, ""), ("rsg_int", "_V6", 10, 1e-3, ""),
                 ("b5_int", "_V7", 10, 5e-4, ""), ("rs_int", "_V8", 10, 5e-4, " --family-balance"),
                 ("rsg_int", "_V9", 10, 5e-4, " --family-balance"), ("rsg_wt", "_V0", 0, 5e-4, "")]
        for cond, tag, lam, lr, extra in cands:
            print(f"--cond {cond} --tag {tag} --seed 11 {COMMON} --lam {lam} --lr {lr}{extra} {VAL_FAM} --out dockmut/runs_val")
    elif a.mode == "val2":
        fa, fb_ = "--val-families gpcr_classA,cyp450,serine_protease_S1", "--val-families nuclear_receptor"
        jobs = [("rs_wt", "_W_A", 0, fa), ("rsg_int", "_G1_A", 1, fa), ("rse_int", "_E3_A", 3, fa), ("rse_int", "_E10_A", 10, fa), ("rse_wt", "_EW_A", 0, fa),
                ("rs_int", "_R10_B", 10, fb_), ("rs_int", "_R3_B", 3, fb_), ("rs_wt", "_W_B", 0, fb_), ("rsg_wt", "_GW_B", 0, fb_),
                ("rsg_int", "_G1_B", 1, fb_), ("rsg_int", "_G3_B", 3, fb_), ("rsg_int", "_G10_B", 10, fb_), ("rse_int", "_E3_B", 3, fb_),
                ("rse_int", "_E10_B", 10, fb_), ("rse_wt", "_EW_B", 0, fb_)]
        for cond, tag, lam, vf in jobs:
            print(f"--cond {cond} --tag {tag} --seed 11 {COMMON} --lam {lam} --lr 5e-4 {vf} --out dockmut/runs_val2")
    elif a.mode == "final":
        arch = a.arch
        conds = [f"{arch}_wt", f"{arch}_int", f"{arch}_int_shl", f"{arch}_int_shm", f"{arch}_int_sha", f"{arch}_const"]
        for other in [x for x in ("rs", "rsg", "rse") if x != arch]:
            conds += [f"{other}_wt", f"{other}_int"]
        conds += ["b5_wt", "b5_int"]
        for c in conds:
            for seed in (11, 22, 33):
                lam = a.lam if "_int" in c else 0
                print(f"--cond {c} --seed {seed} {COMMON} --lam {lam} --lr {a.lr}{fb if not c.startswith('b5') else ''} --heldout-eval --out dockmut/runs_final")
    else:
        cond = f"{a.arch}_int"
        base = f"--cond {cond} {COMMON} --lam {a.lam} --lr {a.lr}{fb} --heldout-eval --out dockmut/runs_scale"
        for seed in (11, 22):
            for n in (4, 8, 16, 24):
                print(f"{base} --seed {seed} --tag _nt{n} --n-train-targets {n}")
            for k in (2, 4, 8, 12):
                print(f"{base} --seed {seed} --tag _mk{k} --mut-keep {k}")
        print(f"--cond {cond} --seed 11 {COMMON} --lam {a.lam} --lr {a.lr}{fb} --val-targets ABL1,EGFR,SRC,JAK2 --tag _kin --out dockmut/runs_scale")


if __name__ == "__main__":
    main()
