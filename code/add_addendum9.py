"""Append addendum 9 to dockmut/PREREG.md and record its hash in prereg_ledger.json (run once, before any job of the extension is trained or timed).

Usage: python dockmut/add_addendum9.py   (needs data/splits/v4_cv_folds.json from dockmut/v4_make_folds.py; its hash is written into the text)
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
prereg = here / "PREREG.md"
ledger = here / "prereg_ledger.json"
folds = json.loads((root / "data/splits/v4_cv_folds.json").read_text(encoding="utf-8"))
FOLD_SHA = folds.get("fold_assignment_sha256")
assert FOLD_SHA, "fold hash missing in v4_cv_folds.json"
TEXT = """

## Addendum 9 (written AFTER the review of the manuscript that followed addendum 8 and BEFORE any job of this extension was trained or timed; post hoc relative to the registered plan)

Motivation: the review of the manuscript found three weaknesses of the comparison of addendum 8. (a) The recall at the 10% budget of the nine
test targets cannot separate networks that read the pocket from networks that do not: a network with one constant dummy pocket recovers 0.82
against 0.83, and every network, the cross-attention network included, outputs almost the same ligand ranking for every target. (b) The nine test
targets are targets on which the shared ligand ranking works, and the cross-attention network was trained once with the recipe of the dual encoders.
(c) The cost ratio of the joint network rests on a linear extrapolation of an unoptimized implementation measured on a desktop RTX 4060.
This addendum registers two measurements on rented GPUs. Neither reads the sealed test labels.

Experiment E1 (cross-validation on 49 targets, scripts/v4_cv_train.py, dockmut/v4_make_folds.py, outputs in runs/v4): the 49 targets of the training
and development roles are split into five folds of whole families (kinase; nuclear receptor; GPCR class A, CYP450 and MAOB; aspartyl protease,
metalloprotease, serine protease S1, NOS1 and PARP1; PDE5A, PTGS2 and PTPN1), the assignment has the sha256 """ + FOLD_SHA + """ (data/splits/v4_cv_folds.json).
For every fold the networks are trained on the targets outside the fold (training ligands of the preceding analysis, 40,000 per target) and
evaluated on the held-out targets against the 10,000 development ligands, whose labels are open cells. Early stopping and the checkpoint rule use the
macro Recall_1%@10% over six targets drawn from the training targets of the fold against 10,000 ligands of the probability-calibration split
(a deviation from the cold-development-target rule of the preceding analysis, because the folds consume the targets). Conditions: dual encoder,
pooled-concatenation network and cross-attention network, each trained with the real pockets and with one constant synthetic pocket for every
target (six conditions), recipe R0 (AdamW, learning rate 3e-4, weight decay 1e-4, batches of 256 ligands of one target, binary cross-entropy on
the top-1% hits plus 0.2 x Huber loss on standardized scores, at most 60,000 updates or 10 epochs, evaluation every 6,094 updates, patience of
three evaluations), seed 11 first and seed 22 added if the first seed finishes within the budget of the rented time. A second recipe R1 (learning
rate 1e-4, patience of eight evaluations, at most 150,000 updates or 30 epochs) is run for the dual encoder and the cross-attention network with
the real pockets, to test whether the cross-attention network was undertrained. No hyperparameter is tuned and nothing is changed after the first
job starts.

Outcomes of E1, all reported whatever the result: (1) top-1% recall at the 1, 5, 10 and 20% budgets per held-out target and as the macro mean over
the 49 targets, with 95% percentile intervals from 5,000 bootstrap resamples of the targets; (2) the pocket gain of each architecture, the paired
difference between the network and its constant-pocket twin; (3) the selectivity, the correlation of the ligand-centered predictions with the
ligand-centered docking scores within a fold, and the reversal accuracy on quadruplets drawn within each fold; (4) the same outcomes on the targets
whose shared-ranking oracle recall (the recall of the ranking by the mean docking score over the training targets) is below 0.5.

Claim rules of E1: (a) the statement that the dual encoder reaches the screening recall of a joint network on 49 targets requires that the lower
limit of the 95% interval of the paired difference (dual encoder minus network, 10% budget) is at or above -0.03, for the cross-attention network
in recipes R0 and R1 and for the pooled network; (b) a network is said to read the pocket only if the lower limit of its pocket gain at the 10%
budget is above zero and its selectivity exceeds that of the constant-pocket twin (zero by construction); (c) otherwise the manuscript states that the
network does not read the pocket under this recipe.

Experiment E2 (cost at the full scale, dockmut/v4_fair_timing.py): on one rented RTX 4090 the complete DOCKSTRING library of 260,155 ligands is
screened from SMILES against 1, 5, 20 and 57 wild-type pockets, and against 1,000 pockets obtained by cycling the 57 pockets, with the dual encoder
(matrix product), the pooled network (decomposed first layer) and the cross-attention network in its original padded float32 form, in a ragged float32
form and in a ragged float16 form, after a check that every optimized scorer reproduces the reference module (rank correlation of at least 0.9999 in
float32 and 0.999 in float16). The stages (featurization, embedding, pocket states, scoring) and the end-to-end time are reported as measured, and the
ratio of the cross-attention network to the dual encoder replaces the extrapolated ratio of addendum 8 in the manuscript, whatever its value.
"""
s = prereg.read_text(encoding="utf-8")
if "Addendum 9" not in s:
    prereg.write_text(s.rstrip("\n") + TEXT, encoding="utf-8")
sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
d = json.loads(ledger.read_text(encoding="utf-8"))
if not any(e["sha256"] == sha for e in d["entries"]):
    d["entries"].append({"file": "dockmut/PREREG.md", "sha256": sha, "utc": datetime.now(timezone.utc).isoformat(),
                         "note": "addendum 9: 49-target cross-validation with constant-pocket controls and full-scale timing, written before any job"})
    ledger.write_text(json.dumps(d, indent=1), encoding="utf-8")
print(sha)
