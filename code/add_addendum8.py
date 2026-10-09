"""Append addendum 8 to dockmut/PREREG.md and record its hash in prereg_ledger.json (run once, before the joint networks are trained)."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

here = Path(__file__).resolve().parent
prereg = here / "PREREG.md"
ledger = here / "prereg_ledger.json"
TEXT = """

## Addendum 8 (written AFTER the revision of the manuscript to the Jev-like framing and BEFORE any model of this comparison was trained; post hoc relative to the registered plan)

Motivation: the manuscript claims that the Jev-like organization (pocket encoded once into a state, ligand embeddings that do not
depend on the pocket, a cheap readout) gives surrogates that cost far less than joint networks for a library-against-many-pockets
screen, at no loss of screening recall. Test: two joint surrogates are trained with the recipe of the baseline dual encoders
(scripts/v3_train_joint.py, outputs in runs/v3): xattn, the PocketGate network in which ligand atoms attend to the residues of
the pocket so that the ligand representation depends on the pocket, and pooled, PooledConcatNet, which concatenates pooled ligand
and pocket vectors and reads them with a multilayer perceptron.

Conditions: seeds 11, 22 and 33 for each of the two networks. Settings are those of the baseline dual encoders of runs/v2 (AdamW,
learning rate 3e-4, weight decay 1e-4, batches of 256 ligands of one training target, binary cross-entropy on the top-1% hits plus
0.2 x Huber loss on standardized scores, at most 60,000 updates or 10 epochs, evaluation every 6,094 updates, patience of three
evaluations, checkpoint = best macro Recall_1%@10% on the cold development targets). No hyperparameter is tuned and nothing is
changed after the first training run starts.

Outcomes: (1) top-1% recall at the 1, 5, 10 and 20% budgets on the 10 development targets and, after the checkpoints are frozen
(sha256 in runs/v3/v3_registry.json), on the 9 test targets with the frozen frames of the preceding analysis; the sealed test labels
are read a THIRD time, once, only for these six models (logged in runs/label_access.log); (2) the reversal accuracy on the frozen
quadruplets; (3) the time to score the 260,155 ligands against the 57 wild-type pockets and against K pocket variants on one GPU for
the joint networks (ligand atom states cached across pockets where the network allows it) and for the Jev-like networks. All
outcomes are reported whatever the result.

Claim rule: the statement that the Jev-like dual encoders reach the screening recall of the joint networks requires that the lower
limit of the 95% interval over targets of the paired difference (dual encoder with its hit-probability head minus the joint network,
test targets, 10% budget) is at or above -0.03. If the joint networks exceed this margin, the manuscript reports the cost and the
recall of both and states the trade-off.
"""
s = prereg.read_text(encoding="utf-8")
if "Addendum 8" not in s:
    prereg.write_text(s.rstrip("\n") + TEXT, encoding="utf-8")
sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
d = json.loads(ledger.read_text(encoding="utf-8"))
if not any(e["sha256"] == sha for e in d["entries"]):
    d["entries"].append({"file": "dockmut/PREREG.md", "sha256": sha, "utc": datetime.now(timezone.utc).isoformat(),
                         "note": "addendum 8: joint (cross-attention and pooled) surrogates trained with the baseline recipe, written before training"})
    ledger.write_text(json.dumps(d, indent=1), encoding="utf-8")
print(sha)
