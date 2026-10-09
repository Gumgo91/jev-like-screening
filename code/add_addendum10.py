"""Append addendum 10 to dockmut/PREREG.md and record its hash in prereg_ledger.json (run once, before the known-pocket evaluation is computed).

Usage: python dockmut/add_addendum10.py
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

here = Path(__file__).resolve().parent
prereg = here / "PREREG.md"
ledger = here / "prereg_ledger.json"
TEXT = """

## Addendum 10 (written AFTER the experiments of addendum 9 and BEFORE the known-pocket evaluation was computed; post hoc relative to the registered plan)

Scope decision: the manuscript is restricted to the pockets of the training data (known pockets), the setting in which the surrogates are used
as multi-pocket screening engines after a docked training sample exists for each pocket. The cross-validation on held-out target families of
addendum 9 (runs/v4, dockmut/eval/v4_cv_summary.json) was completed and is kept in the data archive; it is not part of the Results of the
manuscript, which states the restriction to known pockets in the Discussion and lists addendum 9 in the Supporting Information.

Experiment E3 (known-pocket evaluation, dockmut/v5_known_pockets_eval.py, outputs in runs/v5): the existing checkpoints of the dual encoder
(runs/v1 and runs/v2, seeds 11, 22 and 33), the dual encoder with one constant synthetic pocket (three seeds), the pooled-concatenation network
and the cross-attention network (runs/v3, three seeds each) are evaluated on the 39 training targets against the 10,000 development ligands
(labels of the open cell train x dev; no sealed label is read). The development ligands were never training ligands of these networks; the
checkpoint rule of these networks used the cold development targets only. Outcomes, all reported whatever the result: top-1% recall at the 1, 5, 10
and 20% budgets per target and as the macro mean over the 39 targets (mean over seeds per target), with 95% percentile intervals from 5,000
bootstrap resamples of the targets, and paired differences between networks; the model times of addendum 9 (experiment E2) give the cost side.
The manuscript states the recall of the shared-state networks as a fraction of the recall of the cross-attention network with its interval.
"""
s = prereg.read_text(encoding="utf-8")
if "Addendum 10" not in s:
    prereg.write_text(s.rstrip("\n") + TEXT, encoding="utf-8")
sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
d = json.loads(ledger.read_text(encoding="utf-8"))
if not any(e["sha256"] == sha for e in d["entries"]):
    d["entries"].append({"file": "dockmut/PREREG.md", "sha256": sha, "utc": datetime.now(timezone.utc).isoformat(),
                         "note": "addendum 10: manuscript restricted to known pockets; known-pocket evaluation of the existing checkpoints, written before it was computed"})
    ledger.write_text(json.dumps(d, indent=1), encoding="utf-8")
print(sha)
