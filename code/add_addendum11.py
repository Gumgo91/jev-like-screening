"""Append addendum 11 to dockmut/PREREG.md and record its hash in prereg_ledger.json (run once, before the cross-fitted known-pocket evaluation is computed)."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

here = Path(__file__).resolve().parent
prereg = here / "PREREG.md"
ledger = here / "prereg_ledger.json"
TEXT = """

## Addendum 11 (written AFTER the outcomes of experiment E3 of addendum 10 were known and BEFORE the cross-fitted known-pocket evaluation was computed; post hoc)

Reason: the checkpoint rule of the networks of experiment E3 (macro recall on the cold development targets) selects checkpoints for new targets, and it can
select early checkpoints that do not suit the pockets of the training data. E3 gave top-1% recall at the 10% budget of 0.860 (dual encoder), 0.849 (pooled
network), 0.847 (cross-attention network) and 0.833 (constant pocket) on the 39 training pockets. The networks of the cross-validation runs of addendum 9 choose their
checkpoint on seen targets (six validation targets of the training targets of the fold against 10,000 calibration ligands) and therefore suit the known-pocket setting.
Experiment E4 (dockmut/v5_known_cv_eval.py, outputs in runs/v5): every network of a fold (conditions dual, pooled and cross-attention with the real pockets and with the constant
pocket, recipe R0, and the recipe R1 for the dual encoder and the cross-attention network, seed 11) is evaluated on the pockets of its training targets against the 10,000
development ligands (open labels; neither training nor checkpoint choice used them). The recall of a pocket is the mean over the folds in which the pocket is a training target
(four of five); summaries are macro means over the 49 pockets with 95% bootstrap intervals and paired differences, as in E3. Both E3 and E4 are reported whatever the result.
"""
s = prereg.read_text(encoding="utf-8")
if "Addendum 11" not in s:
    prereg.write_text(s.rstrip("\n") + TEXT, encoding="utf-8")
sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
d = json.loads(ledger.read_text(encoding="utf-8"))
if not any(e["sha256"] == sha for e in d["entries"]):
    d["entries"].append({"file": "dockmut/PREREG.md", "sha256": sha, "utc": datetime.now(timezone.utc).isoformat(),
                         "note": "addendum 11: cross-fitted known-pocket evaluation of the cross-validation networks, written after E3 and before E4 was computed"})
    ledger.write_text(json.dumps(d, indent=1), encoding="utf-8")
print(sha)
