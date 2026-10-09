"""Check every entry of prereg_ledger.json against PREREG.md (run: python verify_ledger.py [PREREG.md] [prereg_ledger.json]).
An entry is the SHA-256 of the plan as it stood when the entry was written, which is a line-aligned prefix of the file.
The first eight entries were hashed with LF line endings and the later ones with CRLF line endings, so each prefix is tried in both forms."""
import hashlib, json, sys
from pathlib import Path
here = Path(__file__).resolve().parent
plan = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "PREREG.md"
ledger = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "prereg_ledger.json"
lf = plan.read_bytes().replace(b"\r\n", b"\n")
forms = {"LF": lf, "CRLF": lf.replace(b"\n", b"\r\n")}
cuts = {k: [len(v)] + [i + 1 for i, c in enumerate(v) if c == 10] for k, v in forms.items()}
bad = 0
for n, e in enumerate(json.loads(ledger.read_text(encoding="utf-8"))["entries"], 1):
    hit = next(((k, c) for k, v in forms.items() for c in cuts[k] if hashlib.sha256(v[:c]).hexdigest() == e["sha256"]), None)
    print(n, e["sha256"][:16], f"match {hit[0]} prefix of {hit[1]} bytes" if hit else "NO MATCH")
    bad += hit is None
sys.exit(1 if bad else 0)
