"""Convert dockmut/PREREG.md and its hash ledger into LaTeX for the Supporting Information.

Writes si_prereg.tex (plan text) and si_ledger.tex (table of hashes). Headings become unnumbered subsections and
blank-line separated blocks become paragraphs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent


def esc(s: str) -> str:
    s = s.replace("\\", "\\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("^", r"\^{}"), ("~", r"\~{}"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    s = re.sub(r"`([^`]+)`", r"\\texttt{\1}", s)
    s = s.replace("+-", r"$\pm$").replace(">=", r"$\ge$").replace("<=", r"$\le$")
    s = s.replace(" > ", r" $>$ ").replace(" < ", r" $<$ ")
    s = re.sub(r"\b1e-3\b", r"$10^{-3}$", s)
    return s


text = (ROOT / "dockmut/PREREG.md").read_text(encoding="utf-8")
blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
lines = []
for b in blocks:
    if b.startswith("#"):
        head, _, rest = b.partition("\n")
        lines.append(r"\subsection*{" + esc(head.lstrip("# ").strip()) + "}")
        if rest.strip():
            lines.append(esc(" ".join(x.strip() for x in rest.splitlines())))
    else:
        para = " ".join(x.strip() for x in b.splitlines())
        lines.append(esc(para))
out_text = "\n\n".join(lines) + "\n"
# corrections of the SI that do not change the hashed text of PREREG.md
ERR6 = (r" Erratum: the hashed text gives 0.72 for the regression head on the nine test targets, and the value recomputed for Table~\ref{tab:wt} is \wtRTestCZero{}.")
anchor6 = "The registered outcomes and conditions are unchanged."
i6 = out_text.index("While reviewing the screening comparison")
j6 = out_text.index(anchor6, i6) + len(anchor6)
if ERR6 not in out_text:
    out_text = out_text[:j6] + ERR6 + out_text[j6:]
out_text = out_text.replace("They replace the laptop timings of the first draft.", "They replace the timings on the local RTX 4060 desktop of the first draft.")
(OUT / "si_prereg.tex").write_text(out_text, encoding="utf-8")

led = json.loads((ROOT / "dockmut/prereg_ledger.json").read_text())["entries"]
rows = [r"\begin{table}[h]", r"\caption{Hash ledger of the analysis plan. Each entry is the SHA-256 of \texttt{PREREG.md} at the time of writing, with the UTC time and the status of the plan.}",
        r"\label{tab:si_ledger}", r"\small", r"\begin{tabular}{llp{0.4\linewidth}}", r"\toprule", r"UTC time & SHA-256 (first 16 characters) & Status \\", r"\midrule"]
for e in led:
    rows.append(f"{e['utc'][:19].replace('T', ' ')} & \\texttt{{{e['sha256'][:16]}}} & {esc(e['note'])} " + r"\\")
rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "si_ledger.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("si_prereg.tex", len(lines), "blocks; ledger", len(led), "entries")
