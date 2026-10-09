"""Create fallback definitions for every number macro used in the result text.

Each macro prints a visible placeholder until numbers.tex (generated from analysis outputs) defines it.
"""
import re
import sys
from pathlib import Path

here = Path(__file__).parent
macros = set()
for f in sys.argv[1:]:
    macros |= set(re.findall(r"\\([A-Za-z]+)\{\}", (here / f).read_text(encoding="utf-8")))
skip = {"kcal", "dS", "AA", "TBD"}
lines = ["% Fallback definitions: a macro prints a visible placeholder until numbers.tex defines it."]
for m in sorted(macros - skip):
    lines.append("\\providecommand{\\%s}{\\TBD{%s}}" % (m, m))
(here / "numbers_placeholder.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(lines) - 1, "macros:", ", ".join(sorted(macros - skip)))
