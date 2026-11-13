"""Fill the verdict column of metrics/audit_disagreements_malayalam.csv.

Run from the repo root:  python evaluation/apply_malayalam_verdicts.py

Also repairs mojibake in the text column where it can. Mojibake here is UTF-8
bytes that were decoded as cp1252 and written back out, so re-encoding the
string as cp1252 and decoding it as UTF-8 recovers the original Malayalam.
Rows that do not round-trip are left exactly as they were.
"""
import csv, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "metrics/audit_disagreements_malayalam.csv"

V = {
 # agree cell -- praise, thanks, anniversary wishes
 "ma0386":"guideline","ma0185":"guideline","ma0080":"guideline",
 "ma0206":"guideline","ma0193":"guideline","ma0174":"guideline",
 "ma0038":"guideline",
 # disagree cell -- praise of a debater
 "ma0165":"guideline","ma0184":"guideline",
 # gold said Non_hope, defensibly
 "ma0091":"guideline","ma0282":"guideline",
 # hostility, mockery, slurs, neutral statements -- all marked Hope by gold
 "ma0343":"gold_error","ma0104":"gold_error","ma0116":"gold_error",
 "ma0036":"gold_error","ma0368":"gold_error","ma0154":"gold_error",
 "ma0176":"gold_error","ma0125":"gold_error","ma0190":"gold_error",
 "ma0128":"gold_error","ma0063":"gold_error","ma0110":"gold_error",
 "ma0290":"gold_error","ma0234":"gold_error","ma0002":"gold_error",
 "ma0373":"gold_error",
 # plainly defends the group, marked Non_hope
 "ma0214":"gold_error",
 # a positive and a dig in one comment, or sarcasm that cuts both ways
 "ma0375":"ambiguous","ma0369":"ambiguous","ma0094":"ambiguous",
}

def load(p):
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(p, encoding=enc, newline="") as f:
                return list(csv.DictReader(f)), enc
        except UnicodeDecodeError:
            continue
    sys.exit(f"could not decode {p}")

def repair(s):
    try:
        fixed = s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s, False
    return fixed, fixed != s

rows, enc = load(SRC)
rows = [r for r in rows if (r.get("row_id") or "").strip()]
print(f"read {len(rows)} rows as {enc}")

ids = {r["row_id"] for r in rows}
if ids != set(V):
    sys.exit(f"MISMATCH missing={sorted(ids - set(V))} extra={sorted(set(V) - ids)}")

repaired = 0
for r in rows:
    r["verdict"] = V[r["row_id"]]
    if r.get("text"):
        r["text"], changed = repair(r["text"])
        repaired += changed

with open(SRC, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

c = Counter(r["verdict"] for r in rows)
print(f"text repaired     {repaired}/{len(rows)}")
for v in ("gold_error", "guideline", "ambiguous"):
    inA = sum(1 for r in rows if r["cell"] == "agree" and r["verdict"] == v)
    print(f"  {v:<11} {c[v]:>3}   of which agree cell: {inA}")
print(f"\nwrote {SRC.name} as utf-8")
