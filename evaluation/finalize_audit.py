"""Fold the adjudication verdicts back into the noise-rate estimate.

Usage:  python evaluation/finalize_audit.py english

Reads   metrics/audit_disagreements_<lang>.csv   (verdict column filled in)
        metrics/audit_scored_<lang>.json         (pool and sample sizes)

Writes  metrics/audit_verdicts_<lang>.csv        backup, so a stray re-run of
                                                 score_audit.py cannot destroy
                                                 the hand-entered verdicts
        metrics/audit_final_<lang>.json

Produces three nested rates rather than one, because they answer three
different objections:

  strict     gold_error only.  Cannot be explained away as a guideline
             difference, so this is the defensible headline.
  defensible strict + guideline.  Everything a careful annotator would
             relabel, whatever the reason.
  all        every overturn, ambiguous included.  The upper bound, and the
             figure score_audit.py already reported.
"""
import json, math, shutil, sys
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
VERDICTS = ["gold_error", "guideline", "ambiguous"]

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    c = (k + z * z / 2) / (n + z * z)
    h = z / (n + z * z) * math.sqrt(k * (n - k) / n + z * z / 4)
    return (max(0.0, c - h), min(1.0, c + h))

def pct(x):
    return "   n/a" if x != x else f"{100 * x:5.1f}%"


def main(lang):
    src = ROOT / f"metrics/audit_disagreements_{lang}.csv"
    d = pd.read_csv(src, encoding="utf-8-sig")

    # back the verdicts up before anything else can clobber them
    bak = ROOT / f"metrics/audit_verdicts_{lang}.csv"
    shutil.copyfile(src, bak)

    if "verdict" not in d.columns:
        sys.exit("ERROR: no 'verdict' column in " + str(src))
    d["verdict"] = d["verdict"].fillna("").astype(str).str.strip().str.lower()

    blank = int((d["verdict"] == "").sum())
    if blank:
        sys.exit(f"ERROR: {blank} rows have an empty verdict. Fill them in first.")
    bad = sorted(set(d["verdict"]) - set(VERDICTS))
    if bad:
        sys.exit(f"ERROR: unrecognised verdicts {bad}. Allowed: {VERDICTS}")

    scored = json.load(open(ROOT / f"metrics/audit_scored_{lang}.json"))
    N, nA, nD = scored["pool_usable"], scored["pool_agree"], scored["pool_disagree"]
    mA = scored["cells"]["agree"]["sampled"]
    mD = scored["cells"]["disagree"]["sampled"]

    print(f"\n=== {lang.upper()} ===  {len(d)} overturns adjudicated")
    print(f"\n  {'verdict':<12} {'agree':>7} {'disagree':>9} {'total':>7}")
    print("  " + "-" * 38)
    for v in VERDICTS:
        s = d[d["verdict"] == v]
        print(f"  {v:<12} {int((s['cell']=='agree').sum()):>7} "
              f"{int((s['cell']=='disagree').sum()):>9} {len(s):>7}")
    print("  " + "-" * 38)
    print(f"  {'ALL':<12} {int((d['cell']=='agree').sum()):>7} "
          f"{int((d['cell']=='disagree').sum()):>9} {len(d):>7}")

    bands = {"strict":     ["gold_error"],
             "defensible": ["gold_error", "guideline"],
             "all":        VERDICTS}

    print(f"\n  {'band':<12} {'agree cell':>11} {'disagree cell':>14} "
          f"{'pool rate':>10}   95% CI")
    print("  " + "-" * 68)

    final = {}
    for name, keep in bands.items():
        s  = d[d["verdict"].isin(keep)]
        kA = int((s["cell"] == "agree").sum())
        kD = int((s["cell"] == "disagree").sum())
        rA, rD = kA / mA, kD / mD
        loA, hiA = wilson(kA, mA)
        loD, hiD = wilson(kD, mD)

        wA, wD = nA / N, nD / N
        rate = wA * rA + wD * rD
        lo   = wA * loA + wD * loD
        hi   = wA * hiA + wD * hiD

        print(f"  {name:<12} {f'{kA}/{mA}':>11} {f'{kD}/{mD}':>14} "
              f"{pct(rate):>10}   [{pct(lo)}, {pct(hi)}]")

        final[name] = dict(agree=dict(k=kA, n=mA, rate=rA, ci=[loA, hiA]),
                           disagree=dict(k=kD, n=mD, rate=rD, ci=[loD, hiD]),
                           pool_rate=rate, pool_ci=[lo, hi])

    st, al = final["strict"], final["all"]
    print(f"\n  -> {lang}: {100*st['pool_rate']:.1f}% of labels are indefensible "
          f"(95% CI {100*st['pool_ci'][0]:.1f}-{100*st['pool_ci'][1]:.1f}%)")
    print(f"     {100*al['pool_rate']:.1f}% would be relabelled by a careful "
          f"annotator (upper bound)")

    json.dump(dict(lang=lang, n_overturns=len(d),
                   verdict_counts=d["verdict"].value_counts().to_dict(),
                   pool_usable=N, pool_agree=nA, pool_disagree=nD,
                   kappa_vs_gold=scored.get("kappa_vs_gold"),
                   kappa_vs_llm=scored.get("kappa_vs_llm"),
                   bands=final),
              open(ROOT / f"metrics/audit_final_{lang}.json", "w"), indent=2)

    print(f"\n  verdict backup -> metrics/audit_verdicts_{lang}.csv")
    print(f"  results        -> metrics/audit_final_{lang}.json\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python evaluation/finalize_audit.py {english|tamil|malayalam}")
    main(sys.argv[1])
