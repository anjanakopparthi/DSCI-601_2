"""Score a filled-in blind audit sheet against the key.

Usage:  python evaluation/score_audit.py english
        python evaluation/score_audit.py tamil
        python evaluation/score_audit.py malayalam

Reads   metrics/audit_sheet_<lang>.csv   (your labels)
        metrics/audit_key_<lang>.csv     (gold + llm + which cell)
        the original llm_validation file (for the cell sizes in the full pool)

Writes  metrics/audit_scored_<lang>.json
        metrics/audit_disagreements_<lang>.csv   <- second pass goes here

Reports
  kappa(you, gold)  how far your reading is from the dataset's
  kappa(you, llm)   how far your reading is from the LLM's.  This is the number
                    that licenses using the LLM on languages you cannot read.
  noise rate        the fraction of the whole usable pool whose gold label you
                    would overturn, estimated from the two cells separately and
                    recombined by their true sizes.
"""
import json, math, sys
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "english":   ROOT / "data_collection/collected/llm_validation_english.csv",
    "tamil":     ROOT / "data_collection/collected/llm_validation.csv",
    "malayalam": ROOT / "data_collection/collected/llm_validation_malayalam.csv",
}

def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion. Behaves sensibly at k=0 and k=n,
    which a normal-approximation interval does not."""
    if n == 0:
        return (float("nan"), float("nan"))
    c = (k + z * z / 2) / (n + z * z)
    h = z / (n + z * z) * math.sqrt(k * (n - k) / n + z * z / 4)
    return (max(0.0, c - h), min(1.0, c + h))

def weighted_kappa(a, b, w):
    """Cohen's kappa on a confusion matrix built with per-row weights.

    The audit sheet over-samples the disagreement cell on purpose, so an
    unweighted kappa measures the sampling design as much as the labelling.
    Giving each row the weight (cell size in pool)/(rows sampled from cell)
    puts the two cells back in their true proportions.
    """
    import numpy as np
    a, b, w = np.asarray(a), np.asarray(b), np.asarray(w, dtype=float)
    M = np.zeros((2, 2))
    for ai, bi, wi in zip(a, b, w):
        M[int(ai), int(bi)] += wi
    T = M.sum()
    if T == 0:
        return float("nan")
    p_o = np.trace(M) / T
    p_e = float((M.sum(1) / T) @ (M.sum(0) / T))
    return float("nan") if p_e == 1 else (p_o - p_e) / (1 - p_e)


def pct(x):
    return "   n/a" if x != x else f"{100 * x:5.1f}%"


def main(lang):
    sheet = pd.read_csv(ROOT / f"metrics/audit_sheet_{lang}.csv", encoding="utf-8-sig")
    key   = pd.read_csv(ROOT / f"metrics/audit_key_{lang}.csv")

    # --- validate the sheet before trusting any number out of it ----------
    blank = sheet["my_label"].isna().sum()
    if blank:
        sys.exit(f"ERROR: {blank} rows have an empty my_label. Fill them in first.")
    sheet["my_label"] = sheet["my_label"].astype(int)
    bad = set(sheet["my_label"].unique()) - {0, 1}
    if bad:
        sys.exit(f"ERROR: my_label contains {sorted(bad)}; only 0 and 1 are allowed.")

    d = sheet.merge(key, on="row_id", how="inner", validate="one_to_one")
    if len(d) != len(sheet):
        sys.exit(f"ERROR: {len(sheet) - len(d)} sheet rows did not match the key.")

    d["note"] = d["note"].fillna("").astype(str).str.strip().str.lower()
    skipped = d[d["note"] == "not_judgeable"]
    d = d[d["note"] != "not_judgeable"].copy()

    # --- the full pool, so the two cells can be reweighted to their real sizes
    pool = pd.read_csv(FILES[lang])
    pool = pool[pool["llm_label"].isin([0, 1])]
    N  = len(pool)
    nA = int((pool["label"] == pool["llm_label"]).sum())
    nD = N - nA

    # --- headline agreement ----------------------------------------------
    k_gold = cohen_kappa_score(d["my_label"], d["label"])
    k_llm  = cohen_kappa_score(d["my_label"], d["llm_label"])
    a_gold = float((d["my_label"] == d["label"]).mean())
    a_llm  = float((d["my_label"] == d["llm_label"]).mean())

    # the same four figures with the two cells put back in their true
    # proportions -- these are the ones comparable to a full-pool number
    m = d["cell"].value_counts().to_dict()
    wmap = {"agree": (nA / N) / m.get("agree", 1),
            "disagree": (nD / N) / m.get("disagree", 1)}
    d["w"] = d["cell"].map(wmap)
    W = d["w"].sum()
    rk_gold = weighted_kappa(d["my_label"], d["label"], d["w"])
    rk_llm  = weighted_kappa(d["my_label"], d["llm_label"], d["w"])
    ra_gold = float((d.loc[d["my_label"] == d["label"], "w"].sum()) / W)
    ra_llm  = float((d.loc[d["my_label"] == d["llm_label"], "w"].sum()) / W)

    # --- per-cell overturn rates -----------------------------------------
    out = {}
    for cell, size in (("agree", nA), ("disagree", nD)):
        s = d[d["cell"] == cell]
        k = int((s["my_label"] != s["label"]).sum())
        n = len(s)
        lo, hi = wilson(k, n)
        out[cell] = dict(sampled=n, overturned=k,
                         rate=(k / n if n else float("nan")),
                         ci=[lo, hi], pool_size=size)

    # --- recombine into one pool-level rate ------------------------------
    rate = (nA / N) * out["agree"]["rate"] + (nD / N) * out["disagree"]["rate"]
    lo   = (nA / N) * out["agree"]["ci"][0] + (nD / N) * out["disagree"]["ci"][0]
    hi   = (nA / N) * out["agree"]["ci"][1] + (nD / N) * out["disagree"]["ci"][1]

    # --- report -----------------------------------------------------------
    print(f"\n=== {lang.upper()} ===  {len(d)} adjudicated"
          + (f", {len(skipped)} skipped as not_judgeable" if len(skipped) else ""))
    print("\n  on the sampled sheet (disagreements over-sampled -- NOT comparable")
    print("  to a full-pool kappa; shown only for the record):")
    print(f"    you vs gold    kappa {k_gold:5.3f}   agreement {pct(a_gold)}")
    print(f"    you vs LLM     kappa {k_llm:5.3f}   agreement {pct(a_llm)}")
    print("\n  REWEIGHTED to the true cell proportions -- report these:")
    print(f"    you vs gold    kappa {rk_gold:5.3f}   agreement {pct(ra_gold)}")
    print(f"    you vs LLM     kappa {rk_llm:5.3f}   agreement {pct(ra_llm)}")

    print(f"\n  pool: {N} usable = {nA} agree ({100*nA/N:.1f}%) "
          f"+ {nD} disagree ({100*nD/N:.1f}%)")
    print(f"\n  {'cell':<10} {'sampled':>8} {'overturned':>11} {'rate':>8}   95% CI")
    print("  " + "-" * 56)
    for cell in ("agree", "disagree"):
        o = out[cell]
        print(f"  {cell:<10} {o['sampled']:>8} {o['overturned']:>11} "
              f"{pct(o['rate']):>8}   [{pct(o['ci'][0])}, {pct(o['ci'][1])}]")
    print("  " + "-" * 56)
    print(f"  {'POOL':<10} {'':>8} {'':>11} {pct(rate):>8}   [{pct(lo)}, {pct(hi)}]")
    print(f"\n  -> estimated label-noise rate for {lang}: {100*rate:.1f}% "
          f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)")

    # --- second pass: the rows you would overturn ------------------------
    dis = d[d["my_label"] != d["label"]].copy()
    dis["verdict"] = ""        # gold_error | guideline | ambiguous
    cols = [c for c in ["row_id", "text", "my_label", "label", "llm_label",
                        "cell", "note", "script", "verdict"] if c in dis.columns]
    dis[cols].to_csv(ROOT / f"metrics/audit_disagreements_{lang}.csv", index=False)

    json.dump(dict(lang=lang, n_adjudicated=len(d), n_skipped=len(skipped),
                   kappa_vs_gold_sample=k_gold, kappa_vs_llm_sample=k_llm,
                   agreement_vs_gold_sample=a_gold, agreement_vs_llm_sample=a_llm,
                   kappa_vs_gold=rk_gold, kappa_vs_llm=rk_llm,
                   agreement_vs_gold=ra_gold, agreement_vs_llm=ra_llm,
                   pool_usable=N, pool_agree=nA, pool_disagree=nD,
                   cells=out, noise_rate=rate, noise_ci=[lo, hi]),
              open(ROOT / f"metrics/audit_scored_{lang}.json", "w"), indent=2)

    print(f"\n  {len(dis)} rows to classify in "
          f"metrics/audit_disagreements_{lang}.csv")
    print(f"  scores -> metrics/audit_scored_{lang}.json\n")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in FILES:
        sys.exit("usage: python evaluation/score_audit.py {english|tamil|malayalam}")
    main(sys.argv[1])
