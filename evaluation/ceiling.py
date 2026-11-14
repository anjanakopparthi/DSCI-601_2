"""Estimate the label-noise ceiling: the macro F1 a PERFECT classifier would
score when graded against the observed, noisy test labels.

Run from the repo root:  python evaluation/ceiling.py

Idea. If a share of the test labels is wrong, then a classifier that outputs
the true label for every comment still disagrees with the answer key on those
rows, and so cannot score 1.0. The attainable score is bounded by how noisy
the key is. Comparing that bound against what the models actually reached says
how much of the remaining gap is the model's fault and how much is the data's.

Noise is modelled class-conditionally, because the audit found it is strongly
asymmetric -- most errors are comments labelled Hope that are not. So two
rates per language:

  rho_1 = P(the label is wrong | the key says Hope)
  rho_0 = P(the label is wrong | the key says Non_hope)

Both are estimated from the audit sheets, with each sampled row weighted by
(cell size in the pool) / (rows sampled from that cell), since the sheets
deliberately over-sample the disagreement cell.

Two bands, as elsewhere: strict counts only overturns adjudicated gold_error;
all counts every overturn.

Then, for the real test split of each language, the observed labels are
perturbed by those rates to produce a draw of the true labels, a perfect
classifier is taken to output those true labels, and macro F1 is computed
against the observed labels. Averaged over many draws.

Assumptions, which belong in the write-up:
 - the annotator's judgement is treated as ground truth;
 - noise is independent of the text given the class, which is unlikely to hold
   exactly and makes this an optimistic bound on the ceiling;
 - rates estimated on the 400-row validation pool are applied to the test
   split.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

ROOT  = Path(__file__).resolve().parents[1]
LANGS = ["english", "tamil", "malayalam"]
DRAWS = 4000
RNG   = np.random.default_rng(42)

ACHIEVED = {}   # filled from sweep_summary.json plus the English single run

TEXTCOLS  = ["text", "comment", "Comment", "sentence", "text_clean"]
LABELCOLS = ["label", "Label", "class"]

def pick(d, names):
    return next((c for c in names if c in d.columns), None)

def rates(lang):
    """Weighted class-conditional noise rates for one language."""
    sheet = pd.read_csv(ROOT / f"metrics/audit_sheet_{lang}.csv",
                        encoding="utf-8-sig")
    key   = pd.read_csv(ROOT / f"metrics/audit_key_{lang}.csv")
    ver   = pd.read_csv(ROOT / f"metrics/audit_verdicts_{lang}.csv",
                        encoding="utf-8-sig")[["row_id", "verdict"]]
    sc    = json.load(open(ROOT / f"metrics/audit_scored_{lang}.json"))

    d = sheet.merge(key, on="row_id").merge(ver, on="row_id", how="left")
    d["note"] = d["note"].fillna("").astype(str).str.strip().str.lower()
    d = d[d["note"] != "not_judgeable"].copy()
    d["my_label"] = d["my_label"].astype(int)
    d["verdict"]  = d["verdict"].fillna("").astype(str).str.strip().str.lower()

    N, nA, nD = sc["pool_usable"], sc["pool_agree"], sc["pool_disagree"]
    m = d["cell"].value_counts().to_dict()
    w = {"agree": (nA / N) / m.get("agree", 1),
         "disagree": (nD / N) / m.get("disagree", 1)}
    d["w"] = d["cell"].map(w)

    out = {}
    for band, keep in (("strict", {"gold_error"}),
                       ("all", {"gold_error", "guideline", "ambiguous"})):
        r = {}
        for g in (0, 1):
            s = d[d["label"] == g]
            denom = s["w"].sum()
            wrong = s[(s["my_label"] != s["label"]) & s["verdict"].isin(keep)]
            r[g] = (wrong["w"].sum() / denom) if denom else 0.0
        out[band] = r
    return out, len(d)

def ceiling(obs, r0, r1):
    """Expected macro F1 of a perfect classifier graded against noisy labels."""
    obs = np.asarray(obs)
    p   = np.where(obs == 1, r1, r0)          # per-row chance the key is wrong
    scores = []
    for _ in range(DRAWS):
        truth = np.where(RNG.random(len(obs)) < p, 1 - obs, obs)
        scores.append(f1_score(obs, truth, average="macro", zero_division=0))
    a = np.array(scores)
    return a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5)

# ---- what the models actually reached -------------------------------------
sw = json.load(open(ROOT / "metrics/sweep_summary.json"))
for k, v in sw.items():
    model, lang = k.split("|")
    cur = ACHIEVED.get(lang)
    if cur is None or v["test_macro_f1_mean"] > cur[1]:
        ACHIEVED[lang] = (model, v["test_macro_f1_mean"], v.get("test_macro_f1_std"))
p = ROOT / "metrics/xlmr_english_metrics.json"
if p.exists():
    e = json.load(open(p))
    ACHIEVED["english"] = ("XLM-R", e["test"]["macro_f1"], None)

print(f"Label-noise ceiling, {DRAWS} draws per estimate\n")
print(f"  {'language':<11} {'band':<7} {'rho_0':>7} {'rho_1':>7} "
      f"{'ceiling':>9} {'95% range':>17} {'achieved':>10} {'headroom':>9}")
print("  " + "-" * 86)

report = {}
for lang in LANGS:
    tp = ROOT / f"processed/{lang}_test_processed.csv"
    if not tp.exists():
        print(f"  {lang:<11} test split not found at {tp}")
        continue
    t  = pd.read_csv(tp)
    lc = pick(t, LABELCOLS)
    obs = pd.to_numeric(t[lc], errors="coerce").dropna().astype(int).values

    rs, n_adj = rates(lang)
    got = ACHIEVED.get(lang)
    report[lang] = dict(n_test=len(obs), n_adjudicated=n_adj,
                        achieved=got[1] if got else None,
                        achieved_model=got[0] if got else None, bands={})
    for band in ("strict", "all"):
        r = rs[band]
        mu, lo, hi = ceiling(obs, r[0], r[1])
        head = (mu - got[1]) if got else float("nan")
        print(f"  {lang:<11} {band:<7} {r[0]:>7.3f} {r[1]:>7.3f} {mu:>9.4f} "
              f"  [{lo:.4f}, {hi:.4f}] {got[1]:>10.4f} {head:>9.4f}")
        report[lang]["bands"][band] = dict(rho_0=r[0], rho_1=r[1],
                                           ceiling=mu, ci=[lo, hi],
                                           headroom=head)
    print()

json.dump(report, open(ROOT / "metrics/noise_ceiling.json", "w"), indent=2)
print("  rho_0 = chance a Non_hope label is wrong; rho_1 = same for Hope")
print("  headroom = ceiling minus achieved; near zero means the model is at "
      "the limit the labels permit")
print("\nwrote metrics/noise_ceiling.json")
