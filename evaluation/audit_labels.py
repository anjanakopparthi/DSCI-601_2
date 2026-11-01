"""Audit HopeEDI gold labels against an independent LLM labelling.

Reports per-language agreement, Cohen's kappa and class base rates; splits Tamil
by writing system to test whether disagreement is an artefact of romanization;
and dumps the gold=Hope / LLM=Non_hope cases for manual inspection.
"""
import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "tamil":     ROOT / "data_collection/collected/llm_validation.csv",
    "english":   ROOT / "data_collection/collected/llm_validation_english.csv",
    "malayalam": ROOT / "data_collection/collected/llm_validation_malayalam.csv",
}

def script_of(s):
    s = str(s)
    tam = sum(1 for ch in s if "\u0b80" <= ch <= "\u0bff")
    lat = sum(1 for ch in s if ch.isascii() and ch.isalpha())
    if tam + lat == 0:
        return "other"
    return "native" if tam / (tam + lat) > 0.5 else "romanized"

def stats(g):
    return {
        "n": int(len(g)),
        "agreement": round(float(accuracy_score(g["label"], g["llm_label"])), 4),
        "kappa": round(float(cohen_kappa_score(g["label"], g["llm_label"])), 4),
        "macro_f1": round(float(f1_score(g["label"], g["llm_label"], average="macro")), 4),
        "llm_hope_rate": round(float(g["llm_label"].mean()), 4),
        "gold_hope_rate": round(float(g["label"].mean()), 4),
    }

out = {}
for lang, path in FILES.items():
    d = pd.read_csv(path)
    usable = d[d["llm_label"].isin([0, 1])]
    rec = {"rows": int(len(d)), "usable": int(len(usable)),
           "unparsed": int((d["llm_label"] == -1).sum())}
    rec.update(stats(usable))
    out[lang] = rec
    print(f"{lang:<10} n={rec['usable']:<4}/{rec['rows']:<4} "
          f"kappa {rec['kappa']:.3f}  agreement {rec['agreement']:.3f}  "
          f"LLM Hope {100*rec['llm_hope_rate']:.0f}% vs gold {100*rec['gold_hope_rate']:.0f}%"
          + (f"  [{rec['unparsed']} unparsed]" if rec["unparsed"] else ""))

# does romanization explain Tamil's disagreement?
print("\nTamil, split by writing system:")
d = pd.read_csv(FILES["tamil"])
d["script"] = d["text"].map(script_of)
out["tamil_by_script"] = {}
for sc, g in d.groupby("script"):
    g = g[g["llm_label"].isin([0, 1])]
    if len(g) < 20:
        continue
    out["tamil_by_script"][sc] = stats(g)
    print(f"  {sc:<10} n={len(g):<4} kappa {out['tamil_by_script'][sc]['kappa']:.3f}  "
          f"agreement {out['tamil_by_script'][sc]['agreement']:.3f}  "
          f"LLM Hope {100*g['llm_label'].mean():.0f}% vs gold {100*g['label'].mean():.0f}%")

# cases for manual inspection
cases = d[(d["label"] == 1) & (d["llm_label"] == 0) & (d["script"] == "native")]
out["inspection"] = {
    "cell": "gold=Hope, LLM=Non_hope, native script",
    "candidates": int(len(cases)),
    "inspected": 20,
    "judged_hope_by_annotator": 2,
    "note": "Manual inspection of the first 20, via translation. Two were plausibly "
            "hope (one clearly, one weakly). The remainder were content requests, "
            "questions, political commentary or complaints, and one was explicit "
            "dehumanising hate speech. This is the disagreement cell, not a random "
            "sample, so it is an existence proof and not a dataset noise rate.",
}
cases.head(20)[["text", "label", "llm_label"]].to_csv(
    ROOT / "metrics/label_audit_inspection_tamil.csv", index=False)

with open(ROOT / "metrics/label_audit.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved -> metrics/label_audit.json and metrics/label_audit_inspection_tamil.csv")