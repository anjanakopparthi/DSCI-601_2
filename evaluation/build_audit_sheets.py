"""Build blind adjudication sheets for the three-language label audit.

For each language, samples from both the agreement cell and the disagreement cell
between the HopeEDI gold labels and the independent LLM labelling, shuffles them
together, and writes two files:

  metrics/audit_sheet_<lang>.csv   the blind sheet: id + text + blank my_label
  metrics/audit_key_<lang>.csv     the key: id + gold + llm + which cell it came from

Sampling both cells is what makes a dataset-level noise rate estimable. Sampling
only disagreements, as the first pass did, can show mislabelling exists but cannot
say how common it is.

The sheet is blind so the annotator cannot anchor to the gold label.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_DISAGREE = 40      # capped; English has fewer, so all are taken
N_AGREE     = 30     # enough to bound the error rate in the agreement cell

FILES = {
    "english":   ROOT / "data_collection/collected/llm_validation_english.csv",
    "tamil":     ROOT / "data_collection/collected/llm_validation.csv",
    "malayalam": ROOT / "data_collection/collected/llm_validation_malayalam.csv",
}

def script_of(s):
    s = str(s)
    nat = sum(1 for ch in s if "஀" <= ch <= "௿" or "ഀ" <= ch <= "ൿ")
    lat = sum(1 for ch in s if ch.isascii() and ch.isalpha())
    if nat + lat == 0:
        return "other"
    return "native" if nat / (nat + lat) > 0.5 else "romanized"

print(f"{'language':<11} {'usable':>7} {'agree':>7} {'disagree':>9}   sampled")
print("-" * 62)

for lang, path in FILES.items():
    d = pd.read_csv(path).reset_index(drop=True)
    d["row_id"] = [f"{lang[:2]}{i:04d}" for i in range(len(d))]
    usable = d[d["llm_label"].isin([0, 1])].copy()

    agree    = usable[usable["label"] == usable["llm_label"]]
    disagree = usable[usable["label"] != usable["llm_label"]]

    # stratify each cell by the gold class so neither class is missed
    def take(df, n):
        if len(df) <= n:
            return df
        per = max(1, n // df["label"].nunique())
        out = (df.groupby("label", group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), per), random_state=SEED)))
        if len(out) < n:                       # top up from whatever is left
            rest = df.drop(out.index)
            out = pd.concat([out, rest.sample(min(len(rest), n - len(out)),
                                              random_state=SEED)])
        return out

    sa, sd = take(agree, N_AGREE), take(disagree, N_DISAGREE)
    sa = sa.assign(cell="agree"); sd = sd.assign(cell="disagree")

    sample = pd.concat([sa, sd]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample["script"] = sample["text"].map(script_of)

    # the blind sheet — no gold, no llm label
    sheet = sample[["row_id", "text"]].copy()
    sheet["my_label"] = ""      # 1 = Hope, 0 = Non_hope
    sheet["note"] = ""          # gold error / guideline / ambiguous / sarcasm / unclear
    sheet.to_csv(ROOT / f"metrics/audit_sheet_{lang}.csv", index=False)

    # the key, kept separate
    sample[["row_id", "label", "llm_label", "cell", "script"]].to_csv(
        ROOT / f"metrics/audit_key_{lang}.csv", index=False)

    print(f"{lang:<11} {len(usable):>7} {len(agree):>7} {len(disagree):>9}   "
          f"{len(sa)} agree + {len(sd)} disagree = {len(sample)}")

print()
print("Blind sheets  -> metrics/audit_sheet_<lang>.csv   (fill in my_label: 1 = Hope, 0 = Non_hope)")
print("Keys          -> metrics/audit_key_<lang>.csv     (do not open until the sheet is filled)")
