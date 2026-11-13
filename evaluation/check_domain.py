"""Profile what the three HopeEDI language subsets are actually about.

Run from the repo root:  python evaluation/check_domain.py

Motivation: the Tamil subset reads like a consumer-tech and news comment
section rather than content about equality, diversity and inclusion. If that is
so, a Hope label on half of it is not a boundary a model could learn, which
would explain a ceiling that survives every modelling intervention. Tested
three ways, none of which needs the reader to know Tamil or Malayalam.

1. Lexicon prevalence. All three subsets code-mix heavily with English, so an
   English-loanword scan reaches Tamil and Malayalam script as well as
   romanised text. Two lexicons: EDI topics, and consumer-tech / news topics.
   Read the RATIO rather than either absolute figure: absolute prevalence
   depends on how long the word lists are, which is arbitrary.
2. Top tokens. The most frequent Latin-script content words per language,
   computed independently of the lexicons.
3. Script mix, comment length and class balance, as context.

Runs over the full corpora the models were trained on (processed/*.csv,
train + dev + test), and repeats the lexicon figures on the 400-row LLM
validation samples for comparison.
"""
import re, sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT  = Path(__file__).resolve().parents[1]
LANGS = ["english", "tamil", "malayalam"]

CORPUS = {l: [ROOT / f"processed/{l}_{s}_processed.csv" for s in
              ("train", "dev", "test")] for l in LANGS}
SAMPLE = {
    "english":   [ROOT / "data_collection/collected/llm_validation_english.csv"],
    "tamil":     [ROOT / "data_collection/collected/llm_validation.csv"],
    "malayalam": [ROOT / "data_collection/collected/llm_validation_malayalam.csv"],
}

EDI = {
 "gay","gays","lgbt","lgbtq","lesbian","trans","transgender","queer","homosexual",
 "homosexuality","homo","gender","feminism","feminist","equality","equal","caste",
 "dalit","racism","racist","race","disability","disabled","discrimination",
 "discriminate","pride","acceptance","accept","minority","minorities","rights",
 "religion","religious","christian","muslim","hindu","church","bible","quran",
 "atheist","atheism","marriage","married","wedding","couple","relationship",
 "sexuality","orientation","identity","inclusion","inclusive","bisexual",
}
TECH_NEWS = {
 "phone","phones","mobile","app","apps","video","videos","channel","subscribe",
 "subscribers","review","camera","android","samsung","xiaomi","redmi","vivo",
 "oppo","realme","micromax","jio","china","chinese","corona","coronavirus",
 "covid","vaccine","vaccin","covaxin","covacine","tiktok","pubg","freefire",
 "netflix","whatsapp","google","youtube","gaming","game","media","news","modi",
 "india","indian","pakistan","america","upload","content","link","download",
 "update","price","launch","battery","processor",
}
STOP = {
 "the","and","for","you","are","not","but","all","can","has","have","was","with",
 "this","that","they","them","their","from","what","when","which","will","would",
 "there","here","your","just","like","also","only","even","than","then","does",
 "about","very","more","some","any","out","who","how","why","its","his","her",
 "him","she","our","one","two","get","got","yes","man","now","see","say","said",
 "bro","anna","sir","please","pls","thanks","thank","good","nice","super",
}

WORD   = re.compile(r"[a-z]{3,}")
LATIN  = re.compile(r"[A-Za-z]")
NATIVE = re.compile(r"[஀-௿ഀ-ൿ]")   # Tamil + Malayalam blocks

TEXTCOLS  = ["text", "comment", "Comment", "sentence", "text_clean", "clean_text"]
LABELCOLS = ["label", "Label", "category", "class"]


def pick(d, names):
    for c in names:
        if c in d.columns:
            return c
    return None


def load(paths):
    frames, used = [], []
    for p in paths:
        if not p.exists():
            continue
        d = pd.read_csv(p)
        tc = pick(d, TEXTCOLS) or next(
            (c for c in d.columns if d[c].dtype == object), None)
        if tc is None:
            continue
        lc = pick(d, LABELCOLS)
        frames.append(pd.DataFrame({"text": d[tc].astype(str),
                                    "label": d[lc] if lc else pd.NA}))
        used.append(p.name)
    if not frames:
        return None, []
    return pd.concat(frames, ignore_index=True), used


def profile(texts):
    edi = tech = nat = rom = 0
    toks, lens = Counter(), []
    for t in texts:
        low = t.lower()
        w = set(WORD.findall(low))
        edi  += bool(w & EDI)
        tech += bool(w & TECH_NEWS)
        toks.update(x for x in WORD.findall(low) if x not in STOP)
        nc, lc = len(NATIVE.findall(t)), len(LATIN.findall(t))
        if nc + lc:
            nat += nc / (nc + lc) > 0.5
            rom += nc / (nc + lc) <= 0.5
        lens.append(len(t))
    n = len(texts)
    return dict(n=n, edi=edi, tech=tech, toks=toks, native=nat, romanized=rom,
                medlen=int(pd.Series(lens).median()) if lens else 0)


def table(title, store):
    print(f"\n{title}\n")
    print(f"  {'language':<11} {'n':>7} {'EDI topics':>12} {'tech / news':>13} "
          f"{'ratio':>8}")
    print("  " + "-" * 56)
    for l in LANGS:
        r = store.get(l)
        if not r:
            print(f"  {l:<11}   -- files not found --")
            continue
        ratio = (r["edi"] / r["tech"]) if r["tech"] else float("inf")
        print(f"  {l:<11} {r['n']:>7} {100*r['edi']/r['n']:>11.1f}% "
              f"{100*r['tech']/r['n']:>12.1f}% {ratio:>8.2f}")


corpus, sample, balance, sources = {}, {}, {}, {}
for l in LANGS:
    d, used = load(CORPUS[l])
    if d is not None:
        corpus[l] = profile(d["text"])
        sources[l] = used
        if d["label"].notna().any():
            vc = d["label"].value_counts(normalize=True).sort_index()
            balance[l] = ", ".join(f"{k}={100*v:.1f}%" for k, v in vc.items())
    s, _ = load(SAMPLE[l])
    if s is not None:
        sample[l] = profile(s["text"])

if not corpus:
    sys.exit("no processed/ corpora found -- check the paths in CORPUS")

print("Files read")
for l in LANGS:
    print(f"  {l:<11} {', '.join(sources.get(l, ['--']))}")

table("FULL CORPUS (train + dev + test) -- lexicon prevalence", corpus)
table("400-row LLM validation sample -- same measure, for comparison", sample)

print("\nScript, length and class balance (full corpus)\n")
print(f"  {'language':<11} {'native':>8} {'romanized':>11} {'med chars':>11}   "
      f"class balance")
print("  " + "-" * 72)
for l in LANGS:
    r = corpus.get(l)
    if not r:
        continue
    print(f"  {l:<11} {r['native']:>8} {r['romanized']:>11} {r['medlen']:>11}   "
          f"{balance.get(l, '--')}")

print("\nTop Latin-script content words, full corpus "
      "(stopwords and generic praise removed)\n")
for l in LANGS:
    r = corpus.get(l)
    if not r:
        continue
    print(f"  {l}:\n    " + ", ".join(w for w, _ in r['toks'].most_common(25)) + "\n")
