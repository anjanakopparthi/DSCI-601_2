"""
Filter and clean a scraped YouTube comment corpus (multi-language).

Reads data_collection/collected/{lang}_comments_raw.csv (from
fetch_comments.py) and produces a cleaned, deduplicated corpus for that
language:

    data_collection/collected/{lang}_corpus_clean.csv   (columns: text, kind)

Per-language keep rules:
    english   : Latin-script text that looks like English (stopword evidence);
                other languages and script-mixed junk dropped. kind="english"
    tamil     : native Tamil script / mixed / romanized Tamil (lexicon +
                Tamil suffix morphology). kinds: native|mixed|romanized
    malayalam : native Malayalam script / mixed / romanized Malayalam
                ("Manglish", lexicon + suffix morphology). same kinds

Cleaning matches the supervised pipeline: URLs and @mentions removed,
'#tag' -> 'tag', video timestamps removed, whitespace collapsed;
punctuation/emoji/case preserved.

Usage (from anywhere):
    $ python data_collection/filter_corpus.py --lang english
    $ python data_collection/filter_corpus.py --lang tamil
    $ python data_collection/filter_corpus.py --lang malayalam
"""

import argparse
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- cleaning (mirrors preprocess/preprocess_all.py) ---
URL_RE = re.compile(r"(?:https?://|www\.)\S+")
MENTION_RE = re.compile(r"@\w+")
HASH_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
LATIN_RE = re.compile(r"[A-Za-z]")
TAMIL_CHAR_RE = re.compile(r"[\u0B80-\u0BFF]")
MALAYALAM_CHAR_RE = re.compile(r"[\u0D00-\u0D7F]")

ENGLISH_STOPWORDS = {
    "the", "is", "are", "was", "were", "and", "or", "of", "to", "in", "on",
    "for", "with", "this", "that", "you", "your", "it", "its", "very", "so",
    "have", "has", "will", "would", "can", "could", "should", "a", "an", "i",
    "im", "me", "my", "we", "our", "he", "she", "they", "his", "her", "be",
    "been", "do", "does", "did", "not", "no", "but", "if", "as", "at", "by",
    "from", "up", "out", "all", "just", "what", "when", "how", "who", "why",
    "there", "here", "one", "get", "got", "like", "love", "never", "give",
    "keep", "going", "thank", "thanks", "god", "people", "life", "time",
}

TAMIL_ROMANIZED_LEXICON = {
    "nambikkai", "nambikkaiya", "vazhkai", "valkai", "vazhkaila", "nanba",
    "nanbargal", "thala", "thalaiva", "anna", "anne", "akka", "semma",
    "vera", "level", "mass", "padam", "pathi", "romba", "rombha", "iruku",
    "irukku", "illa", "illai", "venum", "vendam", "panna", "pannunga",
    "seri", "sari", "aana", "ana", "epdi", "eppadi", "enna", "yenna",
    "unga", "ungal", "namma", "nammal", "avanga", "intha", "antha", "ithu",
    "athu", "ella", "ellam", "elarum", "vantha", "varum", "poga", "pona",
    "solli", "sollunga", "kelunga", "paaru", "paarunga", "paatha",
    "kandippa", "kandipa", "mudiyum", "mudiyathu", "mudiyaathu", "vetri",
    "velvom", "jeipom", "jeippom", "da", "dei", "bro", "machan", "machi",
    "sir", "ji", "saar", "super", "superb", "nalla", "nallavan", "azhaga",
    "alaga", "kashtam", "kastam", "sandhosham", "santhosham", "magizhchi",
    "edhu", "ethu", "adhu", "idhu", "theriyum", "therihum", "theriyadha",
    "therila", "therilla", "teriyum", "pattha", "paakkala", "pakkala",
    "udane", "udanea", "avalo", "evalo", "evlo", "ivlo", "ippo", "ipo",
    "apram", "appuram", "apdi", "ipdi", "appadi", "ippadi", "yaen", "yen",
    "yenda", "engada", "thozha", "machaan", "ayyo", "aiyo", "aiyoo",
    "yappa", "podu", "podunga", "pannu", "panalum", "pannalam", "poren",
    "pora", "povaen", "pova", "polam", "polama", "mudila", "mudiyala",
    "mudingada", "ayiduchu", "aachu", "aayiduchu", "varaikum", "vanthu",
    "vanga", "vaanga", "vanka", "pakkam", "mattum", "matum", "innum",
    "avan", "aval", "avar", "ivan", "veetla", "veedu", "irunthu",
    "irundhu", "iruka", "irukinga", "irukke", "irruku", "irukum",
    "irukkum", "makkal", "makkalukku", "varusham", "varushame", "manikku",
    "kelvi", "sami", "saami", "mamiyar", "purushan", "pondatti",
    "thangachi", "thambi", "samathuva", "arasiyal", "padikanum",
    "padichen", "padicha", "sollanum", "solranga", "soldringa", "sonna",
    "sonnanga", "kudukalam", "kuduthanga", "vangi", "sapdunga", "sapduga",
    "saptiya", "thoonga", "thookam", "yosanai", "nenachu", "nenaikiren",
}
TAMIL_SUFFIX_RE = re.compile(
    r"(nga|ngal|inga|unga|uchu|ichu|udhu|athu|adhu|kittu|itten|iruken)$")
TAMIL_WEAK = {"da", "dei", "bro", "sir", "ji", "super", "superb", "mass",
              "level"}

# Romanized Malayalam ("Manglish") — common function words + frequent tokens
MALAYALAM_ROMANIZED_LEXICON = {
    "njan", "njn", "ente", "ningal", "ningalude", "nammal", "namuk",
    "namukku", "avan", "aval", "avar", "avante", "avalude", "athe",
    "athu", "ithu", "ith", "alle", "alla", "illa", "illatha", "undu",
    "und", "undo", "aanu", "anu", "aan", "aano", "ano", "cheyyuka",
    "cheyyum", "cheythu", "cheyyu", "nannayi", "nalla", "nallathu",
    "kollam", "kollaam", "adipoli", "pwoli", "poli", "kidu", "kidukki",
    "chetta", "chettan", "chechi", "chechy", "machane", "machan", "mone",
    "mole", "eda", "edi", "enthu", "entha", "enthina", "engane",
    "enganeya", "evide", "evideya", "ippo", "ippol", "appo", "appol",
    "pinne", "onnum", "ellam", "ellarum", "oru", "koode", "koodi",
    "venam", "venda", "vannu", "varum", "varu", "pokum", "poyi", "poku",
    "nokku", "nokkam", "nokki", "para", "parayu", "paranju", "ariyilla",
    "ariyam", "ariyaam", "sughamano", "sukhamano", "santhosham",
    "sneham", "prateeksha", "pratheeksha", "vishwasam", "jeevitham",
    "jeevithathil", "padikkuka", "padichu", "padikkanam", "pani",
    "sherikkum", "sheriyanu", "sheri", "shariyanu", "onnu", "randu",
    "kure", "orupad", "orupadu", "valare", "vare", "mathi", "mathram",
    "polum", "thanne", "thanneya", "kandu", "kanam", "kelkkuka", "kettu",
    "ishtam", "ishtamanu", "istam",
    # round 2: mined from the dropped-sample audit
    "kurich", "kurichu", "cheyyo", "cheyuna", "cheythal", "manassil",
    "thattiya", "vaangum", "vangum", "vaangi", "kittum", "kitti",
    "kittunnu", "kittiyilla", "aayi", "aayirunnu", "enik", "enikku",
    "eniku", "ithilum", "athilum", "engane", "enganya", "ariyam",
    "ariyathondu", "padikunnathu", "padikkum", "padikan", "padikkan",
    "kazhikkan", "kazhichu", "parnju", "paranjathu", "thanna", "tharu",
    "tharumo", "kidumo", "kollilla", "kollavunna", "maatramanu",
    "mathramanu", "avatharana", "ottakaran", "pakshe", "bhayangara",
    "aaradhana", "prayasam", "vilichille", "odiyilla", "najn",
}
MALAYALAM_WEAK = {"mass", "poli", "super", "bro", "sir", "kidu"}
# unambiguous even as a single hit (can't be English/Tamil/Hindi)
MALAYALAM_DISTINCT = {
    "njan", "njn", "najn", "ente", "aanu", "alle", "kollam", "adipoli",
    "pwoli", "chetta", "chettan", "chechi", "chechy", "machane", "mone",
    "enthina", "evide", "ippol", "appol", "venda", "ariyilla", "paranju",
    "sherikkum", "enikku", "manassil", "njangal", "nammude", "ningalude",
}
MALAYALAM_SUFFIX_RE = re.compile(
    r"(unnu|unna|aanu|anu|illa|alla|alle|ille|ittund|ittu|ukayanu|"
    r"aayirunnu|ikkum|aayi|umo|unnathu)$")


def clean_text(text: str) -> str:
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = HASH_RE.sub(r"\1", text)
    text = MENTION_RE.sub(" ", text)
    text = TIMESTAMP_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


# ============================================================
# Per-language classifiers -> kind or None (= drop)
# ============================================================

def _romanized(tokens, lexicon, weak, suffix_re):
    hits = {t for t in tokens if t in lexicon}
    pattern_hits = {t for t in tokens if len(t) >= 5 and suffix_re.search(t)}
    strong = (hits - weak) | pattern_hits
    return len(hits | pattern_hits) >= 2 and bool(strong)


def classify_dravidian(text, script_re, lexicon, weak, suffix_re):
    letters = LETTER_RE.findall(text)
    if not letters:
        return None
    ratio = len(script_re.findall(text)) / len(letters)
    if ratio >= 0.30:
        return "native"
    if ratio > 0:
        return "mixed"
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]
    if tokens and _romanized(tokens, lexicon, weak, suffix_re):
        return "romanized"
    return None


def classify_english(text):
    letters = LETTER_RE.findall(text)
    if not letters:
        return None
    latin_ratio = len(LATIN_RE.findall(text)) / len(letters)
    if latin_ratio < 0.90:              # substantial non-Latin script
        return None
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]
    if not tokens:
        return None
    # not English if it reads as romanized Tamil/Malayalam
    if _romanized(tokens, TAMIL_ROMANIZED_LEXICON, TAMIL_WEAK,
                  TAMIL_SUFFIX_RE):
        return None
    if _romanized(tokens, MALAYALAM_ROMANIZED_LEXICON, MALAYALAM_WEAK,
                  MALAYALAM_SUFFIX_RE):
        return None
    stop_ratio = sum(t in ENGLISH_STOPWORDS for t in tokens) / len(tokens)
    if stop_ratio >= 0.15 or (len(tokens) <= 6 and stop_ratio > 0):
        return "english"
    return None


def classify_malayalam(text):
    kind = classify_dravidian(text, MALAYALAM_CHAR_RE,
                              MALAYALAM_ROMANIZED_LEXICON, MALAYALAM_WEAK,
                              MALAYALAM_SUFFIX_RE)
    if kind is not None:
        return kind
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]
    if any(t in MALAYALAM_DISTINCT for t in tokens):
        return "romanized"
    return None


CLASSIFIERS = {
    "english": classify_english,
    "tamil": lambda t: classify_dravidian(
        t, TAMIL_CHAR_RE, TAMIL_ROMANIZED_LEXICON, TAMIL_WEAK,
        TAMIL_SUFFIX_RE),
    "malayalam": lambda t: classify_malayalam(t),
}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Filter scraped corpus")
    parser.add_argument("--lang", default="tamil",
                        choices=list(CLASSIFIERS.keys()))
    parser.add_argument("--min-tokens", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=1000)
    args = parser.parse_args()

    raw_path = HERE / "collected" / f"{args.lang}_comments_raw.csv"
    out_path = HERE / "collected" / f"{args.lang}_corpus_clean.csv"
    if not raw_path.exists():
        sys.exit(f"{raw_path} not found — run fetch_comments.py --lang "
                 f"{args.lang} first")

    with raw_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[{args.lang}] raw comments:  {len(rows)}")

    classify = CLASSIFIERS[args.lang]
    seen, kept, counts, dropped = set(), [], Counter(), []
    for row in rows:
        text = clean_text(row["text"])
        if not text or len(text) > args.max_chars:
            counts["dropped_len"] += 1
            continue
        if len(text.split()) < args.min_tokens:
            counts["dropped_short"] += 1
            continue
        key = text.lower()
        if key in seen:
            counts["dropped_dup"] += 1
            continue
        seen.add(key)
        kind = classify(text)
        if kind is None:
            counts["dropped_other"] += 1
            dropped.append(text)
        else:
            counts[kind] += 1
            kept.append({"text": text, "kind": kind})

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "kind"])
        w.writeheader()
        w.writerows(kept)

    print(f"  dropped: short {counts['dropped_short']}, "
          f"dup {counts['dropped_dup']}, len {counts['dropped_len']}, "
          f"other-language/junk {counts['dropped_other']}")
    for kind in ("english", "native", "mixed", "romanized"):
        if counts[kind]:
            print(f"  kept {kind}: {counts[kind]}")
    print(f"  TOTAL KEPT: {len(kept)}")
    print(f"  saved -> {out_path}")

    if dropped:
        random.seed(0)
        sample = random.sample(dropped, min(60, len(dropped)))
        sample_path = out_path.parent / f"dropped_sample_{args.lang}.txt"
        sample_path.write_text("\n".join(sample), encoding="utf-8")
        print(f"  sample of dropped rows -> {sample_path}")


if __name__ == "__main__":
    main()
