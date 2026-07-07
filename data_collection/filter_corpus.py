"""
Filter and clean the scraped YouTube comment corpus for DAPT.

Reads data_collection/collected/tamil_comments_raw.csv (from
fetch_comments.py) and produces a cleaned, deduplicated, Tamil-only corpus:

    data_collection/collected/tamil_corpus_clean.csv   (columns: text, kind)

where kind is:
    - "native"     : contains Tamil-script characters (>= 30% of letters)
    - "romanized"  : Latin-script text matching a romanized-Tamil lexicon
    - "mixed"      : some Tamil script, below the native threshold

Everything else (pure English, other languages, spam, too short/long) is
dropped. Prints a breakdown so you can judge the yield.

Cleaning matches the supervised pipeline (preprocess_all.clean_text):
URLs and @mentions removed, '#tag' -> 'tag', whitespace collapsed;
punctuation/emoji/case preserved.

Usage (from anywhere):
    $ python data_collection/filter_corpus.py
    $ python data_collection/filter_corpus.py --min-tokens 4
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "collected" / "tamil_comments_raw.csv"
OUT_PATH = HERE / "collected" / "tamil_corpus_clean.csv"

# --- cleaning (mirrors preprocess/preprocess_all.py) ---
URL_RE = re.compile(r"(?:https?://|www\.)\S+")
MENTION_RE = re.compile(r"@\w+")
HASH_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")   # video timestamps

TAMIL_CHAR_RE = re.compile(r"[\u0B80-\u0BFF]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
TAMIL_SUFFIX_RE = re.compile(
    r"(nga|ngal|inga|unga|uchu|ichu|udhu|athu|adhu|kittu|itten|iruken)$")

# Frequent romanized-Tamil tokens (function words + very common words that are
# unambiguous vs English). Two or more hits => romanized Tamil.
ROMANIZED_LEXICON = {
    "nambikkai", "nambikkaiya", "vazhkai", "valkai", "vazhkaila",
    "nanba", "nanbargal", "thala", "thalaiva", "anna", "anne", "akka",
    "semma", "vera", "level", "mass", "padam", "pathi", "romba", "rombha",
    "iruku", "irukku", "irukku", "illa", "illai", "venum", "vendam",
    "panna", "pannunga", "seri", "sari", "aana", "ana", "epdi", "eppadi",
    "enna", "yenna", "unga", "ungal", "namma", "nammal", "avanga",
    "intha", "antha", "ithu", "athu", "ella", "ellam", "elarum",
    "vantha", "varum", "poga", "pona", "solli", "sollunga", "kelunga",
    "paaru", "paarunga", "paatha", "kandippa", "kandipa", "mudiyum",
    "mudiyathu", "mudiyaathu", "vetri", "velvom", "jeipom", "jeippom",
    "da", "dei", "bro", "machan", "machi", "sir", "ji", "saar",
    "super", "superb", "nalla", "nallavan", "azhaga", "alaga",
    "kashtam", "kastam", "sandhosham", "santhosham", "magizhchi",
    # round 2: mined from the dropped-'other' sample
    "edhu", "ethu", "adhu", "idhu", "theriyum", "therihum", "theriyadha",
    "therila", "therilla", "teriyum", "pattha", "paakkala", "pakkala",
    "udane", "udanea", "avalo", "evalo", "evlo", "ivlo", "ippo", "ipo",
    "apram", "appuram", "apdi", "ipdi", "appadi", "ippadi", "epdi",
    "yaen", "yen", "yenda", "engada", "thozha", "machaan", "ayyo",
    "aiyo", "aiyoo", "yappa", "podu", "podunga", "pannu", "panalum",
    "pannalam", "poren", "pora", "povaen", "pova", "polam", "polama",
    "mudila", "mudiyala", "mudingada", "ayiduchu", "aachu", "aayiduchu",
    "varaikum", "vanthu", "vanga", "vaanga", "vanka", "pakkam",
    "mattum", "matum", "innum", "avan", "aval", "avar", "ivan",
    "veetla", "veedu", "irunthu", "irundhu", "iruka", "irukinga",
    "irukke", "irruku", "irukum", "irukkum", "makkal", "makkalukku",
    "varusham", "varushame", "manikku", "kelvi", "sami", "saami",
    "mamiyar", "purushan", "pondatti", "thangachi", "thambi",
    "samathuva", "arasiyal", "padikanum", "padichen", "padicha",
    "sollanum", "solranga", "soldringa", "sonna", "sonnanga",
    "kudukalam", "kuduthanga", "vangi", "sapdunga", "sapduga",
    "saptiya", "thoonga", "thookam", "yosanai", "nenachu", "nenaikiren",
}
# "da","bro","sir","super" alone are weak -> require >= 2 distinct hits
WEAK_TOKENS = {"da", "bro", "sir", "ji", "super", "superb", "mass", "level"}

ENGLISH_STOPWORDS = {
    "the", "is", "are", "was", "were", "and", "or", "of", "to", "in",
    "for", "with", "this", "that", "you", "your", "it", "its", "very",
    "have", "has", "will", "would", "can", "could", "should", "a", "an",
}


def clean_text(text: str) -> str:
    text = str(text)
    text = URL_RE.sub(" ", text)
    text = HASH_RE.sub(r"\1", text)
    text = MENTION_RE.sub(" ", text)
    text = TIMESTAMP_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


def classify(text: str) -> str:
    """Return 'native' | 'mixed' | 'romanized' | 'english' | 'other'."""
    letters = LETTER_RE.findall(text)
    if not letters:
        return "other"
    tamil_ratio = len(TAMIL_CHAR_RE.findall(text)) / len(letters)
    if tamil_ratio >= 0.30:
        return "native"
    if tamil_ratio > 0:
        return "mixed"

    tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]
    if not tokens:
        return "other"
    hits = {t for t in tokens if t in ROMANIZED_LEXICON}
    # morphology booster: word endings that are distinctly Tamil
    # (verb/plural/politeness suffixes rare in English/Hindi)
    pattern_hits = {t for t in tokens if len(t) >= 5 and
                    TAMIL_SUFFIX_RE.search(t)}
    strong_hits = (hits - WEAK_TOKENS) | pattern_hits
    if len(hits | pattern_hits) >= 2 and strong_hits:
        return "romanized"

    stop_ratio = sum(t in ENGLISH_STOPWORDS for t in tokens) / len(tokens)
    if stop_ratio >= 0.15:
        return "english"
    return "other"


def main():
    parser = argparse.ArgumentParser(description="Filter scraped corpus")
    parser.add_argument("--min-tokens", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=1000)
    args = parser.parse_args()

    if not RAW_PATH.exists():
        sys.exit(f"{RAW_PATH} not found — run fetch_comments.py first")

    with RAW_PATH.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Raw comments:        {len(rows)}")

    seen, kept, counts = set(), [], Counter()
    dropped_other = []
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
        counts[kind] += 1
        if kind in ("native", "mixed", "romanized"):
            kept.append({"text": text, "kind": kind})
        elif kind == "other":
            dropped_other.append(text)

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "kind"])
        w.writeheader()
        w.writerows(kept)

    print(f"After length/dup:    {sum(counts[k] for k in ('native','mixed','romanized','english','other'))}"
          f"  (short: {counts['dropped_short']}, dup: {counts['dropped_dup']},"
          f" len: {counts['dropped_len']})")
    print(f"  native Tamil:      {counts['native']}")
    print(f"  mixed script:      {counts['mixed']}")
    print(f"  romanized Tamil:   {counts['romanized']}")
    print(f"  english (dropped): {counts['english']}")
    print(f"  other (dropped):   {counts['other']}")
    print(f"\nKept for corpus:     {len(kept)}")
    print(f"Saved -> {OUT_PATH}")

    # sample of the 'other' bucket for manual inspection — if these look like
    # romanized Tamil, the lexicon needs more words
    if dropped_other:
        import random
        random.seed(0)
        sample = random.sample(dropped_other, min(60, len(dropped_other)))
        sample_path = OUT_PATH.parent / "dropped_other_sample.txt"
        sample_path.write_text("\n".join(sample), encoding="utf-8")
        print(f"Sample of dropped 'other' rows -> {sample_path}")


if __name__ == "__main__":
    main()
