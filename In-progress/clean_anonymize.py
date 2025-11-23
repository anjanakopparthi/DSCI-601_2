# clean_anonymize.py
import re
import pandas as pd
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # makes language detection stable

# Regex patterns to remove PII
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+")
MENTION_RE = re.compile(r"@\w+")
HASH_RE = re.compile(r"#\w+")
NON_PRINTABLE_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")

def anonymize(text):
    if not isinstance(text, str): return ""
    t = text
    t = URL_RE.sub(" [URL] ", t)
    t = EMAIL_RE.sub(" [EMAIL] ", t)
    t = MENTION_RE.sub(" [USER] ", t)
    t = HASH_RE.sub(" ", t)
    t = NON_PRINTABLE_RE.sub(" ", t)
    t = re.sub(r"\b\d{4,}\b", " [NUM] ", t)  # remove long numbers
    return " ".join(t.split())

def detect_lang(text):
    try:
        return detect(text)
    except:
        return "unknown"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", default="comments_clean.csv")
    args = parser.parse_args()
    
    df = pd.read_csv(args.infile)
    df["text_clean"] = df["text"].fillna("").apply(anonymize)
    df["lang"] = df["text_clean"].apply(detect_lang)
    
    # Remove PII columns
    if "authorDisplayName" in df.columns:
        df = df.drop(columns=["authorDisplayName"])
    
    df.to_csv(args.outfile, index=False)
    print(f"Wrote cleaned comments to {args.outfile}")
