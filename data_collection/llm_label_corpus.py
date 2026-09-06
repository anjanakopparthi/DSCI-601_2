"""
Batch-label the scraped corpora with the validated LLM labeler.

Uses the SAME setup that was validated against gold labels
(gemini-3.1-flash-lite + 30 gold few-shot examples; measured agreement:
English 0.782, Malayalam 0.855): labels every comment in
{lang}_corpus_clean.csv as HOPE (1) / NON_HOPE (0).

Fully RESUMABLE: labeled rows are appended to the output CSV incrementally
and progress is tracked in a state file, so quota exhaustion, rate limits,
or Ctrl+C lose nothing — rerun the same command to continue. On what looks
like a daily-quota wall (repeated 429s), it saves and exits cleanly.

Outputs:
    data_collection/collected/{lang}_corpus_labeled.csv   (text, kind, label)
    data_collection/collected/label_state_{lang}.json

Usage (from anywhere):
    $ python data_collection/llm_label_corpus.py --lang malayalam
    $ python data_collection/llm_label_corpus.py --lang tamil
    $ python data_collection/llm_label_corpus.py --lang english
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

BATCH_SIZE = 20
SLEEP_BETWEEN = 4.0          # seconds between requests
FEW_SHOT_K = 30              # same construction as llm_label_validate.py
MAX_CONSECUTIVE_429 = 5      # then assume daily quota -> save & exit

SYSTEM_PROMPT_TEMPLATE = """You are labeling YouTube comments for HOPE SPEECH \
detection, following the HopeEDI (LT-EDI) annotation guidelines.

Label a comment HOPE if it expresses or promotes positivity in a broad sense: \
support, reassurance, encouragement, inspiration, optimism about the future, \
resilience, gratitude, well-wishes, appreciation of someone's effort or \
achievement, or promotion of equality and inclusion. In this dataset, \
annotators labeled generously: warm, positive, appreciative, or uplifting \
comments generally count as HOPE even if brief.

Label a comment NON_HOPE if it is neutral/informational, a plain question, \
sarcastic or joking without warmth, critical, complaining, hostile, hopeless, \
or otherwise not positive in tone.

The comments are in {language}, possibly romanized (written with Latin \
letters) or code-mixed with English. Judge the meaning, not the script.

You will receive numbered comments. Respond with ONLY a JSON object of the form
{{"labels": [{{"id": 1, "label": "HOPE"}}, {{"id": 2, "label": "NON_HOPE"}}, ...]}}
with exactly one entry per comment, no other text."""


# ============================================================
# 1. Plumbing (matches the validated setup)
# ============================================================

def load_api_key() -> str:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        sys.exit(f".env not found at {env_path}")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key:
                return key
    sys.exit("GEMINI_API_KEY not set in .env")


def build_system_prompt(lang: str) -> str:
    """Few-shot prompt built exactly like the validated one (seed 7)."""
    prompt = SYSTEM_PROMPT_TEMPLATE.format(language=lang.capitalize())
    train = pd.read_csv(BASE_DIR / "processed" / f"{lang}_train_processed.csv")
    train["text"] = train["text"].fillna("").astype(str)
    train = train[train["text"].str.strip() != ""]
    k = FEW_SHOT_K // 2
    shots = pd.concat([
        train[train["label"] == 1].sample(k, random_state=7),
        train[train["label"] == 0].sample(k, random_state=7),
    ]).sample(frac=1.0, random_state=7)
    example_lines = "\n".join(
        f'- "{t[:200]}" -> {"HOPE" if l == 1 else "NON_HOPE"}'
        for t, l in zip(shots["text"], shots["label"]))
    prompt += ("\n\nHere are examples labeled by the dataset's own "
               "annotators. Match THEIR conventions, even where they differ "
               "from your intuition:\n" + example_lines)
    return prompt


def build_prompt(texts: list[str]) -> str:
    lines = [f"{i + 1}. {t}" for i, t in enumerate(texts)]
    return "Label these comments:\n\n" + "\n".join(lines)


def parse_labels(reply: str, n_expected: int) -> list[str] | None:
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        items = {int(x["id"]): str(x["label"]).strip().upper()
                 for x in data["labels"]}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    labels = []
    for i in range(1, n_expected + 1):
        lab = items.get(i, "")
        if lab not in ("HOPE", "NON_HOPE"):
            return None
        labels.append(lab)
    return labels


class DailyQuotaExhausted(Exception):
    pass


def label_batch(texts, api_key, model, system_prompt,
                consecutive_429, max_retries=4):
    gen_cfg = {"temperature": 0, "maxOutputTokens": 4000,
               "responseMimeType": "application/json"}
    if "2.5" in model:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": build_prompt(texts)}]}],
        "generationConfig": gen_cfg,
    }
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(f"{API_BASE}/{model}:generateContent",
                              params={"key": api_key}, json=body, timeout=120)
        except requests.RequestException as e:
            print(f"  network error: {e} (attempt {attempt + 1})")
            time.sleep(10)
            continue
        if r.status_code == 404:
            sys.exit("Model not available: " + r.text[:300])
        if r.status_code == 429:
            consecutive_429[0] += 1
            if consecutive_429[0] >= MAX_CONSECUTIVE_429:
                raise DailyQuotaExhausted()
            wait = int(r.headers.get("retry-after", 30))
            print(f"  rate limited, waiting {wait}s "
                  f"({consecutive_429[0]}/{MAX_CONSECUTIVE_429})")
            time.sleep(wait)
            continue
        if r.status_code == 503:
            print(f"  503 overloaded, waiting 30s (attempt {attempt + 1})")
            time.sleep(30)
            continue
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}: {r.text[:200]}")
            time.sleep(10)
            continue
        consecutive_429[0] = 0
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            print(f"  unexpected response shape (attempt {attempt + 1})")
            time.sleep(5)
            continue
        reply = "".join(pt.get("text", "") for pt in parts)
        labels = parse_labels(reply, len(texts))
        if labels is not None:
            return labels
        print(f"  malformed reply, retrying (attempt {attempt + 1})")
    return None


# ============================================================
# 2. Main (resumable)
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Batch-label a corpus")
    parser.add_argument("--lang", required=True,
                        choices=["english", "tamil", "malayalam"])
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    args = parser.parse_args()

    corpus_path = HERE / "collected" / f"{args.lang}_corpus_clean.csv"
    out_path = HERE / "collected" / f"{args.lang}_corpus_labeled.csv"
    state_path = HERE / "collected" / f"label_state_{args.lang}.json"
    if not corpus_path.exists():
        sys.exit(f"{corpus_path} not found — run filter_corpus.py first")

    api_key = load_api_key()
    system_prompt = build_system_prompt(args.lang)

    corpus = pd.read_csv(corpus_path)
    corpus["text"] = corpus["text"].astype(str)
    n_total = len(corpus)

    start = 0
    if state_path.exists():
        start = json.loads(state_path.read_text())["next_index"]
    print(f"[{args.lang}] {n_total} comments, resuming at index {start}")
    est_req = (n_total - start + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  ~{est_req} requests remaining "
          f"(~{est_req * SLEEP_BETWEEN / 60:.0f}+ minutes)")

    consecutive_429 = [0]
    new_file = not out_path.exists()
    labeled_this_run = skipped = 0

    try:
        with out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "kind", "label"])
            if new_file:
                writer.writeheader()
            for i in range(start, n_total, BATCH_SIZE):
                chunk = corpus.iloc[i:i + BATCH_SIZE]
                labels = label_batch(chunk["text"].tolist(), api_key,
                                     args.model, system_prompt,
                                     consecutive_429)
                if labels is None:
                    print(f"  batch at {i}: failed permanently, "
                          f"writing label=-1 (filter later)")
                    labels = ["SKIP"] * len(chunk)
                    skipped += len(chunk)
                for (_, row), lab in zip(chunk.iterrows(), labels):
                    writer.writerow({
                        "text": row["text"], "kind": row["kind"],
                        "label": {"HOPE": 1, "NON_HOPE": 0}.get(lab, -1)})
                f.flush()
                labeled_this_run += len(chunk)
                state_path.write_text(json.dumps(
                    {"next_index": i + len(chunk)}))
                done = i + len(chunk)
                if (done // BATCH_SIZE) % 10 == 0 or done >= n_total:
                    print(f"  labeled {done}/{n_total}")
                time.sleep(SLEEP_BETWEEN)
    except DailyQuotaExhausted:
        print("\nDaily quota appears exhausted — progress saved.")
        print("Rerun the same command after the quota resets "
              "(midnight Pacific) to continue.")
    except KeyboardInterrupt:
        print("\nInterrupted — progress saved. Rerun to continue.")

    print(f"\nThis run: {labeled_this_run} labeled ({skipped} skipped)")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("next_index", 0) >= n_total:
        df = pd.read_csv(out_path)
        ok = df[df["label"] != -1]
        print("=" * 50)
        print(f"COMPLETE: {len(ok)}/{len(df)} labeled")
        print(f"Label distribution: "
              f"hope {int((ok['label'] == 1).sum())} "
              f"({(ok['label'] == 1).mean():.1%}) / "
              f"non-hope {int((ok['label'] == 0).sum())}")
        print(f"Saved -> {out_path}")
    else:
        print(f"Progress: {state.get('next_index', 0)}/{n_total} "
              f"— rerun to continue.")


if __name__ == "__main__":
    main()
