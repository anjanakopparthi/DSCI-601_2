"""
Validate LLM labeling quality BEFORE using it for data augmentation.

Samples gold-labeled comments from the HopeEDI Tamil train split, asks Claude
to label them (without seeing the gold labels), and measures agreement.

Decision rule (rough guide):
    - agreement >= ~0.85          -> LLM labels are useful training signal
    - agreement ~0.75-0.85        -> usable with a confidence filter, marginal
    - agreement < ~0.75           -> too noisy; skip the augmentation experiment

Setup (free — no billing account needed):
    1. Go to https://aistudio.google.com -> Get API key -> Create API key
    2. Add to the repo-root .env:  GEMINI_API_KEY=AIza...

Usage (from anywhere):
    $ python data_collection/llm_label_validate.py
    $ python data_collection/llm_label_validate.py --n 400 --model gemini-2.5-pro
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
# paths are chosen per --lang inside main()

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
BATCH_SIZE = 20

SYSTEM_PROMPT = """You are labeling YouTube comments for HOPE SPEECH detection, \
following the HopeEDI (LT-EDI) annotation guidelines.

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
{"labels": [{"id": 1, "label": "HOPE"}, {"id": 2, "label": "NON_HOPE"}, ...]}
with exactly one entry per comment, no other text."""


# ============================================================
# 1. Plumbing
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
    sys.exit("GEMINI_API_KEY not set in .env "
             "(add a line: GEMINI_API_KEY=AIza... — free key from aistudio.google.com)")


def build_prompt(texts: list[str]) -> str:
    lines = [f"{i + 1}. {t}" for i, t in enumerate(texts)]
    return "Label these comments:\n\n" + "\n".join(lines)


def parse_labels(reply: str, n_expected: int) -> list[str] | None:
    """Extract {"labels":[...]} from the model reply; None if malformed."""
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


def label_batch(texts: list[str], api_key: str, model: str,
                max_retries: int = 4) -> list[str] | None:
    """One API call labeling up to BATCH_SIZE comments; retries on failures."""
    gen_cfg = {"temperature": 0, "maxOutputTokens": 4000,
               "responseMimeType": "application/json"}
    if "2.5" in model:                      # thinkingConfig: 2.5-only param
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": build_prompt(texts)}]}],
        "generationConfig": gen_cfg,
    }
    url = f"{API_BASE}/{model}:generateContent"
    headers = {"content-type": "application/json"}
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, params={"key": api_key},
                              headers=headers, json=body, timeout=120)
        except requests.RequestException as e:
            print(f"  network error: {e} (attempt {attempt + 1})")
            time.sleep(5)
            continue
        if r.status_code == 429:            # rate limited — back off
            wait = int(r.headers.get("retry-after", 15))
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 404:      # model doesn't exist for this key
            sys.exit(f"Model not available: {r.text[:300]}\n"
                     f"Run with --list-models to see what your key can use.")
        if r.status_code == 503:      # model overloaded — wait it out
            print(f"  503 overloaded, waiting 30s (attempt {attempt + 1})")
            time.sleep(30)
            continue
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}: {r.text[:200]} (attempt {attempt + 1})")
            time.sleep(10)
            continue
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
# 2. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Validate LLM labels vs gold")
    parser.add_argument("--n", type=int, default=400,
                        help="sample size (default 400, stratified)")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--lang", default="tamil",
                        choices=["english", "tamil", "malayalam"])
    parser.add_argument("--few-shot", type=int, default=0, metavar="K",
                        help="prepend K gold-labeled examples to the prompt "
                             "(teaches the annotators' conventions)")
    parser.add_argument("--list-models", action="store_true",
                        help="list models available to this API key and exit")
    args = parser.parse_args()

    if args.list_models:
        key = load_api_key()
        r = requests.get(f"{API_BASE}", params={"key": key}, timeout=30)
        r.raise_for_status()
        print("Models available to your key (generateContent-capable):")
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(" -", m["name"].split("/")[-1])
        return

    global SYSTEM_PROMPT
    train_path = BASE_DIR / "processed" / f"{args.lang}_train_processed.csv"
    out_path = (Path(__file__).resolve().parent / "collected"
                / f"llm_validation_{args.lang}.csv")
    SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{language}", args.lang.capitalize())

    api_key = load_api_key()
    df = pd.read_csv(train_path)
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]

    # few-shot examples: separate stratified draw, EXCLUDED from evaluation
    if args.few_shot:
        k = args.few_shot // 2
        shots = pd.concat([
            df[df["label"] == 1].sample(k, random_state=7),
            df[df["label"] == 0].sample(k, random_state=7),
        ]).sample(frac=1.0, random_state=7)
        df = df.drop(shots.index)
        example_lines = "\n".join(
            f'- "{t[:200]}" -> {"HOPE" if l == 1 else "NON_HOPE"}'
            for t, l in zip(shots["text"], shots["label"]))
        SYSTEM_PROMPT += (
            "\n\nHere are examples labeled by the dataset's own annotators. "
            "Match THEIR conventions, even where they differ from your "
            "intuition:\n" + example_lines)
        print(f"Few-shot: {len(shots)} gold examples added to the prompt")

    # stratified sample: half hope, half non-hope
    per_class = args.n // 2
    sample = pd.concat([
        df[df["label"] == 1].sample(per_class, random_state=42),
        df[df["label"] == 0].sample(per_class, random_state=42),
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print("(free tier is rate-limited: ~7 min for 400 comments)")
    print(f"Validating {args.model} on {args.lang}: {len(sample)} gold-labeled comments "
          f"({per_class} per class)")

    llm_labels: list[str | None] = []
    for start in range(0, len(sample), BATCH_SIZE):
        chunk = sample.iloc[start:start + BATCH_SIZE]
        labels = label_batch(chunk["text"].tolist(), api_key, args.model)
        if labels is None:
            print(f"  batch {start // BATCH_SIZE + 1}: FAILED, skipping")
            labels = [None] * len(chunk)
        llm_labels.extend(labels)
        done = min(start + BATCH_SIZE, len(sample))
        print(f"  labeled {done}/{len(sample)}")
        time.sleep(6.5)   # free-tier rate limit (~10 req/min)

    sample["llm_label"] = [{"HOPE": 1, "NON_HOPE": 0}.get(l, -1)
                           if l else -1 for l in llm_labels]
    valid = sample[sample["llm_label"] != -1]
    out_path.parent.mkdir(exist_ok=True)
    sample.to_csv(out_path, index=False)

    # --- agreement report ---
    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 confusion_matrix, f1_score)
    y_gold, y_llm = valid["label"], valid["llm_label"]
    print("\n" + "=" * 55)
    print(f"AGREEMENT REPORT  ({len(valid)}/{len(sample)} usable)")
    print("=" * 55)
    print(f"Agreement (accuracy vs gold): {accuracy_score(y_gold, y_llm):.3f}")
    print(f"Cohen's kappa:                {cohen_kappa_score(y_gold, y_llm):.3f}")
    print(f"Macro F1 vs gold:             {f1_score(y_gold, y_llm, average='macro'):.3f}")
    print(f"Hope-class F1 vs gold:        {f1_score(y_gold, y_llm, pos_label=1):.3f}")
    print("Confusion [[gold0-llm0, gold0-llm1], [gold1-llm0, gold1-llm1]]:")
    print(confusion_matrix(y_gold, y_llm))
    print(f"LLM label distribution:       "
          f"{dict(valid['llm_label'].value_counts())}")
    print(f"\nRow-level results saved -> {out_path}")
    print("\nGuide: >=0.85 proceed | 0.75-0.85 proceed with care | <0.75 skip")


if __name__ == "__main__":
    main()
