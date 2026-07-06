"""
Quick prediction demo — replaces testing_english.py / testing_tamil.py /
testing_malayalam.py.

Shows live 0/1 predictions (0 = Non-hope, 1 = Hope) from the base model and
the rule-adjusted model side by side, for built-in sample sentences or your
own text.

Usage (from anywhere):
    $ python evaluation/predict.py english
    $ python evaluation/predict.py tamil
    $ python evaluation/predict.py malayalam
    $ python evaluation/predict.py english --text "never lose hope" "no hope left"
    $ python evaluation/predict.py tamil --interactive     # type sentences live
"""

import argparse
import sys
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "training"))

from train_all import predict_with_rules  # noqa: E402

LABELS = {0: "Non-hope", 1: "Hope"}

SAMPLES = {
    "english": [
        "there is hope",
        "no hope left",
        "never lose hope",
        "not hope",
        "Just smile and stop being weak",
        "I don't love you",
    ],
    "tamil": [
        "எனக்கு நம்பிக்கை இருக்கிறது",      # I have hope -> Hope
        "நம்பிக்கை இல்லை",                  # There is no hope -> Non-hope
        "இது நல்ல நாள் வரும்",               # Good days will come -> Hope
        "நம்மால் முடியாது",                  # We cannot do it -> Non-hope
        "உற்சாகமாக இரு நண்பா",              # Stay enthusiastic friend -> Hope
    ],
    "malayalam": [
        "ആശ ഇല്ല",                          # No hope -> Non-hope
        "ഞങ്ങൾ വിജയിക്കും",                  # We will win -> Hope
        "നല്ല ദിവസം വരും",                    # Good days will come -> Hope
        "ഇത് സാധ്യമല്ല",                      # This is not possible -> Non-hope
    ],
}


def load_models(lang: str):
    models_dir = BASE_DIR / "models"
    base_path = models_dir / f"hope_{lang}_model.pkl"
    rules_path = models_dir / f"hope_{lang}_model_with_rules.pkl"
    if not base_path.exists():
        sys.exit(f"Model not found: {base_path} — run training/train_all.py")
    model = joblib.load(base_path)
    pos, neg = [], []
    if rules_path.exists():
        bundle = joblib.load(rules_path)
        pos, neg = bundle["pos_patterns"], bundle["neg_patterns"]
    return model, pos, neg


def show(model, pos, neg, texts):
    base = model.predict(texts)
    adjusted = predict_with_rules(model, texts, pos, neg)
    print(f"\n{'text':<50}{'base':>6}{'rules':>7}")
    print("-" * 66)
    for t, b, a in zip(texts, base, adjusted):
        mark = "  <- rule override" if a != b else ""
        shown = t if len(t) <= 47 else t[:44] + "..."
        print(f"{shown:<50}{b:>6}{a:>7}{mark}")
    print(f"\nBase predictions:          {list(map(int, base))}")
    print(f"Rule-adjusted predictions: {list(map(int, adjusted))}")
    print(f"(0 = {LABELS[0]}, 1 = {LABELS[1]})")


def main():
    parser = argparse.ArgumentParser(description="Live hope-speech predictions")
    parser.add_argument("language", choices=["english", "tamil", "malayalam"])
    parser.add_argument("--text", nargs="+", help="your own sentence(s)")
    parser.add_argument("--interactive", action="store_true",
                        help="type sentences one at a time (empty line to quit)")
    args = parser.parse_args()

    model, pos, neg = load_models(args.language)
    print(f"Loaded {args.language} model "
          f"({len(pos)} positive / {len(neg)} negative rule patterns)")

    if args.interactive:
        print("Type a sentence and press Enter (empty line to quit):")
        while True:
            line = input("> ").strip()
            if not line:
                break
            show(model, pos, neg, [line])
    else:
        texts = args.text if args.text else SAMPLES[args.language]
        show(model, pos, neg, texts)


if __name__ == "__main__":
    main()
