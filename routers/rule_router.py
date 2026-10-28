"""Rule-based router: hand-written features -> weighted score -> thresholds -> tier.

One threshold per tier boundary (1 for small/large, 2 for small/medium/large).

Run this file to grid-search weights/thresholds on TRAIN, save rules.json,
and write test-set predictions to results/rule_preds.csv.
"""
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

RULES_PATH = Path(__file__).resolve().parent / "rules.json"
TIERS = config.TIERS

MATH_WORDS = r"\b(how many|how much|total|sum|percent|average|cost|price|each|per|times|twice|half|remaining|left|profit|rate|ratio)\b"
REASONING_WORDS = r"\b(why|explain|because|therefore|which of the following|best|most likely|implies|infer|conclude|except)\b"
MULTISTEP = r"\b(then|after|before|if|while|later|finally|each|every|remaining|first|second|third)\b"
SHORT_FACT = r"^(who|what|when|where|which|in which|in what|name)\b"
FEATURES = ["words", "numbers", "math", "reasoning", "multistep", "short_fact", "multiple_choice"]


def features(question):
    """Turn a question into 7 numbers in roughly [0, 1]. short_fact is negative (it means 'easier')."""
    q = question.lower()
    words = len(q.split())
    return [
        min(words, 120) / 120,
        min(len(re.findall(r"\d+(?:\.\d+)?", q)), 10) / 10,
        min(len(re.findall(MATH_WORDS, q)), 5) / 5,
        min(len(re.findall(REASONING_WORDS, q)), 3) / 3,
        min(len(re.findall(MULTISTEP, q)), 6) / 6,
        -1.0 if (re.match(SHORT_FACT, q) and words <= 20) else 0.0,
        1.0 if re.search(r"^[a-d]\)", q, flags=re.M) else 0.0,
    ]


def route(question, rules):
    """Weighted score; each threshold the score reaches moves the question one tier up."""
    score = float(np.dot(rules["weights"], features(question)))
    return rules["tiers"][sum(score >= t for t in rules["thresholds"])]


def load_rules():
    return json.loads(RULES_PATH.read_text())


def grid_search(X, y):
    """Try every weight combo in {0,1,2}^7 and thresholds from score quantiles; keep the best train accuracy."""
    best = (-1, None)
    for w in itertools.product([0, 1, 2], repeat=len(FEATURES)):
        scores = X @ np.array(w)
        cands = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 19)))
        for ths in itertools.combinations_with_replacement(cands, len(TIERS) - 1):
            pred = (scores[:, None] >= np.array(ths)[None, :]).sum(axis=1)
            acc = (pred == y).mean()
            if acc > best[0]:
                best = (acc, {"tiers": TIERS, "weights": list(w), "thresholds": [float(t) for t in ths]})
    return best


if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix

    df = pd.read_csv(ROOT / "data" / "labels.csv")
    train, test = df[df.split == "train"], df[df.split == "test"]
    X = np.array([features(q) for q in train.question])
    y = train.label.map(TIERS.index).values

    acc, rules = grid_search(X, y)
    rules["features"] = FEATURES
    RULES_PATH.write_text(json.dumps(rules, indent=2))
    print(f"Best TRAIN routing accuracy: {acc:.3f}  -> saved {RULES_PATH.name}: {rules}")

    # Evaluate once on TEST (no tuning here)
    preds, ms = [], []
    for q in test.question:
        t0 = time.perf_counter()
        preds.append(route(q, rules))
        ms.append((time.perf_counter() - t0) * 1000)
    test_acc = (np.array(preds) == test.label.values).mean()
    print(f"TEST routing accuracy: {test_acc:.3f} | avg {np.mean(ms):.4f} ms per route")
    print("Confusion matrix (rows = true label, cols = predicted):")
    print(pd.DataFrame(confusion_matrix(test.label, preds, labels=TIERS), index=TIERS, columns=TIERS))

    pd.DataFrame({"id": test.id, "variant": "rule_based", "pred_tier": preds, "router_ms": ms}).to_csv(
        ROOT / "results" / "rule_preds.csv", index=False)
