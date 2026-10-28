"""Grade every answer and label each question with the cheapest correct tier.

Output: data/labels.csv
"""
import json
import re
import string
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config


def grade_gsm8k(answer, gold):
    """Number after 'Answer:' (fallback: last number in the reply), compared numerically."""
    text = answer.replace(",", "").replace("$", "")
    m = re.findall(r"Answer:\s*\**\s*(-?\d+(?:\.\d+)?)", text, flags=re.I)
    nums = m or re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return False
    try:
        return abs(float(nums[-1]) - float(gold)) < 1e-6
    except ValueError:
        return False


def grade_mmlu(answer, gold):
    """First standalone letter A-D in the reply."""
    m = re.search(r"\b([ABCD])\b", answer)
    return bool(m) and m.group(1) == gold


def normalize(s):
    """Lowercase, drop punctuation and articles, collapse spaces (standard TriviaQA normalization)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def grade_trivia(answer, aliases):
    """Correct if any alias equals the answer, or appears as whole words inside it."""
    ans = f" {normalize(answer)} "
    for a in aliases:
        na = normalize(a)
        if na and (ans.strip() == na or f" {na} " in ans):
            return True
    return False


def grade(q, answer):
    if q["source"] == "gsm8k":
        return grade_gsm8k(answer, q["gold"])
    if q["source"] == "mmlu":
        return grade_mmlu(answer, q["gold"])
    return grade_trivia(answer, q["aliases"])


if __name__ == "__main__":
    questions = [json.loads(l) for l in open(ROOT / "data" / "questions.jsonl", encoding="utf-8")]
    raw = {}
    for t in config.TIERS:
        raw[t] = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "results" / f"raw_{t}.jsonl", encoding="utf-8")}

    rows = []
    for q in questions:
        if not all(q["id"] in raw[t] for t in config.TIERS):
            continue  # skip questions not yet answered by every tier
        row = {"id": q["id"], "source": q["source"], "split": q["split"], "question": q["question"]}
        for t in config.TIERS:
            r = raw[t][q["id"]]
            row[f"correct_{t}"] = grade(q, r["answer"])
            row[f"cost_{t}"] = r["cost"]
            row[f"latency_{t}"] = r["latency_ms"]
        # Label = cheapest tier that was correct; if none, large
        row["label"] = next((t for t in config.TIERS if row[f"correct_{t}"]), "large")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "labels.csv", index=False)
    print(f"Saved {len(df)} labeled questions (of {len(questions)}) to data/labels.csv\n")

    print("Accuracy per tier:")
    acc = df.groupby("source")[[f"correct_{t}" for t in config.TIERS]].mean()
    acc.loc["OVERALL"] = df[[f"correct_{t}" for t in config.TIERS]].mean()
    print((acc * 100).round(1).to_string(), "\n")

    print("Label distribution:")
    print(df["label"].value_counts().reindex(config.TIERS).to_string())
    print("\nNo tier correct:", int((~df[[f"correct_{t}" for t in config.TIERS]].any(axis=1)).sum()), "\n")

    # 10 random graded examples for a manual sanity check
    by_id = {q["id"]: q for q in questions}
    for _, r in df.sample(10, random_state=config.SEED).iterrows():
        q = by_id[r["id"]]
        ans = raw["small"][r["id"]]["answer"].replace("\n", " ")
        print(f"[{r['source']}] gold={q['gold']!r} | small said: {ans[-120:]!r} -> {r['correct_small']}")
