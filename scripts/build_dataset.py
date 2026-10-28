"""Build data/questions.jsonl: 900 questions (300 per source) with gold answers."""
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

# Make "import config" work when running from the scripts folder
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

N_PER_SOURCE = 300              # questions taken from each dataset
TRAIN_FRACTION = 0.70           # 70% train, 30% test
OUT_PATH = ROOT / "data" / "questions.jsonl"


# ---- prompt formats (what is sent to the LLM) ----
def gsm8k_prompt(question):
    return question + "\n\nSolve step by step. End your reply with 'Answer: <number>'."


def mmlu_prompt(question, choices):
    options = "\n".join(f"{'ABCD'[i]}) {choices[i]}" for i in range(4))
    return question + "\n" + options + "\n\nReply with only the letter of the correct option."


def trivia_prompt(question):
    return question + "\n\nAnswer in as few words as possible."


def sample_indices(n_total):
    """Same 300 random row numbers every run (fixed seed)."""
    return random.Random(config.SEED).sample(range(n_total), N_PER_SOURCE)


def build_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i in sample_indices(len(ds)):
        row = ds[i]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k_{i}", "source": "gsm8k", "question": row["question"],
                    "prompt": gsm8k_prompt(row["question"]), "gold": gold, "aliases": []})
    return out


def build_mmlu():
    ds = load_dataset("cais/mmlu", "all", split="test")
    out = []
    for i in sample_indices(len(ds)):
        row = ds[i]
        # Raw question text for routers includes the options, as a user would type it
        raw = row["question"] + "\n" + "\n".join(f"{'ABCD'[j]}) {c}" for j, c in enumerate(row["choices"]))
        out.append({"id": f"mmlu_{i}", "source": "mmlu", "question": raw,
                    "prompt": mmlu_prompt(row["question"], row["choices"]),
                    "gold": "ABCD"[row["answer"]], "aliases": []})
    return out


def build_trivia():
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    out = []
    for i in sample_indices(len(ds)):
        row = ds[i]
        ans = row["answer"]
        aliases = sorted(set([ans["value"]] + list(ans["aliases"])))
        out.append({"id": f"trivia_{i}", "source": "triviaqa", "question": row["question"],
                    "prompt": trivia_prompt(row["question"]), "gold": ans["value"], "aliases": aliases})
    return out


def assign_split(records):
    """70/30 train/test split, stratified by source (each source split separately)."""
    rng = random.Random(config.SEED)
    for source in {r["source"] for r in records}:
        group = [r for r in records if r["source"] == source]
        rng.shuffle(group)
        cut = int(len(group) * TRAIN_FRACTION)
        for k, r in enumerate(group):
            r["split"] = "train" if k < cut else "test"


if __name__ == "__main__":
    records = build_gsm8k() + build_mmlu() + build_trivia()
    assign_split(records)
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} questions to {OUT_PATH}")
    for s in ["gsm8k", "mmlu", "triviaqa"]:
        tr = sum(1 for r in records if r["source"] == s and r["split"] == "train")
        te = sum(1 for r in records if r["source"] == s and r["split"] == "test")
        print(f"  {s:9} train={tr} test={te}")
