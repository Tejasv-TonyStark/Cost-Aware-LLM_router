"""Laya router: ask Laya a "choice" question with options small_model / medium_model / large_model.

API used (from the official model card and fine-tuning notebook, convaiinnovations/laya):
  laya.load(repo_id)                -> agent (base model from Hugging Face)
  laya.Agent(local_dir, device=...) -> agent (local fine-tuned checkpoint)
  agent.predict(state, questions)   -> {"answers": {qid: {"choice": key, "probabilities": {key: p}}}}

Run this file to:
  1. write data/laya_split.csv (TRAIN split into laya_train / validation 85/15, plus test)
  2. save validation + test predictions to results/laya_preds.csv for zero-shot, and for the
     fine-tuned variant too if routers/laya_checkpoint/ exists (both timed on this machine)
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

CHECKPOINT_DIR = Path(__file__).resolve().parent / "laya_checkpoint"  # fine-tuned model from Colab goes here
TIERS = config.TIERS

# What each option means; only the active tiers are offered to Laya
DESCRIPTIONS = {
    "small": "simple, short factual recall, an easy well-known fact, or a simple question",
    "medium": "moderate knowledge or a few reasoning steps",
    "large": "multi-step math, hard reasoning, or rare specialist knowledge",
}
QUESTIONS = {
    "tier": {
        "type": "choice",
        "instructions": "Which is the cheapest language model that will still answer this question correctly?",
        "criteria": {f"{t}_model": DESCRIPTIONS[t] for t in TIERS},
    }
}


def load(prefer_finetuned=True):
    import laya
    if prefer_finetuned and (CHECKPOINT_DIR / "rl_agent_config.json").exists():
        return laya.Agent(str(CHECKPOINT_DIR), device="cpu")
    return laya.load("convaiinnovations/laya")


def route(question, agent):
    """Return (tier, {tier: probability})."""
    ans = agent.predict({"question": question}, QUESTIONS)["answers"]["tier"]
    probs = {t: float(ans["probabilities"].get(f"{t}_model", 0.0)) for t in TIERS}
    return ans["choice"].replace("_model", ""), probs


def make_split(labels, seed=42):
    """Hold out 15% of TRAIN (stratified by label) as validation; test stays untouched."""
    from sklearn.model_selection import train_test_split
    train = labels[labels.split == "train"]
    tr, val = train_test_split(train, test_size=0.15, stratify=train.label, random_state=seed)
    split = pd.concat([tr[["id"]].assign(laya_split="laya_train"),
                       val[["id"]].assign(laya_split="validation"),
                       labels[labels.split == "test"][["id"]].assign(laya_split="test")])
    split.to_csv(ROOT / "data" / "laya_split.csv", index=False)
    return split


def predict_rows(agent, df, variant):
    rows = []
    for _id, q, s in zip(df.id, df.question, df.laya_split):
        t0 = time.perf_counter()
        tier, p = route(q, agent)
        rows.append({"id": _id, "variant": variant, "split": s, "pred_tier": tier,
                     **{f"p_{t}": p[t] for t in TIERS},
                     "router_ms": (time.perf_counter() - t0) * 1000})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    labels = pd.read_csv(ROOT / "data" / "labels.csv")
    split = make_split(labels)
    df = labels.merge(split, on="id")
    df = df[df.laya_split.isin(["validation", "test"])]

    # Both variants are predicted here, on the same machine, so router_ms is comparable
    variants = [("laya_zero_shot", False)]
    if (CHECKPOINT_DIR / "rl_agent_config.json").exists():
        variants.append(("laya_finetuned", True))
    else:
        print(f"No fine-tuned checkpoint in {CHECKPOINT_DIR} -> zero-shot only")

    all_preds = []
    for variant, finetuned in variants:
        agent = load(prefer_finetuned=finetuned)
        route(df.question.iloc[0], agent)  # warm-up so the first timing is fair
        preds = predict_rows(agent, df, variant)
        all_preds.append(preds)
        z = preds.merge(labels[["id", "label"]], on="id")
        for s in ["validation", "test"]:
            part = z[z.split == s]
            print(f"{variant} {s}: routing accuracy {(part.pred_tier == part.label).mean():.3f} "
                  f"| avg {part.router_ms.mean():.0f} ms | predicted: {part.pred_tier.value_counts().to_dict()}")
    pd.concat(all_preds).to_csv(ROOT / "results" / "laya_preds.csv", index=False)
