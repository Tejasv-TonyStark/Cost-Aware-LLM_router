"""Confidence rule: if the router's top probability is below a threshold, move one tier up.

Run this file to pick the threshold on Laya's VALIDATION predictions, save
policy.json, and report how many TEST questions get bumped.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

POLICY_PATH = Path(__file__).resolve().parent / "policy.json"
TIERS = config.TIERS
PROB_COLS = [f"p_{t}" for t in TIERS]
COST_WEIGHT = 0.2  # objective = accuracy - COST_WEIGHT * (cost / cost of always_large)


def apply(pred_tier, max_prob, threshold):
    """Bump one tier up (the top tier stays where it is) when the router is unsure."""
    if max_prob < threshold:
        return TIERS[min(TIERS.index(pred_tier) + 1, len(TIERS) - 1)]
    return pred_tier


def simulate(preds, labels, threshold):
    """Return (accuracy, cost fraction of always_large, n bumped) for a threshold."""
    m = preds.merge(labels, on="id")
    pmax = m[PROB_COLS].max(axis=1)
    tiers = [apply(t, p, threshold) for t, p in zip(m.pred_tier, pmax)]
    correct = np.mean([m[f"correct_{t}"].iloc[i] for i, t in enumerate(tiers)])
    cost = sum(m[f"cost_{t}"].iloc[i] for i, t in enumerate(tiers)) / m.cost_large.sum()
    return correct, cost, int(sum(t != p for t, p in zip(tiers, m.pred_tier)))


def load_policy():
    return json.loads(POLICY_PATH.read_text())


if __name__ == "__main__":
    labels = pd.read_csv(ROOT / "data" / "labels.csv")
    preds = pd.read_csv(ROOT / "results" / "laya_preds.csv")
    # Use the fine-tuned variant if it exists, otherwise zero-shot
    variant = "laya_finetuned" if (preds.variant == "laya_finetuned").any() else "laya_zero_shot"
    preds = preds[preds.variant == variant]
    val, test = preds[preds.split == "validation"], preds[preds.split == "test"]

    best = None
    print(f"Tuning on VALIDATION ({variant}, n={len(val)}):")
    for th in np.round(np.arange(0.40, 0.901, 0.05), 2):
        acc, cost, bumped = simulate(val, labels, th)
        score = acc - COST_WEIGHT * cost
        print(f"  threshold={th:.2f}  acc={acc:.3f}  cost={cost:.1%}  bumped={bumped}  score={score:.3f}")
        if best is None or score > best[0]:
            best = (score, th)

    policy = {"variant": variant, "threshold": float(best[1]), "cost_weight": COST_WEIGHT}
    POLICY_PATH.write_text(json.dumps(policy, indent=2))
    acc, cost, bumped = simulate(test, labels, best[1])
    print(f"Saved {POLICY_PATH.name}: {policy}")
    print(f"TEST: {bumped} of {len(test)} questions bumped up one tier")
