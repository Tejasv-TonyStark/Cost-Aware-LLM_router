"""Compare all routing strategies on the TEST split by simulation (no new LLM calls).

Outputs: results/comparison.csv, results/comparison.md, results/cost_vs_accuracy.png,
results/confusion_matrices.png, and an error analysis printed to the console.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config
from routers import policy

TIERS = config.TIERS
RES = ROOT / "results"

labels = pd.read_csv(ROOT / "data" / "labels.csv")
test = labels[labels.split == "test"].reset_index(drop=True)


# ---- each strategy = a tier per test question (+ router time per question) ----
strategies = {}
for t in TIERS:
    strategies[f"always_{t}"] = (pd.Series([t] * len(test)), pd.Series(0.0, index=test.index))
rng = np.random.default_rng(config.SEED)
strategies["random"] = (pd.Series(rng.choice(TIERS, len(test))), pd.Series(0.0, index=test.index))


def from_preds(df):
    m = test[["id"]].merge(df, on="id", how="left")
    if m.pred_tier.isna().any():
        raise ValueError(f"missing predictions for {m.pred_tier.isna().sum()} test ids")
    return m.pred_tier, m.router_ms


for name, file in [("rule_based", "rule_preds.csv"), ("ml_classifier", "ml_preds.csv")]:
    if (RES / file).exists():
        strategies[name] = from_preds(pd.read_csv(RES / file))

if (RES / "laya_preds.csv").exists():
    laya = pd.read_csv(RES / "laya_preds.csv")
    laya = laya[laya.split == "test"]
    for v in ["laya_zero_shot", "laya_finetuned"]:
        if (laya.variant == v).any():
            strategies[v] = from_preds(laya[laya.variant == v])
    pol_path = Path(policy.POLICY_PATH)
    if pol_path.exists():
        pol = policy.load_policy()
        sub = test[["id"]].merge(laya[laya.variant == pol["variant"]], on="id")
        pmax = sub[policy.PROB_COLS].max(axis=1)
        bumped = pd.Series([policy.apply(t, p, pol["threshold"]) for t, p in zip(sub.pred_tier, pmax)])
        strategies[f"{pol['variant']}+confidence"] = (bumped, sub.router_ms)

strategies["oracle"] = (test.label, pd.Series(0.0, index=test.index))


# ---- metrics ----
def outcome(tiers):
    """Per-question correctness, cost and model latency for the chosen tiers."""
    idx = np.arange(len(test))
    pick = lambda col: np.array([test[f"{col}_{t}"].iloc[i] for i, t in zip(idx, tiers)])
    return pick("correct").astype(float), pick("cost"), pick("latency")


large_cost = test.cost_large.sum()
rows, per_q = [], {}
for name, (tiers, rms) in strategies.items():
    tiers = list(tiers)
    correct, cost, lat = outcome(tiers)
    per_q[name] = correct
    rows.append({
        "strategy": name,
        "accuracy": correct.mean(),
        "total_cost": cost.sum(),
        "cost_pct_of_large": 100 * cost.sum() / large_cost,
        "avg_latency_ms": lat.mean() + np.mean(rms),
        "router_ms": np.mean(rms),
        "routing_accuracy": np.mean(np.array(tiers) == test.label.values),
        **{f"pct_{t}": 100 * np.mean(np.array(tiers) == t) for t in TIERS},
    })
comp = pd.DataFrame(rows)
comp.to_csv(RES / "comparison.csv", index=False)
# Markdown table without the optional "tabulate" dependency
r4 = comp.round(4).astype(str)
md = "| " + " | ".join(r4.columns) + " |\n|" + "---|" * len(r4.columns) + "\n"
md += "\n".join("| " + " | ".join(row) + " |" for row in r4.values)
(RES / "comparison.md").write_text(md, encoding="utf-8")
print(comp.round(3).to_string(index=False), "\n")


# ---- cost vs accuracy plot ----
fig, ax = plt.subplots(figsize=(8, 5.5))
placed = []  # label positions so far, to stack labels of points that sit close together
x_span = max(comp.cost_pct_of_large.max(), 1)
y_span = max(100 * (comp.accuracy.max() - comp.accuracy.min()), 1)
for _, r in comp.iterrows():
    x, y = r.cost_pct_of_large, 100 * r.accuracy
    base = r.strategy.startswith("always_") or r.strategy in ("random", "oracle")
    ax.scatter(x, y, s=60, color="#9aa0a6" if base else "#2b6cb0", edgecolor="white", linewidth=2, zorder=3)
    near = sum(abs(x - px) < 0.12 * x_span and abs(y - py) < 0.04 * y_span for px, py in placed)
    placed.append((x, y))
    ax.annotate(r.strategy, (x, y), xytext=(6, 4 + 11 * near), textcoords="offset points", fontsize=9,
                color="#333", arrowprops={"arrowstyle": "-", "color": "#bbb", "lw": 0.6} if near else None)
ax.set_xlabel("Cost (% of always_large)")
ax.set_ylabel("Final answer accuracy (%)")
ax.set_title(f"Cost vs accuracy on the test set (n={len(test)}); gray = baselines, blue = routers", fontsize=10)
ax.grid(alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(RES / "cost_vs_accuracy.png", dpi=150)


# ---- confusion matrices: rule_based vs best Laya variant ----
laya_name = next((n for n in ["laya_finetuned", "laya_zero_shot"] if n in strategies), None)
compare = [n for n in ["rule_based", "ml_classifier", laya_name] if n in strategies]
fig, axes = plt.subplots(1, len(compare), figsize=(4.2 * len(compare), 4))
for ax, name in zip(np.atleast_1d(axes), compare):
    cm = confusion_matrix(test.label, list(strategies[name][0]), labels=TIERS)
    ax.imshow(cm, cmap="Blues")
    n = len(TIERS)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(n), TIERS)
    ax.set_yticks(range(n), TIERS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true label")
    ax.set_title(name)
fig.tight_layout()
fig.savefig(RES / "confusion_matrices.png", dpi=150)


# ---- significance: paired bootstrap of the accuracy difference ----
def bootstrap_diff(a, b, n=2000):
    r = np.random.default_rng(config.SEED)
    m = len(per_q[a])  # number of test questions
    idx = r.integers(0, m, (n, m))
    d = per_q[a][idx].mean(1) - per_q[b][idx].mean(1)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


if laya_name and "rule_based" in strategies:
    L, R = comp.set_index("strategy").loc[laya_name], comp.set_index("strategy").loc["rule_based"]
    d, lo, hi = bootstrap_diff(laya_name, "rule_based")
    sig = "significant" if lo > 0 or hi < 0 else "NOT significant"
    print(f"HEADLINE: {laya_name} reached {100*L.accuracy:.1f}% accuracy at {L.cost_pct_of_large:.1f}% of "
          f"always_large cost, vs rule_based {100*R.accuracy:.1f}% at {R.cost_pct_of_large:.1f}% "
          f"(accuracy difference {100*d:+.1f} pts, 95% CI [{100*lo:+.1f}, {100*hi:+.1f}], {sig}).")
    for other in ["ml_classifier"]:
        if other in strategies:
            d, lo, hi = bootstrap_diff(laya_name, other)
            print(f"  vs {other}: {100*d:+.1f} pts, 95% CI [{100*lo:+.1f}, {100*hi:+.1f}]")

    # ---- error analysis ----
    lt, rt = list(strategies[laya_name][0]), list(strategies["rule_based"][0])
    t = test.assign(laya=lt, rule=rt)
    t["laya_ok"], t["rule_ok"] = per_q[laya_name], per_q["rule_based"]
    for title, mask in [(f"{laya_name} right, rules wrong", (t.laya_ok == 1) & (t.rule_ok == 0)),
                        (f"rules right, {laya_name} wrong", (t.laya_ok == 0) & (t.rule_ok == 1))]:
        sub = t[mask]
        print(f"\n--- {title}: {len(sub)} questions (by source: {sub.source.value_counts().to_dict()}) ---")
        for _, r in sub.head(5).iterrows():
            q = r.question.replace("\n", " ")[:110]
            print(f"[{r.source}] label={r.label} laya={r.laya} rule={r.rule} | {q}")
