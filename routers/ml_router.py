"""Classic ML router: TF-IDF features + logistic regression.

Run this file to tune C with 5-fold cross-validation on TRAIN only, save
routers/ml_router.joblib, and write test predictions to results/ml_preds.csv.
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

MODEL_PATH = Path(__file__).resolve().parent / "ml_router.joblib"
TIERS = config.TIERS


def load_model():
    return joblib.load(MODEL_PATH)


def route(question, model):
    """Return (tier, {tier: probability})."""
    probs = model.predict_proba([question])[0]
    p = dict(zip(model.classes_, probs))
    return max(p, key=p.get), {t: float(p.get(t, 0.0)) for t in TIERS}


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "labels.csv")
    train, test = df[df.split == "train"], df[df.split == "test"]

    # class_weight="balanced" stops the model from only predicting the majority tier
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(class_weight="balanced", max_iter=2000))
    search = GridSearchCV(pipe, {"logisticregression__C": [0.1, 0.3, 1, 3, 10]}, cv=5, scoring="accuracy")
    search.fit(train.question, train.label)
    model = search.best_estimator_
    joblib.dump(model, MODEL_PATH)
    print(f"Best C={search.best_params_['logisticregression__C']} | CV train accuracy={search.best_score_:.3f}")

    rows = []
    for _id, q in zip(test.id, test.question):
        t0 = time.perf_counter()
        tier, p = route(q, model)
        rows.append({"id": _id, "variant": "ml_classifier", "pred_tier": tier,
                     **{f"p_{t}": p[t] for t in TIERS},
                     "router_ms": (time.perf_counter() - t0) * 1000})
    preds = pd.DataFrame(rows)
    preds.to_csv(ROOT / "results" / "ml_preds.csv", index=False)

    print(f"TEST routing accuracy: {(preds.pred_tier.values == test.label.values).mean():.3f} "
          f"| avg {preds.router_ms.mean():.3f} ms per route")
    print(pd.DataFrame(confusion_matrix(test.label, preds.pred_tier, labels=TIERS), index=TIERS, columns=TIERS))
