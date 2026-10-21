#!/usr/bin/env python3
"""Fake news detector — proper Python/NLP version.

Trains TF-IDF + LogisticRegression on the bundled corpus in this folder
(corpus.jsonl: {"text": ..., "label": 0|1}) with a larger vocabulary and
character n-grams, and reports honest stratified holdout + 5-fold CV metrics.

The bundled corpus is a small curated starter — point --data at a bigger
open JSONL dataset (same schema) for real-world training. Open datasets that
need no account: the ISOT fake news set on GitHub mirrors, LIAR (TSV).

Usage:
    pip install -r requirements.txt
    python pipeline.py [--data corpus.jsonl]
"""
import argparse, json
import numpy as np

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    X = [r["text"] for r in rows]
    y = np.array([r["label"] for r in rows])
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="corpus.jsonl")
    args = ap.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
    from sklearn.pipeline import Pipeline

    X, y = load(args.data)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                  min_df=2, max_df=0.95, sublinear_tf=True,
                                  strip_accents="unicode")),
        ("clf", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced")),
    ])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    cv = cross_validate(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=7),
                        scoring=["accuracy", "f1"])
    metrics = {
        "data": args.data, "samples": len(y), "positive_rate": float(y.mean()),
        "pipeline": "TfidfVectorizer(word 1-2gram, min_df=2, sublinear) + LogisticRegression(C=2, balanced)",
        "holdout": {
            "accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
        },
        "cv5_accuracy": {"mean": round(float(cv["test_accuracy"].mean()), 4),
                         "std": round(float(cv["test_accuracy"].std()), 4)},
        "cv5_f1": {"mean": round(float(cv["test_f1"].mean()), 4),
                   "std": round(float(cv["test_f1"].std()), 4)},
        "vocab_size": len(pipe.named_steps["tfidf"].vocabulary_),
        "note": "Text-pattern model: flags writing style, not truth. Always verify claims against primary sources.",
    }
    import joblib
    joblib.dump(pipe, "model.joblib")
    json.dump(metrics, open("metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
