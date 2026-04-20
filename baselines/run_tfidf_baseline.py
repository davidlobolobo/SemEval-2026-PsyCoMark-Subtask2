"""
SemEval-2026 Task 10 — PsyCoMark Subtask 2
Author: David Rodriguez Gutierrez

Baseline: TF-IDF (unigrams + bigrams) with Logistic Regression.
Codabench submission ID 515446 — F1 weighted: 0.65
"""

import json
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from load_jsonl import load_jsonl

train_ids, train_texts, train_labels = load_jsonl("data/train.jsonl", "train")
test_ids, test_texts, _ = load_jsonl("data/test.jsonl", "dev")

train_labels = np.array(train_labels, dtype=int)

os.makedirs("submissions", exist_ok=True)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        min_df=3,           # ignore terms appearing in fewer than 3 documents
        ngram_range=(1, 2), # unigrams and bigrams
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        max_iter=2000       # increased from default to ensure convergence
    ))
])

model.fit(train_texts, train_labels)
test_pred = model.predict(test_texts)

out_path = "submissions/submission_tfidf_baseline.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for _id, pred in zip(test_ids, test_pred):
        label = "Yes" if int(pred) == 1 else "No"
        f.write(json.dumps({"_id": _id, "conspiracy": label}) + "\n")

print(f"Submission saved → {out_path}")
