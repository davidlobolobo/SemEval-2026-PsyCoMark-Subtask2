"""
SemEval-2026 Task 10 — PsyCoMark Subtask 2
Author: David Rodriguez Gutierrez

TF-IDF (up to trigrams, min_df=2) with Logistic Regression.
Applies light text preprocessing before vectorization.
Codabench submission ID 515453 — F1 weighted: 0.68
"""

import json
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from load_jsonl import load_jsonl
from utils_preprocess import preprocess

train_ids, train_texts_raw, train_labels = load_jsonl("data/train.jsonl", "train")
test_ids, test_texts_raw, _ = load_jsonl("data/test.jsonl", "dev")

y_train = np.array(train_labels, dtype=int)
train_texts = [preprocess(t) for t in train_texts_raw]
test_texts = [preprocess(t) for t in test_texts_raw]

os.makedirs("submissions", exist_ok=True)

vectorizer = TfidfVectorizer(
    min_df=2,           # lower threshold than baseline: keeps more rare terms
    ngram_range=(1, 3), # unigrams, bigrams, trigrams
    sublinear_tf=True,  # apply log normalization to term frequencies
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

clf = LogisticRegression(
    C=3.0,                    # less regularization than default (C=1.0)
    max_iter=3500,            # extended to ensure convergence
    class_weight="balanced"   # compensates for Yes/No class imbalance
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

out_path = "submissions/submission_tfidf_ngram3.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for _id, pred in zip(test_ids, y_pred):
        label = "Yes" if pred == 1 else "No"
        f.write(json.dumps({"_id": _id, "conspiracy": label}) + "\n")

print(f"Submission saved → {out_path}")
