"""
SemEval-2026 Task 10 — PsyCoMark Subtask 2
Author: David Rodriguez Gutierrez

Utility: loads PsyCoMark JSONL files.
For training splits, examples labeled "Can't tell" are discarded entirely.
For test splits, only ids and texts are returned (no labels available).
"""

import json


def load_jsonl(path, split):
    ids, texts, labels = [], [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            lab = obj.get("conspiracy", None)

            if split == "train":
                if lab == "Can't tell":
                    continue  # excluded: ambiguous annotation
                if lab == "Yes":
                    ids.append(obj["_id"])
                    texts.append(obj["text"])
                    labels.append(1)
                elif lab == "No":
                    ids.append(obj["_id"])
                    texts.append(obj["text"])
                    labels.append(0)
            else:
                ids.append(obj["_id"])
                texts.append(obj["text"])
                labels.append(None)

    return ids, texts, labels
