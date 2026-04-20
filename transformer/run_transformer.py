"""
SemEval-2026 Task 10 — PsyCoMark Subtask 2
Author: David Rodriguez Gutierrez

Fine-tuning of DistilRoBERTa-base for binary conspiracy detection.
Trained on CPU. Examples labeled "Can't tell" are excluded from training.
Codabench submission ID 515564 — F1 weighted: 0.75
"""

import json
import numpy as np
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "distilroberta-base"
TRAIN_PATH = "data/train.jsonl"
TEST_PATH = "data/test.jsonl"
OUT_FILE = "submissions/submission_transformer.jsonl"


def load_jsonl(path, has_labels):
    ids, texts, labels = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if has_labels:
                lab = obj.get("conspiracy")
                if lab == "Can't tell":
                    continue  # excluded: ambiguous annotation
                if lab == "Yes":
                    label = 1
                elif lab == "No":
                    label = 0
                else:
                    continue
                ids.append(obj["_id"])
                texts.append(obj["text"])
                labels.append(label)
            else:
                ids.append(obj["_id"])
                texts.append(obj["text"])
    return ids, texts, labels


print("Loading data...")
_, train_texts, train_labels = load_jsonl(TRAIN_PATH, has_labels=True)
test_ids, test_texts, _ = load_jsonl(TEST_PATH, has_labels=False)

train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_ds = Dataset.from_dict({"text": test_texts})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=384  # extended context window (default is 128)
    )


train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask"])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

training_args = TrainingArguments(
    output_dir="./tmp",
    num_train_epochs=3,               # full passes over training data
    per_device_train_batch_size=8,    # batch size per device
    learning_rate=2e-5,               # standard fine-tuning learning rate
    weight_decay=0.01,                # L2 regularization
    warmup_ratio=0.06,                # linear warmup over first 6% of steps
    logging_steps=100,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
)

print("Training transformer (CPU)...")
trainer.train()

print("Generating predictions on test set...")
preds = trainer.predict(test_ds).predictions
y_pred = np.argmax(preds, axis=1)

os.makedirs("submissions", exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for _id, pred in zip(test_ids, y_pred):
        label = "Yes" if pred == 1 else "No"
        f.write(json.dumps({"_id": _id, "conspiracy": label}) + "\n")

print(f"Submission saved → {OUT_FILE}")
