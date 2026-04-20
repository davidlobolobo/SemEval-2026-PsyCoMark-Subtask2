# SemEval-2026 Task 10 — PsyCoMark Subtask 2: Conspiracy Detection

System submitted to SemEval-2026 Task 10 (PsyCoMark), Subtask 2: binary classification of conspiracy-related content.

**Author:** David Rodriguez Gutierrez

---

## Results

| Model | F1 (weighted) |
|---|---|
| TF-IDF + LR (ngram_range=(1,2)) | 0.65 |
| TF-IDF + LR (ngram_range=(1,4)) | 0.67 |
| TF-IDF + LR (ngram_range=(1,3)) | 0.68 |
| DistilRoBERTa-base (fine-tuned) | **0.75** |

---

## Repository Structure

```
├── baselines/
│   ├── load_jsonl.py                  # data loader (filters "Can't tell")
│   ├── run_tfidf.py                   # TF-IDF + LR, ngram_range=(1,2) → 0.65
│   ├── 08_run_tfidf_ngram4.py         # TF-IDF + LR, ngram_range=(1,4) → 0.67
│   └── 09_run_tfidf_ngram3_mindf2.py  # TF-IDF + LR, ngram_range=(1,3) → 0.68
├── transformer/
│   └── transformer_run.py             # DistilRoBERTa-base fine-tuning → 0.75
├── data/
│   └── README.md                      # data info (not redistributed)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place the official task data files at:
- `data/train.jsonl`
- `data/test.jsonl`

---

## Running the baselines

```bash
cd baselines
python run_tfidf.py
python 08_run_tfidf_ngram4.py
python 09_run_tfidf_ngram3_mindf2.py
```

## Running the transformer

```bash
cd transformer
python transformer_run.py
```

Trained on CPU. Expected training time: ~3 hours.

---

## Hyperparameters

**DistilRoBERTa-base**
- learning_rate: 2e-5
- num_train_epochs: 3
- per_device_train_batch_size: 8
- max_length: 384
- weight_decay: 0.01
- warmup_ratio: 0.06

**TF-IDF + LR — 0.68** (`09_run_tfidf_ngram3_mindf2.py`)
- ngram_range: (1, 3)
- min_df: 2
- sublinear_tf: True
- C: 3.0
- max_iter: 3500
- class_weight: balanced

**TF-IDF + LR — 0.67** (`08_run_tfidf_ngram4.py`)
- ngram_range: (1, 4)
- min_df: 3
- sublinear_tf: True
- C: 3.0
- max_iter: 3500
- class_weight: balanced

**TF-IDF + LR — 0.65** (`run_tfidf.py`)
- ngram_range: (1, 2)
- min_df: 3
- C: 1.0 (default)
- max_iter: 2000
