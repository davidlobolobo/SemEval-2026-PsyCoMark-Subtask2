"""
SemEval-2026 Task 10 — PsyCoMark Subtask 2
Author: David Rodriguez Gutierrez

Utility: light text cleaning applied before TF-IDF vectorization.
Replaces URLs, user mentions, and numbers with generic tokens.
"""

import re

URL_RE = re.compile(r"http\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
NUM_RE = re.compile(r"\d+")
WS_RE = re.compile(r"\s+")


def preprocess(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" <URL> ", text)
    text = USER_RE.sub(" <USER> ", text)
    text = NUM_RE.sub(" <NUM> ", text)
    text = WS_RE.sub(" ", text).strip()
    return text
