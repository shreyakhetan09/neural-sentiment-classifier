"""Pretrained word embeddings.

Reads GloVe vectors for the words in a vocabulary and assembles an embedding
matrix indexed by token id. Words without a pretrained vector (and the special
tokens) are initialised from a small Gaussian.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .vocab import Vocabulary


def load_glove(path: str, vocab: Vocabulary) -> Dict[str, np.ndarray]:
    """Load only the GloVe vectors whose word is in ``vocab``."""
    wanted = vocab.wordset
    vectors: Dict[str, np.ndarray] = {}
    with open(path, "rb") as handle:
        for line in handle:
            try:
                parts = line.decode("utf-8").rstrip().split(" ")
                word = parts[0]
                if word in wanted:
                    vectors[word] = np.asarray(parts[1:], dtype=np.float32)
            except Exception:
                continue  # skip malformed lines (URLs, bad bytes)
    return vectors


def build_embedding_matrix(
    vocab: Vocabulary,
    dim: int = 300,
    glove: Dict[str, np.ndarray] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Create an ``(vocab_size, dim)`` matrix, filling rows from GloVe when known."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(scale=0.1, size=(vocab.size, dim)).astype(np.float32)
    if glove:
        for word_id, word in enumerate(vocab.ordered_words()):
            vec = glove.get(word)
            if vec is not None and len(vec) == dim:
                matrix[word_id] = vec
    return matrix
