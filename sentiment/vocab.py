"""Vocabulary: maps tokens to integer ids and back.

Builds a frequency-ranked vocabulary from a stream of training tokens and
reserves slots for sentence-boundary and unknown markers.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<unk>"


def canonicalize_word(word: str, wordset: set | None = None) -> str:
    """Lowercase, fold digit sequences to ``DG`` tokens, and map OOV to ``<unk>``."""
    word = word.lower()
    if not any(c.isalpha() for c in word):
        folded = re.sub(r"\d", "DG", word)
        if folded.startswith("DG"):
            folded = folded.replace(",", "")
        word = folded
    if wordset is None or word in wordset:
        return word
    return UNK_TOKEN


class Vocabulary:
    def __init__(self, tokens: Iterable[str], size: int | None = 20000) -> None:
        counts = Counter(tokens)
        reserved = [START_TOKEN, END_TOKEN, UNK_TOKEN]
        top = counts.most_common(None if size is None else size - len(reserved))
        vocab = reserved + [w for w, _ in top]

        self.id_to_word = dict(enumerate(vocab))
        self.word_to_id = {w: i for i, w in self.id_to_word.items()}
        self.wordset = set(self.word_to_id)
        self.size = len(self.id_to_word)

        self.START_ID = self.word_to_id[START_TOKEN]
        self.END_ID = self.word_to_id[END_TOKEN]
        self.UNK_ID = self.word_to_id[UNK_TOKEN]

    def words_to_ids(self, words: Sequence[str]) -> List[int]:
        return [self.word_to_id.get(w, self.UNK_ID) for w in words]

    def ids_to_words(self, ids: Sequence[int]) -> List[str]:
        return [self.id_to_word[i] for i in ids]

    def ordered_words(self) -> List[str]:
        return [self.id_to_word[i] for i in range(self.size)]

    def encode(self, text: str) -> List[int]:
        tokens = [canonicalize_word(t, self.wordset) for t in text.split()]
        return self.words_to_ids(tokens)
